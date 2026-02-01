# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy datasets transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed + comparison
triality = 3
dim = 384
latent_dim = 8
seq_len = 256  # reduced for speed
batch_size = 16
epochs = 10000  # fast convergence for comparison
lr = 5e-5
use_amp = True
use_checkpoint = True

# Real CodeParrot data (clean GitHub Python subset — HF streaming)
dataset = load_dataset("codeparrot/github-code", languages=["Python"], split="train", streaming=True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Get batch of tokenized code sequences
codes = []
for i, sample in enumerate(dataset):
    if i >= batch_size:
        break
    code = sample["code"]
    tokens = tokenizer(code, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")["input_ids"].squeeze()
    codes.append(tokens)

codes = torch.stack(codes).to(device)  # (batch, seq_len)

# Apply masking (40–70% missing tokens for E8)
missing_rate = torch.linspace(0.4, 0.7, batch_size, device=device).view(batch_size, 1)
mask = torch.rand_like(codes.float()) < missing_rate
masked_codes = codes.clone()
masked_codes[mask] = tokenizer.pad_token_id  # mask for E8

# E8 roots – precompute
def get_e8_roots():
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for signs in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = torch.zeros(8)
                v[i] = signs[0]; v[j] = signs[1]
                roots.append(v); roots.append(-v)
    for signs in range(1 << 8):
        v = torch.tensor([(1 if (signs & (1<<k)) else -1) for k in range(8)], dtype=torch.float32) * 0.5
        if bin(signs).count('1') % 2 == 0:
            roots.append(v); roots.append(-v)
    roots = torch.stack(roots[:240])
    return roots / roots.norm(dim=-1, keepdim=True)

e8_roots = get_e8_roots().to(device)

# Triality Cycle Block (detached step)
class CodeCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        step_float = float(step)
        pump = 0.8 * torch.sin(torch.tensor(step_float, device=device) * 0.006 * 2 * math.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# E8 Model (reduced depth for speed)
class E8CodeFusion(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.embed = nn.Embedding(tokenizer.vocab_size, dim)
        self.cycle = CodeCycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, tokenizer.vocab_size)

    def forward(self, x, step):
        x = self.embed(x)
        x = self.cycle(x, step)
        for layer in self.layers:
            if use_checkpoint:
                attn, _ = checkpoint(layer, x, x, x, use_reentrant=False)
            else:
                attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return self.head(x)

# GPT-2 baseline (frozen for comparison)
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2.eval()  # no training, just inference comparison

# E8 model
model = E8CodeFusion().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Metrics history
e8_loss_hist = []
gpt2_loss_hist = []

for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        # E8 triality
        logits_e8 = model(masked_codes, epoch)
        loss_e8 = loss_fn(logits_e8.view(-1, tokenizer.vocab_size), codes.view(-1))

        # GPT-2 baseline (causal LM on masked — shift for next token prediction)
        with torch.no_grad():
            shifted_codes = codes[:, :-1]
            labels = codes[:, 1:]
            logits_gpt2 = gpt2(shifted_codes).logits
            loss_gpt2 = loss_fn(logits_gpt2.view(-1, tokenizer.vocab_size), labels.view(-1))

    scaler.scale(loss_e8).backward() if use_amp else loss_e8.backward()
    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    e8_loss_hist.append(loss_e8.item())
    gpt2_loss_hist.append(loss_gpt2.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | E8 Loss {loss_e8.item():.6f} | GPT-2 Loss {loss_gpt2.item():.6f}")

# Sigma Test (E8 vs GPT-2 on loss — lower better)
e8_mean = np.mean(e8_loss_hist)
gpt2_mean = np.mean(gpt2_loss_hist)
std = np.std(e8_loss_hist + gpt2_loss_hist)
sigma = (gpt2_mean - e8_mean) / std if std > 0 else 0  # positive = E8 better

print(f"Final Sigma (E8 vs GPT-2): {sigma:.2f} (higher = E8 advantage)")

# Visualization (loss curves)
plt.figure(figsize=(12,6))
plt.plot(e8_loss_hist, label='E8 Triality')
plt.plot(gpt2_loss_hist, label='GPT-2 Baseline')
plt.title("E8 Triality vs GPT-2 Code Generation Loss")
plt.xlabel("Epoch / 500")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma: {sigma:.2f} (E8 advantage)", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig("e8_vs_gpt2_code_loss_comparison.png")
plt.show()

print("Visualization saved as e8_vs_gpt2_code_loss_comparison.png")