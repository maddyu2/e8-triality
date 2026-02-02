# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy datasets transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed + stability
triality = 3
dim = 384
latent_dim = 8
seq_len = 512
batch_size = 16
epochs = 20000
lr = 5e-5
use_amp = True
use_checkpoint = True

# Real code data (bigcode/the-stack-dedup Python subset — modern HF, no script error)
dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get batch of tokenized code sequences
codes = []
for i, sample in enumerate(dataset):
    if i >= batch_size:
        break
    code = sample["content"]
    tokens = tokenizer(code, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"].squeeze()
    if tokens.shape[0] < seq_len:
        tokens = F.pad(tokens, (0, seq_len - tokens.shape[0]))
    codes.append(tokens)

codes = torch.stack(codes).to(device)  # (batch, seq_len)

# Embed tokens to dim
embed = nn.Embedding(tokenizer.vocab_size, dim).to(device)
real_data = embed(codes)

# Apply masking (40–70% missing tokens)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = embed(codes)

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
        step_float = float(step)  # detached
        pump = 0.8 * torch.sin(torch.tensor(step_float, device=device) * 0.006 * 2 * math.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model (reduced depth for speed)
class E8CodeFusion(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.cycle = CodeCycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, tokenizer.vocab_size)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            if use_checkpoint:
                attn, _ = checkpoint(layer, x, x, x, use_reentrant=False)
            else:
                attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return self.head(x)

model = E8CodeFusion().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        logits = model(real_data, epoch)
        loss = loss_fn(logits.view(-1, tokenizer.vocab_size), codes.view(-1))

    scaler.scale(loss).backward() if use_amp else loss.backward()
    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Visualization (masked vs reconstructed code tokens proxy — heatmaps)
with torch.no_grad():
    logits = model(real_data, 0)
    recon_tokens = logits.argmax(dim=-1).cpu()
    original_tokens = codes.cpu()

num_vis = 8
fig, axes = plt.subplots(2, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original_tokens[i].unsqueeze(0).numpy(), cmap='viridis', aspect='auto')
    axes[0, i].set_title("Masked Code Tokens")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon_tokens[i].unsqueeze(0).numpy(), cmap='viridis', aspect='auto')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

plt.suptitle("CodeParrot Real Code: Masked vs Triality Reconstructed Tokens")
plt.tight_layout()
plt.show()

print("Visualization displayed above")