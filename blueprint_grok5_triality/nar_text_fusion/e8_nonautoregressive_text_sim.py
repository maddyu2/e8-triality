# e8_nonautoregressive_text_sim.py
# Purpose: Parallel triality fusion for sparse text generation (non-autoregressive)
#          Compare vs autoregressive baseline on latency, perplexity/accuracy, entropy

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Optional

# === Import your TrialityCycleBlock ===
# Adjust path based on where you placed it
try:
    from ...grok_architecture.components.triality_cycle_block import TrialityCycleBlock
except ImportError:
    # Fallback: copy-paste or adjust sys.path
    # For now, assume it's in the same folder or paste the class here
    class TrialityCycleBlock(nn.Module):
        def __init__(self, dim: int = 128, hidden: Optional[int] = None, dropout: float = 0.0):
            super().__init__()
            hidden = hidden or dim * 4
            self.proj_vector   = nn.Linear(dim, hidden)
            self.proj_spinor   = nn.Linear(dim, hidden)
            self.proj_cospinor = nn.Linear(dim, hidden)
            self.merge = nn.Linear(3 * hidden, dim)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(dim)
            self.gate = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            p1 = F.gelu(self.proj_vector(x))
            p2 = F.gelu(self.proj_spinor(x))
            p3 = F.gelu(self.proj_cospinor(x))
            fused = torch.cat([p1, p2, p3], dim=-1)
            out = self.merge(fused)
            out = self.dropout(out)
            out = self.norm(residual + out * self.gate.sigmoid())
            return out

# === Tiny GPT-like model (for simplicity) ===
class TinyGPT(nn.Module):
    def __init__(self, vocab_size=10000, dim=128, depth=4, heads=4, use_triality_nar=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, dim))  # max seq ~512
        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
            if use_triality_nar:
                ffn = TrialityCycleBlock(dim=dim, hidden=dim*4)
            else:
                ffn = nn.Sequential(
                    nn.Linear(dim, dim*4),
                    nn.GELU(),
                    nn.Linear(dim*4, dim),
                    nn.Dropout(0.1)
                )
            self.layers.append(nn.ModuleList([nn.LayerNorm(dim), attn, nn.LayerNorm(dim), ffn]))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.use_triality_nar = use_triality_nar

    def forward(self, x, mask=None):
        B, T = x.shape
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed[:, :T, :]
        x = tok_emb + pos_emb

        for ln1, attn, ln2, ffn in self.layers:
            x_norm = ln1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
            x = x + attn_out
            x = x + ffn(ln2(x))

        x = self.norm(x)
        logits = self.head(x)
        return logits

# Simple entropy (nats) on logits
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    return entropy

# Mask 50% of tokens randomly (for sparse/noisy input)
def random_mask(inputs, mask_ratio=0.5):
    mask = torch.rand_like(inputs.float()) < mask_ratio
    masked_inputs = inputs.clone()
    masked_inputs[mask] = 0  # assume 0 is pad/mask token for simplicity
    return masked_inputs, mask

# === Dummy text dataset (tiny wiki-like sentences for proxy) ===
# In real run: replace with WikiText-2 / PTB via torchtext or huggingface datasets
sentences = [
    "the quick brown fox jumps over the lazy dog",
    "e8 triality enables parallel fusion in sparse regimes",
    "grok scaling with geometric priors is promising",
    # Add 100–500 more dummy or real short sentences...
] * 50  # repeat for ~few thousand tokens

vocab = sorted(set(" ".join(sentences)))
vocab_size = len(vocab) + 2  # + pad, unk
word2idx = {w: i+2 for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
word2idx['<PAD>'] = 0
word2idx['<UNK>'] = 1

def tokenize(texts, max_len=64):
    seqs = []
    for text in texts:
        tokens = [word2idx.get(w, 1) for w in text.split()]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        seqs.append(tokens)
    return torch.tensor(seqs)

data = tokenize(sentences)
dataloader = DataLoader(data, batch_size=32, shuffle=True)

# === Training / Eval loop ===
def run_experiment(use_nar=False, epochs=3):
    device = torch.device("cpu")
    model = TinyGPT(vocab_size=vocab_size, use_triality_nar=use_nar).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            if use_nar:
                inputs, mask = random_mask(batch, 0.5)
                logits = model(inputs)  # parallel full sequence
                loss = criterion(logits.view(-1, vocab_size), batch.view(-1))
            else:
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | {'NAR' if use_nar else 'AR'} Loss: {avg_loss:.4f}")

    latency = time.time() - start_time
    print(f"{'NAR' if use_nar else 'AR'} Total time: {latency:.2f}s")

    # Quick eval: entropy on a masked batch
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))[:16].to(device)
        masked, _ = random_mask(test_batch, 0.5)
        logits = model(masked)
        ent = compute_entropy(logits)
        ppl = torch.exp(torch.tensor(ent)).item()  # rough proxy perplexity

    print(f"{'NAR' if use_nar else 'AR'} Entropy: {ent:.4f} nats | Approx PPL: {ppl:.2f}")
    return latency, ent, ppl

# Run both
print("=== Autoregressive Baseline ===")
ar_latency, ar_ent, ar_ppl = run_experiment(use_nar=False, epochs=3)

print("\n=== Non-Autoregressive + Triality ===")
nar_latency, nar_ent, nar_ppl = run_experiment(use_nar=True, epochs=3)

print("\n=== Comparison ===")
print(f"Latency delta: NAR {nar_latency/ar_latency:.2f}x faster")
print(f"Entropy delta: {nar_ent - ar_ent:+.4f} nats")
print(f"PPL delta:     {nar_ppl - ar_ppl:+.2f}")