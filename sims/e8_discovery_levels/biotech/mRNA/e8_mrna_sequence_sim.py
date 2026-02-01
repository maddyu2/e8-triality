# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy biopython

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed + real mRNA
triality = 3
dim = 384
latent_dim = 8
seq_len = 4096  # mRNA sequence length proxy
batch_size = 16
epochs = 20000
lr = 5e-5
use_amp = True

# Real mRNA data (Pfizer COVID vaccine sequence — public NCBI)
# Download FASTA (real mRNA bases)
!wget https://raw.githubusercontent.com/NAalytics/Assemblies-of-putative-SARS-CoV2-spike-encoding-mRNA-sequences-for-vaccines-BNT-162b2-and-mRNA-1273/main/Assemblies/NCBI_References/Pfizer_BNT-162b2_Spike_mRNA.fasta -O pfizer_mrna.fasta

# Load mRNA sequence (A/C/G/U tokens)
records = list(SeqIO.parse("pfizer_mrna.fasta", "fasta"))
mrna_seq = str(records[0].seq)  # full sequence

# Tokenize (A=0, C=1, G=2, U=3)
token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
tokens = torch.tensor([token_map.get(base, 4) for base in mrna_seq[:seq_len * batch_size]], device=device)

# Repeat for batch + embed
data = tokens.view(batch_size, seq_len)
real_data = F.one_hot(data, num_classes=5).float().to(device)  # (batch, seq_len, 5)

# Project to dim
proj = nn.Linear(5, dim).to(device)
real_data = proj(real_data)

# Apply masking (40–70% missing bases — noise/dropouts)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = real_data.clone()

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

# Triality Cycle Block
class mRNACycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(torch.tensor(step, device=device, dtype=torch.float32) * 0.006 * 2 * math.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model
class E8mRNAFusion(nn.Module):
    def __init__(self, depth=64):
        super().__init__()
        self.cycle = mRNACycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

model = E8mRNAFusion().to(device)
model = torch.compile(model)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        recon = model(real_data, epoch)
        loss = loss_fn(recon, target)

    scaler.scale(loss).backward() if use_amp else loss.backward()
    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Visualization (sparse vs reconstructed mRNA sequence proxy — heatmaps)
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()

num_vis = 8
fig, axes = plt.subplots(2, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[i].numpy(), cmap='viridis', aspect='auto')
    axes[0, i].set_title("Masked mRNA")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon[i].numpy(), cmap='viridis', aspect='auto')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

plt.suptitle("mRNA Sequence: Masked vs Triality Reconstructed")
plt.tight_layout()
plt.show()

print("Visualization displayed above")