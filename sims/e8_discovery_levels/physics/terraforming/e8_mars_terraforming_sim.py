# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy pandas

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for Pro A100
triality = 3
dim = 768
latent_dim = 8
seq_len = 512  # time steps (daily readings proxy)
batch_size = 32
epochs = 20000
lr = 5e-5
use_amp = True

# Real Curiosity Mars weather CSV (The Pudding cleaned NASA data — public, stable)
url = "https://raw.githubusercontent.com/the-pudding/data/master/mars-weather/mars-weather.csv"
df = pd.read_csv(url)

# Proxy features (temp, pressure, wind — normalize)
features = df[['min_temp', 'max_temp', 'pressure', 'wind_speed']].dropna()
features = (features - features.min()) / (features.max() - features.min() + 1e-6)  # normalize
data = torch.from_numpy(features.values).float().to(device)

# Pad/crop to batch * seq_len
data = data[:batch_size * seq_len]
data = data.view(batch_size, seq_len, -1)  # (batch, seq_len, features)

# Project to dim
proj = nn.Linear(data.shape[-1], dim).to(device)
real_data = proj(data)

# Apply masking (40–70% missing readings — sparse sensors)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = proj(data)

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
class TerraformCycleBlock(nn.Module):
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
class E8TerraformingFusion(nn.Module):
    def __init__(self, depth=64):
        super().__init__()
        self.cycle = TerraformCycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

model = E8TerraformingFusion().to(device)
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

# Visualization (sparse vs reconstructed planetary readings proxy)
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()

num_vis = 8
fig, axes = plt.subplots(2, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[i].numpy(), cmap='viridis', aspect='auto')
    axes[0, i].set_title("Masked Readings")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon[i].numpy(), cmap='viridis', aspect='auto')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

plt.suptitle("Mars Terraforming Proxy: Masked vs Triality Reconstructed Sensor Readings")
plt.tight_layout()
plt.show()

print("Visualization displayed above")