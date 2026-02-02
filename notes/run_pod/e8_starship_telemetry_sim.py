# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for Colab/Gemini speed (fast epochs, no OOM)
triality = 3
dim = 192  # reduced
latent_dim = 8
seq_len = 1024  # time steps (telemetry samples)
batch_size = 8  # reduced for Colab
epochs = 20000
lr = 5e-5
use_amp = True
use_checkpoint = True

# Synthetic Starship telemetry proxy (velocity, altitude, acceleration + noise/occlusion)
# Mimics real Falcon/Starship profiles (boost, coast, reentry)
time = torch.linspace(0, 600, seq_len, device=device)  # 10 min flight proxy
telemetry = []
for b in range(batch_size):
    # Velocity (m/s) - boost + coast + reentry
    vel = 0.5 * torch.sin(time * 0.05) + 100 * time / seq_len  # ramp up
    vel = vel + torch.randn_like(time) * 10  # noise

    # Altitude (km)
    alt = 100 * torch.sin(time * 0.03) + 50 * time / seq_len
    alt = alt + torch.randn_like(time) * 5

    # Acceleration (g)
    acc = torch.sin(time * 0.1) * 3 + torch.randn_like(time) * 0.5

    # Stack features
    sample = torch.stack([vel, alt, acc], dim=-1)  # (seq_len, 3)
    telemetry.append(sample)

telemetry = torch.stack(telemetry).to(device)  # (batch, seq_len, features)

# Project to dim (batch, seq_len, dim)
proj = nn.Linear(telemetry.shape[-1], dim).to(device)
real_data = proj(telemetry)

# Apply masking (40–70% missing readings — sensor occlusion/noise)
missing_rate = torch.linspace(0.4, 0.7, batch_size, device=device).view(batch_size, 1, 1)
mask = torch.rand_like(real_data) < missing_rate
real_data[mask] = 0

target = proj(telemetry)

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

# Triality Cycle Block (detached step + fixed pump broadcast)
class StarshipCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]  # (seq_len, 8)
        low_dim = self.proj(pos_emb)  # (seq_len, dim//3)
        emb = low_dim.repeat(1, triality)  # (seq_len, dim)
        # Detached pump scalar
        with torch.no_grad():
            pump_scalar = 0.8 * math.sin(step * 0.006 * 2 * math.pi)
        pump = torch.full((1, x.shape[1], 1), pump_scalar, device=device)  # (1, seq_len, 1)
        emb_broadcast = emb.unsqueeze(0)  # (1, seq_len, dim)
        x_rot1 = x * (emb_broadcast.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=1) * emb_broadcast.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=1) * emb_broadcast.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model (reduced depth for speed)
class E8StarshipFusion(nn.Module):
    def __init__(self, depth=16):
        super().__init__()
        self.cycle = StarshipCycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            if use_checkpoint:
                attn, _ = checkpoint(layer, x, x, x, use_reentrant=False)
            else:
                attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

model = E8StarshipFusion().to(device)

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

# Heatmap + Line Plot Visualization (telemetry curves)
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()
    clean = target.cpu()

# First batch sample
masked = original[0].numpy()  # (seq_len, dim)
reconstructed = recon[0].numpy()
clean_tele = clean[0].numpy()

time = np.arange(seq_len)

# Heatmaps (dim as "features")
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
im1 = axes[0].imshow(masked.T, aspect='auto', cmap='viridis')
axes[0].set_title("Masked Telemetry (Noise/Occlusion)")
axes[0].set_ylabel("Features")
fig.colorbar(im1, ax=axes[0], orientation='horizontal')

im2 = axes[1].imshow(reconstructed.T, aspect='auto', cmap='viridis')
axes[1].set_title("Triality Reconstructed Telemetry")
axes[1].set_ylabel("Features")
fig.colorbar(im2, ax=axes[1], orientation='horizontal')

im3 = axes[2].imshow(clean_tele.T, aspect='auto', cmap='viridis')
axes[2].set_title("Clean Telemetry (Ground Truth)")
axes[2].set_ylabel("Features")
axes[2].set_xlabel("Time Steps")
fig.colorbar(im3, ax=axes[2], orientation='horizontal')

plt.suptitle("Starship Telemetry Proxy: Heatmap Visualization")
plt.tight_layout()
plt.show()

# Line Plots (top 8 features as telemetry curves)
num_features = 8
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

for c in range(num_features):
    axes[0].plot(time, masked[:, c], label=f'Feat {c}' if c == 0 else None, alpha=0.7)
axes[0].set_title("Masked Telemetry Features")
axes[0].set_ylabel("Value")
axes[0].grid(True)
axes[0].legend()

for c in range(num_features):
    axes[1].plot(time, reconstructed[:, c], label=f'Feat {c}' if c == 0 else None, alpha=0.7)
axes[1].set_title("Triality Reconstructed Telemetry Features")
axes[1].set_ylabel("Value")
axes[1].grid(True)

for c in range(num_features):
    axes[2].plot(time, clean_tele[:, c], label=f'Feat {c}' if c == 0 else None, alpha=0.7)
axes[2].set_title("Clean Telemetry Features")
axes[2].set_ylabel("Value")
axes[2].set_xlabel("Time Steps")
axes[2].grid(True)

plt.suptitle("Starship Telemetry Proxy: Line Plot Visualization (Top Features)")
plt.tight_layout()
plt.show()

print("Heatmap and line plot visualizations displayed above")