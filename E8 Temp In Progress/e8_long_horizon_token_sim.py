# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CONFIG – optimized for Pro A100 (long-horizon proxy)
# ────────────────────────────────────────────────
triality = 3
dim = 240
latent_dim = 8
seq_len = 131072  # 128k token proxy (long-horizon)
batch_size = 32   # memory-safe on Pro
epochs = 100000   # 100k steps for eternal lock test
lr = 5e-5
use_amp = True

# Long-horizon sparse text proxy (linspace + noise)
long_seq = torch.linspace(0.1, 0.9, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
real_data = long_seq.repeat(1, 1, dim) * torch.randn(batch_size, seq_len, dim, device=device) * 0.002

# Apply masking (40–70% missing tokens)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = long_seq.repeat(1, 1, dim)  # clean coherence target

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
class LongHorizonCycleBlock(nn.Module):
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

# Model with ablation support
class E8LongHorizon(nn.Module):
    def __init__(self, depth=64, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.cycle = LongHorizonCycleBlock() if use_triality else nn.Identity()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

# Training – optimized with AMP
model = E8LongHorizon(use_triality=True).to(device)
model = torch.compile(model)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

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
        ent = -recon * torch.log(recon + 1e-12)
        p = recon.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Ablation: triality disabled
model_ablation = E8LongHorizon(use_triality=False).to(device)
opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

abl_prec_hist = []
abl_ent_hist = []

for epoch in range(epochs):
    opt_ablation.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        recon = model_ablation(real_data, epoch)
        loss = loss_fn(recon, target)

    scaler_ablation.scale(loss).backward() if use_amp else loss.backward()
    scaler_ablation.unscale_(opt_ablation) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model_ablation.parameters(), 1e6)
    scaler_ablation.step(opt_ablation) if use_amp else opt_ablation.step()
    scaler_ablation.update() if use_amp else None

    if epoch % 500 == 0:
        ent = -recon * torch.log(recon + 1e-12)
        p = recon.mean().item()
        e = ent.mean().item()
        abl_prec_hist.append(p)
        abl_ent_hist.append(e)

# Sigma Test
e8_prec_mean = np.mean(prec_hist)
abl_prec_mean = np.mean(abl_prec_hist)
prec_std = np.std(np.concatenate([prec_hist, abl_prec_hist]))
sigma_prec = (e8_prec_mean - abl_prec_mean) / prec_std if prec_std > 0 else 0

e8_ent_mean = np.mean(ent_hist)
abl_ent_mean = np.mean(abl_ent_hist)
ent_std = np.std(np.concatenate([ent_hist, abl_ent_hist]))
sigma_ent = (abl_ent_mean - e8_ent_mean) / ent_std if ent_std > 0 else 0

print(f"Sigma Precision: {sigma_prec:.2f}")
print(f"Sigma Entropy: {sigma_ent:.2f}")

# Visualization (long-horizon precision/entropy curves)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(prec_hist, label='E8 Triality')
plt.plot(abl_prec_hist, label='Ablation', linestyle='--')
plt.title("Precision Convergence (128k Token Proxy)")
plt.xlabel("Epoch / 500")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Precision: {sigma_prec:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.subplot(1,2,2)
plt.plot(ent_hist, label='E8 Triality')
plt.plot(abl_ent_hist, label='Ablation', linestyle='--')
plt.title("Entropy Convergence (128k Token Proxy)")
plt.xlabel("Epoch / 500")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Entropy: {sigma_ent:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig("long_horizon_token_sigma_visualization.png")
plt.show()

print("Sigma visualization saved as long_horizon_token_sigma_visualization.png")