# UCF101 Video Fusion Sim with E8 Triality + Sigma Visualization
# Run on Colab with GPU (Pro recommended for full subset)

# Install if needed
!pip install decord av torch torchvision matplotlib numpy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import decord
from decord import VideoReader, cpu
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CONFIG – optimized for Colab
# ────────────────────────────────────────────────
triality = 3
dim = 768  # flattened frame dim (64x64x3 = 12288, embedded to 768)
latent_dim = 8
seq_len = 32  # frames per clip
batch_size = 16  # small for free T4, increase on Pro
epochs = 10000  # test — increase on Pro
lr = 5e-5
use_amp = True
use_checkpoint = True

# ────────────────────────────────────────────────
# Real UCF101 sample video (download one clip — use full dataset on Pro)
# ────────────────────────────────────────────────
!wget https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/v_Basketball_g01_c01.avi -O ucf101_sample.avi

vr = VideoReader('ucf101_sample.avi', ctx=cpu(0))
frames = vr.get_batch(range(0, seq_len * batch_size, seq_len)).asnumpy()  # batch of clips
frames = torch.from_numpy(frames).to(device).float() / 255.0  # normalize

# Resize for memory
frames = F.interpolate(frames.permute(0, 3, 1, 2), size=(64, 64)).permute(0, 2, 3, 1)
frames = frames.view(batch_size, seq_len, -1)  # (batch, seq, dim=12288)

# Simple projection to lower dim (for efficiency)
proj = nn.Linear(12288, dim).to(device)
real_data = proj(frames)

# Apply masking (40–70% missing frames)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = proj(frames)  # clean for reconstruction

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
class VideoCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(torch.tensor(step, device=device, dtype=torch.float32) * 0.006 * 2 * torch.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model
class E8UCF101Fusion(nn.Module):
    def __init__(self, depth=64):
        super().__init__()
        self.cycle = VideoCycleBlock()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

model = E8UCF101Fusion().to(device)
model = torch.compile(model)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []
abl_prec_hist = []
abl_ent_hist = []

# Training loop (triality)
for epoch in range(epochs):
    opt.zero_grad()
    recon = model(real_data, epoch)
    loss = loss_fn(recon, target)
    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        ent = -recon * torch.log(recon + 1e-12)
        p = recon.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)

# Ablation (no triality, sequential attention proxy)
model_ablation = E8UCF101Fusion(depth=64)
model_ablation.cycle = nn.Identity()  # disable triality
model_ablation = model_ablation.to(device)
opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)

for epoch in range(epochs):
    opt_ablation.zero_grad()
    recon = model_ablation(real_data, epoch)
    loss = loss_fn(recon, target)
    loss.backward()
    opt_ablation.step()

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

# Sigma Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(prec_hist, label='E8 Triality')
plt.plot(abl_prec_hist, label='Ablation', linestyle='--')
plt.title("Precision Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Final Sigma Precision: {sigma_prec:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white", ec="black"))

plt.subplot(1,2,2)
plt.plot(ent_hist, label='E8 Triality')
plt.plot(abl_ent_hist, label='Ablation', linestyle='--')
plt.title("Entropy Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Final Sigma Entropy: {sigma_ent:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white", ec="black"))

plt.tight_layout()
plt.savefig("ucf101_sigma_visualization.png")
plt.show()

# Frame Visualization
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()
    clean = target.cpu()

num_vis = 8
fig, axes = plt.subplots(3, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[0, i].view(64, 64, 3).numpy())
    axes[0, i].set_title("Masked Frame")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon[0, i].view(64, 64, 3).clip(0,1).numpy())
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

    axes[2, i].imshow(clean[0, i].view(64, 64, 3).numpy())
    axes[2, i].set_title("Clean Frame")
    axes[2, i].axis('off')

plt.suptitle("UCF101 Real Video Frames: Masked vs Triality Reconstructed")
plt.tight_layout()
plt.show()

print("Sigma visualization saved as ucf101_sigma_visualization.png")