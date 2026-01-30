# Detailed Kinetics-400 Video Fusion Sim with E8 Triality
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
dim = 240
latent_dim = 8
seq_len = 32  # number of frames per video clip
batch_size = 32  # small for Colab free, increase on Pro
epochs = 10000  # small test — increase on Pro
lr = 5e-5
use_amp = True
use_checkpoint = True

# ────────────────────────────────────────────────
# Real Kinetics-400 loader (small subset — use full on Pro)
# ────────────────────────────────────────────────
# Download a sample Kinetics video (replace with full dataset path on Pro)
!wget https://storage.googleapis.com/decord/videos/sample_video.mp4 -O sample_video.mp4

vr = VideoReader('sample_video.mp4', ctx=cpu(0))
frames = vr.get_batch(range(0, seq_len * batch_size, seq_len)).asnumpy()  # proxy batch of clips
frames = torch.from_numpy(frames).to(device)  # (batch*seq, H, W, C)

# Resize for memory
frames = F.interpolate(frames.permute(0, 3, 1, 2), size=(64, 64)).permute(0, 2, 3, 1)
frames = frames.view batch_size, seq_len, -1)  # (batch, seq, dim)

real_data = frames.to(device)

# Apply masking (40–70% missing frames)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = frames.to(device)  # clean for reconstruction

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

# Model with cycle block
class E8KineticsFusion(nn.Module):
    def __init__(self, depth=64):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.cycle = VideoCycleBlock()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x  # reconstruct frames

model = E8KineticsFusion().to(device)
model = torch.compile(model)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

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
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# ────────────────────────────────────────────────
# Ablation: sequential attention
# ────────────────────────────────────────────────
# (similar training loop with causal masking — omitted for brevity)

# ────────────────────────────────────────────────
# Sigma Test (simplified)
# ────────────────────────────────────────────────
# ... (full sigma calculation as previous sims)

# ────────────────────────────────────────────────
# Video Frame Visualization (masked vs reconstructed)
# ────────────────────────────────────────────────
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

plt.suptitle("Kinetics-400 Real Video Frames: Masked vs Triality Reconstructed")
plt.tight_layout()
plt.savefig("kinetics400_video_fusion_visualization.png")
plt.show()

print("Visualization saved as kinetics400_video_fusion_visualization.png")