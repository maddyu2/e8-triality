# Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy av

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

# CONFIG – optimized for speed + stability
triality = 3
dim = 384
latent_dim = 8
seq_len = 16  # frames per clip (reduced for speed)
batch_size = 8  # reduced
epochs = 20000
lr = 5e-5
use_amp = True
use_checkpoint = True

# Synthetic video proxy (real motion patterns — moving objects + occlusion)
# Generate simple moving sine wave + noise (proxy for action coherence)
t = torch.linspace(0, 2*math.pi, seq_len, device=device)
clips = []
for b in range(batch_size):
    clip = []
    for f in range(seq_len):
        # Moving pattern + noise
        frame = torch.sin(t[f] + torch.randn(128, 128, device=device) * 0.1)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)  # normalize
        frame = frame.unsqueeze(0).repeat(3, 1, 1)  # RGB
        clip.append(frame)
    clip = torch.stack(clip)  # (seq_len, C, H, W)
    clips.append(clip)

clips = torch.stack(clips).to(device)  # (batch, seq_len, C, H, W)

# Flatten + project to dim
real_data = clips.flatten(2).permute(0, 2, 1)  # (batch, pixels*frames, C)
real_data = F.pad(real_data, (0, dim - real_data.shape[-1]))

target = real_data.clone()  # clean before masking

# Apply masking (40–70% missing frames/pixels — occlusion/noise)
missing_rate = torch.linspace(0.4, 0.7, batch_size, device=device).view(batch_size, 1, 1)
mask = torch.rand_like(real_data) < missing_rate  # broadcasts correctly
real_data[mask] = 0

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
class VideoCycleBlock(nn.Module):
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
class E8VideoFusion(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.cycle = VideoCycleBlock()
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

model = E8VideoFusion().to(device)

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

# Visualization (masked vs reconstructed video frames — synthetic motion proxy)
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()
    clean = target.cpu()

num_vis = 8
fig, axes = plt.subplots(3, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    frame_idx = i % seq_len
    masked_frame = original[0, frame_idx].view(3, 128, 128).permute(1,2,0).numpy()
    recon_frame = recon[0, frame_idx].view(3, 128, 128).permute(1,2,0).clip(0,1).numpy()
    clean_frame = clean[0, frame_idx].view(3, 128, 128).permute(1,2,0).numpy()

    axes[0, i].imshow(masked_frame)
    axes[0, i].set_title("Masked Frame")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon_frame)
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

    axes[2, i].imshow(clean_frame)
    axes[2, i].set_title("Clean Frame")
    axes[2, i].axis('off')

plt.suptitle("Synthetic Video Motion Proxy: Masked vs Triality Reconstructed Frames")
plt.tight_layout()
plt.show()

print("Visualization displayed above")