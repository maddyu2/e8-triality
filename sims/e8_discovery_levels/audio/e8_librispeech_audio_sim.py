# First cell: Install dependencies (run once)
!apt-get update -qq
!apt-get install -y libsndfile1  # soundfile backend support
!pip install torch torchaudio librosa matplotlib numpy

# Second cell: The sim code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from torchaudio.datasets import LIBRISPEECH
import librosa
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
seq_len = 64
batch_size = 32
epochs = 20000
lr = 5e-5
use_amp = True
use_checkpoint = True

# LibriSpeech (dev-clean subset — real speech audio, soundfile backend)
dataset = LIBRISPEECH(root="./", url="dev-clean", download=True)

# Extract waveform → melspectrogram (librosa for stability)
def waveform_to_melspec(waveform, sr=16000, n_mels=128):
    waveform = waveform.squeeze().numpy()
    spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=400, hop_length=160)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = torch.from_numpy(spec).float()
    if spec.shape[1] > seq_len:
        spec = spec[:, :seq_len]
    else:
        pad = seq_len - spec.shape[1]
        spec = F.pad(spec, (0, pad))
    return spec  # (freq, time)

# Get batch of melspectrograms
specs = []
for i in range(batch_size):
    waveform, sample_rate, *_ = dataset[i % len(dataset)]  # safe unpack
    spec = waveform_to_melspec(waveform, sample_rate)
    specs.append(spec)

specs = torch.stack(specs)  # (batch, freq, seq_len)
specs = specs.permute(0, 2, 1)  # (batch, seq_len, freq)

# Project to dim
proj = nn.Linear(specs.shape[-1], dim).to(device)
real_data = proj(specs.to(device))

# Apply masking (40–70% missing frames)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = proj(specs.to(device))

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
class AudioCycleBlock(nn.Module):
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
class E8AudioFusion(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.cycle = AudioCycleBlock()
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

model = E8AudioFusion().to(device)

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

# Visualization (masked vs reconstructed melspectrograms)
with torch.no_grad():
    recon = model(real_data, 0).cpu()
    original = real_data.cpu()
    clean = target.cpu()

num_vis = 8
fig, axes = plt.subplots(3, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[0, i].view(128, -1).numpy(), cmap='viridis', aspect='auto')
    axes[0, i].set_title("Masked Melspec")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon[0, i].view(128, -1).clip(0,1).numpy(), cmap='viridis', aspect='auto')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

    axes[2, i].imshow(clean[0, i].view(128, -1).numpy(), cmap='viridis', aspect='auto')
    axes[2, i].set_title("Clean Melspec")
    axes[2, i].axis('off')

plt.suptitle("LibriSpeech Real Audio: Masked vs Triality Reconstructed Melspectrograms")
plt.tight_layout()
plt.show()

print("Visualization displayed above")