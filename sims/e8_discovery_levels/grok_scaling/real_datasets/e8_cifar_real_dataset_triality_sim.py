import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
triality = 3
dim = 3072  # CIFAR flattened (32x32x3)
latent_dim = 8
seq_len = 1  # one image per "sequence"
batch_size = 128  # adjust down if memory low
epochs = 10000  # small for laptop test (scale up on cloud)
lr = 5e-5

# Real CIFAR-10 loader
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Get one batch for demo (replace with full loop for training)
images, _ = next(iter(trainloader))
real_data = images.view(batch_size, seq_len, dim).to(device)

# Apply masking (40–70%)
missing = torch.linspace(0.4, 0.7, batch_size, device=device).view(batch_size, 1, 1)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

# Target (clean images for reconstruction loss)
target = images.view(batch_size, seq_len, dim).to(device)

# E8 roots (same as before)
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

# Triality Cycle Block (same)
class CifarCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * torch.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model (same as before)
class E8CifarRealTriality(nn.Module):
    def __init__(self, depth=128):  # reduced depth for laptop
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.cycle_block = CifarCycleBlock()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)  # reconstruct full image

    def forward(self, x, step):
        x = self.cycle_block(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x  # direct reconstruction

# Training loop (simplified for laptop)
model = E8CifarRealTriality().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    opt.zero_grad()
    recon = model(real_data, epoch)
    loss = loss_fn(recon, target)
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Visualization (original masked vs reconstructed)
with torch.no_grad():
    recon = model(real_data, 0).view(batch_size, 3, 32, 32).cpu()
    original = real_data.view(batch_size, 3, 32, 32).cpu()
    target_vis = target.view(batch_size, 3, 32, 32).cpu()

num_vis = 8
fig, axes = plt.subplots(3, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[i].permute(1, 2, 0))
    axes[0, i].set_title("Masked")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon[i].permute(1, 2, 0).clip(0,1))
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

    axes[2, i].imshow(target_vis[i].permute(1, 2, 0))
    axes[2, i].set_title("Original Clean")
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig("cifar_real_reconstruction.png")
plt.show()