# First cell: Install dependencies (run once — fast)
!pip install torch matplotlib numpy torchvision

# Second cell: The sim code (with visualization added at end)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed (fast epochs)
triality = 3
dim = 384
latent_dim = 8
patch_size = 4  # for 28x28 MNIST-like (49 patches)
image_size = 28
num_patches = (image_size // patch_size) ** 2
epochs = 5000  # fast for visualization demo
batch_size = 64
lr = 5e-5
use_amp = True
use_checkpoint = True

# MNIST proxy (real images — small/fast)
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=16, shuffle=False)  # small for viz

# Patch embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

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

# Triality Cycle Block (detached pump scalar)
class ViTCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        with torch.no_grad():
            pump_scalar = 0.8 * math.sin(step * 0.006 * 2 * math.pi)
        pump = torch.full((1, x.shape[1], 1), pump_scalar, device=device)
        emb_broadcast = emb.unsqueeze(0)
        x_rot1 = x * (emb_broadcast.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=1) * emb_broadcast.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=1) * emb_broadcast.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Dummy cycle for ablation
class DummyCycle(nn.Module):
    def forward(self, x, step=None):
        return x

# E8 ViT Model with ablation support
class E8ViT(nn.Module):
    def __init__(self, depth=16, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cycle = ViTCycleBlock() if use_triality else DummyCycle()
        self.blocks = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, image_size * image_size)  # pixel reconstruction

    def forward(self, x, step):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        if self.use_triality:
            x = self.cycle(x, step)

        for block in self.blocks:
            attn, _ = block(x, x, x)
            x = x + attn
            x = self.norm(x)

        x = x[:, 1:]  # remove cls
        x = self.head(x)
        x = x.view(x.shape[0], 1, image_size, image_size)
        return x

# Models
model = E8ViT(use_triality=True).to(device)
model_ablation = E8ViT(use_triality=False).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

loss_fn = nn.MSELoss()

loss_hist = []
loss_abl_hist = []

# Training
for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)
    opt_ablation.zero_grad(set_to_none=True)

    for images, _ in train_loader:
        images = images.to(device)
        # High masking (70–90%)
        mask_rate = torch.linspace(0.7, 0.9, images.shape[0], device=device).view(-1, 1, 1, 1)
        mask = torch.rand_like(images) < mask_rate
        masked_images = images.clone()
        masked_images[mask] = 0

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
            recon = model(masked_images, epoch)
            loss = loss_fn(recon, images)

            recon_abl = model_ablation(masked_images, epoch)
            loss_abl = loss_fn(recon_abl, images)

        scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler_ablation.scale(loss_abl).backward() if use_amp else loss_abl.backward()

    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    scaler_ablation.unscale_(opt_ablation) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model_ablation.parameters(), 1e6)
    scaler_ablation.step(opt_ablation) if use_amp else opt_ablation.step()
    scaler_ablation.update() if use_amp else None

    loss_hist.append(loss.item())
    loss_abl_hist.append(loss_abl.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Triality Loss {loss.item():.6f} | Ablation Loss {loss_abl.item():.6f}")

# Final Sigma Test
triality_mean = np.mean(loss_hist)
abl_mean = np.mean(loss_abl_hist)
std = np.std(loss_hist + loss_abl_hist)
sigma = (abl_mean - triality_mean) / std if std > 0 else 0

print(f"Final Sigma (Triality vs Ablation): {sigma:.2f} (higher = triality advantage)")

# Visualization of reconstructed images
model.eval()
model_ablation.eval()

with torch.no_grad():
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    # High masking for demo
    mask = torch.rand_like(test_images) < 0.8  # 80% masked
    masked_images = test_images.clone()
    masked_images[mask] = 0

    recon = model(masked_images, 0)
    recon_abl = model_ablation(masked_images, 0)

    # To numpy for plot
    orig = test_images.cpu().numpy()[:8].squeeze()
    masked = masked_images.cpu().numpy()[:8].squeeze()
    tri = recon.cpu().numpy()[:8].squeeze()
    abl = recon_abl.cpu().numpy()[:8].squeeze()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(8):
        axes[0, i].imshow(orig[i], cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(masked[i], cmap='gray')
        axes[1, i].set_title("Masked (80%)")
        axes[1, i].axis('off')

        axes[2, i].imshow(tri[i], cmap='gray')
        axes[2, i].set_title("Triality Recon")
        axes[2, i].axis('off')

        axes[3, i].imshow(abl[i], cmap='gray')
        axes[3, i].set_title("Ablation Recon")
        axes[3, i].axis('off')

    plt.suptitle("E8 Triality Image Reconstruction Visualization")
    plt.tight_layout()
    plt.show()

print("Visualization displayed — triality coherence cosmic!")

print("Sim complete — epochs + sigma test + visualization done")