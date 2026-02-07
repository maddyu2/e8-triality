# First cell: Keep-alive + installs (run once — prevents disconnects)
from IPython.display import display, Javascript
display(Javascript('''
function ClickConnect(){
  console.log("Keeping alive"); 
  document.querySelector("colab-connect-button")?.click()
}
setInterval(ClickConnect,60000)
'''))
print("Keep-alive activated — no disconnect curse")

!pip install torch matplotlib numpy torchvision

# Second cell: The sim code (optimized — 5000 epochs, larger batch, single backward)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
import numpy as np
from contextlib import nullcontext
import math
import os
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed (fast epochs)
triality = 3
dim = 384
latent_dim = 8
patch_size = 4
image_size = 28  # MNIST proxy for vision
num_patches = (image_size // patch_size) ** 2
seq_len = num_patches + 1  # + cls token
batch_size = 128
epochs = 5000  # reduced for fast run (sigma trend visible)
lr = 5e-5
use_amp = True
use_checkpoint = True

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "vit_multimodal_checkpoint.pth")

# MNIST proxy for vision + synthetic text features for multimodal
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Synthetic text features (aligned "captions" for multimodal)
features_text = 192
text_data = torch.randn(batch_size, seq_len, features_text, device=device) * 0.5  # proxy tokens

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
class TrialityCycleBlock(nn.Module):
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

# Patch embedding for vision
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# E8 ViT Multimodal Model with ablation support
class E8ViTMultimodal(nn.Module):
    def __init__(self, depth=16, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.patch_embed = PatchEmbed()
        self.text_proj = nn.Linear(features_text, dim)  # project text features
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cycle = TrialityCycleBlock() if use_triality else DummyCycle()
        self.blocks = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, image_size * image_size)  # reconstruction head (vision proxy)

    def forward(self, vision, text, step):
        # Vision path
        v = self.patch_embed(vision)
        cls_tokens = self.cls_token.expand(v.shape[0], -1, -1)
        v = torch.cat((cls_tokens, v), dim=1)
        v = v + self.pos_embed

        # Text path (project + fake seq_len=1)
        t = self.text_proj(text.mean(dim=1, keepdim=True))  # simple avg for proxy

        # Concat vision + text tokens
        x = torch.cat([v, t], dim=1)

        if self.use_triality:
            x = self.cycle(x, step)

        for block in self.blocks:
            attn, _ = block(x, x, x)
            x = x + attn
            x = self.norm(x)

        # Reconstruction from vision tokens
        x = x[:, 1:num_patches+1]  # vision patches
        x = self.head(x)
        x = x.view(x.shape[0], 1, image_size, image_size)
        return x

# Models
model = E8ViTMultimodal(use_triality=True).to(device)
model_ablation = E8ViTMultimodal(use_triality=False).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

loss_fn = nn.MSELoss()

loss_hist = []
loss_abl_hist = []

start_epoch = 0

# Load checkpoint if exists (resume on disconnect)
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model_ablation.load_state_dict(checkpoint['model_ablation'])
    opt.load_state_dict(checkpoint['opt'])
    opt_ablation.load_state_dict(checkpoint['opt_ablation'])
    scaler.load_state_dict(checkpoint['scaler'])
    scaler_ablation.load_state_dict(checkpoint['scaler_ablation'])
    start_epoch = checkpoint['epoch'] + 1
    loss_hist = checkpoint['loss_hist']
    loss_abl_hist = checkpoint['loss_abl_hist']
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, epochs):
    opt.zero_grad(set_to_none=True)
    opt_ablation.zero_grad(set_to_none=True)

    for images, _ in loader:
        images = images.to(device)
        # Synthetic text features (aligned to batch)
        text = torch.randn(batch_size, seq_len, features_text, device=device) * 0.5

        # High masking on vision (70–90%)
        mask_rate = torch.linspace(0.7, 0.9, images.shape[0], device=device).view(-1, 1, 1, 1)
        mask = torch.rand_like(images) < mask_rate
        masked_images = images.clone()
        masked_images[mask] = 0

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
            recon = model(masked_images, text, epoch)
            loss = loss_fn(recon, images)

            recon_abl = model_ablation(masked_images, text, epoch)
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

    # Checkpoint every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'model_ablation': model_ablation.state_dict(),
            'opt': opt.state_dict(),
            'opt_ablation': opt_ablation.state_dict(),
            'scaler': scaler.state_dict(),
            'scaler_ablation': scaler_ablation.state_dict(),
            'loss_hist': loss_hist,
            'loss_abl_hist': loss_abl_hist,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

# Final Sigma Test
triality_mean = np.mean(loss_hist)
abl_mean = np.mean(loss_abl_hist)
std = np.std(loss_hist + loss_abl_hist)
sigma = (abl_mean - triality_mean) / std if std > 0 else 0

print(f"Final Sigma (Triality vs Ablation): {sigma:.2f} (higher = triality advantage)")

# Loss Curves Visualization
plt.figure(figsize=(12, 6))
plt.plot(loss_hist, label='Triality Loss', color='blue')
plt.plot(loss_abl_hist, label='Ablation Loss', color='orange')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('E8 Triality vs Ablation Loss Curves — Vision-Text Multimodal')
plt.legend()
plt.grid(True)
plt.show()

print("Sim complete — loss curves plotted!")