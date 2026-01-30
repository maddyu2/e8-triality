# Quantized Edge Inference Sim with E8 Triality
# Run on Colab Pro A100 (High-RAM + GPU)

!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import torch.quantization
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for Pro A100 (edge proxy)
triality = 3
dim = 768
latent_dim = 8
seq_len = 32
batch_size = 128  # large for Pro
epochs = 10000
lr = 5e-5
use_amp = True

# Sparse multimodal proxy (40–70% masking)
missing = torch.linspace(0.4, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
real_data = torch.randn(batch_size, seq_len, dim, device=device)
mask = torch.rand_like(real_data) < missing
real_data[mask] = 0

target = torch.randn(batch_size, seq_len, dim, device=device)  # reconstruction target

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
class QuantCycleBlock(nn.Module):
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
class E8QuantEdgeInference(nn.Module):
    def __init__(self, depth=64, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.cycle = QuantCycleBlock() if use_triality else nn.Identity()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

# Full precision model
model_fp = E8QuantEdgeInference(use_triality=True).to(device)
model_fp = torch.compile(model_fp)

# Quantized model (dynamic 8-bit)
model_quant = E8QuantEdgeInference(use_triality=True).to(device)
model_quant = torch.quantization.quantize_dynamic(
    model_quant, {nn.Linear, nn.MultiheadAttention}, dtype=torch.qint8
)

opt_fp = torch.optim.AdamW(model_fp.parameters(), lr=lr)
opt_quant = torch.optim.AdamW(model_quant.parameters(), lr=lr)

scaler_fp = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
scaler_quant = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.MSELoss()

prec_fp_hist = []
ent_fp_hist = []
prec_quant_hist = []
ent_quant_hist = []

# Training full precision
for epoch in range(epochs):
    opt_fp.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        recon = model_fp(real_data, epoch)
        loss = loss_fn(recon, target)

    scaler_fp.scale(loss).backward() if use_amp else loss.backward()
    scaler_fp.unscale_(opt_fp) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model_fp.parameters(), 1e6)
    scaler_fp.step(opt_fp) if use_amp else opt_fp.step()
    scaler_fp.update() if use_amp else None

    if epoch % 500 == 0:
        ent = -recon * torch.log(recon + 1e-12)
        p = recon.mean().item()
        e = ent.mean().item()
        prec_fp_hist.append(p)
        ent_fp_hist.append(e)

# Training quantized
for epoch in range(epochs):
    opt_quant.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        recon = model_quant(real_data, epoch)
        loss = loss_fn(recon, target)

    scaler_quant.scale(loss).backward() if use_amp else loss.backward()
    scaler_quant.unscale_(opt_quant) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model_quant.parameters(), 1e6)
    scaler_quant.step(opt_quant) if use_amp else opt_quant.step()
    scaler_quant.update() if use_amp else None

    if epoch % 500 == 0:
        ent = -recon * torch.log(recon + 1e-12)
        p = recon.mean().item()
        e = ent.mean().item()
        prec_quant_hist.append(p)
        ent_quant_hist.append(e)

# Sigma Test (quant vs full precision)
fp_prec_mean = np.mean(prec_fp_hist)
quant_prec_mean = np.mean(prec_quant_hist)
prec_std = np.std(np.concatenate([prec_fp_hist, prec_quant_hist]))
sigma_prec = (fp_prec_mean - quant_prec_mean) / prec_std if prec_std > 0 else 0

fp_ent_mean = np.mean(ent_fp_hist)
quant_ent_mean = np.mean(ent_quant_hist)
ent_std = np.std(np.concatenate([ent_fp_hist, ent_quant_hist]))
sigma_ent = (quant_ent_mean - fp_ent_mean) / ent_std if ent_std > 0 else 0

print(f"Sigma Precision (FP vs Quant): {sigma_prec:.2f}")
print(f"Sigma Entropy (FP vs Quant): {sigma_ent:.2f}")

# Visualization
with torch.no_grad():
    recon_fp = model_fp(real_data, 0).cpu()
    recon_quant = model_quant(real_data, 0).cpu()
    original = real_data.cpu()

num_vis = 8
fig, axes = plt.subplots(3, num_vis, figsize=(num_vis*2, 6))
for i in range(num_vis):
    axes[0, i].imshow(original[0, i].numpy(), aspect='auto', cmap='viridis')
    axes[0, i].set_title("Masked Input")
    axes[0, i].axis('off')

    axes[1, i].imshow(recon_fp[0, i].numpy(), aspect='auto', cmap='viridis')
    axes[1, i].set_title("Full Precision")
    axes[1, i].axis('off')

    axes[2, i].imshow(recon_quant[0, i].numpy(), aspect='auto', cmap='viridis')
    axes[2, i].set_title("Quantized 8-bit")
    axes[2, i].axis('off')

plt.suptitle(f"Quantized Edge Inference: Full Precision vs 8-bit | Sigma Precision: {sigma_prec:.2f}")
plt.tight_layout()
plt.savefig("quantized_edge_inference_visualization.png")
plt.show()

print("Visualization saved as quantized_edge_inference_visualization.png")