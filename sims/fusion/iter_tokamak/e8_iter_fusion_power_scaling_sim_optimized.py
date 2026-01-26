import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp  # for mixed precision
from torch.utils.checkpoint import checkpoint  # gradient checkpointing
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

# ────────────────────────────────────────────────
# CONFIG – optimized settings
# ────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
triality = 3
dim = 240
latent_dim = 8
seq_len = 1024
noise_scale = 0.002
batch_size = 256             # increased – adjust to VRAM limit (e.g. 512 on 24 GB card)
micro_batch_size = 64        # for gradient accumulation
grad_accum_steps = batch_size // micro_batch_size
epochs = 3000000             # can be reduced for testing
use_amp = True               # mixed precision (AMP)
use_checkpoint = True        # gradient checkpointing (memory + speed)
lr = 5e-5                    # slightly higher starting LR with warmup
warmup_steps = 2000

# ────────────────────────────────────────────────
# Data (same as original)
# ────────────────────────────────────────────────
fusion_power = torch.linspace(400, 600, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
n_tau_e = torch.linspace(8e19, 1.2e21, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
conf_time = torch.linspace(3, 5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
scaling_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([fusion_power, n_tau_e, conf_time, scaling_sym], dim=-1)\
             .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# ────────────────────────────────────────────────
# E8 roots – precompute once
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
# Rotary – optimized
# ────────────────────────────────────────────────
class FusionPowerRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        # Precompute pos_emb once per forward if possible (batch-invariant)
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * np.pi)
        return x * (emb.cos() + pump) + torch.roll(x, shifts=1, dims=-1) * emb.sin()

# ────────────────────────────────────────────────
# Model – with checkpointing support
# ────────────────────────────────────────────────
class E8ITERFusionPowerScaling(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.rotary = FusionPowerRotary()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(dim, triality, batch_first=True, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward_layer(self, layer, x):
        attn, _ = layer(x, x, x)
        return x + self.norm(attn)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            if use_checkpoint:
                x = checkpoint(self.forward_layer, layer, x, use_reentrant=False)
            else:
                x = self.forward_layer(layer, x)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# ────────────────────────────────────────────────
# Training loop – optimized with AMP & gradient accumulation
# ────────────────────────────────────────────────
model = E8ITERFusionPowerScaling().to(device)
model = torch.compile(model)  # Torch 2.0+ compile – huge speedup on GPU

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-10, fused=True)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
scheduler = CosineAnnealingLR(opt, T_max=epochs)

# Warmup scheduler
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)

    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
            prec = model(real_data, epoch)
            loss = loss_fn(prec, torch.ones_like(prec)) / grad_accum_steps

        scaler.scale(loss).backward() if use_amp else loss.backward()

    # Clip and step
    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    # LR scheduling
    if epoch < warmup_steps:
        warmup_scheduler.step()
    else:
        scheduler.step()

    if epoch % 500 == 0:
        ent = -prec * torch.log(prec + 1e-12)
        p = prec.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)
        print(f"Epoch {epoch:5d} | Prec {p:.6f} | Ent {e:.6f} | LR {opt.param_groups[0]['lr']:.2e}")

# ────────────────────────────────────────────────
# Plots & Save
# ────────────────────────────────────────────────
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(prec_hist, label='Precision')
plt.title("Precision Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(ent_hist, label='Entropy (nats)', color='orange')
plt.title("Entropy Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("iter_fusion_power_scaling_optimized_precision_entropy.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as iter_fusion_power_scaling_optimized_precision_entropy.png")
print("Final precision:", prec_hist[-1])
print("Final entropy:", ent_hist[-1])