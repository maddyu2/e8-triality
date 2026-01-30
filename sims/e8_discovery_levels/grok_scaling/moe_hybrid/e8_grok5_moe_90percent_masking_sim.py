import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext

# ────────────────────────────────────────────────
# CONFIG – optimized for Pro A100
# ────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
triality = 3
dim = 240
latent_dim = 8
seq_len = 512
batch_size = 256
epochs = 50000  # large for Pro
lr = 5e-5
use_amp = True
num_experts = 8  # Grok-style MoE
top_k = 2        # top-k routing

# ────────────────────────────────────────────────
# 90% masking sparse multimodal proxies
# ────────────────────────────────────────────────
sparsity_90 = torch.full((batch_size, seq_len, 1), 0.9, device=device)  # 90% masking
multimodal_emb = torch.linspace(0.1, 0.9, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = multimodal_emb.repeat(1, 1, dim) * torch.randn(batch_size, seq_len, dim, device=device) * 0.002

# Apply 90% masking
mask = torch.rand_like(real_data) < sparsity_90
real_data[mask] = 0

target = torch.ones(batch_size, 1, device=device)  # coherence target

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
class Grok5CycleBlock(nn.Module):
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

# Simple expert for Sparse MoE
class MoEExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)

# Sparse MoE layer with top-k routing (Grok 5-style)
class Grok5SparseMoELayer(nn.Module):
    def __init__(self, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([MoEExpert() for _ in range(num_experts)])

    def forward(self, x):
        b, s, d = x.shape
        gate_logits = self.gate(x)  # (b, s, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = top_k_weights[..., i:i+1][mask].unsqueeze(-1)
                output[mask] += expert_out * weight

        return output

# Model with Grok 5-style sparse MoE + triality
class E8Grok5MoE90Masking(nn.Module):
    def __init__(self, depth=128, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.layers = nn.ModuleList([Grok5SparseMoELayer() for _ in range(depth)])
        self.cycle_block = Grok5CycleBlock() if use_triality else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.cycle_block(x, step)  # triality cycle first
        for layer in self.layers:
            if use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            x = x + self.norm(x)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# ────────────────────────────────────────────────
# Training – optimized with AMP
# ────────────────────────────────────────────────
model = E8Grok5MoE90Masking(use_triality=True).to(device)
model = torch.compile(model)

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-10, fused=True)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        prec = model(real_data, epoch)
        loss = loss_fn(prec, target)

    scaler.scale(loss).backward() if use_amp else loss.backward()
    scaler.unscale_(opt) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    scaler.step(opt) if use_amp else opt.step()
    scaler.update() if use_amp else None

    if epoch % 500 == 0:
        ent = -prec * torch.log(prec + 1e-12)
        p = prec.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)
        print(f"Epoch {epoch:5d} | Prec {p:.6f} | Ent {e:.6f} | Loss {loss.item():.6f}")

# ────────────────────────────────────────────────
# Ablation: triality disabled (pure Grok 5-style MoE)
# ────────────────────────────────────────────────
model_ablation = E8Grok5MoE90Masking(use_triality=False).to(device)
opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr, fused=True)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

abl_prec_hist = []
abl_ent_hist = []

for epoch in range(epochs):
    opt_ablation.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
        abl_prec = model_ablation(real_data, epoch)
        abl_loss = loss_fn(abl_prec, target)

    scaler_ablation.scale(abl_loss).backward() if use_amp else abl_loss.backward()
    scaler_ablation.unscale_(opt_ablation) if use_amp else None
    torch.nn.utils.clip_grad_norm_(model_ablation.parameters(), 1e6)
    scaler_ablation.step(opt_ablation) if use_amp else opt_ablation.step()
    scaler_ablation.update() if use_amp else None

    if epoch % 500 == 0:
        abl_ent = -abl_prec * torch.log(abl_prec + 1e-12)
        ap = abl_prec.mean().item()
        ae = abl_ent.mean().item()
        abl_prec_hist.append(ap)
        abl_ent_hist.append(ae)

# ────────────────────────────────────────────────
# Sigma Test + Visualization
# ────────────────────────────────────────────────
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

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(prec_hist, label='E8 Triality MoE')
plt.plot(abl_prec_hist, label='MoE Only Ablation', linestyle='--')
plt.title("Precision Convergence (90% Masking)")
plt.xlabel("Epoch / 500")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Precision: {sigma_prec:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.subplot(1,2,2)
plt.plot(ent_hist, label='E8 Triality MoE')
plt.plot(abl_ent_hist, label='MoE Only Ablation', linestyle='--')
plt.title("Entropy Convergence (90% Masking)")
plt.xlabel("Epoch / 500")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Entropy: {sigma_ent:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig("grok5_moe_90percent_masking_ablation_sigma.png")
plt.show()

print("Sigma visualization saved as grok5_moe_90percent_masking_ablation_sigma.png")