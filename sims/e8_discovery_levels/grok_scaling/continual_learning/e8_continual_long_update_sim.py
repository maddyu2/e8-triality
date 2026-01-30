# E8 Triality Continual Long Update Sim
# Run on Colab Pro A100 (High-RAM + GPU)

!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.amp
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ────────────────────────────────────────────────
# CONFIG – optimized for Pro A100 (continual long updates)
# ────────────────────────────────────────────────
triality = 3
dim = 240
latent_dim = 8
seq_len = 512
batch_size = 256
num_tasks = 5  # sequential tasks
epochs_per_task = 20000  # 100k+ total steps
lr = 5e-5
use_amp = True

# Continual task proxy (different noise patterns per task)
tasks = []
for task_id in range(num_tasks):
    # Task-specific shift + noise
    shift = task_id * 0.2
    task_data = torch.linspace(shift, shift + 0.8, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
    task_data = task_data.repeat(1, 1, dim) * torch.randn(batch_size, seq_len, dim, device=device) * 0.002
    tasks.append(task_data)

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
class ContinualCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(torch.tensor(step, device=device, dtype=torch.float32) * 0.006 * 2 * math.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Model with ablation support
class E8ContinualLongUpdate(nn.Module):
    def __init__(self, depth=128, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.cycle = ContinualCycleBlock() if use_triality else nn.Identity()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x, step):
        x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return x

# Training function
def train_on_task(model, opt, scaler, task_data, task_id):
    prec_hist_task = []
    ent_hist_task = []

    for epoch in range(epochs_per_task):
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
            recon = model(task_data, epoch + task_id * epochs_per_task)
            loss = loss_fn(recon, task_data)  # reconstruct current task

        scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler.unscale_(opt) if use_amp else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        scaler.step(opt) if use_amp else opt.step()
        scaler.update() if use_amp else None

        if epoch % 5000 == 0:
            ent = -recon * torch.log(recon + 1e-12)
            p = recon.mean().item()
            e = ent.mean().item()
            prec_hist_task.append(p)
            ent_hist_task.append(e)
            print(f"Task {task_id} Epoch {epoch} | Loss {loss.item():.6f}")

    return prec_hist_task, ent_hist_task

# Models
model = E8ContinualLongUpdate(use_triality=True).to(device)
model = torch.compile(model)

model_ablation = E8ContinualLongUpdate(use_triality=False).to(device)
model_ablation = torch.compile(model_ablation)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

loss_fn = nn.MSELoss()

# Continual training across tasks
all_prec_hist = []
all_ent_hist = []
all_abl_prec_hist = []
all_abl_ent_hist = []

for task_id, task_data in enumerate(tasks):
    print(f"\n=== Training on Task {task_id} ===")
    
    # Triality model
    prec_task, ent_task = train_on_task(model, opt, scaler, task_data, task_id)
    all_prec_hist.extend(prec_task)
    all_ent_hist.extend(ent_task)
    
    # Ablation model
    prec_task_abl, ent_task_abl = train_on_task(model_ablation, opt_ablation, scaler_ablation, task_data, task_id)
    all_abl_prec_hist.extend(prec_task_abl)
    all_abl_ent_hist.extend(ent_task_abl)

    # Test retention on previous tasks (example on task 0)
    if task_id > 0:
        with torch.no_grad():
            recon_old = model(tasks[0], task_id * epochs_per_task)
            print(f"Retention on Task 0 after Task {task_id}: Loss {loss_fn(recon_old, tasks[0]).item():.6f}")

# Sigma Test (overall)
e8_prec_mean = np.mean(all_prec_hist)
abl_prec_mean = np.mean(all_abl_prec_hist)
prec_std = np.std(np.concatenate([all_prec_hist, all_abl_prec_hist]))
sigma_prec = (e8_prec_mean - abl_prec_mean) / prec_std if prec_std > 0 else 0

e8_ent_mean = np.mean(all_ent_hist)
abl_ent_mean = np.mean(all_abl_ent_hist)
ent_std = np.std(np.concatenate([all_ent_hist, all_abl_ent_hist]))
sigma_ent = (abl_ent_mean - e8_ent_mean) / ent_std if ent_std > 0 else 0

print(f"Final Sigma Precision: {sigma_prec:.2f}")
print(f"Final Sigma Entropy: {sigma_ent:.2f}")

# Visualization (continual precision/entropy curves)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(all_prec_hist, label='E8 Triality')
plt.plot(all_abl_prec_hist, label='Ablation', linestyle='--')
plt.title("Precision Across Continual Tasks")
plt.xlabel("Steps")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Precision: {sigma_prec:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.subplot(1,2,2)
plt.plot(all_ent_hist, label='E8 Triality')
plt.plot(all_abl_ent_hist, label='Ablation', linestyle='--')
plt.title("Entropy Across Continual Tasks")
plt.xlabel("Steps")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.text(0.95, 0.95, f"Sigma Entropy: {sigma_ent:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig("continual_long_update_sigma_visualization.png")
plt.show()

print("Visualization saved as continual_long_update_sigma_visualization.png")