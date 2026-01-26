import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import deepspeed
import os

# DDP Setup (for Colossus multi-GPU)
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

triality = 3
dim = 240
latent_dim = 8
seq_len = 1024
noise_scale = 0.001
batch_size = 1024  # Colossus-scale (effective after accumulation)
grad_accum_steps = 4  # effective batch = 1024 × 4 × world_size

# ────────────────────────────────────────────────
# Data: GENE turbulence proxies (ETG/ITG gradients a/L_Te,i 1–10, magnetic shear ~0.1–1.0)
# ────────────────────────────────────────────────

a_L_Te = torch.linspace(1.0, 10.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
a_L_Ti = torch.linspace(0.5, 5.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
magnetic_shear = torch.linspace(0.1, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
turb_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_turb_data = torch.cat([a_L_Te, a_L_Ti, magnetic_shear, turb_sym], dim=-1)\
                 .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

dataset = TensorDataset(real_turb_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# ────────────────────────────────────────────────
# E8 Roots
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
projection = nn.Linear(8, dim, bias=False).to(device)

# ────────────────────────────────────────────────
# Rotary Layer
# ────────────────────────────────────────────────

class TurbRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality)

    def forward(self, x, step):
        pos_emb = e8_roots[torch.arange(x.shape[1]) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * np.pi)
        return x * (emb.cos() + pump) + torch.roll(x, shifts=1, dims=-1) * emb.sin()

# ────────────────────────────────────────────────
# E8 Model
# ────────────────────────────────────────────────

class E8GENETurbulence(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.root_inits = nn.Parameter(projection(e8_roots.repeat(seq_len // 240 + 1, 1)[:seq_len]))
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = TurbRotary()
        self.norm = nn.LayerNorm(dim)
        self.precision_head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = x + self.root_inits
        x = self.rotary(x, step)
        for layer in self.layers:
            attn_out, _ = checkpoint(layer, x, x, x)
            split = attn_out.chunk(triality, dim=-1)
            rotated = torch.roll(torch.stack(split, dim=0), shifts=1, dim=0)
            fused = torch.cat(rotated.unbind(0), dim=-1)
            fused = self.norm(fused)
            noise = noise_scale * torch.randn_like(fused)
            x = x + (fused + noise).clamp(-1e12, 1e12)
        precision = torch.sigmoid(self.precision_head(x.mean(dim=1)))
        entropy = -precision * torch.log(precision + 1e-12)
        return precision.mean(), entropy.mean()

# ────────────────────────────────────────────────
# ZeRO-3 + DDP Initialization
# ────────────────────────────────────────────────

model = E8GENETurbulence().to(device)
model = DDP(model, device_ids=[local_rank])

ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "reduce_bucket_size": 0,
        "stage3_prefetch_bucket_size": 0,
        "stage3_param_persistence_threshold": 0
    },
    "fp16": {"enabled": True},
    "train_batch_size": batch_size * grad_accum_steps * world_size,
    "gradient_accumulation_steps": grad_accum_steps,
    "gradient_clipping": 1.0
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=500)

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    model_engine.zero_grad()
    prec = model_engine(real_data, epoch)
    loss = torch.nn.functional.mse_loss(prec, torch.ones_like(prec))
    model_engine.backward(loss)
    model_engine.step()

    if epoch < 500:
        warmup_scheduler.step()
    else:
        scheduler.step()

    if epoch % 500 == 0 and local_rank == 0:
        ent = -prec * torch.log(prec + 1e-12)
        p = prec.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)
        print(f"Rank {local_rank} | Epoch {epoch:5d} | Prec {p:.6f} | Ent {e:.6f}")

# Save convergence plot (only rank 0)
if local_rank == 0:
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
    plt.savefig("gene_zer03_turbulence_precision_entropy.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Plots saved as gene_zer03_turbulence_precision_entropy.png")
    print("Final precision:", prec_hist[-1])
    print("Final entropy:", ent_hist[-1])

destroy_process_group()