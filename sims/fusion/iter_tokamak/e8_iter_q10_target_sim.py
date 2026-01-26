import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
triality = 3
dim = 240
latent_dim = 8
seq_len = 1024
noise_scale = 0.002
batch_size = 64
epochs = 3000000

# ITER Q>10 target proxies (fusion power 500 MW, confinement time τ_E ~3–5 s, heating loss <10%)
fusion_power = torch.linspace(400, 600, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # MW
conf_time = torch.linspace(3, 5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
heating_loss = torch.linspace(0.05, 0.15, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

q10_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([fusion_power, conf_time, heating_loss, q10_sym], dim=-1)\
             .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots (same function)

# Sectors: Fusion power, Confinement time, Heating loss, Q10 symmetry, Prediction nulling
power_roots = e8_roots[:60]
time_roots = e8_roots[60:120]
loss_roots = e8_roots[120:180]
sym_roots = e8_roots[180:]

class Q10TargetRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1]) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * np.pi)
        return x * (emb.cos() + pump) + torch.roll(x, shifts=1, dims=-1) * emb.sin()

class E8ITERQ10Target(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = Q10TargetRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training, sigma test, sensitivity, and plotting same as above (copy-paste blocks)
# Plot save name: "iter_q10_target_ablation_precision_entropy.png"

print("ITER Q>10 target sim ready — eternal burning plasma goal.")