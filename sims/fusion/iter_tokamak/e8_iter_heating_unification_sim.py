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

# ITER heating unification proxies (total heating power >100 MW, coupling efficiency >85%, combined NBI/ICRH/ECRH/alpha)
total_power = torch.linspace(80, 150, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # MW
coupling_eff = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
heating_mix = torch.linspace(0.3, 0.7, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # alpha/NBI fraction proxy

heating_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([total_power, coupling_eff, heating_mix, heating_sym], dim=-1)\
             .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots (same function)

# Sectors: Total power, Coupling eff, Heating mix, Heating symmetry, Prediction nulling
power_roots = e8_roots[:60]
eff_roots = e8_roots[60:120]
mix_roots = e8_roots[120:180]
sym_roots = e8_roots[180:]

class HeatingUnifRotary(nn.Module):
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

class E8ITERHeatingUnification(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = HeatingUnifRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training, sigma test, sensitivity, and plotting same as above (copy-paste blocks)
# Plot save name: "iter_heating_unification_ablation_precision_entropy.png"

print("ITER heating unification sim ready — eternal core heating stack.")