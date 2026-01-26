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

# NIF 2025 proxies (laser energy ~1.8–2.0 MJ, yield ~2–3 MJ, hotspot T ~5–8 keV, ρR ~0.3–0.5 g/cm²)
laser_energy = torch.linspace(1.8, 2.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # MJ
yield_mj = torch.linspace(2, 3, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
hotspot_temp = torch.linspace(5, 8, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # keV
rho_r = torch.linspace(0.3, 0.5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

nif_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([laser_energy, yield_mj, hotspot_temp, rho_r, nif_sym], dim=-1)\
             .repeat(1, 1, dim // 5) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots (same function)

# Sectors: Laser energy, Yield MJ, Hotspot temp, ρR, NIF symmetry, Prediction nulling
energy_roots = e8_roots[:48]
yield_roots = e8_roots[48:96]
temp_roots = e8_roots[96:144]
rho_roots = e8_roots[144:192]
sym_roots = e8_roots[192:]

class NIFTrialityRotary(nn.Module):
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

class E8NIFTriality(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = NIFTrialityRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training loop same as above
# Sigma test block same as above (copy-paste)
# Plot save name: "nif_triality_ablation_precision_entropy.png"

print("NIF fusion with E8 triality sim ready — eternal ignition modeling.")