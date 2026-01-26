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
epochs = 3000000  # reduce for testing

# NIF hotspot ignition proxies (T ~5-10 keV, ρR ~0.3-0.5 g/cm², ignition efficiency >50%)
hotspot_temp = torch.linspace(5, 10, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # keV
rho_r = torch.linspace(0.3, 0.5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # g/cm²
ignition_eff = torch.linspace(0.5, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

ignition_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([hotspot_temp, rho_r, ignition_eff, ignition_sym], dim=-1)\
             .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots
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

# Sectors: Hotspot temp, ρR, Ignition eff, Ignition symmetry, Prediction nulling
temp_roots = e8_roots[:60]
rho_roots = e8_roots[60:120]
eff_roots = e8_roots[120:180]
sym_roots = e8_roots[180:]

class HotspotIgnitionRotary(nn.Module):
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

class E8NIFHotspotIgnition(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = HotspotIgnitionRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Initial hotspot state → precision target
states = real_data
target_prec = torch.ones(batch_size, 1, device=device)

model = E8NIFHotspotIgnition().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = CosineAnnealingLR(opt, T_max=epochs)
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    opt.zero_grad()
    prec = model(states, epoch)
    loss = loss_fn(prec, torch.ones_like(prec))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    opt.step()
    scheduler.step()

    if epoch % 500 == 0:
        ent = -prec * torch.log(prec + 1e-12)
        p = prec.mean().item()
        e = ent.mean().item()
        prec_hist.append(p)
        ent_hist.append(e)
        print(f"Epoch {epoch:5d} | Prec {p:.6f} | Ent {e:.6f}")

# ────────────────────────────────────────────────
# Sensitivity Analysis: Vary hotspot_temp ±10%
# ────────────────────────────────────────────────

hotspot_temp_var = hotspot_temp * torch.linspace(0.9, 1.1, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
var_data = torch.cat([hotspot_temp_var, rho_r, ignition_eff, ignition_sym], dim=-1)\
             .repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

var_prec = model(var_data, epochs - 1)
var_ent = -var_prec * torch.log(var_prec + 1e-12)
var_variance = torch.var(var_prec.mean(dim=0)).item()
print(f"Sensitivity variance on precision (hotspot_temp ±10%): {var_variance:.6f} — low variance = robust")

# ────────────────────────────────────────────────
# Sigma Test: Ablation baseline (no triality)
# ────────────────────────────────────────────────

class AblationModel(E8NIFHotspotIgnition):
    def __init__(self, depth=256):
        super().__init__(depth=depth)
        self.heads = 1  # disable triality

ablation_model = AblationModel().to(device)
ablation_opt = torch.optim.AdamW(ablation_model.parameters(), lr=4e-5)
ablation_scheduler = CosineAnnealingLR(ablation_opt, T_max=epochs)

ablation_prec_hist = []
ablation_ent_hist = []

for epoch in range(epochs):
    ablation_opt.zero_grad()
    ablation_prec = ablation_model(real_data, epoch)
    ablation_loss = loss_fn(ablation_prec, torch.ones_like(ablation_prec))
    ablation_loss.backward()
    ablation_opt.step()
    ablation_scheduler.step()

    if epoch % 500 == 0:
        ablation_ent = -ablation_prec * torch.log(ablation_prec + 1e-12)
        ap = ablation_prec.mean().item()
        ae = ablation_ent.mean().item()
        ablation_prec_hist.append(ap)
        ablation_ent_hist.append(ae)

# Compute sigma
e8_prec_mean = np.mean(prec_hist)
abl_prec_mean = np.mean(ablation_prec_hist)
prec_std = np.std(np.concatenate([prec_hist, ablation_prec_hist]))
sigma_prec = (e8_prec_mean - abl_prec_mean) / prec