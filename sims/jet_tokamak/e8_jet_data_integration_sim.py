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

# JET 2025 proxy data integration (JET-ILW high-performance pulses: Te ~8–12 keV, ne ~8–12×10^{19} m^{-3}, β_N ~2.0–3.0, confinement H98 ~1.0–1.2)
te_core = torch.linspace(8, 12, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)   # keV
ne_core = torch.linspace(8e19, 12e19, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # m^{-3}
beta_n_jet = torch.linspace(2.0, 3.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
h98_factor = torch.linspace(1.0, 1.2, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

jet_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_jet_data = torch.cat([te_core, ne_core, beta_n_jet, h98_factor, jet_sym], dim=-1)\
                 .repeat(1, 1, dim // 5) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

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

# Sectors: Core Te, Core ne, β_N JET, H98 factor, JET symmetry, Prediction nulling
te_roots = e8_roots[:48]
ne_roots = e8_roots[48:96]
beta_roots = e8_roots[96:144]
h98_roots = e8_roots[144:192]
sym_roots = e8_roots[192:]

class JetDataRotary(nn.Module):
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

class E8JETDataIntegration(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = JetDataRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Initial JET state → precision target
states = real_jet_data
target_prec = torch.ones(batch_size, 1, device=device)

model = E8JETDataIntegration().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = CosineAnnealingLR(opt, T_max=epochs)
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    opt.zero_grad()
    prec = model(states, epoch)
    loss = loss_fn(prec, target_prec)
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

# Convergence Plots & Save
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
plt.savefig("jet_data_integration_precision_entropy.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as jet_data_integration_precision_entropy.png")
print("Final precision:", prec_hist[-1])
print("Final entropy:", ent_hist[-1])