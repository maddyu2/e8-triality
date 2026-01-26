import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

device = 'cuda' if torch.cuda.is_available() else 'cpu'
triality = 3
heads = triality
dim = 240
latent_dim = 8
seq_len = 1024
noise_scale = 0.002
batch_size = 64
epochs = 3000000  # scale as needed

# LQG proxies (Planck scale ~1e-35 m, loop density ~1e99 m^{-3}, horizon entropy S = A/4)
# Vary planck_scale / loop_density / horizon_entropy as input vectors
planck_scale = torch.linspace(1e-36, 1e-34, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # m
loop_density = torch.linspace(1e95, 1e103, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # m^{-3}
horizon_entropy = torch.linspace(1e3, 1e5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # Planck units

# Quantum gravity symmetry proxy (~0.85–1.0)
qg_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_lqg_data = torch.cat([planck_scale, loop_density, horizon_entropy, qg_sym], dim=-1).repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

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

# Sectors: Planck scale, Loop density, Horizon entropy, QG symmetry, Prediction nulling
planck_roots = e8_roots[:60]
loop_roots = e8_roots[60:120]
entropy_roots = e8_roots[120:180]
sym_roots = e8_roots[180:]

class LQGRotary(nn.Module):
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

class E8LoopQuantumGravity(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = LQGRotary()
        self.norm = nn.LayerNorm(dim)
        self.precision_head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = x + self.rotary(x, step)
        for layer in self.layers:
            attn_out, _ = layer(x, x, x)
            split = attn_out.chunk(triality, dim=-1)
            rotated = torch.roll(torch.stack(split, dim=0), shifts=1, dim=0)
            fused = torch.cat(rotated.unbind(0), dim=-1)
            fused = self.norm(fused)
            noise = noise_scale * torch.randn_like(fused)
            x = x + (fused + noise).clamp(-1e12, 1e12)
        precision = torch.sigmoid(self.precision_head(x.mean(dim=1)))
        entropy = -precision * torch.log(precision + 1e-12)
        return precision.mean(), entropy.mean()

# Initial LQG state → precision target
states = real_lqg_data
target_prec = torch.ones(batch_size, 1, device=device)

model = E8LoopQuantumGravity().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-10)
scheduler = CosineAnnealingLR(opt, T_max=epochs)
loss_fn = nn.MSELoss()

prec_hist = []
ent_hist = []

for epoch in range(epochs):
    opt.zero_grad()
    prec, ent = model(states, epoch)
    loss = loss_fn(prec, target_prec) + 0.02 * ent
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
    opt.step()
    scheduler.step()

    if epoch % 500 == 0:
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
plt.savefig("lqg_precision_entropy.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as lqg_precision_entropy.png")
print("Final precision:", prec_hist[-1])
print("Final entropy:", ent_hist[-1])