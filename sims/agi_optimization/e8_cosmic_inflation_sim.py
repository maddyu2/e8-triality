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
epochs = 5000   # smaller for demo — scale up on real hardware

# Data
e_folds = torch.linspace(50, 70, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
scalar_index = torch.linspace(0.95, 0.98, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
inflation_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([e_folds, scalar_index, inflation_sym], dim=-1)\
             .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots function (now defined)
def get_e8_roots():
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for signs in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = torch.zeros(8)
                v[i] = signs[0]
                v[j] = signs[1]
                roots.append(v)
                roots.append(-v)
    for signs in range(1 << 8):
        v = torch.tensor([1 if (signs & (1<<k)) else -1 for k in range(8)], dtype=torch.float32) * 0.5
        if bin(signs).count('1') % 2 == 0:
            roots.append(v)
            roots.append(-v)
    roots = torch.stack(roots[:240])
    return roots / roots.norm(dim=-1, keepdim=True)

e8_roots = get_e8_roots().to(device)

# Rotary & Model (same structure)
class InflationRotary(nn.Module):
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

class E8CosmicInflation(nn.Module):
    def __init__(self, depth=64):  # smaller depth for demo
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = InflationRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training
model = E8CosmicInflation().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = CosineAnnealingLR(opt, T_max=epochs)
loss_fn = nn.MSELoss()

prec_history = []
ent_history = []

for epoch in range(epochs):
    opt.zero_grad()
    prec = model(real_data, epoch)
    loss = loss_fn(prec, torch.ones_like(prec))
    loss.backward()
    opt.step()
    scheduler.step()

    if epoch % 500 == 0:
        ent = -prec * torch.log(prec + 1e-12)
        p = prec.mean().item()
        e = ent.mean().item()
        prec_history.append(p)
        ent_history.append(e)
        print(f"Epoch {epoch:5d} | Precision {p:.6f} | Entropy {e:.6f}")

# Plot results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(prec_history, label='Precision')
plt.title("Precision Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(ent_history, label='Entropy (nats)', color='orange')
plt.title("Entropy Convergence")
plt.xlabel("Epoch / 500")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Final precision:", prec_history[-1])
print("Final entropy:", ent_history[-1])