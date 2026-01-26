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

# HSX + GENE proxies (ITG growth γ ~0.1–0.5, zonal damping ~0.05–0.2, QHS symmetry ~0.9–1.0)
gene_growth = torch.linspace(0.1, 0.5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
zonal_damping = torch.linspace(0.05, 0.2, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
qhs_sym = torch.linspace(0.9, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

qhs_sym = torch.linspace(0.9, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_hsx_gene_data = torch.cat([gene_growth, zonal_damping, qhs_sym], dim=-1)\
                      .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots (same function as above)

class HSXGeneComparisonRotary(nn.Module):
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

class E8HSXGeneComparison(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.rotary = HSXGeneComparisonRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Initial HSX GENE comparison state → precision target
states = real_hsx_gene_data
target_prec = torch.ones(batch_size, 1, device=device)

model = E8HSXGeneComparison().to(device)
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
plt.savefig("hsx_gene_comparison_precision_entropy.png", dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as hsx_gene_comparison_precision_entropy.png")
print("Final precision:", prec_hist[-1])
print("Final entropy:", ent_hist[-1])