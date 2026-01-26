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
epochs = 3000000  # Scale as needed

# Modular Data Generation
def generate_dark_energy_data():
    omega_lambda = torch.linspace(0.65, 0.75, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
    h0 = torch.linspace(67, 73, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
    de_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
    
    data = torch.cat([omega_lambda, h0, de_sym], dim=-1)\
              .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale
    return data

real_data = generate_dark_energy_data()

# E8 Roots Function (Modular)
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

# Rotary Class (Modular)
class DarkEnergyRotary(nn.Module):
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

# Model Class (Modular)
class E8DarkEnergy(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = DarkEnergyRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training Function (Modular)
def train_model(model, states, target_prec, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    prec_history = []
    ent_history = []

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
            prec_history.append(p)
            ent_history.append(e)
            print(f"Epoch {epoch:5d} | Precision {p:.6f} | Entropy {e:.6f}")

    return prec_history, ent_history

# Main
model = E8DarkEnergy().to(device)
target_prec = torch.ones(batch_size, 1, device=device)
prec_history, ent_history = train_model(model, real_data, target_prec, epochs)

# Visualization (Modular Function)
def plot_results(prec_history, ent_history):
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

plot_results(prec_history, ent_history)
print("Final precision:", prec_history[-1])
print("Final entropy:", ent_history[-1])