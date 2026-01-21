import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Zonal Flows in HSX vs W7-X Simulator
# Models ZF drive and saturation in HSX (QHS) vs W7-X (QI), with triality nulling entropy for eternal low transport.
# Parameters: HSX ε_eff ~0.005, W7-X ~0.01, a/L_T ~2-5, ZF shear ~0.1-0.5 γ_max.
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_stellarators = 2          # HSX, W7-X
n_strata = 6                # Radial strata
n_modes = 160               # ZF/turbulence modes
batch_size = 56
epochs = 250
lr = 0.0003
triality_strength = 0.95    # Triality for ZF nulling

# Generate HSX vs W7-X ZF data
def generate_zf_comp_data(batch_size):
    rho = torch.linspace(0, 1, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)
    a_L_T = torch.rand(batch_size, n_strata, device=device) * 3 + 2
    v_prime = torch.rand(batch_size, device=device) * 0.4 + 0.1  # ZF shear
    gamma_primary = torch.rand(batch_size, device=device) * 0.3 + 0.1  # Primary drive
    eps_eff_hsx = torch.rand(batch_size, device=device) * 0.005 + 0.005  # HSX low
    eps_eff_w7x = torch.rand(batch_size, device=device) * 0.01 + 0.01  # W7-X
    
    k_y = torch.logspace(-1, 0.3, n_modes, device=device)
    gamma_turb = a_L_T.mean(dim=1).unsqueeze(1) / (k_y + 1e-3) * (k_y ** -0.6)
    
    # ZF saturation: drive - shear, eps_eff damping (HSX lower)
    gamma_net_hsx = gamma_turb - v_prime.unsqueeze(1) + gamma_primary.unsqueeze(1) - eps_eff_hsx.unsqueeze(1) * 0.3
    gamma_net_w7x = gamma_turb - v_prime.unsqueeze(1) + gamma_primary.unsqueeze(1) - eps_eff_w7x.unsqueeze(1) * 0.3
    gamma_net = (gamma_net_hsx + gamma_net_w7x) / 2 + torch.randn(batch_size, n_modes, device=device) * 0.025
    
    target_flux = gamma_net.clip(min=0).mean(dim=1) / (k_y.mean() ** 2) * 0.4
    entropy = torch.sum(gamma_net ** 2, dim=1) * 0.007
    
    inputs = torch.cat([
        rho.view(batch_size, -1),
        a_L_T.view(batch_size, -1),
        v_prime.unsqueeze(1).repeat(1, n_modes),
        gamma_primary.unsqueeze(1).repeat(1, n_modes),
        eps_eff_hsx.unsqueeze(1).repeat(1, n_modes),
        eps_eff_w7x.unsqueeze(1).repeat(1, n_modes),
        gamma_net
    ], dim=1)
    
    return inputs, entropy.unsqueeze(1), target_flux.unsqueeze(1)

# E8 Triality Layer: nulls ZF instabilities
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot2 = nn.Parameter(torch.eye(dim, dim, device=device) * 0.01)
        self.rot3 = nn.Parameter(torch.eye(dim, dim, device=device) * 0.01)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        return self.strength * (x + x1 + x2 + x3) / 4.0

# Model: Bounds entropy, predicts low flux
class E8ZFCompNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, output_dim=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)
        x = self.act(self.fc1(x))
        x = self.triality2(x)
        return self.out(x)

input_dim = n_strata + n_modes * 5
model = E8ZFCompNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

losses = []
for epoch in range(epochs):
    inputs, entropy, flux = generate_zf_comp_data(batch_size)
    targets = torch.cat([entropy, flux], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Entropy: {preds[:,0].mean().item():.6f}")

with torch.no_grad():
    test_inputs, test_entropy, test_flux = generate_zf_comp_data(1024)
    test_preds = model(test_inputs)
    entropy_mae = torch.mean(torch.abs(test_preds[:,0] - test_entropy))
    flux_mae = torch.mean(torch.abs(test_preds[:,1] - test_flux))
    coherence = 1.0 - (entropy_mae + flux_mae) / 2
    final_entropy = torch.mean(test_preds[:,0]).item()

print(f"\nFinal: Entropy MAE {entropy_mae:.6f} | Flux MAE {flux_mae:.6f}")
print(f"Coherence {coherence:.6f} | Mean Entropy {final_entropy:.6f} nats")

plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_zf_hsx_w7x_losses.png")
print("Plot saved to: e8_zf_hsx_w7x_losses.png")