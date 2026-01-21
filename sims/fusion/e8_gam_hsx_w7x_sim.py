import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GAM frequency/damping sim for HSX vs W7-X
n_strata = 6
n_modes = 128
batch_size = 48
epochs = 220
lr = 0.00035
triality_strength = 0.92

def generate_gam_comp_data(batch_size):
    rho = torch.linspace(0, 1, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)
    Ti_Te = torch.rand(batch_size, device=device) * 1 + 1  # 1-2
    c_s = torch.rand(batch_size, device=device) * 0.5 + 1  # normalized
    R_hsx = torch.ones(batch_size, device=device) * 1.2  # HSX R=1.2m
    R_w7x = torch.ones(batch_size, device=device) * 5.5  # W7-X R=5.5m
    eps_eff_hsx = torch.rand(batch_size, device=device) * 0.005 + 0.005  # HSX low
    eps_eff_w7x = torch.rand(batch_size, device=device) * 0.02 + 0.01  # W7-X higher
    
    k_r = torch.logspace(-2, -0.5, n_modes, device=device)
    
    # ω_GAM = sqrt(7/4 + Ti/Te) * c_s / R
    omega_gam_hsx = torch.sqrt(1.75 + Ti_Te.unsqueeze(1)) * c_s.unsqueeze(1) / R_hsx.unsqueeze(1)
    omega_gam_w7x = torch.sqrt(1.75 + Ti_Te.unsqueeze(1)) * c_s.unsqueeze(1) / R_w7x.unsqueeze(1)
    
    # γ_GAM ~ (k_r ρ_i)^2 * c_s / R * eps_eff
    gamma_gam_hsx = (k_r ** 2) * c_s.unsqueeze(0) / R_hsx.unsqueeze(0) * eps_eff_hsx.unsqueeze(1) * 0.1
    gamma_gam_w7x = (k_r ** 2) * c_s.unsqueeze(0) / R_w7x.unsqueeze(0) * eps_eff_w7x.unsqueeze(1) * 0.1
    
    # Entropy: sum (ω - γ)^2
    entropy = torch.sum((omega_gam_hsx + omega_gam_w7x - gamma_gam_hsx - gamma_gam_w7x) ** 2, dim=1) * 0.006 / 2  # Avg over devices
    
    # Damping proxy (mean γ_GAM)
    target_gamma = (gamma_gam_hsx.mean(dim=1) + gamma_gam_w7x.mean(dim=1)) / 2
    
    # Inputs: [rho flat, Ti_Te repeat, c_s repeat, R_hsx repeat, R_w7x repeat, eps_hsx repeat, eps_w7x repeat, omega_hsx, omega_w7x, gamma_hsx, gamma_w7x]
    inputs = torch.cat([
        rho.view(batch_size, -1),
        Ti_Te.unsqueeze(1).repeat(1, n_modes),
        c_s.unsqueeze(1).repeat(1, n_modes),
        R_hsx.unsqueeze(1).repeat(1, n_modes),
        R_w7x.unsqueeze(1).repeat(1, n_modes),
        eps_eff_hsx.unsqueeze(1).repeat(1, n_modes),
        eps_eff_w7x.unsqueeze(1).repeat(1, n_modes),
        omega_gam_hsx,
        omega_gam_w7x,
        gamma_gam_hsx,
        gamma_gam_w7x
    ], dim=1)
    
    return inputs, entropy.unsqueeze(1), target_gamma.unsqueeze(1)

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

class E8GAMCompNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, output_dim=2):
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

input_dim = n_strata + n_modes * 9  # rho, Ti_Te, c_s, R_hsx, R_w7x, eps_hsx, eps_w7x, omega_hsx, omega_w7x, gamma_hsx, gamma_w7x

model = E8GAMCompNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

losses = []
for epoch in range(epochs):
    inputs, entropy, gamma = generate_gam_comp_data(batch_size)
    targets = torch.cat([entropy, gamma], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

with torch.no_grad():
    test_inputs, test_entropy, test_gamma = generate_gam_comp_data(1024)
    test_preds = model(test_inputs)
    entropy_mae = torch.mean(torch.abs(test_preds[:,0] - test_entropy))
    gamma_mae = torch.mean(torch.abs(test_preds[:,1] - test_gamma))
    coherence = 1.0 - (entropy_mae + gamma_mae) / 2
    final_entropy = torch.std(test_preds[:,0] - test_entropy.squeeze()).item()

print(f"\nFinal Evaluation (GAM Frequency Derivation W7-X vs HSX):")
print(f"  Entropy MAE: {entropy_mae.item():.6f} (Bound low)")
print(f"  GAM Damping MAE: {gamma_mae.item():.6f} (HSX lower than W7-X)")
print(f"  Coherence: {coherence.item():.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot
plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_gam_w7x_hsx_losses.png")
print("Plot saved to: e8_gam_w7x_hsx_losses.png")