import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Starship Shield Quantum Gravity Simulator
# Integrates curved metrics (full gravity) with triality nulling for radiation shielding.
# Models spacetime curvature effects on radiation byproducts in strata.
# =============================================

# Hyperparameters
e8_spectrum_dim = 80  # Reduced 240 roots
e8_strata_dim = 8     # Strata dims
n_layers = 3          # Shield layers
batch_size = 64
epochs = 220
lr = 0.00025
triality_strength = 0.92

# Generate synthetic data: curved metrics, radiation, shield
def generate_qg_shield_data(batch_size):
    # Curved metrics: simplified Ricci scalar R (gravity proxy, curved spacetime)
    ricci_scalar = torch.randn(batch_size, device=device) * 10 + 5  # Varied curvature
    
    # Spectrum: incoming radiation flux under gravity (warped by R)
    gcr_flux = torch.logspace(0, 3, e8_spectrum_dim, device=device).unsqueeze(0).repeat(batch_size, 1)
    gcr_flux = gcr_flux ** -1.5 * torch.exp(-ricci_scalar.unsqueeze(1) / 100)  # Gravity warp
    
    # Strata: shield layers with gravity-induced strata (density warped)
    strata_density = torch.randn(batch_size, n_layers, e8_strata_dim, device=device) * 0.4 + 1.2
    strata_density *= (1 + ricci_scalar.unsqueeze(1).unsqueeze(1) / 50)  # Curvature effect
    
    # Shield: H2O/photon proxies under gravity
    h2o_absorb = torch.rand(batch_size, device=device) * 15 + 5
    photon_phase = torch.rand(batch_size, device=device) * 2 * np.pi
    
    # Byproduct: radiation in strata, amplified by curvature
    byproduct = torch.sum(gcr_flux * strata_density.mean(dim=1), dim=1) * 0.015 * ricci_scalar
    
    # Target: nulled under triality + gravity
    target_nulled = byproduct * torch.exp(-h2o_absorb / 12) * torch.sin(photon_phase) / (1 + ricci_scalar / 10)
    
    # Inputs: [ricci_scalar, gcr_flux, strata_density flat, h2o, photon]
    inputs = torch.cat([ricci_scalar.unsqueeze(1), gcr_flux, strata_density.view(batch_size, -1), 
                        h2o_absorb.unsqueeze(1), photon_phase.unsqueeze(1)], dim=1)
    return inputs, byproduct.unsqueeze(1), target_nulled.unsqueeze(1)

# Triality Layer
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.008)
        self.rot2 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.008)
        self.rot3 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.008)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Integrates gravity metrics with triality
class E8QGShieldNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Null gravity-warped byproducts
        x = self.act(self.fc1(x))
        x = self.triality2(x)
        return self.out(x)

# Input dim
input_dim = 1 + e8_spectrum_dim + (n_layers * e8_strata_dim) + 2

# Initialize
model = E8QGShieldNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
coherences = []
for epoch in range(epochs):
    inputs, byproducts, targets = generate_qg_shield_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    with torch.no_grad():
        residual = torch.mean(torch.abs(preds - targets))
        coherence = 1.0 - residual / (torch.std(targets) + 1e-6)
        coherences.append(coherence.item())
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Coherence: {coherence.item():.5f}")

# Test
with torch.no_grad():
    test_inputs, test_byproducts, test_targets = generate_qg_shield_data(2048)
    test_preds = model(test_inputs)
    test_residual = torch.mean(torch.abs(test_preds - test_targets))
    final_coherence = 1.0 - test_residual / (torch.std(test_targets) + 1e-6)
    final_entropy = torch.std(test_preds - test_targets).item()

print(f"\nFinal Evaluation:")
print(f"  Nulling Coherence: {final_coherence:.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(coherences, label='Coherence', color='purple')
plt.title('Coherence Evolution')
plt.grid(True)
plt.tight_layout()
plt.savefig("e8_qg_shield.png")
print("Plots saved to: e8_qg_shield.png")