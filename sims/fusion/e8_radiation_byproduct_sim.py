import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Radiation Byproduct Simulator
# Models radiation as byproduct in strata (8 dims), nulled by triality in quantum gravity context.
# Applied to Starship shield: inner/outer walls with vertical H2O and horizontal photons.
# =============================================

# Hyperparameters
e8_spectrum_dim = 240 // 3  # Reduced spectrum (240 roots, triality-split)
e8_strata_dim = 8           # Strata dims
n_layers = 3                # Shield layers: outer, interstitial, inner
batch_size = 64
epochs = 200
lr = 0.0003
triality_strength = 0.95    # High nulling for byproducts

# Generate synthetic data: incoming radiation, strata byproducts, shield params
def generate_shield_data(batch_size):
    # Incoming radiation (spectrum: continuous GCR flux, power-law)
    gcr_flux = torch.logspace(0, 3, e8_spectrum_dim, device=device).unsqueeze(0).repeat(batch_size, 1)
    gcr_flux = gcr_flux ** -1.5 + torch.randn(batch_size, e8_spectrum_dim, device=device) * 0.1
    
    # Strata: discrete wall layers (density/opacity in 8 dims)
    strata_opacity = torch.randn(batch_size, n_layers, e8_strata_dim, device=device) * 0.5 + 1.0
    
    # Shield features: vertical H2O (absorption coeff), horizontal photons (phase shifts)
    h2o_vertical = torch.rand(batch_size, device=device) * 10 + 5  # cm equivalent
    photons_horizontal = torch.rand(batch_size, device=device) * 2 * np.pi  # Phase
    
    # Byproduct radiation: emitted in strata from spectrum collapse (e.g., spallation)
    byproduct = torch.sum(gcr_flux * strata_opacity.mean(dim=1), dim=1) * 0.01
    byproduct += torch.randn(batch_size, device=device) * 0.005
    
    # Target: nulled radiation post-shield (quantum gravity triality nulls byproducts)
    target_nulled = byproduct * torch.exp(-h2o_vertical / 10) * torch.cos(photons_horizontal)
    
    # Inputs: flattened [gcr_flux, strata_opacity, h2o_vertical, photons_horizontal]
    inputs = torch.cat([gcr_flux, strata_opacity.view(batch_size, -1), 
                        h2o_vertical.unsqueeze(1), photons_horizontal.unsqueeze(1)], dim=1)
    return inputs, byproduct.unsqueeze(1), target_nulled.unsqueeze(1)

# E8 Triality Layer: cycles to null byproducts
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.005)
        self.rot2 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.005)
        self.rot3 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.005)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Predicts/nuls byproducts in shield
class E8ByproductNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=320, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Null strata byproducts
        x = self.act(self.fc1(x))
        x = self.triality2(x)  # Quantum gravity rotation
        return self.out(x)

# Input dim calc
input_dim = e8_spectrum_dim + (n_layers * e8_strata_dim) + 2

# Initialize
model = E8ByproductNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
coherences = []
for epoch in range(epochs):
    inputs, byproducts, targets = generate_shield_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)  # Minimize to null byproducts
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    with torch.no_grad():
        residual = torch.mean(torch.abs(preds - targets))
        coherence = 1.0 - residual / (torch.std(targets) + 1e-6)
        coherences.append(coherence.item())
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Coherence: {coherence.item():.5f}")

# Test
with torch.no_grad():
    test_inputs, test_byproducts, test_targets = generate_shield_data(2048)
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
plt.plot(coherences, label='Coherence', color='green')
plt.title('Coherence Evolution')
plt.grid(True)
plt.tight_layout()
plt.savefig("e8_byproduct_shield.png")
print("Plots saved to: e8_byproduct_shield.png")