import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Radiation Cosmo Simulator
# Blends JWST radiation spectra (e.g., CMB-like gradients) with void nulling
# using triality to smooth cosmic radiation fields.
# =============================================

# Hyperparameters
e8_effective_dim = 64
n_void_strata = 5           # Discrete void density strata (under/over-dense)
n_spectrum_bins = 128       # CMB/JWST spectral bins
batch_size = 56
epochs = 140
lr = 0.0007
triality_strength = 0.75

# Generate synthetic cosmo data: void strata + radiation spectra
def generate_cosmo_data(batch_size):
    # Strata: discrete void levels (density contrast δρ/ρ)
    strata_delta = torch.linspace(-0.8, 0.8, n_void_strata + 1, device=device)  # Voids to clusters
    stratum_idx = torch.randint(0, n_void_strata, (batch_size,), device=device)
    delta = strata_delta[stratum_idx] + torch.randn(batch_size, device=device) * 0.1
    
    # Redshift z (JWST high-z)
    z = torch.rand(batch_size, device=device) * 10 + 1  # z=1-11
    
    # CMB-like spectrum: temperature fluctuations δT/T ~ 10^{-5}, spectral
    k = torch.logspace(-3, 1, n_spectrum_bins, device=device)  # Wavenumbers
    power_spec = (k ** -1.5) * (1 + torch.randn(n_spectrum_bins, device=device) * 0.05)  # Simplified CMB power
    delta_t = power_spec.mean() * 1e-5 * (1 + delta)  # Modulated by void strata
    
    # Target gradient: d(δT)/dz for nulling voids
    target_grad = -0.001 * delta * torch.log(1 + z) + torch.randn(batch_size, device=device) * 0.0001
    
    # Inputs: [delta, z, delta_t]
    inputs = torch.stack([delta, z, delta_t.repeat(batch_size)], dim=1)
    return inputs, target_grad.unsqueeze(1)

# Triality Layer
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

# Model
class E8CosmoNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=224, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)
        x = self.act(self.fc1(x))
        x = self.triality2(x)
        return self.out(x)

# Initialize
model = E8CosmoNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
coherences = []
for epoch in range(epochs):
    inputs, targets = generate_cosmo_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    with torch.no_grad():
        mae = torch.mean(torch.abs(preds - targets))
        coherence = 1.0 - mae / (torch.std(targets) + 1e-6)
        coherences.append(coherence.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.8f} | Coherence: {coherence.item():.5f}")

# Test
with torch.no_grad():
    test_inputs, test_targets = generate_cosmo_data(1024)
    test_preds = model(test_inputs)
    test_mae = torch.mean(torch.abs(test_preds - test_targets))
    final_coherence = 1.0 - test_mae / (torch.std(test_targets) + 1e-6)
    final_entropy = torch.std(test_preds - test_targets).item()

print(f"\nFinal Evaluation:")
print(f"  Test Coherence: {final_coherence:.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses)
plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(coherences, color='green')
plt.title('Coherence')
plt.savefig("e8_cosmo_plots.png")
print("Plots saved to: e8_cosmo_plots.png")