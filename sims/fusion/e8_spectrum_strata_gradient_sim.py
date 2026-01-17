import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Spectrum-Strata Gradient Simulator
# Blends continuous spectral turbulence (wavenumbers k)
# with stratified plasma density layers (discrete bins)
# Triality rotations smooth gradients across the interface
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced from 248 for compute
n_strata = 8                # Number of discrete density layers (strata)
n_k_modes = 128             # Number of continuous wavenumber modes (spectrum)
batch_size = 64
epochs = 120
lr = 0.0008
triality_strength = 0.7     # Controls how strongly triality mixes strata ↔ spectrum

# Generate synthetic plasma data: stratified density + spectral turbulence
def generate_plasma_data(batch_size):
    # Strata: discrete density bins (piecewise constant, with sharp jumps)
    strata_bins = torch.linspace(0.1, 2.5, n_strata + 1, device=device)
    stratum_idx = torch.randint(0, n_strata, (batch_size,), device=device)
    density = strata_bins[stratum_idx] + torch.randn(batch_size, device=device) * 0.03  # small intra-stratum noise
    
    # Spectrum: continuous turbulence wavenumbers (power-law like k^{-5/3})
    k = torch.logspace(np.log10(0.1), np.log10(100), n_k_modes, device=device)
    power_spectrum = k ** (-5/3) * (1 + torch.randn(n_k_modes, device=device) * 0.1)
    # Sample spectral amplitude at random k for each batch item
    k_sample_idx = torch.randint(0, n_k_modes, (batch_size,), device=device)
    turb_amplitude = power_spectrum[k_sample_idx]
    
    # Target: gradient we want to learn/predict (shear across strata + spectral decay)
    # Simplified: radial shear gradient + spectral slope
    r = torch.rand(batch_size, device=device) * 3.0 + 0.5  # normalized radius
    target_gradient = -0.4 * torch.sin(2 * np.pi * r) + 0.15 * torch.log(1 + turb_amplitude)
    
    # Input features: [density (strata), turb_amplitude (spectrum), r]
    inputs = torch.stack([density, turb_amplitude, r], dim=1)
    return inputs, target_gradient.unsqueeze(1)

# E8 Triality Layer: simulates triality rotation mixing strata ↔ spectrum
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Learnable rotation matrices for triality (simplified: 3 orthogonal-ish transforms)
        self.rot1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.rot2 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.rot3 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        # Triality cycle: x → rot1(x) → rot2(rot1(x)) → rot3(rot2(rot1(x))) → mix back
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        # Weighted cyclic sum (triality signature)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Full model: Spectrum-Strata Gradient Predictor
class E8SpectrumStrataNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)           # First triality mixing
        x = self.act(self.fc1(x))
        x = self.triality2(x)           # Second triality mixing
        return self.out(x)

# Initialize
model = E8SpectrumStrataNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()

# Training loop
losses = []
coherences = []
for epoch in range(epochs):
    inputs, targets = generate_plasma_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Quick coherence metric (1 - normalized MAE)
    with torch.no_grad():
        mae = torch.mean(torch.abs(preds - targets))
        coherence = 1.0 - mae / (torch.std(targets) + 1e-6)
        coherences.append(coherence.item())
    
    if epoch % 15 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Coherence: {coherence.item():.5f}")

# Final evaluation on larger test set
with torch.no_grad():
    test_inputs, test_targets = generate_plasma_data(2048)
    test_preds = model(test_inputs)
    test_mae = torch.mean(torch.abs(test_preds - test_targets))
    final_coherence = 1.0 - test_mae / (torch.std(test_targets) + 1e-6)
    final_entropy = torch.std(test_preds - test_targets).item()  # proxy for residual entropy

print(f"\nFinal Evaluation:")
print(f"  Test Coherence: {final_coherence:.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot results (will save to file when running on laptop)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(coherences, label='Coherence', color='green')
plt.title('Coherence Evolution')
plt.xlabel('Epoch')
plt.ylabel('Coherence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("e8_spectrum_strata_gradients.png")
print("Plots saved to: e8_spectrum_strata_gradients.png")