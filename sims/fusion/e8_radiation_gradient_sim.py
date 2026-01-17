import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Radiation Gradient Simulator
# Models radiation acting on gradients in spectrum (continuous EM/power-law) 
# vs. strata (discrete density/opacity layers), using triality to blend/null
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_strata = 6                # Discrete opacity strata (e.g., ionization levels)
n_spectrum_modes = 96       # Continuous frequency modes (radiation spectrum)
batch_size = 48
epochs = 150
lr = 0.0005
triality_strength = 0.8     # Triality mixing for spectrum-strata hybridization

# Generate synthetic radiation data: strata opacity + spectral emission
def generate_radiation_data(batch_size):
    # Strata: discrete opacity layers (step-like, with jumps)
    strata_bounds = torch.linspace(0.2, 3.0, n_strata + 1, device=device)
    stratum_idx = torch.randint(0, n_strata, (batch_size,), device=device)
    opacity = strata_bounds[stratum_idx] + torch.randn(batch_size, device=device) * 0.02  # intra-stratum variance
    
    # Spectrum: continuous radiation frequencies (blackbody-like, with Planck gradient)
    freq = torch.logspace(np.log10(1e-3), np.log10(1e3), n_spectrum_modes, device=device)
    # Simplified Planck spectrum: B(ν) ≈ ν^3 / (exp(hν/kT) - 1), approx power-law for sim
    temp = torch.rand(batch_size, device=device) * 100 + 50  # Temperature variation
    spectrum_intensity = (freq.unsqueeze(0) ** 3) / (torch.exp(freq.unsqueeze(0) / temp.unsqueeze(1)) - 1 + 1e-6)
    # Sample mean intensity as spectral feature
    mean_intensity = spectrum_intensity.mean(dim=1)
    
    # Position (e.g., optical depth τ)
    tau = torch.rand(batch_size, device=device) * 5.0
    
    # Target: radiation gradient (dI/dτ) - absorptive in strata + emissive in spectrum
    # Simplified: exponential decay in strata + power-law in spectrum
    target_gradient = -opacity * torch.exp(-tau) + 0.2 * torch.log(1 + mean_intensity)
    target_gradient += torch.randn(batch_size, device=device) * 0.05  # noise
    
    # Inputs: [opacity (strata), mean_intensity (spectrum), tau]
    inputs = torch.stack([opacity, mean_intensity, tau], dim=1)
    return inputs, target_gradient.unsqueeze(1)

# E8 Triality Layer: cycles rotations to blend spectrum/strata gradients
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

# Model: Predicts radiation gradients with triality blending
class E8RadiationNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=192, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Blend strata-spectrum
        x = self.act(self.fc1(x))
        x = self.triality2(x)  # Refine gradient prediction
        return self.out(x)

# Initialize
model = E8RadiationNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.SmoothL1Loss()  # Robust to strata jumps

# Training
losses = []
coherences = []
for epoch in range(epochs):
    inputs, targets = generate_radiation_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Coherence: 1 - normalized Huber-like MAE
    with torch.no_grad():
        mae = torch.mean(torch.abs(preds - targets))
        coherence = 1.0 - mae / (torch.max(torch.abs(targets)) + 1e-6)
        coherences.append(coherence.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Coherence: {coherence.item():.5f}")

# Test eval
with torch.no_grad():
    test_inputs, test_targets = generate_radiation_data(1024)
    test_preds = model(test_inputs)
    test_mae = torch.mean(torch.abs(test_preds - test_targets))
    final_coherence = 1.0 - test_mae / (torch.max(torch.abs(test_targets)) + 1e-6)
    final_entropy = torch.std(test_preds - test_targets).item()

print(f"\nFinal Evaluation:")
print(f"  Test Coherence: {final_coherence:.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Smooth L1')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(coherences, label='Coherence', color='orange')
plt.title('Coherence Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Coherence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("e8_radiation_gradients.png")
print("Plots saved to: e8_radiation_gradients.png")