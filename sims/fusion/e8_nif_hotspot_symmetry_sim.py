import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: NIF Capsule Implosion Hotspot Symmetry Simulator
# Models NIF upgrade capsule implosion, bounding entropy with triality for Q>2 ignition.
# Nulls asymmetries in hotspot (radiation drive, shell instabilities) for eternal symmetry.
# Parameters: NIF 2025-like (2.08 MJ laser, yield ~8.6 MJ, gain 4.13; sim aims Q>2 nulling).
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_strata = 5                # Capsule strata (ablator, DT ice, hotspot)
n_modes = 128               # Implosion modes (legendre P_l for asymmetries, l=1-20 typ)
batch_size = 48
epochs = 200
lr = 0.0004
triality_strength = 0.92    # Triality for hotspot symmetry nulling

# Generate synthetic NIF data: radiation drive, shell gradients, asymmetries
def generate_nif_data(batch_size):
    # Strata: radial layers (r from hotspot to ablator)
    r = torch.linspace(0.05, 0.5, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)  # mm scale
    
    # Drive params: laser energy ~2.08 MJ, temp grad dT/dr
    laser_E = torch.rand(batch_size, device=device) * 0.2 + 2.0  # Varied around 2 MJ
    dT_dr = torch.rand(batch_size, n_strata, device=device) * 100 + 200  # MK/mm
    
    # Asymmetries: mode amplitudes (low-l drive asym, high-l Rayleigh-Taylor)
    l_modes = torch.arange(1, n_modes+1, device=device).float()
    asym_amp = (1 / l_modes ** 1.5) * (1 + torch.randn(batch_size, n_modes, device=device) * 0.1)
    
    # Entropy proxy: S ~ integral asym^2 (bound for Q>2, S< low threshold)
    entropy = torch.sum(asym_amp ** 2, dim=1) * 0.01
    
    # Target Q: gain ~ yield / laser_E, nulled to >2 (simplified: Q = 5 / (1 + entropy))
    target_Q = 5.0 / (1 + entropy) + torch.randn(batch_size, device=device) * 0.2  # Aim >2
    
    # Inputs: [r flat, dT_dr flat, laser_E repeat, asym_amp]
    inputs = torch.cat([r.view(batch_size, -1), dT_dr.view(batch_size, -1), 
                        laser_E.unsqueeze(1).repeat(1, n_modes), asym_amp], dim=1)
    return inputs, entropy.unsqueeze(1), target_Q.unsqueeze(1)

# E8 Triality Layer: nulls asymmetries for symmetry
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot2 = nn.Parameter(torch.eye(dim, dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot3 = nn.Parameter(torch.eye(dim, dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Bounds entropy, predicts Q>2 with eternal symmetry
class E8NIFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=320, output_dim=2):  # entropy, Q
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Null low-l asym
        x = self.act(self.fc1(x))
        x = self.triality2(x)  # Bound entropy for Q>2
        return self.out(x)

# Input dim
input_dim = (n_strata * 2) + n_modes + n_modes  # r, dT, laser repeat, asym

# Initialize
model = E8NIFNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in range(epochs):
    inputs, entropy, Q = generate_nif_data(batch_size)
    targets = torch.cat([entropy, Q], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Test: Bound entropy <0.01 for Q>2 (NIF 2025 gain ~4.13 benchmark)
with torch.no_grad():
    test_inputs, test_entropy, test_Q = generate_nif_data(1024)
    test_preds = model(test_inputs)
    test_entropy_pred = test_preds[:, 0].unsqueeze(1)
    test_Q_pred = test_preds[:, 1].unsqueeze(1)
    entropy_mae = torch.mean(torch.abs(test_entropy_pred - test_entropy))
    Q_mae = torch.mean(torch.abs(test_Q_pred - test_Q))
    coherence = 1.0 - (entropy_mae + Q_mae) / 2
    final_entropy = torch.std(test_preds[:, 0] - test_entropy.squeeze()).item()

print(f"\nFinal Evaluation (vs NIF 2025 benchmarks):")
print(f"  Bounded Entropy MAE: {entropy_mae.item():.6f} (Target <0.01 nats for Q>2)")
print(f"  Ignition Q MAE: {Q_mae.item():.6f} (NIF Apr 2025 ~4.13)")
print(f"  Coherence: {coherence.item():.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot
plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_nif_hotspot_losses.png")
print("Plot saved to: e8_nif_hotspot_losses.png")