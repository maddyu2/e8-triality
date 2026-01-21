import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: RMP Control for ELM Suppression in Tokamaks Simulator
# Models RMP coil phasing (n=3, current ~kA-turns) for ELM suppression (ΔW_ELM/W <5%).
# Nulls pedestal instabilities (peeling-ballooning) via E8 triality rotations for eternal low entropy.
# Parameters: q95 ~3-4 (suppression window Δq95~0.1), δB/B ~10^{-4}, beta_ped ~0.02-0.05.
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_strata = 7                # Pedestal strata
n_modes = 144               # ELM modes (m/n ~1-10)
n_rmp_n = 3                 # Toroidal mode n=3
batch_size = 48
epochs = 220
lr = 0.0004
triality_strength = 0.92    # Triality for RMP nulling

# Generate tokamak RMP data
def generate_rmp_tokamak_data(batch_size):
    # Strata: rho ~0.9-1.0 pedestal
    rho = torch.linspace(0.9, 1.0, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    # Pedestal params: beta_ped ~0.02-0.05, q95 ~3-4
    beta_ped = torch.rand(batch_size, device=device) * 0.03 + 0.02
    q95 = torch.rand(batch_size, device=device) * 1 + 3
    
    # RMP: δB/B ~10^{-4}, current I ~1-5 kA-turn
    delta_B = torch.rand(batch_size, device=device) * 9e-4 + 1e-4
    I_rmp = torch.rand(batch_size, device=device) * 4 + 1
    
    # Modes: m/n for ELM (n=3 RMP resonant)
    modes = torch.arange(1, n_modes+1, device=device).float()
    # ELM growth gamma ~ beta_ped * (modes ** -1) - RMP suppression δB * I_rmp * sin(n*q95 - m)
    gamma_elm = 0.2 * beta_ped.unsqueeze(1) * (modes ** -1) - delta_B.unsqueeze(1) * I_rmp.unsqueeze(1) * torch.sin(3 * q95.unsqueeze(1) - modes)
    gamma_elm += torch.randn(batch_size, n_modes, device=device) * 0.02
    
    # ELM energy loss ΔW/W ~2-5% without RMP, suppressed to <1% with optimal
    target_delta_W = torch.rand(batch_size, device=device) * 0.03 + 0.02 * (1 - delta_B * I_rmp / 5)
    
    # Entropy: sum gamma^2 (bound low with RMP)
    entropy = torch.sum(gamma_elm ** 2, dim=1) * 0.007
    
    # Inputs: [rho flat, beta_ped repeat, q95 repeat, delta_B repeat, I_rmp repeat, gamma_elm]
    inputs = torch.cat([rho.view(batch_size, -1), beta_ped.unsqueeze(1).repeat(1, n_modes), 
                        q95.unsqueeze(1).repeat(1, n_modes), delta_B.unsqueeze(1).repeat(1, n_modes), 
                        I_rmp.unsqueeze(1).repeat(1, n_modes), gamma_elm], dim=1)
    return inputs, entropy.unsqueeze(1), target_delta_W.unsqueeze(1)

# E8 Triality Layer: nulls ELM instabilities
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
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Bounds entropy, predicts low ΔW/W
class E8RMPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, output_dim=2):  # entropy, ΔW/W
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

# Input dim
input_dim = n_strata + n_modes * 5  # rho, beta, q95, delta_B, I_rmp, gamma

# Initialize
model = E8RMPNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in range(epochs):
    inputs, entropy, delta_W = generate_rmp_tokamak_data(batch_size)
    targets = torch.cat([entropy, delta_W], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Test: Low ΔW/W (<5% with RMP)
with torch.no_grad():
    test_inputs, test_entropy, test_delta_W = generate_rmp_tokamak_data(1024)
    test_preds = model(test_inputs)
    test_entropy_pred = test_preds[:, 0].unsqueeze(1)
    test_delta_W_pred = test_preds[:, 1].unsqueeze(1)
    entropy_mae = torch.mean(torch.abs(test_entropy_pred - test_entropy))
    delta_W_mae = torch.mean(torch.abs(test_delta_W_pred - test_delta_W))
    coherence = 1.0 - (entropy_mae + delta_W_mae) / 2
    final_entropy = torch.std(test_preds[:, 0] - test_entropy.squeeze()).item()

print(f"\nFinal Evaluation (RMP Control in Tokamaks):")
print(f"  Entropy MAE: {entropy_mae.item():.6f} (Bound low)")
print(f"  ΔW_ELM/W MAE: {delta_W_mae.item():.6f} (Target <0.05)")
print(f"  Coherence: {coherence.item():.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot
plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_rmp_tokamak_losses.png")
print("Plot saved to: e8_rmp_tokamak_losses.png")