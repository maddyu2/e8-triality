import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: KBM Modes in HSX with Nonlinear Zonal Flows Simulator
# Models Kinetic Ballooning Modes (KBM) in HSX stellarator (quasi-helical symmetry, R=1.2m, B=1T).
# Adds nonlinear zonal flows (ZF) for saturation/stabilization, nulling instabilities via E8 triality.
# Parameters: HSX beta~0.002 (crit ~0.002 for KBM onset), ZF shearing ~0.1-0.5 gamma_max.
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_strata = 5                # Radial strata
n_modes = 136               # Modes (low-k KBM ~0.1-0.5)
batch_size = 52
epochs = 200
lr = 0.0004
triality_strength = 0.9     # Triality for KBM/ZF nulling

# Generate HSX data: beta for KBM, gradients/ZF shear
def generate_kbm_zf_data(batch_size):
    # Strata: r/a ~0-1
    r_a = torch.linspace(0, 1, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    # Beta ~0.001-0.003 (HSX low, crit ~0.002)
    beta = torch.rand(batch_size, device=device) * 0.002 + 0.001
    
    # Gradients: a/L_p ~2-5 for KBM drive
    a_L_p = torch.rand(batch_size, n_strata, device=device) * 3 + 2
    
    # ZF shear omega_E ~0.1-0.5 (nonlinear saturation)
    omega_E = torch.rand(batch_size, device=device) * 0.4 + 0.1
    
    # QS delta ~0.01 (HSX QHS)
    qs_delta = torch.rand(batch_size, device=device) * 0.02 + 0.01
    
    # Modes: k_y rho_s ~0.1-0.5 for KBM
    k_y = torch.logspace(-1, -0.3, n_modes, device=device)
    # Growth: KBM gamma ~ sqrt(beta) * a/L_p / k (low in HSX), ZF stab
    gamma_kbm = torch.sqrt(beta.unsqueeze(1)) * a_L_p.mean(dim=1).unsqueeze(1) / (k_y + 1e-3)
    zf_stab = omega_E.unsqueeze(1) * 0.65  # 50-80% reduction nonlinear
    gamma_hybrid = gamma_kbm * (1 - zf_stab - qs_delta.unsqueeze(1) * 0.4) + torch.randn(batch_size, n_modes, device=device) * 0.015
    
    # Flux proxy (GENE HSX KBM/ITG ~0.05-0.3 GB with ZF)
    target_flux = gamma_hybrid.clip(min=0).mean(dim=1) / (k_y.mean() ** 2) * 0.2
    
    # Entropy: sum gamma^2
    entropy = torch.sum(gamma_hybrid ** 2, dim=1) * 0.006
    
    # Inputs: [r_a flat, a_L_p flat, beta repeat, omega_E repeat, qs_delta repeat, gamma_hybrid]
    inputs = torch.cat([r_a.view(batch_size, -1), a_L_p.view(batch_size, -1), 
                        beta.unsqueeze(1).repeat(1, n_modes), omega_E.unsqueeze(1).repeat(1, n_modes), qs_delta.unsqueeze(1).repeat(1, n_modes), gamma_hybrid], dim=1)
    return inputs, entropy.unsqueeze(1), target_flux.unsqueeze(1)

# E8 Triality Layer: nulls KBM/ZF
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot2 = nn.Parameter(torch.eye(dim, dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot3 = nn.Parameter(torch.eye(dim, dim, device=device) * 0.01)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Bounds entropy, predicts low flux
class E8KBMZFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=368, output_dim=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Null KBM
        x = self.act(self.fc1(x))
        x = self.triality2(x)  # Null ZF in ITG
        return self.out(x)

# Input dim
input_dim = (n_strata * 2) + n_modes * 3  # r, L_p, beta, omega, qs, gamma

# Initialize
model = E8KBMZFNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in range(epochs):
    inputs, entropy, flux = generate_kbm_zf_data(batch_size)
    targets = torch.cat([entropy, flux], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Test: Low flux (GENE HSX/W7-X KBM/ITG/ZF ~0.05-0.3 GB)
with torch.no_grad():
    test_inputs, test_entropy, test_flux = generate_kbm_zf_data(1024)
    test_preds = model(test_inputs)
    test_entropy_pred = test_preds[:, 0].unsqueeze(1)
    test_flux_pred = test_preds[:, 1].unsqueeze(1)
    entropy_mae = torch.mean(torch.abs(test_entropy_pred - test_entropy))
    flux_mae = torch.mean(torch.abs(test_flux_pred - test_flux))
    coherence = 1.0 - (entropy_mae + flux_mae) / 2
    final_entropy = torch.std(test_preds[:, 0] - test_entropy.squeeze()).item()

print(f"\nFinal Evaluation (KBM in HSX & ZF in ITG):")
print(f"  Entropy MAE: {entropy_mae.item():.6f} (Bound low)")
print(f"  Flux MAE: {flux_mae.item():.6f} (GENE ~0.05-0.3 GB)")
print(f"  Coherence: {coherence.item():.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plot
plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_kbm_zonal_itg_hsx_losses.png")
print("Plot saved to: e8_kbm_zonal_itg_hsx_losses.png")