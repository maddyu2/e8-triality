import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Bremsstrahlung Losses Simulator
# Focuses on bremsstrahlung radiation losses in stratified plasma densities,
# using triality-nulling for ITER-like bootstrap current optimization.
# =============================================

# Hyperparameters
e8_effective_dim = 64       # Reduced E8 dim
n_strata = 7                # Discrete density strata (e.g., core-pedestal-divertor)
batch_size = 64
epochs = 160
lr = 0.0006
triality_strength = 0.85    # Strong nulling for losses

# Generate synthetic data: stratified densities + bremsstrahlung losses
def generate_brems_data(batch_size):
    # Strata: discrete density layers (n_e in m^-3, log-scale)
    strata_log_ne = torch.linspace(18, 21, n_strata + 1, device=device)  # 10^18 to 10^21
    stratum_idx = torch.randint(0, n_strata, (batch_size,), device=device)
    log_ne = strata_log_ne[stratum_idx] + torch.randn(batch_size, device=device) * 0.05
    ne = 10 ** log_ne
    
    # Temperature (Te in eV, varied)
    te = torch.rand(batch_size, device=device) * 5000 + 1000  # 1-6 keV for ITER
    
    # Z_eff (effective charge, impurities)
    z_eff = torch.rand(batch_size, device=device) * 2.0 + 1.0  # 1-3 for typical plasmas
    
    # Bremsstrahlung power loss (P_brem ~ n_e^2 * Z_eff^2 * sqrt(Te))
    p_brem = (ne ** 2) * (z_eff ** 2) * torch.sqrt(te) * 5.35e-37  # Simplified formula (W/m^3)
    p_brem += torch.randn(batch_size, device=device) * p_brem * 0.1  # Noise
    
    # Bootstrap current proxy (j_bs ~ grad(p)/B, simplified as target to null losses against)
    grad_p = torch.rand(batch_size, device=device) * 1e6  # Pressure gradient proxy
    target_j_bs = grad_p / (2 * np.pi * 5.0)  # B~5T for ITER, simplified
    
    # Inputs: [ne, te, z_eff, grad_p]
    inputs = torch.stack([ne, te, z_eff, grad_p], dim=1)
    return inputs, p_brem.unsqueeze(1), target_j_bs.unsqueeze(1)

# E8 Triality Layer for nulling
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot2 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.rot3 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.01)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model: Predicts/nulls brems losses while optimizing bootstrap
class E8BremsNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=2):  # Outputs: nulled p_brem, opt j_bs
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)  # Null strata effects
        x = self.act(self.fc1(x))
        x = self.triality2(x)  # Optimize bootstrap
        return self.out(x)

# Initialize
model = E8BremsNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
for epoch in range(epochs):
    inputs, p_brem, j_bs = generate_brems_data(batch_size)
    targets = torch.cat([p_brem, j_bs], dim=1)  # Predict both, but null p_brem
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Test eval (focus on nulling: low residual p_brem)
with torch.no_grad():
    test_inputs, test_p_brem, test_j_bs = generate_brems_data(2048)
    test_preds = model(test_inputs)
    test_p_brem_pred = test_preds[:, 0].unsqueeze(1)
    residual_p_brem = torch.mean(torch.abs(test_p_brem_pred - test_p_brem)) / torch.mean(test_p_brem)
    coherence = 1.0 - residual_p_brem.item()
    entropy = torch.std(test_preds[:, 0] - test_p_brem.squeeze()).item()

print(f"\nFinal Evaluation:")
print(f"  Brems Nulling Coherence: {coherence:.6f}")
print(f"  Residual Entropy: {entropy:.6f} nats")

# Save plot (dummy for now)
plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_brems_losses.png")
print("Plot saved to: e8_brems_losses.png")