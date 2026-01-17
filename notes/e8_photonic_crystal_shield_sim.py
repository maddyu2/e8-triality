import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: Photonic Crystal Shield Simulator
# Models photonic crystals in Starship shield for radiation nulling.
# Optimizes bandgap via triality in strata-spectrum.
# =============================================

# Hyperparameters
e8_dim = 64
n_pc_layers = 5  # Crystal layers
batch_size = 56
epochs = 160
lr = 0.0005
triality_strength = 0.85

# Generate data: radiation, PC params
def generate_pc_shield_data(batch_size):
    # Radiation flux
    rad_flux = torch.rand(batch_size, device=device) * 200 + 100
    
    # PC strata: periodic refractive indices (layers)
    ref_index = torch.randn(batch_size, n_pc_layers, device=device) * 0.3 + 1.5
    
    # Bandgap proxy: forbidden freq range
    bandgap_width = torch.rand(batch_size, device=device) * 0.5 + 0.2
    
    # Byproduct: unblocked radiation
    byproduct = rad_flux * (1 - bandgap_width) * ref_index.mean(dim=1)
    
    # Target: nulled via optimal PC
    target_null = rad_flux * 0.02
    
    # Inputs: [rad_flux, ref_index flat, bandgap_width]
    inputs = torch.cat([rad_flux.unsqueeze(1), ref_index.view(batch_size, -1), 
                        bandgap_width.unsqueeze(1)], dim=1)
    return inputs, byproduct.unsqueeze(1), target_null.unsqueeze(1)

# Triality Layer
class E8TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rot1 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.007)
        self.rot2 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.007)
        self.rot3 = nn.Parameter(torch.eye(dim, device=device) + torch.randn(dim, dim, device=device) * 0.007)
        self.strength = nn.Parameter(torch.tensor(triality_strength))
    
    def forward(self, x):
        x1 = torch.matmul(x, self.rot1)
        x2 = torch.matmul(x1, self.rot2)
        x3 = torch.matmul(x2, self.rot3)
        mixed = self.strength * (x + x1 + x2 + x3) / 4.0
        return mixed

# Model
class E8PCShieldNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=288, output_dim=1):
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

# Input dim
input_dim = 1 + n_pc_layers + 1

# Initialize
model = E8PCShieldNet(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
null_effs = []
for epoch in range(epochs):
    inputs, byproducts, targets = generate_pc_shield_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    with torch.no_grad():
        null_eff = 1.0 - torch.mean(byproducts - preds) / torch.mean(byproducts)
        null_effs.append(null_eff.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Null Efficiency: {null_eff.item():.5f}")

# Test
with torch.no_grad():
    test_inputs, test_byproducts, test_targets = generate_pc_shield_data(1024)
    test_preds = model(test_inputs)
    test_null_eff = 1.0 - torch.mean(test_byproducts - test_preds) / torch.mean(test_byproducts)
    final_entropy = torch.std(test_preds - test_targets).item()

print(f"\nFinal Evaluation:")
print(f"  Null Efficiency: {test_null_eff:.6f}")
print(f"  Residual Entropy: {final_entropy:.6f} nats")

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(null_effs, color='blue')
plt.title('Null Efficiency')
plt.savefig("e8_pc_shield.png")
print("Plots saved to: e8_pc_shield.png")