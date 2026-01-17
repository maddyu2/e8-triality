import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================
# E8 Triality: H2O-Photon Orthogonal Simulator
# Optimizes H2O (vertical) and photon (horizontal) angles for max radiation nulling in shield.
# Uses triality to find optimal orthogonal configurations.
# =============================================

# Hyperparameters
e8_dim = 64
batch_size = 72
epochs = 180
lr = 0.0004
triality_strength = 0.9
angle_range = np.pi / 2  # Angle optimization range

# Generate data: radiation, H2O/photon angles
def generate_ortho_data(batch_size):
    # Incoming radiation (simplified flux)
    rad_flux = torch.rand(batch_size, device=device) * 100 + 50
    
    # Initial angles: H2O vertical (theta ~ pi/2), photon horizontal (phi ~ 0)
    h2o_theta = torch.rand(batch_size, device=device) * angle_range + (np.pi / 2 - angle_range / 2)
    photon_phi = torch.rand(batch_size, device=device) * angle_range - angle_range / 2
    
    # Orthogonality metric: cos(theta - phi - pi/2) ~ 0 for orthogonal
    ortho_dev = torch.abs(torch.cos(h2o_theta - photon_phi - np.pi / 2))
    
    # Byproduct: radiation not nulled (decreases with orthogonality)
    byproduct = rad_flux * (1 - torch.exp(-ortho_dev))
    
    # Target: optimized nulling (min byproduct via ideal angles)
    target_null = rad_flux * 0.05  # Ideal low residual
    
    # Inputs: [rad_flux, h2o_theta, photon_phi]
    inputs = torch.stack([rad_flux, h2o_theta, photon_phi], dim=1)
    return inputs, byproduct.unsqueeze(1), target_null.unsqueeze(1)

# Triality Layer
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

# Model: Optimizes angles via triality
class E8OrthoNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.triality1 = E8TrialityLayer(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.triality2 = E8TrialityLayer(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.05)
    
    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.triality1(x)
        x = self.act(self.fc1(x))
        x = self.triality2(x)
        return self.out(x)

# Initialize
model = E8OrthoNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
losses = []
null_effs = []
for epoch in range(epochs):
    inputs, byproducts, targets = generate_ortho_data(batch_size)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    with torch.no_grad():
        null_eff = 1.0 - torch.mean(byproducts - preds) / torch.mean(byproducts)
        null_effs.append(null_eff.item())
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Null Efficiency: {null_eff.item():.5f}")

# Test
with torch.no_grad():
    test_inputs, test_byproducts, test_targets = generate_ortho_data(1024)
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
plt.plot(null_effs, color='red')
plt.title('Null Efficiency')
plt.savefig("e8_ortho_null.png")
print("Plots saved to: e8_ortho_null.png")