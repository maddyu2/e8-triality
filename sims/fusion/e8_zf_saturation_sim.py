import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ZF saturation sim with tertiary instability & GAM damping
n_strata = 6
n_modes = 160
batch_size = 56
epochs = 250
lr = 0.0003
triality_strength = 0.95

def generate_zf_data(batch_size):
    rho = torch.linspace(0, 1, n_strata, device=device).unsqueeze(0).repeat(batch_size, 1)
    a_L_T = torch.rand(batch_size, n_strata, device=device) * 3 + 2
    omega_E = torch.rand(batch_size, device=device) * 0.4 + 0.1  # ZF shear
    damping_rate = torch.rand(batch_size, device=device) * 0.15 + 0.05  # GAM + tertiary
    qs_factor = torch.rand(batch_size, device=device) * 0.2 + 0.8  # QS enhancement
    
    k_y = torch.logspace(-1, 0.3, n_modes, device=device)
    gamma_turb = a_L_T.mean(dim=1).unsqueeze(1) / (k_y + 1e-3) * (k_y ** -0.6)
    
    # ZF saturation: turbulence drive minus shear + damping
    gamma_net = gamma_turb - omega_E.unsqueeze(1) * qs_factor.unsqueeze(1) + damping_rate.unsqueeze(1)
    gamma_net += torch.randn(batch_size, n_modes, device=device) * 0.025
    
    target_flux = gamma_net.clip(min=0).mean(dim=1) / (k_y.mean() ** 2) * 0.4
    entropy = torch.sum(gamma_net ** 2, dim=1) * 0.007
    
    inputs = torch.cat([
        rho.view(batch_size, -1),
        a_L_T.view(batch_size, -1),
        omega_E.unsqueeze(1).repeat(1, n_modes),
        damping_rate.unsqueeze(1).repeat(1, n_modes),
        qs_factor.unsqueeze(1).repeat(1, n_modes),
        gamma_net
    ], dim=1)
    
    return inputs, entropy.unsqueeze(1), target_flux.unsqueeze(1)

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
        return self.strength * (x + x1 + x2 + x3) / 4.0

class E8ZFSatNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, output_dim=2):
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

input_dim = (n_strata * 2) + n_modes * 3
model = E8ZFSatNet(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

losses = []
for epoch in range(epochs):
    inputs, entropy, flux = generate_zf_data(batch_size)
    targets = torch.cat([entropy, flux], dim=1)
    preds = model(inputs)
    
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Entropy: {preds[:,0].mean().item():.6f}")

with torch.no_grad():
    test_inputs, test_entropy, test_flux = generate_zf_data(1024)
    test_preds = model(test_inputs)
    entropy_mae = torch.mean(torch.abs(test_preds[:,0] - test_entropy))
    flux_mae = torch.mean(torch.abs(test_preds[:,1] - test_flux))
    coherence = 1.0 - (entropy_mae + flux_mae) / 2
    final_entropy = torch.mean(test_preds[:,0]).item()

print(f"\nFinal: Entropy MAE {entropy_mae:.6f} | Flux MAE {flux_mae:.6f}")
print(f"Coherence {coherence:.6f} | Mean Entropy {final_entropy:.6f} nats")

plt.plot(losses)
plt.title('Training Loss')
plt.savefig("e8_zf_saturation_w7x.png")
print("Plot saved to: e8_zf_saturation_w7x.png")