import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# === TrialityCycleBlock (from your gist) ===
class TrialityCycleBlock(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or dim * 4
        self.proj_vector   = nn.Linear(dim, hidden)
        self.proj_spinor   = nn.Linear(dim, hidden)
        self.proj_cospinor = nn.Linear(dim, hidden)
        self.merge = nn.Linear(3 * hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        p1 = F.gelu(self.proj_vector(x))
        p2 = F.gelu(self.proj_spinor(x))
        p3 = F.gelu(self.proj_cospinor(x))
        fused = torch.cat([p1, p2, p3], dim=-1)
        out = self.merge(fused)
        out = self.dropout(out)
        out = self.norm(residual + out * self.gate.sigmoid())
        return out

# === Ultra-light CNN ===
class TinyCNN(nn.Module):
    def __init__(self, use_triality=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc_dim = 64 * 4 * 4  # 1024 after 3 pools on 32×32
        
        if use_triality:
            self.triality = TrialityCycleBlock(dim=self.fc_dim, hidden=512)
        
        self.fc1 = nn.Linear(self.fc_dim, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.use_triality = use_triality

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        
        if self.use_triality:
            x = self.triality(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data (download once)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader  = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)  # bigger test batch = faster eval

# Simple entropy calculator (in nats)
def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean().item()

# Train + eval with entropy
def train_and_eval(use_triality=False, epochs=5):
    device = torch.device("cpu")
    model = TinyCNN(use_triality=use_triality).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n=== {'Triality ON' if use_triality else 'Baseline (Triality OFF)'} ===")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 500 == 499:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i+1:4d} | Loss: {running_loss/500:.3f}")
                running_loss = 0.0
    
    # Eval
    model.eval()
    correct = 0
    total = 0
    total_entropy = 0.0
    count = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            ent = compute_entropy(outputs)
            total_entropy += ent * inputs.size(0)
            count += inputs.size(0)
    
    acc = 100 * correct / total
    avg_entropy = total_entropy / count
    print(f"Acc: {acc:.2f}% | Avg Logit Entropy: {avg_entropy:.4f} nats")
    return acc, avg_entropy

# Run both
print("Starting baseline run...")
baseline_acc, baseline_ent = train_and_eval(use_triality=False, epochs=5)

print("\nStarting triality run...")
triality_acc, triality_ent = train_and_eval(use_triality=True, epochs=5)

print("\n=== Summary ===")
print(f"Baseline:   Acc {baseline_acc:.2f}% | Entropy {baseline_ent:.4f} nats")
print(f"Triality:   Acc {triality_acc:.2f}% | Entropy {triality_ent:.4f} nats")
delta_acc = triality_acc - baseline_acc
delta_ent = triality_ent - baseline_ent
print(f"Delta:      Acc {delta_acc:+.2f}% | Entropy {delta_ent:+.4f} nats")