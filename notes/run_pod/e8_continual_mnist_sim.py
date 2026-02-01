Restart runtime first (Runtime → Restart runtime) for clean memory

!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
import math

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed + stability (fast epochs)
triality = 3
dim = 384  # reduced for speed
latent_dim = 8
num_tasks = 10  # sequential permuted tasks
epochs_per_task = 5000  # reduced — fast now, full proof
lr = 5e-5
use_amp = True
use_checkpoint = True  # memory saver

# Permuted MNIST continual benchmark (real handwritten digits)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(root="./", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)  # balanced batch
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Create permuted versions for tasks
tasks = []
for task_id in range(num_tasks):
    perm = torch.randperm(28*28)
    def permute(x):
        return x.view(-1, 28*28)[:, perm].view(-1, 1, 28, 28)
    tasks.append(permute)

# E8 roots – precompute
def get_e8_roots():
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for signs in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                v = torch.zeros(8)
                v[i] = signs[0]; v[j] = signs[1]
                roots.append(v); roots.append(-v)
    for signs in range(1 << 8):
        v = torch.tensor([(1 if (signs & (1<<k)) else -1) for k in range(8)], dtype=torch.float32) * 0.5
        if bin(signs).count('1') % 2 == 0:
            roots.append(v); roots.append(-v)
    roots = torch.stack(roots[:240])
    return roots / roots.norm(dim=-1, keepdim=True)

e8_roots = get_e8_roots().to(device)

# Triality Cycle Block (detached step)
class ContinualCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        step_float = float(step)  # detached
        pump = 0.8 * torch.sin(torch.tensor(step_float, device=device) * 0.006 * 2 * math.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Dummy cycle for ablation (ignores step)
class DummyCycle(nn.Module):
    def forward(self, x, step=None):
        return x

# Model with ablation support
class E8ContinualLongUpdate(nn.Module):
    def __init__(self, depth=32, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.proj = nn.Linear(784, dim)  # project flattened MNIST to dim
        self.cycle = ContinualCycleBlock() if use_triality else DummyCycle()  # fixed DummyCycle
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)  # MNIST classes

    def forward(self, x, step):
        x = x.view(x.size(0), -1)  # flatten to (batch, 784)
        x = self.proj(x)  # project to dim
        x = x.unsqueeze(1)  # (batch, 1, dim) for attention (single "token" proxy)
        x = self.cycle(x, step)
        for layer in self.layers:
            if use_checkpoint:
                attn, _ = checkpoint(layer, x, x, x, use_reentrant=False)
            else:
                attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return self.head(x.mean(dim=1))  # classification head

# Models
model = E8ContinualLongUpdate(use_triality=True).to(device)

model_ablation = E8ContinualLongUpdate(use_triality=False).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

loss_fn = nn.CrossEntropyLoss()

# Continual training across tasks
accuracy_hist = {i: [] for i in range(num_tasks)}
accuracy_abl_hist = {i: [] for i in range(num_tasks)}

for task_id in range(num_tasks):
    print(f"\n=== Training on Task {task_id} ===")
    
    permute = tasks[task_id]
    
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images = permute(images).to(device)
            labels = labels.to(device)
            
            opt.zero_grad(set_to_none=True)
            opt_ablation.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                outputs = model(images, task_id * epochs_per_task + epoch)
                loss = loss_fn(outputs, labels)

                outputs_abl = model_ablation(images, task_id * epochs_per_task + epoch)
                loss_abl = loss_fn(outputs_abl, labels)

            scaler.scale(loss).backward() if use_amp else loss.backward()
            scaler.unscale_(opt) if use_amp else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
            scaler.step(opt) if use_amp else opt.step()
            scaler.update() if use_amp else None

            scaler_ablation.scale(loss_abl).backward() if use_amp else loss_abl.backward()
            scaler_ablation.unscale_(opt_ablation) if use_amp else None
            torch.nn.utils.clip_grad_norm_(model_ablation.parameters(), 1e6)
            scaler_ablation.step(opt_ablation) if use_amp else opt_ablation.step()
            scaler_ablation.update() if use_amp else None

        if epoch % 2000 == 0:
            print(f"Task {task_id} Epoch {epoch} | Loss {loss.item():.6f}")

    # Test retention on all previous tasks
    with torch.no_grad():
        for prev_task in range(task_id + 1):
            permute_prev = tasks[prev_task]
            acc = 0
            acc_abl = 0
            for images, labels in test_loader:
                images = permute_prev(images).to(device)
                labels = labels.to(device)

                outputs = model(images, task_id * epochs_per_task)
                acc += (outputs.argmax(dim=1) == labels).float().mean().item()

                outputs_abl = model_ablation(images, task_id * epochs_per_task)
                acc_abl += (outputs_abl.argmax(dim=1) == labels).float().mean().item()

            acc /= len(test_loader)
            acc_abl /= len(test_loader)
            accuracy_hist[prev_task].append(acc)
            accuracy_abl_hist[prev_task].append(acc_abl)

# Sigma Retention Test
e8_retention = np.mean([accuracy_hist[i][-1] for i in range(num_tasks)])
abl_retention = np.mean([accuracy_abl_hist[i][-1] for i in range(num_tasks)])
ret_std = np.std([accuracy_hist[i][-1] for i in range(num_tasks)] + [accuracy_abl_hist[i][-1] for i in range(num_tasks)])
sigma_retention = (e8_retention - abl_retention) / ret_std if ret_std > 0 else 0

print(f"Final Retention Sigma: {sigma_retention:.2f}")

# Visualization (retention curves)
plt.figure(figsize=(12,6))
for task_id in range(num_tasks):
    plt.plot(accuracy_hist[task_id], label=f'Task {task_id} Triality')
    plt.plot(accuracy_abl_hist[task_id], label=f'Task {task_id} Ablation', linestyle='--')

plt.title("Continual Long Update: Accuracy Retention Across Tasks")
plt.xlabel("Task Progress")
plt.ylabel("Accuracy on Task")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.text(0.95, 0.95, f"Retention Sigma: {sigma_retention:.2f}", transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig("continual_long_update_retention_visualization.png")
plt.show()

print("Visualization saved as continual_long_update_retention_visualization.png")