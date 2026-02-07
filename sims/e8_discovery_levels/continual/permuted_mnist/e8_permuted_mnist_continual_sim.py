# First cell: Installs (run once)
!pip install torch matplotlib numpy

# Second cell: E8 Triality Continual Learning on Permuted MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed
triality = 3
dim = 384
latent_dim = 8
seq_len = 784  # flattened MNIST
batch_size = 128
tasks = 10
epochs_per_task = 1000  # fast for trend
lr = 5e-5
use_amp = True

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create permuted tasks
task_loaders = []
for t in range(tasks):
    perm = torch.randperm(seq_len)
    train_data = mnist_train.data.float().view(-1, seq_len)[:, perm] / 255.0
    train_labels = mnist_train.targets
    test_data = mnist_test.data.float().view(-1, seq_len)[:, perm] / 255.0
    test_labels = mnist_test.targets
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    task_loaders.append((train_loader, DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)))

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

# Triality Cycle Block
class ContinualCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        with torch.no_grad():
            pump_scalar = 0.8 * math.sin(step * 0.006 * 2 * math.pi)
        pump = torch.full((1, x.shape[1], 1), pump_scalar, device=device)
        emb_broadcast = emb.unsqueeze(0)
        x_rot1 = x * (emb_broadcast.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=1) * emb_broadcast.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=1) * emb_broadcast.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / triality
        return fused

# Dummy cycle for ablation
class DummyCycle(nn.Module):
    def forward(self, x, step=None):
        return x

# Model with ablation support
class E8Continual(nn.Module):
    def __init__(self, depth=32, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.proj = nn.Linear(seq_len, dim)
        self.cycle = ContinualCycleBlock() if use_triality else DummyCycle()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality if use_triality else 8, batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)  # MNIST classes

    def forward(self, x, step):
        x = self.proj(x)
        if self.use_triality:
            x = self.cycle(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + attn
            x = self.norm(x)
        x = x.mean(dim=1)  # global pool
        return self.head(x)

# Models
model = E8Continual(use_triality=True).to(device)
model_ablation = E8Continual(use_triality=False).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

opt_ablation = torch.optim.AdamW(model_ablation.parameters(), lr=lr)
scaler_ablation = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

loss_fn = nn.CrossEntropyLoss()

# Continual training
accuracies_triality = []
accuracies_ablation = []

for task_id, (train_loader, test_loader) in enumerate(task_loaders):
    # Triality training
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                logits = model(images, epoch + task_id*epochs_per_task)  # unique step
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler.unscale_(opt) if use_amp else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
        scaler.step(opt) if use_amp else opt.step()
        scaler.update() if use_amp else None

    # Ablation training
    opt_ablation.zero_grad(set_to_none=True)
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                logits_abl = model_ablation(images, epoch + task_id*epochs_per_task)
                loss_abl = loss_fn(logits_abl, labels)
            scaler_ablation.scale(loss_abl).backward() if use_amp else loss_abl.backward()
        scaler_ablation.unscale_(opt_ablation) if use_amp else None
        torch.nn.utils.clip_grad_norm_(model_ablation.parameters(), 1e6)
        scaler_ablation.step(opt_ablation) if use_amp else opt_ablation.step()
        scaler_ablation.update() if use_amp else None

    # Evaluate on all seen tasks
    model.eval()
    model_ablation.eval()
    correct_tri = 0
    correct_abl = 0
    total = 0
    for seen_task in range(task_id + 1):
        _, test_loader_seen = task_loaders[seen_task]
        for images, labels in test_loader_seen:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images, 0)  # dummy step
                pred = logits.argmax(dim=1)
                correct_tri += (pred == labels).sum().item()

                logits_abl = model_ablation(images, 0)
                pred_abl = logits_abl.argmax(dim=1)
                correct_abl += (pred_abl == labels).sum().item()
            total += labels.size(0)

    acc_tri = 100 * correct_tri / total
    acc_abl = 100 * correct_abl / total
    accuracies_triality.append(acc_tri)
    accuracies_ablation.append(acc_abl)
    print(f"After Task {task_id+1}: Triality Acc {acc_tri:.2f}% | Ablation Acc {acc_abl:.2f}%")

print("E8 Triality Continual Learning Benchmarks on Permuted MNIST complete!")