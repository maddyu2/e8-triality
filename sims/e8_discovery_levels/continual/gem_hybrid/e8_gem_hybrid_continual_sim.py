# First cell: Keep-alive + installs (run once — prevents disconnects)
from IPython.display import display, Javascript
display(Javascript('''
function ClickConnect(){
  console.log("Keeping alive"); 
  document.querySelector("colab-connect-button")?.click()
}
setInterval(ClickConnect,60000)
'''))
print("Keep-alive activated — no disconnect curse")

!pip install torch torchvision matplotlib numpy

# Second cell: The sim code (E8 + GEM + Hybrid on Permuted MNIST)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import os

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed
dim = 384
latent_dim = 8
seq_len = 784  # flattened MNIST
batch_size = 128
tasks = 10
epochs_per_task = 500  # fast (sigma trend early)
lr = 5e-5
use_amp = True
memory_size = 200  # GEM buffer per task

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "gem_hybrid_checkpoint.pth")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Permuted tasks
task_loaders = []
for t in range(tasks):
    perm = torch.randperm(seq_len)
    train_data = mnist_train.data.float().view(-1, seq_len)[:, perm] / 255.0
    train_labels = mnist_train.targets
    test_data = mnist_test.data.float().view(-1, seq_len)[:, perm] / 255.0
    test_labels = mnist_test.targets
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)
    task_loaders.append((train_loader, test_loader))

# E8 roots
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
class TrialityCycleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // 3, bias=False)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1], device=device) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, 3)
        with torch.no_grad():
            pump_scalar = 0.8 * math.sin(step * 0.006 * 2 * math.pi)
        pump = torch.full((1, x.shape[1], 1), pump_scalar, device=device)
        emb_broadcast = emb.unsqueeze(0)
        x_rot1 = x * (emb_broadcast.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=1) * emb_broadcast.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=1) * emb_broadcast.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / 3
        return fused

# Dummy for ablation
class DummyCycle(nn.Module):
    def forward(self, x, step=None):
        return x

# Base Model (shared for E8, Ablation, Hybrid)
class ContinualModel(nn.Module):
    def __init__(self, use_triality=True):
        super().__init__()
        self.use_triality = use_triality
        self.proj = nn.Linear(seq_len, dim)
        self.cycle = TrialityCycleBlock() if use_triality else DummyCycle()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x, step):
        x = self.proj(x).unsqueeze(1)  # fake seq_len=1
        if self.use_triality:
            x = self.cycle(x, step)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

# Models
model_triality = ContinualModel(use_triality=True).to(device)
model_ablation = ContinualModel(use_triality=False).to(device)
model_hybrid = ContinualModel(use_triality=True).to(device)  # hybrid = triality + GEM

opts = [torch.optim.AdamW(m.parameters(), lr=lr) for m in [model_triality, model_ablation, model_hybrid]]
scalers = [torch.amp.GradScaler('cuda') if use_amp else nullcontext() for _ in range(3)]

loss_fn = nn.CrossEntropyLoss()

# Memory buffers for GEM/hybrid
memory_buffers = [[] for _ in range(tasks)]

accuracies = {'triality': [], 'ablation': [], 'hybrid': []}

start_task = 0

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    cp = torch.load(checkpoint_path)
    for i, m in enumerate([model_triality, model_ablation, model_hybrid]):
        m.load_state_dict(cp[f'model_{i}'])
    for i, o in enumerate(opts):
        o.load_state_dict(cp[f'opt_{i}'])
    for i, s in enumerate(scalers):
        if use_amp:
            s.load_state_dict(cp[f'scaler_{i}'])
    start_task = cp['task'] + 1
    accuracies = cp['accuracies']
    memory_buffers = cp['memory_buffers']
    print(f"Resumed from task {start_task}")

# GEM gradient projection
def project_gradients(grads, memories):
    if not memories:
        return grads
    for past_task in memories:
        if past_task:
            past_grad = past_task['grad']
            dot = torch.dot(grads, past_grad)
            if dot < 0:
                grads = grads - dot * past_grad / (past_grad.norm()**2 + 1e-8)
    return grads

for task_id in range(start_task, tasks):
    train_loader, test_loader = task_loaders[task_id]
    
    for model_name, model, opt, scaler in zip(['triality', 'ablation', 'hybrid'], 
                                             [model_triality, model_ablation, model_hybrid],
                                             opts, scalers):
        opt.zero_grad(set_to_none=True)
        memory = []
        for epoch in range(epochs_per_task):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                    logits = model(images, epoch + task_id*epochs_per_task)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward() if use_amp else loss.backward()

                if model_name in ['hybrid', 'gem']:  # apply GEM projection for hybrid/GEM
                    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                    grads = project_gradients(grads, memory_buffers[:task_id])
                    idx = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            size = p.grad.numel()
                            p.grad = grads[idx:idx+size].view(p.grad.shape)
                            idx += size

                scaler.unscale_(opt) if use_amp else None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
                scaler.step(opt) if use_amp else opt.step()
                scaler.update() if use_amp else None

                # Update memory for GEM/hybrid
                if model_name in ['hybrid', 'gem']:
                    with torch.no_grad():
                        memory.append({'inputs': images.detach(), 'labels': labels.detach(), 'grad': torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).detach()})
                    if len(memory) > memory_size:
                        memory.pop(0)

        # Update global memory buffer
        if model_name in ['hybrid', 'gem']:
            memory_buffers[task_id] = memory

    # Evaluate on all seen tasks
    for model_name, model in zip(['triality', 'ablation', 'hybrid'], [model_triality, model_ablation, model_hybrid]):
        model.eval()
        correct = 0
        total = 0
        for seen_task in range(task_id + 1):
            _, test_loader_seen = task_loaders[seen_task]
            for images, labels in test_loader_seen:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    logits = model(images, 0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        accuracies[model_name].append(acc)
        print(f"After Task {task_id+1} | {model_name.capitalize()} Acc {acc:.2f}%")

    # Checkpoint every task
    torch.save({
        'task': task_id,
        'model_0': model_triality.state_dict(),
        'model_1': model_ablation.state_dict(),
        'model_2': model_hybrid.state_dict(),
        'opt_0': opts[0].state_dict(),
        'opt_1': opts[1].state_dict(),
        'opt_2': opts[2].state_dict(),
        'scaler_0': scalers[0].state_dict() if use_amp else None,
        'scaler_1': scalers[1].state_dict() if use_amp else None,
        'scaler_2': scalers[2].state_dict() if use_amp else None,
        'accuracies': accuracies,
        'memory_buffers': memory_buffers,
    }, checkpoint_path)
    print(f"Checkpoint saved after task {task_id+1}")

print("E8 + GEM Hybrid Continual Learning Sim complete!")