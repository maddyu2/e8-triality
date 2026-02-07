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

# Second cell: The sim code (E8 Triality + A-GEM Hybrid on Split CIFAR-100)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import math
import os
import random

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed + stability
dim = 384
latent_dim = 8
batch_size = 128
tasks = 10
epochs_per_task = 300  # fast (sigma trend early)
lr = 5e-5
use_amp = True
buffer_size_per_task = 200  # A-GEM buffer

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "split_cifar100_hybrid_checkpoint.pth")

# CIFAR-100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
cifar_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Split into 10 tasks (10 classes each)
classes_per_task = 10
train_indices = [np.where(np.array(cifar_train.targets) // classes_per_task == t)[0] for t in range(tasks)]
test_indices = [np.where(np.array(cifar_test.targets) // classes_per_task == t)[0] for t in range(tasks)]

train_loaders = [DataLoader(Subset(cifar_train, idx), batch_size=batch_size, shuffle=True) for idx in train_indices]
test_loaders = [DataLoader(Subset(cifar_test, idx), batch_size=batch_size) for idx in test_indices]

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

# Model (ViT-like for CIFAR)
class E8HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)  # 32x32 → 8x8 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, dim))  # + cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cycle = TrialityCycleBlock()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 100)  # 100 classes

    def forward(self, x, step):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.cycle(x, step)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        x = x[:, 0]  # cls token
        return self.head(x)

# Model + optimizer
model = E8HybridModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler('cuda') if use_amp else nullcontext()

criterion = nn.CrossEntropyLoss()

# A-GEM buffers
memory_buffers = [[] for _ in range(tasks)]

def store_in_buffer(buffer, data, labels):
    if len(buffer) < buffer_size_per_task:
        buffer.append((data.detach(), labels.detach()))
    else:
        idx = random.randint(0, buffer_size_per_task - 1)
        buffer[idx] = (data.detach(), labels.detach())

# A-GEM projection
def a_gem_projection(grads, memories):
    if not memories:
        return grads
    past_grad = torch.zeros_like(grads)
    count = 0
    for mem in memories:
        if mem:
            for past_data, past_labels in random.sample(mem, min(32, len(mem))):  # sample batch
                past_outputs = model(past_data)
                past_loss = criterion(past_outputs, past_labels)
                past_grad += torch.autograd.grad(past_loss, model.parameters(), retain_graph=True)[0].flatten()
                count += 1
    if count > 0:
        past_grad /= count
        dot = torch.dot(grads, past_grad)
        if dot < 0:
            grads = grads - dot * past_grad / (past_grad.norm()**2 + 1e-8)
    return grads

accuracies = []

start_task = 0

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    scaler.load_state_dict(cp['scaler']) if use_amp else None
    start_task = cp['task'] + 1
    accuracies = cp['accuracies']
    memory_buffers = cp['memory_buffers']
    print(f"Resumed from task {start_task}")

for task_id in range(start_task, tasks):
    train_loader = train_loaders[task_id]
    
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) - task_id*classes_per_task
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                logits = model(images, epoch + task_id*epochs_per_task)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward() if use_amp else loss.backward()

            # A-GEM projection
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grads = a_gem_projection(grads, memory_buffers[:task_id])
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    size = p.grad.numel()
                    p.grad = grads[idx:idx+size].view(p.grad.shape)
                    idx += size

            scaler.unscale_(optimizer) if use_amp else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer) if use_amp else optimizer.step()
            scaler.update() if use_amp else None

            # Store in buffer
            store_in_buffer(memory_buffers[task_id], images, labels)

    # Evaluate on all seen tasks
    model.eval()
    correct = 0
    total = 0
    for seen_task in range(task_id + 1):
        test_loader = test_loaders[seen_task]
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) - seen_task*classes_per_task
            with torch.no_grad():
                logits = model(images, 0)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"After Task {task_id+1}: Average Accuracy {acc:.2f}%")

    # Checkpoint every task
    torch.save({
        'task': task_id,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if use_amp else None,
        'accuracies': accuracies,
        'memory_buffers': memory_buffers,
    }, checkpoint_path)
    print(f"Checkpoint saved after task {task_id+1}")

print("E8 Triality + A-GEM Hybrid Continual Learning on Split CIFAR-100 complete!")