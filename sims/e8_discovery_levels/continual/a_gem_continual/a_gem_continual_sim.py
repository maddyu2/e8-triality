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

# Second cell: A-GEM Variant on Permuted MNIST (optimized)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import os

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed
batch_size = 128
tasks = 10
epochs_per_task = 500  # fast (trend early)
lr = 0.01
buffer_size_per_task = 200  # optimized (low memory, strong retention)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "a_gem_checkpoint.pth")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Permuted tasks
task_loaders = []
for t in range(tasks):
    perm = torch.randperm(784)
    train_data = mnist_train.data.float().view(-1, 784)[:, perm] / 255.0
    train_labels = mnist_train.targets
    test_data = mnist_test.data.float().view(-1, 784)[:, perm] / 255.0
    test_labels = mnist_test.targets
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)
    task_loaders.append((train_loader, test_loader))

# Simple MLP for MNIST
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# A-GEM buffers
memory_buffers = [[] for _ in range(tasks)]  # one buffer per task

def store_in_buffer(buffer, data, labels):
    if len(buffer) < buffer_size_per_task:
        buffer.append((data.detach(), labels.detach()))
    else:
        idx = np.random.randint(buffer_size_per_task)
        buffer[idx] = (data.detach(), labels.detach())

# A-GEM projection (averaged past gradients)
def a_gem_projection(grad, memories):
    if not memories:
        return grad
    past_grad = torch.zeros_like(grad)
    count = 0
    for mem in memories:
        if mem:
            for past_data, past_labels in mem:
                past_outputs = model(past_data)
                past_loss = criterion(past_outputs, past_labels)
                past_grad += torch.autograd.grad(past_loss, model.parameters(), retain_graph=True)[0].flatten()
                count += 1
    if count > 0:
        past_grad /= count
        dot = torch.dot(grad, past_grad)
        if dot < 0:
            grad = grad - dot * past_grad / (past_grad.norm()**2 + 1e-8)
    return grad

accuracies = []

start_task = 0

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    start_task = cp['task'] + 1
    accuracies = cp['accuracies']
    memory_buffers = cp['memory_buffers']
    print(f"Resumed from task {start_task}")

for task_id in range(start_task, tasks):
    train_loader, test_loader = task_loaders[task_id]
    
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # A-GEM projection
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grads = a_gem_projection(grads, memory_buffers[:task_id])
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    size = p.grad.numel()
                    p.grad = grads[idx:idx+size].view(p.grad.shape)
                    idx += size

            optimizer.step()

            # Store in buffer (random subset)
            if np.random.rand() < 0.1:  # 10% chance to store
                store_in_buffer(memory_buffers[task_id], images, labels)

    # Evaluate on all seen tasks
    model.eval()
    correct = 0
    total = 0
    for seen_task in range(task_id + 1):
        _, test_loader_seen = task_loaders[seen_task]
        for images, labels in test_loader_seen:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                pred = outputs.argmax(dim=1)
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
        'accuracies': accuracies,
        'memory_buffers': memory_buffers,
    }, checkpoint_path)
    print(f"Checkpoint saved after task {task_id+1}")

print("A-GEM Continual Learning Sim complete!")