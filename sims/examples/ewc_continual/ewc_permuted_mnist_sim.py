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

# Second cell: EWC Continual Learning on Permuted MNIST (optimized)
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
lambda_ewc = 10000  # EWC strength

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "ewc_checkpoint.pth")

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

# Simple MLP
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

# EWC variables
fisher = None
old_params = None

def ewc_loss(model, lambda_ewc):
    loss = 0
    if fisher is not None:
        for n, p in model.named_parameters():
            loss += (fisher[n] * (p - old_params[n]) ** 2).sum()
        loss *= lambda_ewc / 2
    return loss

accuracies = []

start_task = 0

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    fisher = cp['fisher']
    old_params = cp['old_params']
    start_task = cp['task'] + 1
    accuracies = cp['accuracies']
    print(f"Resumed from task {start_task}")

for task_id in range(start_task, tasks):
    train_loader, test_loader = task_loaders[task_id]
    
    # Train on current task
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + ewc_loss(model, lambda_ewc)
            loss.backward()
            optimizer.step()

    # Compute Fisher (importance) after task
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad ** 2 / len(train_loader)
    old_params = {n: p.clone().detach() for n, p in model.named_parameters()}

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
        'fisher': fisher,
        'old_params': old_params,
        'accuracies': accuracies,
    }, checkpoint_path)
    print(f"Checkpoint saved after task {task_id+1}")

print("EWC Continual Learning on Permuted MNIST complete!")