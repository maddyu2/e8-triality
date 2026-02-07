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

# Second cell: Experience Replay (ER) on Split CIFAR-100 (optimized)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import random
import os

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CONFIG – optimized for speed
batch_size = 128
tasks = 10
epochs_per_task = 300  # fast (trend early)
lr = 0.001
buffer_size_per_task = 200  # ER buffer

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "er_split_cifar100_checkpoint.pth")

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

# Simple CNN for CIFAR
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ER buffer
buffer = []

def add_to_buffer(buffer, data, labels):
    for d, l in zip(data, labels):
        if len(buffer) < buffer_size_per_task * tasks:
            buffer.append((d, l))
        else:
            idx = random.randint(0, len(buffer)-1)
            buffer[idx] = (d, l)

accuracies = []

start_task = 0

# Load checkpoint if exists
if os.path.exists(checkpoint_path):
    cp = torch.load(checkpoint_path)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    start_task = cp['task'] + 1
    accuracies = cp['accuracies']
    buffer = cp['buffer']
    print(f"Resumed from task {start_task}")

for task_id in range(start_task, tasks):
    train_loader = train_loaders[task_id]
    
    for epoch in range(epochs_per_task):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) - task_id*classes_per_task
            
            # Mix with buffer
            if buffer:
                buffer_images, buffer_labels = zip(*random.sample(buffer, min(batch_size//2, len(buffer))))
                buffer_images = torch.stack(buffer_images).to(device)
                buffer_labels = torch.tensor(buffer_labels).to(device)  # remap if needed
                images = torch.cat([images, buffer_images])
                labels = torch.cat([labels, buffer_labels])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Add current batch to buffer
            add_to_buffer(buffer, images[:batch_size//2].cpu(), labels[:batch_size//2].cpu())

    # Evaluate on all seen tasks
    model.eval()
    correct = 0
    total = 0
    for seen_task in range(task_id + 1):
        test_loader = test_loaders[seen_task]
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) - seen_task*classes_per_task
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
        'buffer': buffer,
    }, checkpoint_path)
    print(f"Checkpoint saved after task {task_id+1}")

print("Experience Replay (ER) Continual Learning on Split CIFAR-100 complete!")