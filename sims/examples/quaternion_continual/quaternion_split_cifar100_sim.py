# First cell: Installs (run once)
!pip install torch torchvision matplotlib numpy

# Second cell: Quaternion Continual Learning on Split CIFAR-100 Proxy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Simple Quaternion Layer (Hamilton product for demo)
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features, 4))

    def hamilton_product(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_features, 4)
        out = self.hamilton_product(self.weight.unsqueeze(0), x.unsqueeze(1))
        out = out.sum(dim=2)
        return out.view(x.shape[0], -1)

# Quaternion CNN for CIFAR-100 proxy
class QuaternionCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = QuaternionLinear(128 * 8 * 8, num_classes * 4)  # to quaternion output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), -1)  # real-valued logits for simplicity

# Load CIFAR-100
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Split into 5 tasks (20 classes each)
num_tasks = 5
classes_per_task = 20
task_datasets = []
for t in range(num_tasks):
    idx = np.where((np.array(trainset.targets) >= t*classes_per_task) & (np.array(trainset.targets) < (t+1)*classes_per_task))[0]
    task_datasets.append(torch.utils.data.Subset(trainset, idx))

# Model + optimizer
model = QuaternionCNN(num_classes=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Continual training (simple fine-tuning for demo — no replay/regularization)
accuracies = []
for task_id, task_dataset in enumerate(task_datasets):
    loader = torch.utils.data.DataLoader(task_dataset, batch_size=128, shuffle=True)
    model.train()
    for epoch in range(10):  # short for demo
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device) - task_id*classes_per_task  # remap labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on all seen tasks
    model.eval()
    correct = 0
    total = 0
    for seen_task in range(task_id + 1):
        seen_loader = torch.utils.data.DataLoader(task_datasets[seen_task], batch_size=128)
        for images, labels in seen_loader:
            images, labels = images.to(device), labels.to(device) - seen_task*classes_per_task
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"After Task {task_id+1}: Average Accuracy on seen tasks: {acc:.2f}%")

print("Quaternion Continual Learning on Split CIFAR-100 proxy complete!")