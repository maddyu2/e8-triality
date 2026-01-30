# Real Dataset Scaling Notes

These notes guide scaling the proxy sims to real datasets on better hardware (GPU cluster/Colossus). Start with small subsets for testing.

## CIFAR-10 (Images)
```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

for images, _ in trainloader:
    real_data = images.view(images.size(0), 1, -1).to(device)  # flatten to (batch, seq=1, dim=3072)
    # Apply masking, then feed to model (e.g., e8_image_fusion_sim.py)
    break  # test one batch first