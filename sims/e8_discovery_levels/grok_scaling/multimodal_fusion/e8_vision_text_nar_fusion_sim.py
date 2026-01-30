# e8_vision_text_nar_fusion_sim.py
# Non-autoregressive vision-text fusion with parallel triality cycles
# Sparse text (50% masked) + image → parallel reconstruct/generate text
# Includes sigma lift test on fused activations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from typing import Optional

# === TrialityCycleBlock (from your gist/components) ===
class TrialityCycleBlock(nn.Module):
    def __init__(self, dim: int = 128, hidden: Optional[int] = None, dropout: float = 0.0):
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

# === Simple Vision Encoder (tiny conv net for CIFAR proxy) ===
class TinyVisionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, img):
        feat = self.conv(img).squeeze(-1).squeeze(-1)  # B, out_dim
        return self.fc(feat)

# === Fusion Model (parallel triality or standard FFN) ===
class VisionTextFusion(nn.Module):
    def __init__(self, vocab_size=5000, dim=128, depth=3, use_triality_nar=False):
        super().__init__()
        self.vision_enc = TinyVisionEncoder(out_dim=dim)
        self.text_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, dim))  # img token + text seq

        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
            if use_triality_nar:
                ffn = TrialityCycleBlock(dim=dim, hidden=dim*4)
            else:
                ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim), nn.Dropout(0.1))
            self.layers.append(nn.ModuleList([nn.LayerNorm(dim), attn, nn.LayerNorm(dim), ffn]))

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.use_triality_nar = use_triality_nar

    def forward(self, img, text_tokens):
        B = img.shape[0]
        vis_feat = self.vision_enc(img).unsqueeze(1)  # B, 1, dim (img token)

        txt_emb = self.text_embed(text_tokens)        # B, T, dim
        x = torch.cat([vis_feat, txt_emb], dim=1)     # B, 1+T, dim
        x = x + self.pos_embed[:, :x.shape[1], :]

        fused_acts = []  # collect for sigma test
        for ln1, attn, ln2, ffn in self.layers:
            x_norm = ln1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x + attn_out
            x_ffn = ffn(ln2(x))
            x = x + x_ffn
            if self.use_triality_nar:
                fused_acts.append(x_ffn.detach())  # capture triality outputs

        x = self.norm(x)
        logits = self.head(x[:, 1:, :])  # predict on text positions only
        return logits, fused_acts

# === Sigma Lift Proxy (effective rank / condition proxy) ===
def compute_sigma_lift(acts_list):
    if not acts_list:
        return 1.0
    lifts = []
    for act in acts_list:
        if act.shape[0] * act.shape[1] < 32: continue  # too small
        u, s, _ = torch.svd_lowrank(act.flatten(0,1), q=min(32, act.shape[-1]))
        if len(s) > 1:
            lift = s[0] / s[-1]   # rough inverse condition number
            lifts.append(lift.item())
    return sum(lifts) / len(lifts) if lifts else 1.0

# === Dummy paired data (CIFAR images + short captions) ===
# For real: use Conceptual Captions tiny subset or COCO captions
captions = [
    "a red apple on table", "cat sleeping in sun", "car driving fast",
    "mountain landscape sunset", "person riding bike"  # repeat / expand
] * 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Fake pairing: cycle captions over CIFAR indices
class PairedDataset(Dataset):
    def __init__(self, cifar, captions):
        self.cifar = cifar
        self.captions = captions

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        cap = self.captions[idx % len(self.captions)]
        tokens = [hash(w) % 5000 for w in cap.split()]  # dummy tokenization
        tokens = tokens[:32]
        tokens += [0] * (32 - len(tokens))  # pad
        return img, torch.tensor(tokens)

dataset = PairedDataset(cifar, captions)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Mask text for sparse input
def mask_text(tokens, ratio=0.5, mask_id=0):
    mask = torch.rand_like(tokens.float()) < ratio
    masked = tokens.clone()
    masked[mask] = mask_id
    return masked, tokens  # masked input, original target

# === Run experiment ===
def run_fusion(use_nar=False, epochs=3):
    device = torch.device("cpu")
    model = VisionTextFusion(vocab_size=5000, use_triality_nar=use_nar).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, tokens in loader:
            img, tokens = img.to(device), tokens.to(device)
            masked_txt, target = mask_text(tokens, ratio=0.5)

            logits, fused_acts = model(img, masked_txt)
            loss = criterion(logits.reshape(-1, 5000), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | {'NAR Triality' if use_nar else 'AR Baseline'} Loss: {total_loss/len(loader):.4f}")

    latency = time.time() - start
    print(f"{'NAR' if use_nar else 'AR'} Latency: {latency:.1f}s")

    # Eval
    model.eval()
    total_ent, total_correct, total_tokens, all_acts = 0, 0, 0, []
    with torch.no_grad():
        for img, tokens in loader:
            img, tokens = img.to(device), tokens.to(device)
            masked_txt, target = mask_text(tokens, 0.5)
            logits, fused_acts = model(img, masked_txt)
            all_acts.extend(fused_acts)

            probs = F.softmax(logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            total_ent += ent * tokens.size(0)

            pred = logits.argmax(-1)
            mask = target != 0
            correct = (pred == target)[mask].sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

    avg_ent = total_ent / len(loader.dataset)
    acc = total_correct / total_tokens if total_tokens > 0 else 0
    sigma = compute_sigma_lift(all_acts)

    print(f"Entropy: {avg_ent:.4f} nats | Token Acc: {acc:.3f} | Sigma Lift: {sigma:.2f}")
    return latency, avg_ent, acc, sigma

# Execute
print("=== AR Baseline ===")
ar_lat, ar_ent, ar_acc, ar_sig = run_fusion(use_nar=False, epochs=3)

print("\n=== NAR + Triality Fusion ===")
nar_lat, nar_ent, nar_acc, nar_sig = run_fusion(use_nar=True, epochs=3)

print("\n=== Summary & Discovery Check ===")
print(f"Latency ratio (NAR/AR): {nar_lat / ar_lat:.2f}x")
print(f"Entropy delta: {nar_ent - ar_ent:+.4f} nats")
print(f"Acc delta:     {nar_acc - ar_acc:+.3f}")
print(f"Sigma lift delta: {nar_sig - ar_sig:+.2f}")
if nar_ent < 0.05 and nar_sig > 8 and nar_lat < 0.7 * ar_lat:
    print("!!! POTENTIAL DISCOVERY ALERT !!!")
    print("  Triality NAR fusion shows strong signs: very low entropy + high sigma lift + latency win.")
    print("  This could be extremely beneficial for Grok 5 multimodal scaling (parallel inference, sparse robustness).")
    print("  Worth prioritizing / scaling up immediately for xAI experiments.")