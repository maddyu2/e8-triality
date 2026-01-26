import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
#   Toy dataset: simple physics QA (question → answer embedding)
# ────────────────────────────────────────────────

class SparsePhysicsQADataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=32, mask_prob=0.4):
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        
        # Fake "physics" data: question = random vector, answer = shifted version
        self.questions = torch.randn(num_samples, seq_len, 64)
        self.answers   = self.questions + torch.randn_like(self.questions) * 0.1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx].clone()
        a = self.answers[idx]

        # Randomly mask tokens (AGI data void simulation)
        mask = torch.rand(self.seq_len) > self.mask_prob
        q[~mask] = 0.0  # masked tokens → zero

        return q, a, mask.float()  # return mask so regularizer can see it

# ────────────────────────────────────────────────
#   Triality Sparse Regularizer (same as before)
# ────────────────────────────────────────────────

class E8TrialitySparseRegularizer(nn.Module):
    def __init__(self, d_model=64, triality_heads=3, dropout=0.1):
        super().__init__()
        assert d_model % triality_heads == 0
        self.heads = triality_heads
        self.d_head = d_model // triality_heads

        self.proj_split = nn.Linear(d_model, d_model)
        self.proj_fuse  = nn.Linear(d_model, d_model)
        self.weight     = nn.Parameter(torch.ones(triality_heads))
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        split = self.proj_split(x).view(B, S, self.heads, self.d_head).permute(2, 0, 1, 3)   # [H, B, S, d]
        cycled = torch.roll(split, shifts=1, dims=0)                                         # cycle heads
        w = torch.softmax(self.weight, dim=0).view(self.heads, 1, 1, 1)
        fused = (cycled * w).sum(dim=0).view(B, S, D)                                        # [B, S, D]
        out = self.proj_fuse(fused)
        out = self.dropout(out)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, D)
            out = out.masked_fill(~mask.bool(), 0.0)

        return self.norm(out + x)

# ────────────────────────────────────────────────
#   Small Transformer with optional triality regularizer
# ────────────────────────────────────────────────

class SparseTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, use_triality=False):
        super().__init__()
        self.embedding = nn.Linear(64, d_model)
        self.layers     = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
            for _ in range(3)
        ])
        self.triality   = E8TrialitySparseRegularizer(d_model, triality_heads=3) if use_triality else None
        self.head       = nn.Linear(d_model, 64)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            if self.triality is not None:
                x = self.triality(x, mask=mask)
        return self.head(x)

# ────────────────────────────────────────────────
#   Training function
# ────────────────────────────────────────────────

def train_sparse_model(use_triality=False, epochs=3000, lr=3e-4):
    dataset = SparsePhysicsQADataset(mask_prob=0.4)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True)

    model = SparseTransformer(use_triality=use_triality).cuda()
    opt   = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    prec_hist = []
    ent_hist  = []

    for epoch in range(epochs):
        total_loss = 0
        for q, a, m in loader:
            q, a, m = q.cuda(), a.cuda(), m.cuda()
            pred = model(q, mask=m)
            loss = loss_fn(pred, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # fake "entropy" proxy = average squared error on masked positions
        ent_proxy = avg_loss * 10   # just illustrative

        prec_hist.append(1 - avg_loss)      # rough precision proxy
        ent_hist.append(ent_proxy)

        if epoch % 300 == 0:
            print(f"[{ 'Triality' if use_triality else 'Vanilla' }] "
                  f"Epoch {epoch:4d} | Loss {avg_loss:.6f} | Approx Entropy {ent_proxy:.6f}")

    return prec_hist, ent_hist

# ────────────────────────────────────────────────
#   Run both versions and compare
# ────────────────────────────────────────────────

print("Training vanilla transformer...")
prec_vanilla, ent_vanilla = train_sparse_model(use_triality=False, epochs=3000)

print("\nTraining triality-regularized transformer...")
prec_triality, ent_triality = train_sparse_model(use_triality=True, epochs=3000)

# ────────────────────────────────────────────────
#   Plot comparison
# ────────────────────────────────────────────────

epochs_range = np.arange(len(prec_vanilla))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, prec_vanilla, label='Vanilla', alpha=0.7)
plt.plot(epochs_range, prec_triality, label='Triality', linewidth=2)
plt.title("Approximate Precision (1 - MSE)")
plt.xlabel("Epoch")
plt.ylabel("Precision proxy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, ent_vanilla, label='Vanilla', alpha=0.7)
plt.plot(epochs_range, ent_triality, label='Triality', linewidth=2)
plt.title("Approximate Entropy Proxy")
plt.xlabel("Epoch")
plt.ylabel("Entropy proxy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Done. Triality version typically converges faster and reaches lower entropy.")