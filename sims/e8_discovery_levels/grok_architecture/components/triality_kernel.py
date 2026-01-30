import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrialityKernel(nn.Module):
    def __init__(self, dim, latent_dim=8):
        super().__init__()
        self.dim = dim
        self.triality = 3
        self.proj = nn.Linear(latent_dim, dim // self.triality, bias=False)
        # Precompute normalized E8 roots (full 240)
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
        roots = roots / roots.norm(dim=-1, keepdim=True)
        self.register_buffer('roots', roots)

    def forward(self, x, step):
        # x: (batch, seq, dim)
        b, s, d = x.shape

        # Positional embedding from E8 roots
        pos_idx = torch.arange(s, device=x.device) % 240
        pos_emb = self.roots[pos_idx]  # (seq, 8)
        low_dim = self.proj(pos_emb)   # (seq, dim//3)
        emb = low_dim.repeat(1, self.triality)  # (seq, dim)

        # Pump for dynamic oscillation
        pump = 0.8 * torch.sin(torch.tensor(step, device=x.device, dtype=torch.float32) * 0.006 * 2 * math.pi)

        # Parallel triality rotations with roll (fused ops)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=1) * emb.cos()

        # Parallel fusion
        fused = (x_rot1 + x_rot2 + x_rot3) / self.triality
        return fused

# Example integration in Grok 5-style layer (FlashAttention fused)
class Grok5TrialityLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.triality = TrialityKernel(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, step):
        # FlashAttention fused
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x + self.out_proj(attn_out)

        # Triality kernel (parallel fusion)
        x = self.triality(x, step)
        x = self.norm(x)
        return x

# Test
if __name__ == '__main__':
    dim = 240
    batch = 4
    seq = 1024
    x = torch.randn(batch, seq, dim)
    layer = Grok5TrialityLayer(dim)
    out = layer(x, step=0)
    print(out.shape)  # (batch, seq, dim)