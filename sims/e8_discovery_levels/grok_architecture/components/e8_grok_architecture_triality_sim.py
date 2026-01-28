import torch
import torch.nn as nn

class TrialityCycleBlock(nn.Module):
    def __init__(self, dim, latent_dim=8, triality=3):
        super().__init__()
        self.triality = triality
        self.dim = dim
        self.proj = nn.Linear(latent_dim, dim // triality)
        self.register_buffer('roots', self.get_e8_roots())

    def get_e8_roots(self):
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

    def forward(self, x, step):
        # Precompute position embedding
        pos_emb = self.roots[torch.arange(x.shape[1]) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, self.triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * torch.pi)
        x_rot1 = x * (emb.cos() + pump)
        x_rot2 = torch.roll(x_rot1, shifts=1, dims=-1) * emb.sin()
        x_rot3 = torch.roll(x_rot2, shifts=1, dims=-1) * emb.cos()
        fused = (x_rot1 + x_rot2 + x_rot3) / self.triality  # triality fusion
        return fused

# Example usage in a simple model
class ExampleTrialityModel(nn.Module):
    def __init__(self, dim=240):
        super().__init__()
        self.cycle_block = TrialityCycleBlock(dim=dim)
        self.linear = nn.Linear(dim, 1)  # simple output

    def forward(self, x, step):
        x = self.cycle_block(x, step)
        return torch.sigmoid(self.linear(x.mean(dim=1)))

# Test snippet
if __name__ == '__main__':
    x = torch.randn(64, 1024, 240)  # example input
    step = 0
    model = ExampleTrialityModel()
    output = model(x, step)
    print(output.shape)  # Expected: torch.Size([64])