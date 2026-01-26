# Same imports as above

# Proxies: gravitational acceleration 0.1–10 m/s² (Earth → microgravity), scaling factor 0.1–1.0
grav_accel = torch.linspace(0.1, 10.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
scaling_factor = torch.linspace(0.1, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

grav_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_grav_data = torch.cat([grav_accel, scaling_factor, grav_sym], dim=-1)\
                  .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# E8 roots (same function)

class GravVizRotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(latent_dim, dim // triality)
        self.register_buffer('roots', e8_roots)

    def forward(self, x, step):
        pos_emb = self.roots[torch.arange(x.shape[1]) % 240]
        low_dim = self.proj(pos_emb)
        emb = low_dim.repeat(1, triality)
        pump = 0.8 * torch.sin(step * 0.006 * 2 * np.pi)
        return x * (emb.cos() + pump) + torch.roll(x, shifts=1, dims=-1) * emb.sin()

class E8ScalingGravViz(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = GravVizRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training & plotting code identical to above (change model and data variable names)

print("Scaling gravity visualization sim ready — eternal gravitational scaling.")