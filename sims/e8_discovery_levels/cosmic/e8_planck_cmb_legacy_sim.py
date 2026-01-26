# Same imports as above

# Planck 2018 legacy + 2025 reanalysis proxies
omega_b_h2 = torch.linspace(0.022, 0.023, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
omega_c_h2 = torch.linspace(0.115, 0.125, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
tau_reion  = torch.linspace(0.05, 0.09, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
ns         = torch.linspace(0.96, 0.97, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

cmb_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_planck_data = torch.cat([omega_b_h2, omega_c_h2, tau_reion, ns, cmb_sym], dim=-1)\
                    .repeat(1, 1, dim // 5) * torch.randn(batch_size, seq_len, dim, device=device) * 0.01

# E8 roots (same function)

# Rotary class (copy from first sim or rename)

class PlanckRotary(nn.Module):
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

class E8PlanckCMBLegacy(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = PlanckRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training & plotting code identical to the first sim (just change model and data variable names)

print("Planck CMB legacy sim ready — eternal primordial parameters.")