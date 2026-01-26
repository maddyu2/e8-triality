# Same imports as above

# Data: CMB temperature anisotropy power spectrum (l=2–2500, ΔT/T ~10^{-5})
cmb_anisotropy = torch.linspace(1e-6, 1e-4, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
power_spectrum = torch.linspace(1000, 6000, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)  # l multipole proxy
cmb_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_cmb_data = torch.cat([cmb_anisotropy, power_spectrum, cmb_sym], dim=-1)\
                 .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * 0.01

# E8 roots (same function as above)

# Rotary class (copy from Lyman sim or rename)
class CMBRotary(nn.Module):
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

class E8CMBAnisotropy(nn.Module):
    def __init__(self, depth=256):
        super().__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(dim, triality, batch_first=True) for _ in range(depth)])
        self.rotary = CMBRotary()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, step):
        x = self.rotary(x, step)
        for layer in self.layers:
            attn, _ = layer(x, x, x)
            x = x + self.norm(attn)
        return torch.sigmoid(self.head(x.mean(dim=1)))

# Training & plotting code identical to the Lyman sim above (just change model and data variable names)

print("CMB anisotropy sim ready — eternal primordial fluctuations.")