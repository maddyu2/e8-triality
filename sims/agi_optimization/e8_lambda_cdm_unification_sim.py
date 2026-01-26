# Same imports

# Data
om   = torch.linspace(0.25, 0.35, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
olam = torch.linspace(0.65, 0.75, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
h0   = torch.linspace(67, 73, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
lcdm_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_lcdm = torch.cat([om, olam, h0, lcdm_sym], dim=-1).repeat(1, 1, dim // 4) * torch.randn(batch_size, seq_len, dim, device=device) * 0.01

# Model, training, plotting — same pattern as above