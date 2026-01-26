# Same imports as above

# Data
he4 = torch.linspace(0.24, 0.25, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
dh  = torch.linspace(2.0e-5, 3.0e-5, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
bbn_sym = torch.linspace(0.85, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_bbn = torch.cat([he4, dh, bbn_sym], dim=-1).repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * 0.01

# Model, rotary, etc. same as first sim (reuse classes or copy-paste)

# Training loop same, collect prec_history and ent_history

# Plot same as above