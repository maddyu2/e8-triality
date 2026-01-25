fugaku_nodes = torch.linspace(1e4, 1e6, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
blackwell_fp4_eff = torch.linspace(0.5, 2.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
scale_factor = torch.linspace(1.0, 10.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([fugaku_nodes, blackwell_fp4_eff, scale_factor], dim=-1).repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale