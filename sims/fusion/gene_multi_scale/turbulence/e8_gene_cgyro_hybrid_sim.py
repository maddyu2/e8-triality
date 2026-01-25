gene_time      = torch.linspace(0.1, 10.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
cgyro_time     = torch.linspace(0.1, 5.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
hybrid_gain    = torch.linspace(1.0, 3.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([gene_time, cgyro_time, hybrid_gain], dim=-1)\
             .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# Plot save name: "gene_cgyro_hybrid_precision_entropy.png"