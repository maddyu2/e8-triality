etg_gradient      = torch.linspace(1.0, 10.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
itg_gradient      = torch.linspace(0.5, 5.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)
coupling_strength = torch.linspace(0.1, 1.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([etg_gradient, itg_gradient, coupling_strength], dim=-1)\
             .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# Plot save name: "etg_itg_coupling_precision_entropy.png"