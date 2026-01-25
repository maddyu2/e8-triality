# ────────────────────────────────────────────────
# Replace the data generation block only
# ────────────────────────────────────────────────

alpha_power = torch.linspace(1, 10, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)     # MW
dt_alpha    = torch.linspace(0.01, 0.1, batch_size * seq_len, device=device).view(batch_size, seq_len, 1) # s
fusion_gain = torch.linspace(1.0, 2.0, batch_size * seq_len, device=device).view(batch_size, seq_len, 1)

real_data = torch.cat([alpha_power, dt_alpha, fusion_gain], dim=-1)\
             .repeat(1, 1, dim // 3) * torch.randn(batch_size, seq_len, dim, device=device) * noise_scale

# Then keep the rest of the code exactly the same as the first file
# Change plot save name to: "iter_alpha_heating_precision_entropy.png"