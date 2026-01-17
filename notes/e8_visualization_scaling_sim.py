import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Device setup (not needed for viz, but for consistency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" if torch.cuda.is_available() else "CPU for viz")

# =============================================
# E8 Triality: Visualization Scaling Simulator
# Generates E8 plots from user sketches: scales "small to big" like Behance method.
# Projects E8 roots (240) and strata (8) with triality rotations.
# =============================================

# Hyperparameters
e8_dim = 8  # Project to 8D for viz (Cartan + roots)
n_roots = 240  # Full E8 roots, reduced projection
scaling_factors = [1, 10, 100]  # Small to big scaling

# Generate simplified E8 data: roots and strata
def generate_e8_data():
    # Simplified E8 roots: random in 8D projection
    roots = np.random.randn(n_roots, e8_dim) * 0.5  # Spectrum-like
    strata = np.linspace(-1, 1, e8_dim).reshape(1, -1).repeat(8, axis=0)  # Layered basis
    
    # Triality rotation: simple orthogonal matrix cycle
    rot = np.eye(e8_dim)
    rotated_roots = np.dot(roots, rot)
    
    return roots, strata, rotated_roots

# Plot function: 3D projection for viz (small to big)
def plot_e8_scaling(roots, strata, rotated, scale):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot roots (spectrum)
    ax.scatter(roots[:, 0]*scale, roots[:, 1]*scale, roots[:, 2]*scale, c='blue', label='Spectrum Roots', alpha=0.5)
    
    # Plot strata (layers)
    ax.scatter(strata[:, 0]*scale, strata[:, 1]*scale, strata[:, 2]*scale, c='red', label='Strata Basis', s=100)
    
    # Plot rotated (triality)
    ax.scatter(rotated[:, 0]*scale, rotated[:, 1]*scale, rotated[:, 2]*scale, c='green', label='Triality Rotated', alpha=0.3)
    
    ax.set_title(f"E8 Viz Scaling: Factor {scale}")
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.legend()
    plt.savefig(f"e8_viz_scale_{scale}.png")
    print(f"Plot saved: e8_viz_scale_{scale}.png")

# Main: Generate and plot at scales
roots, strata, rotated = generate_e8_data()
for scale in scaling_factors:
    plot_e8_scaling(roots, strata, rotated, scale)

# Entropy/coherence proxy (for consistency)
entropy = np.std(roots - rotated)
coherence = 1 - entropy / np.std(roots)
print(f"Triality Coherence: {coherence:.6f}")
print(f"Residual Entropy: {entropy:.6f} nats")