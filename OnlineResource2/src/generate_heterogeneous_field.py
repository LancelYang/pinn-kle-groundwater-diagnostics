"""
Generate KLE heterogeneous conductivity field (stage 3b data preparation)

Output:
  outputs/stage3_heterogeneous/K_field.npz
    - X, Y: grid coordinates (nx x ny)
    - logK: logK field (nx x ny)
    - K: K = exp(logK) field (nx x ny)
    - metadata: KLE parameters
"""
import sys
import os
import numpy as np
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from kle import KLE
from paths import resolve_project_path

def main():
    # Read config
    config_path = resolve_project_path('configs/stage3a_3b_baseline.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # KLE parameters (from config, with defaults if not specified)
    kle_cfg = config.get('kle', {})
    Lx = kle_cfg.get('Lx', 1000.0)
    Ly = kle_cfg.get('Ly', 1000.0)
    nx = kle_cfg.get('nx', 50)
    ny = kle_cfg.get('ny', 50)
    lx = kle_cfg.get('lx', 200.0)
    ly = kle_cfg.get('ly', 200.0)
    sigma2 = kle_cfg.get('sigma2', 1.0)
    cov_type = kle_cfg.get('cov_type', 'exponential')  # Note: KLE class uses cov_type, not kernel
    n_modes = kle_cfg.get('n_modes', 50)  # Use only the first n_modes modes

    print("=" * 60)
    print("Stage 3b: Generating KLE heterogeneous conductivity field")
    print("=" * 60)
    print(f"  Lx={Lx}, Ly={Ly}, nx={nx}, ny={ny}")
    print(f"  sigma2={sigma2}, lx={lx}, ly={ly}, cov_type={cov_type}")
    print(f"  n_modes={n_modes}")
    print()

    # Initialize KLE (note parameter order: Lx, Ly, nx, ny, lx, ly, sigma2, mean, cov_type)
    kle = KLE(Lx=Lx, Ly=Ly, nx=nx, ny=ny,
               lx=lx, ly=ly, sigma2=sigma2,
               mean=0.0, cov_type=cov_type)

    # Compute eigenpairs
    print("Computing eigenpairs...")
    eigenvalues, eigenfunctions = kle.compute_eigenpairs()
    print(f"  Eigenvalue range: {eigenvalues[0]:.4f} -> {eigenvalues[-1]:.6f}")

    # Truncate (retain specified order M)
    if n_modes is not None:
        M = min(n_modes, len(eigenvalues))
        kle.truncate(M=M)
        print(f"  Using M={M} KLE modes")
    else:
        M = kle.truncate(var_threshold=0.99)
        print(f"  Auto-truncated to M={M} KLE modes (retaining 99% variance)")

    # Generate random realization (fixed seed for reproducibility)
    print("Generating logK random realization...")
    np.random.seed(42)
    xi = np.random.randn(M)
    logK, xi_actual = kle.sample(xi=xi)

    # K = exp(logK)
    K = np.exp(logK)

    print(f"  logK: mean={logK.mean():.4f}, std={logK.std():.4f}")
    print(f"  K: min={K.min():.6f}, max={K.max():.2f}, mean={K.mean():.4f}")

    # Build grid coordinates
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Save data
    output_dir = resolve_project_path('outputs/stage3_heterogeneous')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'K_field.npz')
    np.savez_compressed(
        output_path,
        X=X, Y=Y, logK=logK, K=K,
        x=x, y=y,
        eigenvalues=eigenvalues[:M],
        eigenfunctions=eigenfunctions[:, :M] if eigenfunctions.ndim == 2 else eigenfunctions[:M],
        xi=xi,
        metadata=dict(Lx=Lx, Ly=Ly, nx=nx, ny=ny,
                       sigma2=sigma2, lx=lx, ly=ly, cov_type=cov_type, M=M)
    )
    print(f"\nSaved to: {output_path}")

    # Also save as figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # logK field
    im1 = axes[0].contourf(X, Y, logK, levels=50, cmap='viridis')
    axes[0].set_title('logK field (KLE realization)')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[0])

    # K field
    im2 = axes[1].contourf(X, Y, K, levels=50, cmap='viridis')
    axes[1].set_title('K field (exp(logK))')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes[1])

    # K histogram
    axes[2].hist(K.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_title('K distribution')
    axes[2].set_xlabel('K (m/s)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xscale('log')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'K_field_realization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {fig_path}")

    # Save metadata
    meta_path = os.path.join(output_dir, 'K_field_metadata.yaml')
    with open(meta_path, 'w') as f:
        yaml.dump(dict(
            generation_time=datetime.now().isoformat(),
            Lx=Lx, Ly=Ly, nx=nx, ny=ny,
            sigma2=sigma2, lx=lx, ly=ly, cov_type=cov_type,
            n_modes=M,
            logK_mean=float(logK.mean()),
            logK_std=float(logK.std()),
            K_min=float(K.min()),
            K_max=float(K.max()),
            K_mean=float(K.mean()),
            output_file='K_field.npz'
        ), f, default_flow_style=False)
    print(f"Saved metadata: {meta_path}")

    print("\nDone!")
    return output_path


if __name__ == '__main__':
    main()
