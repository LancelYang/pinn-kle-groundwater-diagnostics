"""
Stage 4: Finite-difference reference solution solver
=====================================================
Solves div(K*grad(h)) = 0 on [0,1]x[0,1] for steady-state head.

Boundary conditions (consistent with PINN stage 3b):
  - x=0: h = 1.0  (Dirichlet)
  - x=1: h = 0.0  (Dirichlet)
  - y=0: dh/dy = 0 (Neumann, no-flow)
  - y=1: dh/dy = 0 (Neumann, no-flow)

Usage:
  python src/fdm_solver.py                          # default 201x201 grid
  python src/fdm_solver.py --N 101                   # 101x101 grid
  python src/fdm_solver.py --skip-eval               # compute FDM solution only
"""

import os
import argparse
import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from datetime import datetime

from paths import resolve_project_path

DEFAULT_K_FIELD = "outputs/stage3_heterogeneous/K_field.npz"
DEFAULT_OUTPUT_DIR = "outputs/stage4_fdm"


# ============================================================
# K-field loading and interpolation
# ============================================================

def load_K_field(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    K_grid = data['K']
    x_vec = data['x']
    y_vec = data['y']
    logK = data['logK']
    xi = data['xi']
    eigenvalues = data['eigenvalues']
    nx, ny = K_grid.shape
    print(f"  K-field dimensions: {nx}x{ny}, K range: [{K_grid.min():.4f}, {K_grid.max():.4f}]")
    print(f"  logK mean: {logK.mean():.4f}, +/-sigma: {logK.std():.4f}")
    return K_grid, x_vec, y_vec, xi, eigenvalues, logK


def bilinear_interpolate(K_src, x_src, y_src, x_tgt, y_tgt):
    """Bilinear interpolation of K to target grid"""
    Nx, Ny = K_src.shape
    Lx, Ly = x_src[-1] - x_src[0], y_src[-1] - y_src[0]

    xi_idx = (x_tgt[:, None] / Lx) * (Nx - 1)
    yi_idx = (y_tgt[None, :] / Ly) * (Ny - 1)

    x0 = np.clip(np.floor(xi_idx).astype(int), 0, Nx - 2)
    y0 = np.clip(np.floor(yi_idx).astype(int), 0, Ny - 2)
    x1, y1 = x0 + 1, y0 + 1

    wx = np.clip(xi_idx - x0, 0, 1)
    wy = np.clip(yi_idx - y0, 0, 1)

    K00 = K_src[x0, y0]
    K10 = K_src[x1, y0]
    K01 = K_src[x0, y1]
    K11 = K_src[x1, y1]

    K_tgt = (K00 * (1 - wx) * (1 - wy) +
             K10 * wx * (1 - wy) +
             K01 * (1 - wx) * wy +
             K11 * wx * wy)
    return K_tgt


def harmonic_mean(K1, K2):
    return 2.0 * K1 * K2 / (K1 + K2 + 1e-15)


# ============================================================
# FDM solver
# ============================================================

def solve_steady_flow(K_grid, x_vec, y_vec, N_tgt=None):
    """
    Five-point stencil solver for div(K*grad(h)) = 0
    Dirichlet BCs strongly imposed; Neumann via zero-flux half-cell treatment
    """
    if N_tgt is None:
        N_tgt = K_grid.shape[0]

    M = N_tgt
    dx = 1.0 / (M - 1)
    dy = 1.0 / (M - 1)

    x_tgt = np.linspace(0, 1, M)
    y_tgt = np.linspace(0, 1, M)

    if M == K_grid.shape[0] and np.allclose(x_vec, x_tgt):
        K = K_grid.copy()
    else:
        print(f"  Interpolating K: {K_grid.shape[0]}x{K_grid.shape[1]} -> {M}x{M}")
        K = bilinear_interpolate(K_grid, x_vec, y_vec, x_tgt, y_tgt)

    print(f"  Grid: {M}x{M}, dx=dy={dx:.6f}")

    N = M * M
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i + j * M

    for i in range(M):
        for j in range(M):
            row = idx(i, j)

            # Dirichlet boundary
            if i == 0:
                A[row, row] = 1.0
                b[row] = 1.0
                continue
            if i == M - 1:
                A[row, row] = 1.0
                b[row] = 0.0
                continue

            # Interior points (including Neumann boundaries)
            # x-direction
            K_ip = harmonic_mean(K[i, j], K[i + 1, j]) if i < M - 1 else K[i, j]
            K_im = harmonic_mean(K[i - 1, j], K[i, j]) if i > 0 else K[i, j]
            A[row, idx(i + 1, j)] = K_ip / (dx * dx)
            A[row, idx(i - 1, j)] = K_im / (dx * dx)
            A[row, row] = -(K_ip + K_im) / (dx * dx)

            # y-direction
            if j == 0:
                # Neumann bottom: no downward flux, only upward term
                K_jp = harmonic_mean(K[i, j], K[i, j + 1]) if j < M - 1 else K[i, j]
                A[row, idx(i, j + 1)] = K_jp / (dy * dy)
                A[row, row] -= K_jp / (dy * dy)
            elif j == M - 1:
                # Neumann top: no upward flux, only downward term
                K_jm = harmonic_mean(K[i, j - 1], K[i, j]) if j > 0 else K[i, j]
                A[row, idx(i, j - 1)] = K_jm / (dy * dy)
                A[row, row] -= K_jm / (dy * dy)
            else:
                K_jp = harmonic_mean(K[i, j], K[i, j + 1])
                K_jm = harmonic_mean(K[i, j - 1], K[i, j])
                A[row, idx(i, j + 1)] = K_jp / (dy * dy)
                A[row, idx(i, j - 1)] = K_jm / (dy * dy)
                A[row, row] -= (K_jp + K_jm) / (dy * dy)

    A_csr = A.tocsr()
    print(f"  Matrix: {N}x{N}, nnz={A_csr.nnz}")

    h_flat = spsolve(A_csr, b)
    h = h_flat.reshape(M, M, order='F')

    rel_res = np.linalg.norm(A_csr @ h_flat - b) / (np.linalg.norm(b) + 1e-15)
    print(f"  Relative residual: {rel_res:.2e}")
    print(f"  h range: [{h.min():.6f}, {h.max():.6f}]")

    return h, x_tgt, y_tgt, {
        'grid_size': M, 'dx': float(dx), 'dy': float(dy),
        'n_unknowns': N, 'nnz': int(A_csr.nnz),
        'relative_residual': float(rel_res),
        'h_range': [float(h.min()), float(h.max())],
    }


def compute_pde_residual_fdm(h, K, dx, dy):
    """Compute PDE residual of the FDM solution (quality check)"""
    M = h.shape[0]
    res = np.zeros((M, M))
    for i in range(1, M - 1):
        for j in range(1, M - 1):
            Kr = harmonic_mean(K[i, j], K[i + 1, j])
            Kl = harmonic_mean(K[i - 1, j], K[i, j])
            Ku = harmonic_mean(K[i, j], K[i, j + 1])
            Kd = harmonic_mean(K[i, j - 1], K[i, j])
            res[i, j] = (Kr * (h[i + 1, j] - h[i, j]) -
                         Kl * (h[i, j] - h[i - 1, j])) / (dx * dx) + \
                        (Ku * (h[i, j + 1] - h[i, j]) -
                         Kd * (h[i, j] - h[i, j - 1])) / (dy * dy)
    return res


# ============================================================
# PINN evaluation
# ============================================================

def evaluate_pinn_vs_fdm(model_path, config_path, x_fdm, y_fdm, h_fdm, device='cpu'):
    import torch
    import torch.nn as nn
    import yaml

    class PINNNet(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.net_cfg = cfg['network']
            self.physics = cfg['physics']
            self.bc = cfg.get('boundary_conditions', {})
            self.use_hard_dirichlet = self.net_cfg.get('hard_dirichlet', False)

            ff_cfg = self.net_cfg.get('fourier_features', {})
            self.use_fourier = ff_cfg.get('enabled', False)
            self.n_fourier = int(ff_cfg.get('num_frequencies', 0))
            if self.use_fourier and self.n_fourier > 0:
                max_freq = float(ff_cfg.get('max_frequency', self.n_fourier))
                if ff_cfg.get('spacing', 'linear') == 'log':
                    freqs = torch.logspace(0, math.log10(max_freq), self.n_fourier)
                else:
                    freqs = torch.linspace(1.0, max_freq, self.n_fourier)
                self.register_buffer('fourier_freqs', freqs.float())
                encoded_dim = self.net_cfg['input_dim'] + 2 * self.net_cfg['input_dim'] * self.n_fourier
            else:
                self.register_buffer('fourier_freqs', torch.empty(0))
                encoded_dim = self.net_cfg['input_dim']

            layers = []
            nd = self.net_cfg
            for i in range(nd['hidden_layers'] + 1):
                if i == 0:
                    layers.append(nn.Linear(encoded_dim, nd['hidden_width']))
                elif i == nd['hidden_layers']:
                    layers.append(nn.Linear(nd['hidden_width'], nd['output_dim']))
                else:
                    layers.append(nn.Linear(nd['hidden_width'], nd['hidden_width']))
                if i < nd['hidden_layers']:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            features = self._encode_input(x)
            raw = self.net(features)
            if not self.use_hard_dirichlet:
                return raw

            domain = self.physics['domain']
            x_coord = x[:, 0]
            x_min = domain['x_min']
            x_max = domain['x_max']
            s = (x_coord - x_min) / (x_max - x_min)
            h_left = self.bc.get('h_left', 1.0)
            h_right = self.bc.get('h_right', 0.0)
            base = (1.0 - s) * h_left + s * h_right
            gate = s * (1.0 - s)
            return base.unsqueeze(1) + gate.unsqueeze(1) * raw

        def _encode_input(self, x):
            if not self.use_fourier or self.n_fourier <= 0:
                return x
            angles = 2.0 * math.pi * x.unsqueeze(-1) * self.fourier_freqs
            return torch.cat([
                x,
                torch.sin(angles).reshape(x.shape[0], -1),
                torch.cos(angles).reshape(x.shape[0], -1),
            ], dim=1)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    net = PINNNet(cfg).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()

    M = len(x_fdm)
    X_m, Y_m = np.meshgrid(x_fdm, y_fdm, indexing='ij')
    xy = np.stack([X_m.ravel(), Y_m.ravel()], axis=1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)

    with torch.no_grad():
        h_pinn_flat = net(xy_t).cpu().numpy().ravel()

    # Spatial alignment note (easy to get wrong):
    #
    # X_m, Y_m = meshgrid(x_fdm, y_fdm, indexing='ij')  [shape: (M, M)]
    #   -> ravel() defaults to C-order (row-major), so xy[k] = (x_i, y_j), k = i*M + j
    #
    # h_pinn_flat[k] = net(xy[k]), so C-order reshape preserves [i,j] = (x_i, y_j)
    #
    # h_fdm from the FDM solver uses Fortran-order storage (column-major).
    # Use ascontiguousarray to convert to C-order, ensuring h_pinn[i,j] and h_fdm[i,j]
    # refer to the same spatial coordinate (x_i, y_j).
    h_pinn = h_pinn_flat.reshape(M, M)              # C-order: h_pinn[i,j] = (x_i, y_j)
    h_fdm = np.ascontiguousarray(h_fdm)              # C-order: h_fdm[i,j]  = (x_i, y_j)

    # Zone masks
    bc_dir_mask = (X_m <= x_fdm[1]) | (X_m >= x_fdm[-2])
    dy_f = y_fdm[1] - y_fdm[0]
    bc_neu_mask = (Y_m <= dy_f) | (Y_m >= 1.0 - dy_f)
    interior_mask = ~bc_dir_mask & ~bc_neu_mask
    margin = max(5, int(0.05 * M))
    deep_int = np.ones((M, M), dtype=bool)
    deep_int[:margin, :] = deep_int[-margin:, :] = \
        deep_int[:, :margin] = deep_int[:, -margin:] = False
    deep_int[bc_dir_mask | bc_neu_mask] = False

    def metrics(h1, h2, mask=None):
        if mask is not None:
            h1, h2 = h1[mask], h2[mask]
        e = h1 - h2
        return {
            'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae': float(np.mean(np.abs(e))),
            'max_error': float(np.max(np.abs(e))),
            'r2': float(1 - np.sum(e**2) / (np.sum((h2 - h2.mean())**2) + 1e-15)),
            'n': int(h1.size)
        }

    results = {
        'overall': metrics(h_pinn, h_fdm),
        'dirichlet_zone': metrics(h_pinn, h_fdm, bc_dir_mask),
        'neumann_zone': metrics(h_pinn, h_fdm, bc_neu_mask),
        'interior': metrics(h_pinn, h_fdm, interior_mask),
        'deep_interior': metrics(h_pinn, h_fdm, deep_int),
    }
    return results, h_pinn, X_m, Y_m


# ============================================================
# Visualization
# ============================================================

def plot_validation(h_fdm, x_fdm, y_fdm, h_pinn, results, K_grid, K_x, K_y,
                    output_dir, pde_res=None):
    os.makedirs(output_dir, exist_ok=True)
    M = len(x_fdm)
    X_m, Y_m = np.meshgrid(x_fdm, y_fdm, indexing='ij')
    error = h_pinn - h_fdm

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.38)

    # Row 1: h_FDM, h_PINN, error
    for ax_idx, (data, cmap, title) in enumerate(zip(
        [h_fdm, h_pinn, error],
        ['coolwarm', 'coolwarm', 'RdBu_r'],
        [r'$h_{\rm FDM}$ Reference', r'$h_{\rm PINN}$ Prediction',
         r'Error $h_{\rm PINN} - h_{\rm FDM}$']
    )):
        a = fig.add_subplot(gs[0, ax_idx])
        im = a.contourf(X_m, Y_m, data, levels=30, cmap=cmap)
        plt.colorbar(im, ax=a)
        a.set_xlabel('x')
        a.set_ylabel('y')
        a.set_title(title, fontsize=12)

    # Row 2: scatter, error histogram, y=0.5 profile comparison
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_scatter.scatter(h_fdm.ravel(), h_pinn.ravel(), alpha=0.08, s=3, c='steelblue')
    ax_scatter.plot([0, 1], [0, 1], 'r--', lw=2)
    ax_scatter.set_xlabel(r'$h_{\rm FDM}$')
    ax_scatter.set_ylabel(r'$h_{\rm PINN}$')
    ax_scatter.set_title("Scatter h vs FDM\n$R^2$={:.4f}".format(results["overall"]["r2"]))
    ax_scatter.grid(True, alpha=0.3)

    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(error.ravel(), bins=80, edgecolor='black', alpha=0.7, density=True)
    ax_hist.axvline(0, color='red', ls='--', lw=2)
    ax_hist.set_xlabel('Error')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title(f'Error Distribution\n'
                       f'RMSE={results["overall"]["rmse"]:.4f}  '
                       f'MAE={results["overall"]["mae"]:.4f}')
    ax_hist.grid(True, alpha=0.3)

    ax_profile = fig.add_subplot(gs[1, 2])
    mid = M // 2
    ax_profile.plot(x_fdm, h_fdm[:, mid], 'b-', lw=2, label='FDM')
    ax_profile.plot(x_fdm, h_pinn[:, mid], 'r--', lw=2, label='PINN')
    ax_profile.set_xlabel('x')
    ax_profile.set_ylabel('h')
    ax_profile.set_title(f'Profile at y={y_fdm[mid]:.2f}')
    ax_profile.legend()
    ax_profile.grid(True, alpha=0.3)

    # Row 3: zone error bar chart, K field, PDE residual
    ax_bar = fig.add_subplot(gs[2, 0])
    regions = ['Overall', 'Dirichlet\nZone', 'Neumann\nZone', 'Interior',
               'Deep\nInterior']
    keys = ['overall', 'dirichlet_zone', 'neumann_zone', 'interior', 'deep_interior']
    rmses = [results[k]['rmse'] for k in keys]
    maes = [results[k]['mae'] for k in keys]
    x_p = np.arange(len(regions))
    w = 0.35
    ax_bar.bar(x_p - w/2, rmses, w, label='RMSE', color='steelblue')
    ax_bar.bar(x_p + w/2, maes, w, label='MAE', color='coral')
    for bar, v in zip(ax_bar.patches, rmses + maes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{v:.4f}', ha='center', fontsize=8)
    ax_bar.set_xticks(x_p)
    ax_bar.set_xticklabels(regions)
    ax_bar.set_ylabel('Error (m)')
    ax_bar.set_title('Error by Region')
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')

    ax_K = fig.add_subplot(gs[2, 1])
    im = ax_K.contourf(K_x, K_y, K_grid.T, levels=30, cmap='YlOrBr')
    plt.colorbar(im, ax=ax_K, label=r'$K$')
    ax_K.set_xlabel('x')
    ax_K.set_ylabel('y')
    ax_K.set_title(r'$K(x,y)$ Heterogeneous Field')

    ax_res = fig.add_subplot(gs[2, 2])
    if pde_res is not None:
        interior = pde_res[1:-1, 1:-1]
        v = np.log10(np.abs(interior) + 1e-15)
        im = ax_res.contourf(X_m[1:-1, 1:-1], Y_m[1:-1, 1:-1], v,
                              levels=20, cmap='viridis')
        plt.colorbar(im, ax=ax_res, label=r'$\log_{10}|{\rm residual}|$')
        ax_res.set_xlabel('x')
        ax_res.set_ylabel('y')
        ax_res.set_title(r'FDM $\nabla\cdot(K\nabla h)$ Residual (quality check)')
    else:
        ax_res.text(0.5, 0.5, 'PDE residual\nnot computed', ha='center',
                     va='center', transform=ax_res.transAxes)
        ax_res.set_title('FDM Residual')

    plt.savefig(os.path.join(output_dir, 'validation_report.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] validation_report.png")


# ============================================================
# Main program
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 4: Finite-difference reference solution solver')
    parser.add_argument('--N', type=int, default=201,
                        help='Grid size (default 201x201)')
    parser.add_argument('--k-field', type=str, default=DEFAULT_K_FIELD,
                        help='K-field npz path')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--model', type=str,
                        default='outputs/stage3_heterogeneous/best_model.pt',
                        help='PINN model path')
    parser.add_argument('--config', type=str, default='configs/stage3a_3b_baseline.yaml',
                        help='PINN config path')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Compute FDM solution only, skip PINN evaluation')
    args = parser.parse_args()

    args.k_field = str(resolve_project_path(args.k_field))
    args.output_dir = str(resolve_project_path(args.output_dir))
    args.model = str(resolve_project_path(args.model))
    args.config = str(resolve_project_path(args.config))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 4: Finite-difference reference solution solver")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load K field
    print("\n[1/4] Loading K field...")
    K_grid_src, x_src, y_src, xi, eigenvalues, logK = load_K_field(args.k_field)

    # Step 2: FDM solve
    print(f"\n[2/4] FDM solve (N={args.N})...")
    h_fdm, x_fdm, y_fdm, conv = solve_steady_flow(K_grid_src, x_src, y_src, args.N)

    # Step 3: Compare with 1-x (to show how large the deviation is)
    print(f"\n[3/4] Comparing with incorrect baseline (1-x)...")
    X_m, Y_m = np.meshgrid(x_fdm, y_fdm, indexing='ij')
    h_1x = 1.0 - X_m
    diff = h_fdm - h_1x
    rmse_1x = np.sqrt(np.mean(diff ** 2))
    mae_1x = np.mean(np.abs(diff))
    print(f"  h_FDM vs h=1-x: RMSE={rmse_1x:.6f}, MAE={mae_1x:.6f}")
    print(f"  Max deviation: {np.abs(diff).max():.6f}")

    # Save h_true
    h_true_path = os.path.join(output_dir, f'h_true_N{args.N}.npz')
    np.savez_compressed(h_true_path, h=h_fdm, x=x_fdm, y=y_fdm,
                        h_1x=h_1x, diff_1x=diff, conv_info=conv,
                        xi=xi, eigenvalues=eigenvalues, logK=logK)
    print(f"  h_true saved: {h_true_path}")

    # Step 4: Evaluate PINN
    if not args.skip_eval and os.path.exists(args.model):
        print(f"\n[4/4] PINN vs FDM evaluation...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        results, h_pinn, X_pinn, Y_pinn = evaluate_pinn_vs_fdm(
            args.model, args.config, x_fdm, y_fdm, h_fdm, device=device
        )
        # evaluate_pinn_vs_fdm internally converts h_fdm to C-order; sync external h_fdm here
        h_fdm = np.ascontiguousarray(h_fdm)

        print(f"\n  {'='*52}")
        print(f"  Stage 3b model vs FDM reference")
        print(f"  {'='*52}")
        labels = {
            'overall': 'Overall',
            'dirichlet_zone': 'Dirichlet zone',
            'neumann_zone': 'Neumann zone',
            'interior': 'Interior',
            'deep_interior': 'Deep interior',
        }
        for k, label in labels.items():
            r = results[k]
            print(f"  {label:<20} RMSE={r['rmse']:.6f}  MAE={r['mae']:.6f}  "
                  f"MaxE={r['max_error']:.6f}  R2={r['r2']:.4f}  (n={r['n']})")

        # FDM PDE residual (quality check)
        K_fdm = bilinear_interpolate(K_grid_src, x_src, y_src, x_fdm, y_fdm)
        pde_res = compute_pde_residual_fdm(h_fdm, K_fdm, conv['dx'], conv['dy'])
        print(f"\n  FDM solution quality check:")
        print(f"    PDE residual mean: {np.abs(pde_res[1:-1,1:-1]).mean():.2e}")
        print(f"    PDE residual max:  {np.abs(pde_res[1:-1,1:-1]).max():.2e}")

        # Save evaluation results JSON
        eval_path = os.path.join(output_dir, 'evaluation_vs_fdm.json')
        with open(eval_path, 'w') as f:
            json.dump({
                'rmse_vs_1x': float(rmse_1x),
                'mae_vs_1x': float(mae_1x),
                'pinn_vs_fdm': results,
                'n_grid': args.N,
                'model': args.model,
            }, f, indent=2)
        print(f"  Evaluation results saved: {eval_path}")

        # Plot
        K_x_m, K_y_m = np.meshgrid(x_src, y_src, indexing='ij')
        plot_validation(h_fdm, x_fdm, y_fdm, h_pinn, results,
                        K_grid_src, K_x_m, K_y_m, output_dir, pde_res)

        print(f"\nAll results -> {output_dir}/")

    else:
        print(f"\nPINN evaluation skipped (file not found or --skip-eval specified)")
        print(f"h_true path: {h_true_path}")


if __name__ == '__main__':
    main()
