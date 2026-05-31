#!/usr/bin/env python3
"""
Stage 3c: Inverse problem — PINN training with learnable KLE coefficients xi

Core idea:
  K(x,y) = exp( sum_i sqrt(lambda_i) * phi_i(x,y) * xi_i )
  xi starts from random initial values and is jointly optimized with MLP weights.

Architecture:
  Phase 1 (pre-training): Fix xi=0 (K≡1), train MLP only -> learn reasonable h distribution
  Phase 2 (joint training): Jointly optimize MLP + xi -> invert K field

Usage:  python src/train_stage3c.py --config configs/stage3c_inverse.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from paths import PROJECT_ROOT, resolve_project_path


# ===========================================================================
# 1. KLE Invertible PINN
# ===========================================================================

class KLEInvertiblePINN(nn.Module):
    """PINN with learnable KLE coefficients ξ for the inverse problem."""

    def __init__(self, config, kle_basis):
        super().__init__()

        self.cfg = config
        self.domain = config['physics']['domain']
        self.net_cfg = config['network']
        self.bc = config.get('boundary_conditions', {})
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

        # ---- MLP for h(x,y) ----
        layers = []
        in_dim = encoded_dim
        hw = config['network']['hidden_width']
        nl = config['network']['hidden_layers']
        for i in range(nl + 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, hw))
            elif i == nl:
                layers.append(nn.Linear(hw, config['network']['output_dim']))
            else:
                layers.append(nn.Linear(hw, hw))
            if i < nl:
                act = config['network']['activation']
                if act == 'tanh':
                    layers.append(nn.Tanh())
                elif act == 'relu':
                    layers.append(nn.ReLU())
                elif act == 'gelu':
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

        # ---- KLE basis (fixed) ----
        M = kle_basis['M']
        nx, ny = kle_basis['nx'], kle_basis['ny']
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = kle_basis['Lx'], kle_basis['Ly']

        self.register_buffer('sqrt_lambda',
                             torch.tensor(kle_basis['sqrt_lambda'], dtype=torch.float32))
        self.register_buffer('phi_grid',
                             torch.tensor(kle_basis['phi_grid'], dtype=torch.float32))
        self.register_buffer('x_grid',
                             torch.tensor(kle_basis['x_grid'], dtype=torch.float32))
        self.register_buffer('y_grid',
                             torch.tensor(kle_basis['y_grid'], dtype=torch.float32))

        # ---- Learnable KLE coefficients ----
        self.xi = nn.Parameter(torch.zeros(M))

        # ---- Clamp range ----
        self.clamp_logK = config['inverse'].get('clamp_logK', 3.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_method = self.cfg['network'].get('init_method', 'xavier')
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_K_on_grid(self):
        """Compute K values on the physical grid: K = exp(logK).

        logK_grid = Σ sqrt(λᵢ) · φᵢ_grid · ξᵢ     [matrix form]
        shape: (N,) where N = nx * ny
        """
        logK_flat = (self.phi_grid.T * self.sqrt_lambda.unsqueeze(0)
                     * self.xi.unsqueeze(0)).sum(dim=1)  # (N,)
        logK_flat = torch.clamp(logK_flat, -self.clamp_logK, self.clamp_logK)
        return torch.exp(logK_flat)

    def compute_K_at_points(self, x, y):
        """Bilinear interpolation of K from grid to arbitrary query points.

        Parameters
        ----------
        x, y : Tensor (B,) or scalar
            Physical coordinates in [0, 1].

        Returns
        -------
        Tensor (B,) — K values at query points.
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            was_scalar = True
        else:
            was_scalar = False

        K_grid = self.compute_K_on_grid()  # (N,)

        nx, ny = self.nx, self.ny
        Lx, Ly = self.Lx, self.Ly

        x_norm = (x / Lx) * (nx - 1)
        y_norm = (y / Ly) * (ny - 1)

        x0 = torch.floor(x_norm).long().clamp(0, nx - 2)
        y0 = torch.floor(y_norm).long().clamp(0, ny - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = (x_norm - x0.float()).clamp(0.0, 1.0)
        wy = (y_norm - y0.float()).clamp(0.0, 1.0)

        idx00 = x0 * ny + y0
        idx10 = x1 * ny + y0
        idx01 = x0 * ny + y1
        idx11 = x1 * ny + y1

        K00 = K_grid[idx00]
        K10 = K_grid[idx10]
        K01 = K_grid[idx01]
        K11 = K_grid[idx11]

        K = (K00 * (1 - wx) * (1 - wy) +
             K10 * wx * (1 - wy) +
             K01 * (1 - wx) * wy +
             K11 * wx * wy)

        return K if not was_scalar else K.squeeze(0)

    def forward(self, x):
        features = self._encode_input(x)
        raw = self.net(features)
        if not self.use_hard_dirichlet:
            return raw

        x_coord = x[:, 0]
        x_min = self.domain['x_min']
        x_max = self.domain['x_max']
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

    def predict(self, x, y):
        xy = torch.stack([x, y], dim=1)
        with torch.no_grad():
            return self.forward(xy).squeeze().cpu().numpy()

    def pde_residual(self, x, y):
        """Compute ∇·(K∇h) residual using two-step autodiff.

        K(x,y) is computed from learnable ξ via bilinear interpolation.
        """
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        h = self.forward(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1))

        dh_dx = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                                     create_graph=True, retain_graph=True)[0]
        dh_dy = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h),
                                     create_graph=True, retain_graph=True)[0]

        K = self.compute_K_at_points(x.squeeze(), y.squeeze())

        K_dh_dx = K * dh_dx.squeeze()
        K_dh_dy = K * dh_dy.squeeze()

        div_Kh_x = torch.autograd.grad(K_dh_dx, x,
                                        grad_outputs=torch.ones_like(K_dh_dx),
                                        create_graph=True, retain_graph=True)[0]
        div_Kh_y = torch.autograd.grad(K_dh_dy, y,
                                        grad_outputs=torch.ones_like(K_dh_dy),
                                        create_graph=True, retain_graph=True)[0]

        return div_Kh_x.squeeze() + div_Kh_y.squeeze()

    def bc_dirichlet(self, x, y, h_bc):
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        h_pred = self.forward(xy)
        return (h_pred.squeeze() - h_bc) ** 2

    def bc_neumann(self, x, y):
        """Neumann no-flow boundary: dh/dn = 0 (top/bottom edges dh/dy = 0)"""
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        xy.requires_grad_(True)
        h = self.forward(xy)
        grad_h = torch.autograd.grad(
            h, xy, grad_outputs=torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]
        dh_dy = grad_h[:, 1]
        return dh_dy ** 2

    def data_loss(self, x_obs, y_obs, h_obs):
        xy = torch.cat([x_obs.unsqueeze(1), y_obs.unsqueeze(1)], dim=1)
        h_pred = self.forward(xy)
        return (h_pred.squeeze() - h_obs) ** 2


# ===========================================================================
# 2. Sampling helpers
# ===========================================================================

def sample_pde_points(config, n_points, device='cpu'):
    d = config['physics']['domain']
    x = torch.rand(n_points, device=device) * (d['x_max'] - d['x_min']) + d['x_min']
    y = torch.rand(n_points, device=device) * (d['y_max'] - d['y_min']) + d['y_min']
    return x, y

def sample_boundary_points(config, n_per_side, device='cpu'):
    """Sample boundary points: left/right Dirichlet, top/bottom Neumann"""
    bc = config['boundary_conditions']
    d = config['physics']['domain']

    # --- Dirichlet: left + right ---
    y_d = torch.rand(n_per_side, device=device) * (d['y_max'] - d['y_min']) + d['y_min']

    # Left edge (x=0): h = h_left
    x_left = torch.zeros(n_per_side, device=device)
    y_left = y_d
    h_left = torch.full((n_per_side,), bc['h_left'], device=device)

    # Right edge (x=1): h = h_right
    x_right = torch.ones(n_per_side, device=device)
    y_right = y_d
    h_right = torch.full((n_per_side,), bc['h_right'], device=device)

    # --- Neumann: top + bottom ---
    x_n = torch.rand(n_per_side, device=device) * (d['x_max'] - d['x_min']) + d['x_min']

    # Bottom edge (y=0): dh/dy = 0
    x_bottom = x_n
    y_bottom = torch.zeros(n_per_side, device=device)

    # Top edge (y=1): dh/dy = 0
    x_top = x_n
    y_top = torch.ones(n_per_side, device=device)

    return {
        'dirichlet': {
            'x': torch.cat([x_left, x_right]),
            'y': torch.cat([y_left, y_right]),
            'h': torch.cat([h_left, h_right]),
        },
        'neumann': {
            'x': torch.cat([x_bottom, x_top]),
            'y': torch.cat([y_bottom, y_top]),
        }
    }


# ===========================================================================
# 3. Synthetic observation generation
# ===========================================================================

def _bilinear_sample_grid_numpy(values, x_grid, y_grid, xq, yq):
    """Bilinear interpolation for grid values indexed as values[ix, iy]."""
    xq_np = xq.detach().cpu().numpy()
    yq_np = yq.detach().cpu().numpy()
    nx = len(x_grid)
    ny = len(y_grid)

    x_norm = (xq_np - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (nx - 1)
    y_norm = (yq_np - y_grid[0]) / (y_grid[-1] - y_grid[0]) * (ny - 1)

    x0 = np.floor(x_norm).astype(int).clip(0, nx - 2)
    y0 = np.floor(y_norm).astype(int).clip(0, ny - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = np.clip(x_norm - x0, 0.0, 1.0)
    wy = np.clip(y_norm - y0, 0.0, 1.0)

    sampled = (
        values[x0, y0] * (1 - wx) * (1 - wy)
        + values[x1, y0] * wx * (1 - wy)
        + values[x0, y1] * (1 - wx) * wy
        + values[x1, y1] * wx * wy
    )
    return torch.tensor(sampled, dtype=torch.float32, device=xq.device)


def generate_synthetic_observations(config, device='cpu'):
    """Generate synthetic h observations from FDM truth or a stage 3b model."""
    source = config['inverse'].get('observation_source', 'model')
    n_obs = config['inverse']['n_obs_points']
    d = config['physics']['domain']
    torch.manual_seed(config['training'].get('seed', 999))
    x_obs = torch.rand(n_obs, device=device) * (d['x_max'] - d['x_min']) + d['x_min']
    y_obs = torch.rand(n_obs, device=device) * (d['y_max'] - d['y_min']) + d['y_min']

    if source == 'fdm':
        fdm_path = resolve_project_path(config['inverse']['fdm_reference_path'])
        if not fdm_path.exists():
            raise FileNotFoundError(f"FDM reference not found: {fdm_path}")
        fdm = np.load(fdm_path, allow_pickle=True)
        h_grid = np.ascontiguousarray(fdm['h'])
        x_grid = fdm['x']
        y_grid = fdm['y']
        h_obs = _bilinear_sample_grid_numpy(h_grid, x_grid, y_grid, x_obs, y_obs)
        noise_std = config['inverse'].get('obs_noise', 0.0)
        if noise_std > 0:
            h_obs = h_obs + torch.randn(n_obs, device=device) * noise_std
        print(f"[INFO] Generated {n_obs} synthetic observations from FDM truth")
        print(f"       FDM path = {fdm_path}")
        print(f"       Noise std = {noise_std}")
        return x_obs, y_obs, h_obs

    # Project root is one level above src/
    project_root = PROJECT_ROOT
    ref_path = project_root / config['logging']['ref_model_path']
    if not ref_path.exists():
        print(f"[WARNING] Reference model not found: {ref_path}")
        print("          Using linear interpolation as fallback.")
        bc = config['boundary_conditions']
        h_obs = (1 - x_obs) * bc['h_left'] + x_obs * bc['h_right']
    else:
        # Import train module dynamically (src is not a package)
        import importlib.util
        _train_path = Path(__file__).parent / 'train.py'
        _spec = importlib.util.spec_from_file_location("train", _train_path)
        _train = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_train)
        PINN = _train.PINN
        load_K_field_from_npz = _train.load_K_field_from_npz
        # Reconstruct stage 3b config — k_field_file is relative to project root
        k_field = resolve_project_path(config['kle']['k_field_file'])
        K_func = load_K_field_from_npz(k_field, device=device)
        model_3b = PINN(config, K_func=K_func).to(device)
        state = torch.load(ref_path, map_location=device)
        model_3b.load_state_dict(state)
        model_3b.eval()

        with torch.no_grad():
            h_true = model_3b.predict(x_obs, y_obs)
            h_true = torch.tensor(h_true, device=device)
        noise_std = config['inverse']['obs_noise']
        noise = torch.randn(n_obs, device=device) * noise_std
        h_obs = h_true + noise

        print(f"[INFO] Generated {n_obs} synthetic observations from stage 3b model")
        print(f"       Noise std = {noise_std}")

    return x_obs, y_obs, h_obs


def _cosine_to_target(vec, target):
    vec_norm = torch.linalg.norm(vec)
    target_norm = torch.linalg.norm(target)
    if vec_norm.item() == 0.0 or target_norm.item() == 0.0:
        return None
    return float(torch.dot(vec, target).item() / (vec_norm.item() * target_norm.item()))


def _load_xi_true(config, model, device):
    k_field_path = resolve_project_path(config['kle']['k_field_file'])
    if not k_field_path.exists():
        return None
    data = np.load(k_field_path, allow_pickle=True)
    if 'xi' not in data.files:
        return None
    xi_true = data['xi'][:len(model.xi)]
    return torch.tensor(xi_true, dtype=torch.float32, device=device)


def _load_xi_init_file(config, model, device):
    inv_cfg = config.get('inverse', {})
    xi_path = inv_cfg.get('xi_init_file')
    if not xi_path:
        raise ValueError("xi_init=file requires inverse.xi_init_file")

    xi_path = resolve_project_path(xi_path)
    if xi_path.suffix.lower() == '.npz':
        data = np.load(xi_path, allow_pickle=True)
        if 'xi' in data.files:
            xi = data['xi']
        elif 'xi_pred' in data.files:
            xi = data['xi_pred']
        else:
            raise ValueError(f"{xi_path} must contain 'xi' or 'xi_pred'")
    elif xi_path.suffix.lower() == '.json':
        with open(xi_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'xi' in data:
            xi = data['xi']
        elif 'xi_pred' in data:
            xi = data['xi_pred']
        elif 'stage_b' in data and 'xi_pred' in data['stage_b']:
            xi = data['stage_b']['xi_pred']
        else:
            raise ValueError(f"{xi_path} must contain 'xi' or 'xi_pred'")
    else:
        raise ValueError("xi_init_file must be .npz or .json")

    xi = np.asarray(xi, dtype=np.float32).reshape(-1)
    if len(xi) < len(model.xi):
        raise ValueError(
            f"xi_init_file has {len(xi)} values, but model needs {len(model.xi)}"
        )
    return torch.tensor(xi[:len(model.xi)], dtype=torch.float32, device=device)


def _grad_stats(loss, model, xi_true=None):
    grad = torch.autograd.grad(
        loss, model.xi, retain_graph=True, allow_unused=True
    )[0]
    if grad is None:
        return {
            'is_none': True,
            'norm': 0.0,
            'max_abs': 0.0,
            'mean_abs': 0.0,
            'cos_grad_to_target': None,
            'cos_neg_grad_to_target': None,
        }
    grad_abs = grad.detach().abs()
    target = None if xi_true is None else (xi_true - model.xi.detach())
    cos_grad = None if target is None else _cosine_to_target(grad.detach(), target)
    cos_neg_grad = None if target is None else _cosine_to_target(-grad.detach(), target)
    return {
        'is_none': False,
        'norm': float(torch.linalg.norm(grad.detach()).item()),
        'max_abs': float(grad_abs.max().item()),
        'mean_abs': float(grad_abs.mean().item()),
        'cos_grad_to_target': cos_grad,
        'cos_neg_grad_to_target': cos_neg_grad,
    }


def initialize_xi(model, config, device):
    inv_cfg = config.get('inverse', {})
    mode = inv_cfg.get('xi_init', 'zero')
    if mode == 'zero':
        return

    with torch.no_grad():
        if mode == 'random':
            seed = int(inv_cfg.get('xi_init_seed', config.get('training', {}).get('seed', 999)))
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            std = float(inv_cfg.get('xi_init_std', 0.1))
            model.xi.copy_(torch.randn(model.xi.shape, generator=gen, device=device) * std)
        elif mode == 'true_fraction':
            xi_true = _load_xi_true(config, model, device)
            if xi_true is None:
                raise ValueError("xi_init=true_fraction requires xi in k_field_file")
            fraction = float(inv_cfg.get('xi_init_fraction', 0.1))
            model.xi.copy_(fraction * xi_true)
        elif mode == 'file':
            model.xi.copy_(_load_xi_init_file(config, model, device))
        else:
            raise ValueError(f"Unknown inverse.xi_init: {mode}")


def diagnose_xi_gradients(model, config, x_obs, y_obs, h_obs, device, out_dir, label):
    """Report which loss terms can move xi.

    The data term only constrains h directly. In this formulation xi is
    coupled to h through the PDE residual, so loss_data -> xi is expected
    to be none/zero unless the model formulation changes.
    """
    diag_cfg = config.get('diagnostics', {})
    n_pde = int(diag_cfg.get('n_pde_points', min(config['training']['n_pde_points'], 256)))
    n_bc = int(diag_cfg.get('n_bc_points', min(config['training']['n_bc_points'], 80)))
    n_bc_per_side = max(1, n_bc // 4)

    w = config['training']['weights']
    w_bc = w.get('bc', 1.0)
    w_pde = w.get('pde', 1.0)
    w_bc_d = w.get('bc_dirichlet', w_bc)
    w_bc_n = w.get('bc_neumann', w_bc)
    w_data = w.get('data', 10.0)
    w_xi_prior = w.get('xi_prior', 0.0)

    was_training = model.training
    was_requires_grad = model.xi.requires_grad
    model.train()
    model.xi.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    xi_true = _load_xi_true(config, model, device)

    x_pde, y_pde = sample_pde_points(config, n_pde, device)
    bc_points = sample_boundary_points(config, n_bc_per_side, device)

    loss_pde = torch.mean(model.pde_residual(x_pde, y_pde) ** 2)
    loss_bc_d = torch.mean(model.bc_dirichlet(
        bc_points['dirichlet']['x'], bc_points['dirichlet']['y'],
        bc_points['dirichlet']['h']))
    loss_bc_n = torch.mean(model.bc_neumann(
        bc_points['neumann']['x'], bc_points['neumann']['y']))
    loss_data = torch.mean(model.data_loss(x_obs, y_obs, h_obs))
    loss_xi_prior = torch.mean(model.xi ** 2)
    k_interp_mean = torch.mean(model.compute_K_at_points(x_pde, y_pde))

    loss_total = (
        w_pde * loss_pde
        + w_bc_d * loss_bc_d
        + w_bc_n * loss_bc_n
        + w_data * loss_data
        + w_xi_prior * loss_xi_prior
    )

    results = {
        'label': label,
        'xi_norm': float(model.xi.detach().norm().item()),
        'distance_to_xi_true': None if xi_true is None else float(torch.linalg.norm(xi_true - model.xi.detach()).item()),
        'loss_values': {
            'pde': float(loss_pde.detach().item()),
            'bc_dirichlet': float(loss_bc_d.detach().item()),
            'bc_neumann': float(loss_bc_n.detach().item()),
            'data': float(loss_data.detach().item()),
            'xi_prior': float(loss_xi_prior.detach().item()),
            'total': float(loss_total.detach().item()),
            'k_interp_mean': float(k_interp_mean.detach().item()),
        },
        'xi_gradients': {
            'k_interp_mean': _grad_stats(k_interp_mean, model, xi_true),
            'loss_pde': _grad_stats(loss_pde, model, xi_true),
            'loss_bc_dirichlet': _grad_stats(loss_bc_d, model, xi_true),
            'loss_bc_neumann': _grad_stats(loss_bc_n, model, xi_true),
            'loss_data': _grad_stats(loss_data, model, xi_true),
            'loss_xi_prior': _grad_stats(loss_xi_prior, model, xi_true),
            'loss_total_weighted': _grad_stats(loss_total, model, xi_true),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'gradient_diagnostics_{label}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 60)
    print(f"XI GRADIENT DIAGNOSTICS: {label}")
    print("-" * 60)
    for name, stats in results['xi_gradients'].items():
        none_flag = " none" if stats['is_none'] else ""
        print(
            f"  {name:22s} | norm={stats['norm']:.3e} "
            f"max={stats['max_abs']:.3e} mean={stats['mean_abs']:.3e} "
            f"cos(-g,target)={stats['cos_neg_grad_to_target']}{none_flag}"
        )
    print(f"  saved: {out_dir / f'gradient_diagnostics_{label}.json'}")

    model.xi.requires_grad_(was_requires_grad)
    if not was_training:
        model.eval()
    return results


def save_phase2_step0_diagnostics(model, config, out_dir, xi_before, xi_after, raw_grad, loss_value):
    xi_true = _load_xi_true(config, model, xi_after.device)
    target_before = None if xi_true is None else xi_true - xi_before
    update = xi_after - xi_before

    results = {
        'loss_total_before_step': float(loss_value),
        'xi_norm_before': float(torch.linalg.norm(xi_before).item()),
        'xi_norm_after': float(torch.linalg.norm(xi_after).item()),
        'update_norm': float(torch.linalg.norm(update).item()),
        'raw_grad_norm': float(torch.linalg.norm(raw_grad).item()),
        'distance_to_xi_true_before': None if xi_true is None else float(torch.linalg.norm(xi_true - xi_before).item()),
        'distance_to_xi_true_after': None if xi_true is None else float(torch.linalg.norm(xi_true - xi_after).item()),
        'cos_raw_grad_to_target': None if target_before is None else _cosine_to_target(raw_grad, target_before),
        'cos_neg_raw_grad_to_target': None if target_before is None else _cosine_to_target(-raw_grad, target_before),
        'cos_actual_update_to_target': None if target_before is None else _cosine_to_target(update, target_before),
        'xi_before': xi_before.detach().cpu().numpy().tolist(),
        'xi_after': xi_after.detach().cpu().numpy().tolist(),
        'raw_grad': raw_grad.detach().cpu().numpy().tolist(),
    }
    if results['distance_to_xi_true_before'] is not None:
        results['distance_delta_after_minus_before'] = (
            results['distance_to_xi_true_after'] - results['distance_to_xi_true_before']
        )

    with open(out_dir / 'phase2_step0_direction_diagnostics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 60)
    print("PHASE2 STEP0 DIRECTION DIAGNOSTICS")
    print("-" * 60)
    print(f"  cos(-raw_grad, xi_true-xi) = {results['cos_neg_raw_grad_to_target']}")
    print(f"  cos(actual_update, target) = {results['cos_actual_update_to_target']}")
    print(f"  distance delta             = {results.get('distance_delta_after_minus_before')}")
    print(f"  saved: {out_dir / 'phase2_step0_direction_diagnostics.json'}")
    return results


# ===========================================================================
# 4. Training
# ===========================================================================

def train_stage3c(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    # ---- step 1: prepare KLE basis ----
    sys.path.insert(0, str(Path(__file__).parent))
    from kle import KLE

    k_cfg = config['kle']
    # Load the K_field.npz to get the exact eigenpairs used for 3b
    project_root = PROJECT_ROOT
    k_field_path = resolve_project_path(k_cfg['k_field_file'])
    if k_field_path.exists():
        k_data = np.load(k_field_path)
        # Use eigenpairs from npz (guaranteed consistency with 3b)
        eigenvalues_full = k_data['eigenvalues']  # (50,)
        eigenfunctions_full = k_data['eigenfunctions']  # (2500, 50)
        M_learn = config['inverse']['n_modes']  # e.g. 20
        # Truncate to M_learn modes
        sqrt_lambda = np.sqrt(eigenvalues_full[:M_learn])
        phi_grid = eigenfunctions_full[:, :M_learn].T.copy()  # (M, 2500)
        x_grid = k_data.get('x', np.linspace(0, k_cfg['Lx'], k_cfg['nx']))
        y_grid = k_data.get('y', np.linspace(0, k_cfg['Ly'], k_cfg['ny']))
        # Create flat grid coordinates
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        x_flat = X.ravel().astype(np.float32)
        y_flat = Y.ravel().astype(np.float32)
        kle_basis = {
            'sqrt_lambda': sqrt_lambda,
            'phi_grid': phi_grid,
            'x_grid': x_flat,
            'y_grid': y_flat,
            'nx': k_cfg['nx'],
            'ny': k_cfg['ny'],
            'Lx': k_cfg['Lx'],
            'Ly': k_cfg['Ly'],
            'M': M_learn,
        }
        print(f"[KLE] Loaded basis from npz: {k_field_path}")
        print(f"[KLE] Using first {M_learn} modes out of {len(eigenvalues_full)} total")
    else:
        # Fallback: compute from scratch
        kle = KLE(Lx=k_cfg['Lx'], Ly=k_cfg['Ly'],
                  nx=k_cfg['nx'], ny=k_cfg['ny'],
                  lx=k_cfg['lx'], ly=k_cfg['ly'],
                  sigma2=k_cfg['sigma2'],
                  cov_type=k_cfg.get('cov_type', 'exponential'),
                  seed=42)
        kle.compute_eigenpairs()
        kle.truncate(M=k_cfg['n_modes'])
        kle_basis = kle.get_basis_for_pytorch()

    # ---- step 2: create model ----
    model = KLEInvertiblePINN(config, kle_basis).to(device)
    initialize_xi(model, config, device)

    # ---- step 3: generate synthetic observations ----
    x_obs, y_obs, h_obs = generate_synthetic_observations(config, device)

    # ---- step 4: output directory ----
    out_dir = resolve_project_path(config['logging']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- loss weights ----
    w_pde = config['training']['weights'].get('pde', 1.0)
    w_bc = config['training']['weights'].get('bc', 1.0)
    w_bc_d = config['training']['weights'].get('bc_dirichlet', w_bc)
    w_bc_n = config['training']['weights'].get('bc_neumann', w_bc)
    w_data = config['training']['weights'].get('data', 10.0)
    w_xi_prior = config['training']['weights'].get('xi_prior', 0.0)

    # ---- history tracking ----
    history = {'epoch': [], 'loss_total': [], 'loss_pde': [],
               'loss_bc': [], 'loss_bc_dirichlet': [], 'loss_bc_neumann': [],
               'loss_data': [], 'loss_xi_prior': [], 'lr': [],
               'xi_norm': []}

    n_pde = config['training']['n_pde_points']
    n_bc = config['training']['n_bc_points']
    n_bc_per_side = n_bc // 4

    print_every = config['logging']['print_every']
    plot_every = config['logging']['plot_every']
    run_gradient_diagnostics = config.get('diagnostics', {}).get('gradient_check', False)
    if run_gradient_diagnostics:
        diagnose_xi_gradients(
            model, config, x_obs, y_obs, h_obs, device, out_dir, 'initial'
        )
    if config.get('diagnostics', {}).get('diagnose_only', False):
        print("\n[DIAGNOSTICS] diagnose_only=true; skipping training.")
        return model, history

    # ======================================================================
    # Phase 1: Pre-training with ξ=0 (K≡1)
    # ======================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-training MLP (ξ fixed at 0, K≡1)")
    print("=" * 60)

    model.xi.requires_grad_(False)

    n_iter1 = config['training']['phase1_iterations']
    lr1 = config['training']['lr_phase1']
    optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr1, weight_decay=config['training']['weight_decay'])
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'])

    best_loss_phase1 = float('inf')

    for epoch in range(n_iter1):
        # Sample
        x_pde, y_pde = sample_pde_points(config, n_pde, device)
        bc_points = sample_boundary_points(config, n_bc_per_side, device)

        # Loss
        loss_pde = torch.mean(model.pde_residual(x_pde, y_pde) ** 2)
        loss_bc_d = torch.mean(model.bc_dirichlet(
            bc_points['dirichlet']['x'], bc_points['dirichlet']['y'],
            bc_points['dirichlet']['h']))
        loss_bc_n = torch.mean(model.bc_neumann(
            bc_points['neumann']['x'], bc_points['neumann']['y']))
        loss_bc = loss_bc_d + loss_bc_n
        loss_xi_prior = torch.mean(model.xi ** 2)
        loss = w_pde * loss_pde + w_bc_d * loss_bc_d + w_bc_n * loss_bc_n

        # Backward
        optimizer1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer1.step()
        scheduler1.step(loss.item())

        # Track
        if epoch % print_every == 0 or epoch == n_iter1 - 1:
            print(f"[P1] epoch {epoch:5d} | loss {loss.item():.4e} | "
                  f"PDE {loss_pde.item():.2e} | BC {loss_bc.item():.2e} | "
                  f"LR {optimizer1.param_groups[0]['lr']:.2e}")

        history['epoch'].append(epoch)
        history['loss_total'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_bc_dirichlet'].append(loss_bc_d.item())
        history['loss_bc_neumann'].append(loss_bc_n.item())
        history['loss_data'].append(0.0)
        history['loss_xi_prior'].append(loss_xi_prior.item())
        history['lr'].append(optimizer1.param_groups[0]['lr'])
        history['xi_norm'].append(0.0)

        if loss.item() < best_loss_phase1:
            best_loss_phase1 = loss.item()

        if epoch > 0 and epoch % plot_every == 0:
            _plot_prediction(model, out_dir / f'prediction_phase1_{epoch:05d}.png',
                           config, device)

    _save_p1 = out_dir / 'model_phase1_final.pt'
    if _save_p1.exists():
        _save_p1.unlink()
    torch.save(model.state_dict(), _save_p1, _use_new_zipfile_serialization=False)
    print(f"[Phase 1 done] best loss = {best_loss_phase1:.4e}")

    # ======================================================================
    # Phase 2: Joint training (MLP + ξ)
    # ======================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Joint training (MLP + learnable ξ)")
    print("=" * 60)

    freeze_xi_phase2 = config.get('inverse', {}).get('freeze_xi', False)
    model.xi.requires_grad_(not freeze_xi_phase2)
    if run_gradient_diagnostics:
        diagnose_xi_gradients(
            model, config, x_obs, y_obs, h_obs, device, out_dir, 'after_phase1'
        )

    n_iter2 = config['training']['phase2_iterations']
    lr2 = config['training']['lr_phase2']
    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2,
                            weight_decay=config['training']['weight_decay'])
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'])

    best_loss_phase2 = float('inf')
    best_epoch = 0

    for epoch in range(n_iter2):
        # Sample
        x_pde, y_pde = sample_pde_points(config, n_pde, device)
        bc_points = sample_boundary_points(config, n_bc_per_side, device)

        # Loss with data term
        loss_pde = torch.mean(model.pde_residual(x_pde, y_pde) ** 2)
        loss_bc_d = torch.mean(model.bc_dirichlet(
            bc_points['dirichlet']['x'], bc_points['dirichlet']['y'],
            bc_points['dirichlet']['h']))
        loss_bc_n = torch.mean(model.bc_neumann(
            bc_points['neumann']['x'], bc_points['neumann']['y']))
        loss_bc = loss_bc_d + loss_bc_n
        loss_data = torch.mean(model.data_loss(x_obs, y_obs, h_obs))
        loss_xi_prior = torch.mean(model.xi ** 2)
        loss = (
            w_pde * loss_pde
            + w_bc_d * loss_bc_d
            + w_bc_n * loss_bc_n
            + w_data * loss_data
            + w_xi_prior * loss_xi_prior
        )

        # Backward
        optimizer2.zero_grad()
        xi_before_step = None
        raw_xi_grad = None
        if run_gradient_diagnostics and epoch == 0:
            xi_before_step = model.xi.detach().clone()
            raw_xi_grad = torch.autograd.grad(
                loss, model.xi, retain_graph=True, allow_unused=True
            )[0]
            if raw_xi_grad is None:
                raw_xi_grad = torch.zeros_like(model.xi)
            else:
                raw_xi_grad = raw_xi_grad.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer2.step()
        scheduler2.step(loss.item())
        if run_gradient_diagnostics and epoch == 0:
            save_phase2_step0_diagnostics(
                model, config, out_dir, xi_before_step,
                model.xi.detach().clone(), raw_xi_grad, loss.item()
            )

        # Track
        xi_norm = model.xi.norm().item()
        global_epoch = n_iter1 + epoch

        if epoch % print_every == 0 or epoch == n_iter2 - 1:
            print(f"[P2] epoch {epoch:5d} | loss {loss.item():.4e} | "
                  f"PDE {loss_pde.item():.2e} | BC {loss_bc.item():.2e} | "
                  f"data {loss_data.item():.2e} | |ξ|={xi_norm:.3f} | "
                  f"LR {optimizer2.param_groups[0]['lr']:.2e}")

        history['epoch'].append(global_epoch)
        history['loss_total'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_bc_dirichlet'].append(loss_bc_d.item())
        history['loss_bc_neumann'].append(loss_bc_n.item())
        history['loss_data'].append(loss_data.item())
        history['loss_xi_prior'].append(loss_xi_prior.item())
        history['lr'].append(optimizer2.param_groups[0]['lr'])
        history['xi_norm'].append(xi_norm)

        if loss.item() < best_loss_phase2:
            best_loss_phase2 = loss.item()
            best_epoch = global_epoch
            # Save at phase end, not here (avoid Windows file lock)

        if epoch > 0 and epoch % plot_every == 0:
            _plot_prediction(model, out_dir / f'prediction_phase2_{epoch:05d}.png',
                           config, device)
            _plot_xi_convergence(history, out_dir)

    # ---- Final save ----
    _save_final = out_dir / 'model_final.pt'
    if _save_final.exists():
        _save_final.unlink()
    torch.save(model.state_dict(), _save_final, _use_new_zipfile_serialization=False)

    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Phase 2 best loss = {best_loss_phase2:.4e} at epoch {best_epoch}")

    # ---- Plot final results ----
    _plot_prediction(model, out_dir / 'prediction_final.png', config, device)
    _plot_training_curves(history, out_dir)
    _plot_xi_convergence(history, out_dir)

    # ---- Evaluate ----
    evaluate_inverse(model, config, out_dir, device)

    return model, history


# ===========================================================================
# 5. Evaluation
# ===========================================================================

def evaluate_inverse(model, config, out_dir, device):
    """Evaluate inverse problem: compare recovered K vs true K."""
    print("\n" + "=" * 60)
    print("EVALUATION: K field inversion")
    print("=" * 60)

    # Load ground-truth K field (project root is one level above src/)
    project_root = PROJECT_ROOT
    k_field_path = resolve_project_path(config['kle']['k_field_file'])
    if not k_field_path.exists():
        print("[WARNING] No ground-truth K field found. Skipping K evaluation.")
        return

    data = np.load(k_field_path, allow_pickle=True)
    K_true = data['K']  # (50, 50)
    X_grid = data['X']
    Y_grid = data['Y']
    logK_true = data.get('logK', np.log(K_true))
    xi_true_full = data['xi']  # (50,) — full xi used for 3b
    # Only compare the first M_learn modes
    M_learn = len(model.xi)
    xi_true = xi_true_full[:M_learn]

    # Predict K from learned ξ
    model.eval()
    with torch.no_grad():
        K_pred_grid = model.compute_K_on_grid().cpu().numpy().reshape(50, 50)
        logK_pred = np.log(np.clip(K_pred_grid, 1e-6, None))
        xi_pred = model.xi.cpu().numpy()

    if 'eigenfunctions' in data.files and 'eigenvalues' in data.files:
        phi = data['eigenfunctions'][:, :M_learn]
        eig = data['eigenvalues'][:M_learn]
        logK_trunc = (phi * np.sqrt(eig)[None, :] * xi_true[None, :]).sum(axis=1).reshape(K_true.shape)
        K_true_trunc = np.exp(logK_trunc)
    else:
        logK_trunc = logK_true
        K_true_trunc = K_true

    # Metrics
    K_rmse = np.sqrt(np.mean((K_pred_grid - K_true) ** 2))
    K_mae = np.mean(np.abs(K_pred_grid - K_true))
    logK_rmse = np.sqrt(np.mean((logK_pred - logK_true) ** 2))
    K_rmse_trunc = np.sqrt(np.mean((K_pred_grid - K_true_trunc) ** 2))
    logK_rmse_trunc = np.sqrt(np.mean((logK_pred - logK_trunc) ** 2))
    xi_corr = float(np.corrcoef(xi_pred, xi_true)[0, 1]) if M_learn > 1 else float('nan')
    xi_sign_accuracy = float(np.mean(np.sign(xi_pred) == np.sign(xi_true)))

    h_metrics = {}
    fdm_path = config['inverse'].get('fdm_reference_path')
    if fdm_path:
        fdm_path = resolve_project_path(fdm_path)
        if fdm_path.exists():
            fdm = np.load(fdm_path, allow_pickle=True)
            h_fdm = np.ascontiguousarray(fdm['h'])
            x_fdm = fdm['x']
            y_fdm = fdm['y']
            Xf, Yf = np.meshgrid(x_fdm, y_fdm, indexing='ij')
            with torch.no_grad():
                h_pred = model.predict(
                    torch.tensor(Xf.ravel(), dtype=torch.float32, device=device),
                    torch.tensor(Yf.ravel(), dtype=torch.float32, device=device),
                ).reshape(h_fdm.shape)
            err = h_pred - h_fdm
            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((h_fdm - h_fdm.mean()) ** 2)
            h_metrics = {
                'h_rmse_vs_fdm': float(np.sqrt(np.mean(err ** 2))),
                'h_mae_vs_fdm': float(np.mean(np.abs(err))),
                'h_maxe_vs_fdm': float(np.max(np.abs(err))),
                'h_r2_vs_fdm': float(1.0 - ss_res / ss_tot),
            }

    results = {
        'K_rmse': float(K_rmse),
        'K_mae': float(K_mae),
        'logK_rmse': float(logK_rmse),
        'K_rmse_truncated_truth': float(K_rmse_trunc),
        'logK_rmse_truncated_truth': float(logK_rmse_trunc),
        'xi_corr': xi_corr,
        'xi_sign_accuracy': xi_sign_accuracy,
        'xi_true': xi_true.tolist(),
        'xi_pred': xi_pred.tolist(),
        **h_metrics,
    }

    with open(out_dir / 'evaluation_results_inverse.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  K RMSE    = {K_rmse:.4e}")
    print(f"  K MAE     = {K_mae:.4e}")
    print(f"  logK RMSE = {logK_rmse:.4e}")
    print(f"  K RMSE (truncated truth)    = {K_rmse_trunc:.4e}")
    print(f"  logK RMSE (truncated truth) = {logK_rmse_trunc:.4e}")
    print(f"  xi corr   = {xi_corr:.4f}")
    if h_metrics:
        print(f"  h RMSE vs FDM = {h_metrics['h_rmse_vs_fdm']:.4e}")

    # Plot: K_true vs K_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(K_true.min(), K_pred_grid.min())
    vmax = max(K_true.max(), K_pred_grid.max())

    im0 = axes[0].contourf(X_grid, Y_grid, K_true,
                           levels=20, cmap='YlOrRd')
    axes[0].set_title('K True (ground truth)')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X_grid, Y_grid, K_pred_grid,
                           levels=20, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    axes[1].set_title('K Predicted (inverted)')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    axes[2].scatter(K_true.flatten(), K_pred_grid.flatten(),
                    s=1, alpha=0.5, c='#2563eb')
    axes[2].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
    axes[2].set_xlabel('K True')
    axes[2].set_ylabel('K Predicted')
    axes[2].set_title(f'K scatter (RMSE={K_rmse:.3f})')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    fig.savefig(out_dir / 'evaluation_inverse_K.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: evaluation_inverse_K.png")

    # Plot: ξ convergence
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xi_pred, 'o-', markersize=4, linewidth=0.8, color='#2563eb',
            label='Predicted ξ')
    ax.plot(xi_true, 's--', markersize=4, linewidth=0.8, color='#dc2626',
            label='True ξ')
    ax.set_xlabel('Mode index i')
    ax.set_ylabel('ξᵢ')
    ax.set_title('KLE Coefficients: Predicted vs True')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'xi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved: xi_comparison.png")

    return


# ===========================================================================
# 6. Plotting helpers
# ===========================================================================

def _plot_prediction(model, save_path, config, device):
    model.eval()
    d = config['physics']['domain']
    nx, ny = 80, 80
    x = torch.linspace(d['x_min'], d['x_max'], nx, device=device)
    y = torch.linspace(d['y_min'], d['y_max'], ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    with torch.no_grad():
        h_flat = model.predict(X.flatten(), Y.flatten())
        H = h_flat.reshape(nx, ny)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].contourf(X.cpu(), Y.cpu(), H, levels=30, cmap='Spectral_r')
    axes[0].set_title(f'h(x,y) prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='h (m)')

    # PDE residual sample (outside no_grad since pde_residual needs autograd)
    n_sample = 500
    xs = torch.rand(n_sample, device=device) * (d['x_max'] - d['x_min']) + d['x_min']
    ys = torch.rand(n_sample, device=device) * (d['y_max'] - d['y_min']) + d['y_min']

    model.eval()
    with torch.no_grad():
        h_flat = model.predict(X.flatten(), Y.flatten())
        H = h_flat.reshape(nx, ny)

    with torch.enable_grad():
        res = model.pde_residual(xs, ys).detach().cpu().numpy()
    sc = axes[1].scatter(xs.cpu(), ys.cpu(), c=np.abs(res), cmap='hot',
                         s=2, alpha=0.6)
    axes[1].set_title('|PDE residual| distribution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    model.train()


def _plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    epochs = history['epoch']
    # Loss
    axes[0].semilogy(epochs, history['loss_total'], 'k-', linewidth=1.5, label='Total')
    axes[0].semilogy(epochs, history['loss_pde'], 'b--', linewidth=1, label='PDE')
    axes[0].semilogy(epochs, history['loss_bc'], 'r--', linewidth=1, label='BC')
    axes[0].semilogy(epochs, history['loss_data'], 'g--', linewidth=1, label='Data')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # LR
    axes[1].semilogy(epochs, history['lr'], 'm-', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('LR Schedule')
    axes[1].grid(True, alpha=0.3)

    # ξ norm
    if any(v > 0 for v in history['xi_norm']):
        axes[2].plot(epochs, history['xi_norm'], 'c-', linewidth=1.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('||ξ||')
        axes[2].set_title('ξ Vector Norm')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_xi_convergence(history, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = history['epoch']
    xi_norms = history['xi_norm']
    if any(v > 0 for v in xi_norms):
        ax.plot(epochs, xi_norms, 'c-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('||ξ||')
        ax.set_title('ξ Convergence')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / 'xi_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# 7. Main
# ===========================================================================

def load_config(config_path):
    with open(resolve_project_path(config_path), 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3c: Inverse Problem')
    parser.add_argument('--config', type=str, default='configs/stage3c_inverse.yaml',
                        help='Path to config YAML')
    parser.add_argument('--diagnose-only', action='store_true',
                        help='Run gradient diagnostics and skip training')
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(str(config_path))
    if args.diagnose_only:
        config.setdefault('diagnostics', {})
        config['diagnostics']['gradient_check'] = True
        config['diagnostics']['diagnose_only'] = True

    print("=" * 60)
    print("Stage 3c: Inverse Problem — KLE Coefficients Learnable")
    print(f"Config: {config_path}")
    print(f"KLE modes: {config['kle']['n_modes']}")
    print(f"Obs points: {config['inverse']['n_obs_points']}")
    print(f"Data weight: {config['training']['weights']['data']}")
    print("=" * 60)

    model, history = train_stage3c(config)
    print("\nStage 3c complete!")
