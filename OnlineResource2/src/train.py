"""
Stage 3: PINN training framework (homogeneous validation + heterogeneous extension)
=======================================
Usage:
  python src/train.py --config configs/stage3a_3b_baseline.yaml --stage 3a   # homogeneous validation
  python src/train.py --config configs/stage3a_3b_baseline.yaml --stage 3b   # heterogeneous extension (KLE field)
  python src/train.py --config configs/stage3b_50mode_fourier_hardbc.yaml --stage 3b
Goals:
  - 3a: Homogeneous K field, validated against analytical solution
  - 3b: Heterogeneous K field (KLE-generated), PDE residual ∇·(K∇h)=0
  - 3c: Simultaneous inversion of K and h from sparse observations
"""

import os
import sys
import math
import yaml
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

from paths import PROJECT_ROOT, resolve_project_path

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import norm as torch_norm

# ============================================================
# 0. Configuration loading
# ============================================================

def load_config(config_path):
    """Load YAML configuration file"""
    with open(resolve_project_path(config_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ============================================================
# 1. Neural network definition
# ============================================================

class PINN(nn.Module):
    """
    Multi-layer perceptron (MLP) PINN
    Input: (x, y) coordinates
    Output: hydraulic head h(x, y)
    """
    def __init__(self, config, K_func=None):
        super(PINN, self).__init__()
        self.cfg = config['network']
        self.physics = config['physics']
        self.bc = config.get('boundary_conditions', {})
        self.K_func = K_func  # None=homogeneous K=1, callable=heterogeneous K(x,y)
        self.use_hard_dirichlet = self.cfg.get('hard_dirichlet', False)

        ff_cfg = self.cfg.get('fourier_features', {})
        self.use_fourier = ff_cfg.get('enabled', False)
        self.n_fourier = int(ff_cfg.get('num_frequencies', 0))
        if self.use_fourier and self.n_fourier > 0:
            max_freq = float(ff_cfg.get('max_frequency', self.n_fourier))
            if ff_cfg.get('spacing', 'linear') == 'log':
                freqs = torch.logspace(0, math.log10(max_freq), self.n_fourier)
            else:
                freqs = torch.linspace(1.0, max_freq, self.n_fourier)
            self.register_buffer('fourier_freqs', freqs.float())
            encoded_dim = self.cfg['input_dim'] + 2 * self.cfg['input_dim'] * self.n_fourier
        else:
            self.register_buffer('fourier_freqs', torch.empty(0))
            encoded_dim = self.cfg['input_dim']

        layers = []
        in_dim = encoded_dim
        for i in range(self.cfg['hidden_layers'] + 1):
            if i == 0:
                layers.append(nn.Linear(in_dim, self.cfg['hidden_width']))
            elif i == self.cfg['hidden_layers']:
                layers.append(nn.Linear(self.cfg['hidden_width'], self.cfg['output_dim']))
            else:
                layers.append(nn.Linear(self.cfg['hidden_width'], self.cfg['hidden_width']))
            if i < self.cfg['hidden_layers']:
                if self.cfg['activation'] == 'tanh':
                    layers.append(nn.Tanh())
                elif self.cfg['activation'] == 'relu':
                    layers.append(nn.ReLU())
                elif self.cfg['activation'] == 'gelu':
                    layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.cfg['init_method'] == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.cfg['init_method'] == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass"""
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

    def pde_residual(self, x, y):
        """
        Compute PDE residual
        - Homogeneous (stage 3a): ∇²h = 0
        - Heterogeneous (stage 3b): ∇·(K∇h) = 0
        Uses automatic differentiation for second-order derivatives
        """
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Combine into (x, y) input
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        h = self.forward(xy)

        # First-order derivatives
        dh_dx = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                                      create_graph=True, retain_graph=True)[0]
        dh_dy = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h),
                                      create_graph=True, retain_graph=True)[0]

        if self.K_func is None:
            # Homogeneous: K=1, ∇²h = 0
            d2h_dx2 = torch.autograd.grad(dh_dx, x, grad_outputs=torch.ones_like(dh_dx),
                                            create_graph=True, retain_graph=True)[0]
            d2h_dy2 = torch.autograd.grad(dh_dy, y, grad_outputs=torch.ones_like(dh_dy),
                                            create_graph=True, retain_graph=True)[0]
            residual = d2h_dx2 + d2h_dy2
        else:
            # Heterogeneous: ∇·(K∇h) = ∂(K·∂h/∂x)/∂x + ∂(K·∂h/∂y)/∂y
            # K(x,y) provided by K_func (differentiable)
            K = self.K_func(x, y)

            # Second-order mixed derivatives need product rule
            # ∂(K·∂h/∂x)/∂x = K·∂²h/∂x² + ∂K/∂x·∂h/∂x
            # Use autograd to compute full derivatives directly
            K_dh_dx = K * dh_dx
            K_dh_dy = K * dh_dy

            div_Kh_x = torch.autograd.grad(K_dh_dx, x,
                                            grad_outputs=torch.ones_like(K_dh_dx),
                                            create_graph=True, retain_graph=True)[0]
            div_Kh_y = torch.autograd.grad(K_dh_dy, y,
                                            grad_outputs=torch.ones_like(K_dh_dy),
                                            create_graph=True, retain_graph=True)[0]
            residual = div_Kh_x + div_Kh_y

        return residual

    def bc_dirichlet(self, x, y, h_bc):
        """
        Dirichlet boundary loss
        x, y: boundary point coordinates (tensor)
        h_bc: boundary head true values (tensor)
        """
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        h_pred = self.forward(xy)
        return (h_pred.squeeze() - h_bc) ** 2

    def bc_neumann(self, x, y):
        """
        Neumann no-flow boundary loss: dh/dn = 0
        x, y: boundary point coordinates (tensor) — points on y=0 or y=1
        Differentiate along y, penalize dh/dy ≠ 0
        """
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        xy.requires_grad_(True)
        h = self.forward(xy)
        # dh/dy: gradient w.r.t. y component
        grad_h = torch.autograd.grad(
            h, xy, grad_outputs=torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]
        dh_dy = grad_h[:, 1]  # y component
        return dh_dy ** 2      # Penalize (dh/dy)², drive dh/dy → 0

    def predict(self, x, y):
        """Batch prediction (for inference)"""
        xy = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        with torch.no_grad():
            return self.forward(xy).squeeze().numpy()


# ============================================================
# 2. Sampling functions
# ============================================================

def sample_pde_points(config, n_points):
    """Uniformly sample PDE residual points within the domain"""
    x_min = config['physics']['domain']['x_min']
    x_max = config['physics']['domain']['x_max']
    y_min = config['physics']['domain']['y_min']
    y_max = config['physics']['domain']['y_max']

    x = torch.rand(n_points) * (x_max - x_min) + x_min
    y = torch.rand(n_points) * (y_max - y_min) + y_min
    return x, y

def sample_boundary_points(config, n_points_per_side):
    """
    Sample boundary points: left/right Dirichlet, top/bottom Neumann.
    Supports fixed grid points (eliminates random sampling noise).
    """
    x_min = config['physics']['domain']['x_min']
    x_max = config['physics']['domain']['x_max']
    y_min = config['physics']['domain']['y_min']
    y_max = config['physics']['domain']['y_max']
    bc = config['boundary_conditions']
    use_fixed = config['training'].get('use_fixed_boundary_grid', False)

    n_each = n_points_per_side // 2  # Allocate to two Dirichlet edges

    if use_fixed:
        # Fixed equispaced grid points (eliminate random sampling noise)
        y_left = torch.linspace(y_min, y_max, n_each)
        y_right = torch.linspace(y_min, y_max, n_each)
        x_top = torch.linspace(x_min, x_max, n_each)
        x_bottom = torch.linspace(x_min, x_max, n_each)
    else:
        # Random sampling (original method)
        y_left = torch.rand(n_each) * (y_max - y_min) + y_min
        y_right = torch.rand(n_each) * (y_max - y_min) + y_min
        x_top = torch.rand(n_each) * (x_max - x_min) + x_min
        x_bottom = torch.rand(n_each) * (x_max - x_min) + x_min

    # Left edge x=0 (Dirichlet: h=1)
    x_left = torch.zeros(n_each)
    h_left = torch.full((n_each,), bc['h_left'])

    # Right edge x=1 (Dirichlet: h=0)
    x_right = torch.full((n_each,), x_max - x_min)
    h_right = torch.full((n_each,), bc['h_right'])

    # Top edge y=1 (Neumann: dh/dy=0)
    y_top = torch.full((n_each,), y_max - y_min)

    # Bottom edge y=0 (Neumann: dh/dy=0)
    y_bottom = torch.zeros(n_each)

    return {
        'dirichlet': {
            'x': torch.cat([x_left, x_right]),
            'y': torch.cat([y_left, y_right]),
            'h': torch.cat([h_left, h_right]),
        },
        'neumann': {
            'x': torch.cat([x_top, x_bottom]),
            'y': torch.cat([y_top, y_bottom]),
        }
    }

def analytical_solution(x, y, config):
    """
    Analytical solution for homogeneous Laplace equation on rectangular domain
    (Fourier series expansion, retaining first 20 terms)
    Uses the same boundary conditions as PINN
    """
    # Extract parameters
    Lx = config['physics']['domain']['x_max'] - config['physics']['domain']['x_min']
    Ly = config['physics']['domain']['y_max'] - config['physics']['domain']['y_min']
    bc = config['boundary_conditions']

    h0_left   = bc['h_left']
    h0_right  = bc['h_right']
    h0_bottom = bc['h_bottom']
    h0_top    = bc['h_top']

    # Mean correction for boundary values to avoid violating harmonic function BCs
    h_avg = (h0_left + h0_right + h0_bottom + h0_top) / 4.0

    # Fourier series summation (first 20 terms)
    h = torch.zeros_like(x)
    for n in range(1, 21):
        # Compute coefficients An and Bn (simplified approximation, using first term)
        # Exact analytical solution requires boundary condition decomposition; here we use approximate formulas
        term = (2.0 / (n * math.pi)) * (
            (h0_right - h0_left) / math.sinh(n * math.pi * Ly / Lx) * torch.sinh(n * math.pi * y / Lx) * torch.sin(n * math.pi * x / Lx)
            + (h0_top - h0_bottom) / math.sinh(n * math.pi * Lx / Ly) * torch.sinh(n * math.pi * x / Ly) * torch.sin(n * math.pi * y / Ly)
        )
        h = h + term

    # Add linear interpolation as basis function
    h = h + (1 - x / Lx) * h0_left + (x / Lx) * h0_right

    return h


def analytical_K(x, y, Lx=1.0, Ly=1.0, alpha=0.3):
    """
    Analytical heterogeneous conductivity field K(x,y) = 1 + α·sin(2πx/Lx)·cos(2πy/Ly)

    This is the "soft test" function for stage 3b — K is smooth, differentiable,
    suitable for quickly validating PINN's heterogeneous extension capability when no pre-generated K field is available.

    Parameters
    ----------
    x, y : torch.tensor
        Coordinates (arbitrary shape, N-dimensional)
    Lx, Ly : float
        Domain size
    alpha : float
        Heterogeneity strength (0=homogeneous, 0.3=weak, 1.0=strong)

    Returns
    -------
    K : torch.tensor
        K(x,y) values, same shape as x/y
    """
    pi = torch.tensor(math.pi, dtype=x.dtype, device=x.device)
    K = 1.0 + alpha * torch.sin(2.0 * pi * x / Lx) * torch.cos(2.0 * pi * y / Ly)
    return K


def load_K_field_from_npz(npz_path, device='cpu'):
    """
    Load pre-generated KLE heterogeneous conductivity field from npz file,
    and return a differentiable K(x,y) function (PyTorch lambda closure).

    Parameters
    ----------
    npz_path : str
        K_field.npz file path
    device : str
        PyTorch device

    Returns
    -------
    K_func : callable
        K_func(x, y) -> torch.tensor, can be used in pde_residual()
    """
    data = np.load(npz_path)
    K_grid = data['K']  # (nx, ny)
    x_grid = data['x']  # (nx,)
    y_grid = data['y']  # (ny,)

    nx, ny = K_grid.shape
    Lx = x_grid[-1] - x_grid[0]
    Ly = y_grid[-1] - y_grid[0]

    # Convert to torch tensor
    K_tensor = torch.tensor(K_grid, dtype=torch.float32, device=device)
    x_t = torch.tensor(x_grid, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_grid, dtype=torch.float32, device=device)

    def K_func(x, y):
        """
        Differentiable K(x,y) using bilinear interpolation.
        x, y: torch.tensor (N,)
        Returns: K(x,y) (N,)"""
        # Normalize to [0, nx-1], [0, ny-1]
        xi = (x / Lx) * (nx - 1)
        yi = (y / Ly) * (ny - 1)

        # Bilinear interpolation (differentiable soft version)
        x0 = torch.floor(xi).long()
        y0 = torch.floor(yi).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # Boundary clamping
        x0 = torch.clamp(x0, 0, nx - 2)
        x1 = torch.clamp(x1, 1, nx - 1)
        y0 = torch.clamp(y0, 0, ny - 2)
        y1 = torch.clamp(y1, 1, ny - 1)

        # Weights
        wx = (xi - x0.float()).clamp(0, 1)
        wy = (yi - y0.float()).clamp(0, 1)

        # Four corner values
        K00 = K_tensor[x0, y0]
        K10 = K_tensor[x1, y0]
        K01 = K_tensor[x0, y1]
        K11 = K_tensor[x1, y1]

        # Bilinear interpolation
        K = (K00 * (1 - wx) * (1 - wy) +
             K10 * wx * (1 - wy) +
             K01 * (1 - wx) * wy +
             K11 * wx * wy)
        return K

    return K_func

# ============================================================
# 3. Training function
# ============================================================

def train_pinn(config, device='cpu', stage='3a', resume=False, resume_from=None):
    """
    Main training workflow
    stage: '3a' homogeneous validation, '3b' heterogeneous extension (KLE), '3c' inverse problem
    """
    print("=" * 60, flush=True)
    print(f"Stage {stage}: PINN training framework", flush=True)
    print("=" * 60, flush=True)

    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ========== K-field setup (selected by stage) ==========
    K_func = None  # Homogeneous: K=1 (default)
    output_dir = config['logging']['output_dir']

    if stage == '3a':
        print("Mode: homogeneous field (K=1)")
        output_dir = output_dir.replace('stage3_heterogeneous', 'stage3_homogeneous')

    elif stage == '3b':
        print("Mode: heterogeneous field (KLE-generated)")
        # Load K field from npz (config paths are relative to project root)
        k_field_path = resolve_project_path(
            config.get('kle', {}).get('k_field_file', '')
        )
        if os.path.exists(k_field_path):
            K_func = load_K_field_from_npz(k_field_path, device=device)
            print(f"  K-field source: {k_field_path}")
        else:
            # Fallback: use analytical K(x,y)
            print(f"  Warning: K-field file not found, using analytical function analytical_K")
            domain = config['physics']['domain']
            Lx = domain['x_max'] - domain['x_min']
            Ly = domain['y_max'] - domain['y_min']

            def K_func(x, y):
                return analytical_K(x, y, Lx=Lx, Ly=Ly, alpha=0.3)
        output_dir = output_dir.replace('stage3_homogeneous', 'stage3_heterogeneous')

    elif stage == '3c':
        print("Mode: inverse problem (K field to be inverted)")
        output_dir = output_dir.replace('stage3_homogeneous', 'stage3_inverse')

    output_dir = str(resolve_project_path(output_dir))
    print(f"Output directory: {output_dir}")
    config['logging']['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create network (pass in K_func)
    model = PINN(config, K_func=K_func).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Network architecture: {config['network']['hidden_layers']} layers x {config['network']['hidden_width']} width")
    print(f"Total parameters: {total_params:,}")

    # Sampling points
    n_pde = config['training']['n_pde_points']
    n_bc = config['training']['n_bc_points']
    print(f"\nSampling points: PDE residual={n_pde}, boundary={n_bc}")

    # Optimizer
    optimizer_name = config['training']['optimizer']
    lr = config['training']['lr_adam']

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=config['training']['weights'].get('weight_decay', 1e-5))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=config['training']['weights'].get('weight_decay', 1e-5))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2000
    )

    # Loss logging (supports Dirichlet / Neumann separation)
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_pde': [],
        'loss_bc_dirichlet': [],
        'loss_bc_neumann': [],
        'loss_bc': [],
        'loss_data': [],
        'lr': [],
        'rmse_vs_fdm': [],   # FDM reference RMSE (mid-training evaluation)
        'rmse_vs_fdm_epochs': [],  # Epoch for each FDM evaluation
    }

    # Weights (supports Dirichlet / Neumann separation)
    w_pde = config['training']['weights']['pde']
    w_bc_dir = config['training']['weights'].get('bc_dirichlet', 1.0)
    w_bc_neu = config['training']['weights'].get('bc_neumann', 1.0)
    w_bc = config['training']['weights'].get('bc', w_bc_dir + w_bc_neu)  # Backward-compatible with old configs
    w_data = config['training']['weights']['data']

    # FDM reference (for mid-training evaluation)
    fdm_ref = config.get('fdm_reference', {})
    use_fdm_eval = fdm_ref.get('enabled', False)
    eval_every = fdm_ref.get('eval_every', 10000)
    fdm_h = None
    fdm_path = resolve_project_path(fdm_ref.get('path', '')) if fdm_ref.get('path') else None
    if use_fdm_eval and fdm_path and os.path.exists(fdm_path):
        fdm_data = np.load(fdm_path, allow_pickle=True)
        fdm_h = fdm_data['h']   # shape (N, N) Fortran order
        fdm_x = fdm_data['x']
        fdm_y = fdm_data['y']
        print(f"  FDM reference loaded: shape={fdm_h.shape}, range=[{fdm_h.min():.4f}, {fdm_h.max():.4f}]")

    # Training loop
    n_iterations = config['training']['n_iterations']
    print_every = config['logging']['print_every']
    plot_every = config['logging']['plot_every']

    # Resume logic
    start_epoch = 0
    if resume:
        history_path = os.path.join(output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                existing = json.load(f)
            if existing.get('epoch'):
                start_epoch = existing['epoch'][-1] + 1
                history = existing
                print(f"\nResume: from epoch {start_epoch} (already have {len(existing['epoch'])} record points)")
                # Load best model
                model.load_state_dict(torch.load(
                    os.path.join(output_dir, 'best_model.pt'), map_location=device))
                best_loss = min(existing['loss_total'])
                print(f"  Loaded best_model.pt, historical best loss={best_loss:.2e}")
        else:
            print("\nWarning: history file not found, starting training from scratch")

    if start_epoch >= n_iterations:
        print(f"\nTraining already completed ({n_iterations} epochs), skipping.")
        print(f"Available model: {output_dir}/best_model.pt")
    else:
        print(f"\nStarting training ({n_iterations} iterations, from epoch {start_epoch})...")
        print(f"{'Epoch':>8} | {'LossTotal':>12} | {'PDE':>10} | {'BC_D':>10} {'BC_N':>10} | {'LR':>12}")
        print("-" * 75)

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(start_epoch, n_iterations):
        model.train()

        # Sampling
        x_pde, y_pde = sample_pde_points(config, n_pde)
        bc_points = sample_boundary_points(config, n_bc)

        # Transfer to device
        x_pde = x_pde.to(device)
        y_pde = y_pde.to(device)

        # Dirichlet BC points
        x_bc_d = bc_points['dirichlet']['x'].to(device)
        y_bc_d = bc_points['dirichlet']['y'].to(device)
        h_bc_d = bc_points['dirichlet']['h'].to(device)

        # Neumann BC points
        x_bc_n = bc_points['neumann']['x'].to(device)
        y_bc_n = bc_points['neumann']['y'].to(device)

        # PDE residual
        residual_pde = model.pde_residual(x_pde, y_pde)
        loss_pde = torch.mean(residual_pde ** 2)

        # Boundary condition loss (separated Dirichlet / Neumann)
        loss_bc_d = torch.mean(model.bc_dirichlet(x_bc_d, y_bc_d, h_bc_d))
        loss_bc_n = torch.mean(model.bc_neumann(x_bc_n, y_bc_n))
        loss_bc = w_bc_dir * loss_bc_d + w_bc_neu * loss_bc_n

        # Data loss (zero in synthetic stages)
        loss_data = torch.tensor(0.0, device=device)

        # Total loss
        loss = w_pde * loss_pde + loss_bc + w_data * loss_data

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (after backward(), clip current step gradients)
        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config['training']['gradient_clip'])

        optimizer.step()

        scheduler.step(loss.item())

        # Logging
        if epoch % print_every == 0 or epoch == n_iterations - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{epoch:>8} | {loss.item():>12.2e} | {loss_pde.item():>10.2e} "
                  f"| D={loss_bc_d.item():.2e} N={loss_bc_n.item():.2e} | {current_lr:>12.2e}")

            history['epoch'].append(epoch)
            history['loss_total'].append(loss.item())
            history['loss_pde'].append(loss_pde.item())
            history['loss_bc_dirichlet'].append(loss_bc_d.item())
            history['loss_bc_neumann'].append(loss_bc_n.item())
            history['loss_bc'].append(loss.item() - w_pde * loss_pde.item())
            history['loss_data'].append(loss_data.item())
            history['lr'].append(current_lr)

            # FDM mid-training evaluation
            if use_fdm_eval and fdm_h is not None and epoch % eval_every == 0:
                rmse_fdm = _evaluate_vs_fdm(model, fdm_h, fdm_x, fdm_y, device)
                history['rmse_vs_fdm'].append(rmse_fdm)
                history['rmse_vs_fdm_epochs'].append(epoch)
                print(f"           >> FDM RMSE @ epoch {epoch}: {rmse_fdm:.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                # Save best model
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'best_model.pt'))

        # Mid-training plotting
        if (epoch + 1) % plot_every == 0:
            _plot_prediction(model, config, epoch + 1, device, output_dir)

    print("-" * 60)
    print(f"Adam training complete! Best loss: {best_loss:.2e} @ epoch {best_epoch}")
    print(f"Model saved to: {output_dir}/best_model.pt")

    # ============================================================
    # 3b. L-BFGS fine-tuning (optional)
    # ============================================================
    n_lbfgs = config['training'].get('n_lbfgs', 0)
    if n_lbfgs > 0:
        print(f"\nStarting L-BFGS fine-tuning ({n_lbfgs} steps)...")

        # Load best model
        model.load_state_dict(torch.load(
            os.path.join(output_dir, 'best_model.pt'), map_location=device))

        lbfgs_history = {
            'loss_pde': [], 'loss_bc_dirichlet': [], 'loss_bc_neumann': [], 'loss_total': []
        }

        # Build L-BFGS optimizer
        # Note: PyTorch uses LBFGS (all caps), not Lbfgs
        lbfgs_optimizer = optim.LBFGS(
            model.parameters(),
            lr=config['training'].get('lr_lbfgs', 1.0),
            max_iter=n_lbfgs,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        # L-BFGS pre-sample fixed collocation points (avoid re-sampling inside closure)
        print("  L-BFGS: pre-sampling fixed collocation points...")
        lbfgs_pde_x, lbfgs_pde_y = sample_pde_points(config, n_pde)
        lbfgs_bc_pts = sample_boundary_points(config, n_bc)
        lbfgs_pde_x = lbfgs_pde_x.to(device)
        lbfgs_pde_y = lbfgs_pde_y.to(device)
        lbfgs_bc_d_x = lbfgs_bc_pts['dirichlet']['x'].to(device)
        lbfgs_bc_d_y = lbfgs_bc_pts['dirichlet']['y'].to(device)
        lbfgs_bc_d_h = lbfgs_bc_pts['dirichlet']['h'].to(device)
        lbfgs_bc_n_x = lbfgs_bc_pts['neumann']['x'].to(device)
        lbfgs_bc_n_y = lbfgs_bc_pts['neumann']['y'].to(device)

        def lbfgs_closure():
            lbfgs_optimizer.zero_grad()
            res_pde = model.pde_residual(lbfgs_pde_x, lbfgs_pde_y)
            loss_pde = torch.mean(res_pde ** 2)
            loss_bc_d = torch.mean(model.bc_dirichlet(lbfgs_bc_d_x, lbfgs_bc_d_y, lbfgs_bc_d_h))
            loss_bc_n = torch.mean(model.bc_neumann(lbfgs_bc_n_x, lbfgs_bc_n_y))
            loss_bc = w_bc_dir * loss_bc_d + w_bc_neu * loss_bc_n
            loss_data = torch.tensor(0.0, device=device)
            loss = w_pde * loss_pde + loss_bc + w_data * loss_data
            loss.backward()
            lbfgs_history['loss_pde'].append(loss_pde.item())
            lbfgs_history['loss_bc_dirichlet'].append(loss_bc_d.item())
            lbfgs_history['loss_bc_neumann'].append(loss_bc_n.item())
            lbfgs_history['loss_total'].append(loss.item())
            return loss

        lbfgs_optimizer.step(lbfgs_closure)
        print(f"  L-BFGS complete! Final loss: {lbfgs_history['loss_total'][-1]:.2e}")
        print(f"  PDE: {lbfgs_history['loss_pde'][-1]:.2e}, "
              f"BC_D: {lbfgs_history['loss_bc_dirichlet'][-1]:.2e}, "
              f"BC_N: {lbfgs_history['loss_bc_neumann'][-1]:.2e}")

        # Save L-BFGS optimized model
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_lbfgs.pt'))
        # Final evaluation uses L-BFGS model
        _plot_prediction(model, config, f"lbfgs_final", device, output_dir, final=True)

        # Record to history
        history['lbfgs_final'] = lbfgs_history

    # ============================================================
    # 4. Final evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)

    # Load best model (prefer L-BFGS result)
    best_model_path = os.path.join(output_dir,
        'best_model_lbfgs.pt' if n_lbfgs > 0 else 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _plot_training_curves(history, output_dir)
    # Heterogeneous stages (3b/3c) no longer output h=1-x evaluation_results.json
    # That baseline has no physical meaning under heterogeneous K (RMSE vs h=1-x = 0.056)
    # Use the FDM evaluation below as the official metric
    if stage in ('3b', '3c'):
        print("  Heterogeneous stage: skipping h=1-x analytical evaluation (using FDM reference instead)")
    else:
        _evaluate_against_analytical(model, config, device, output_dir)

    # Final FDM evaluation
    if use_fdm_eval and fdm_h is not None:
        print("\nFinal FDM evaluation:")
        final_rmse = _evaluate_vs_fdm(model, fdm_h, fdm_x, fdm_y, device)
        print(f"  Overall RMSE vs FDM: {final_rmse:.6f}")
        history['final_rmse_vs_fdm'] = final_rmse

    # Save history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nAll results saved to: {output_dir}/")
    return model, history


def _evaluate_vs_fdm(model, h_fdm, x_fdm, y_fdm, device):
    """
    Evaluate PINN model on FDM grid and compute RMSE.
    h_fdm: FDM solution, shape (N, N), Fortran order (converted to C-order inside)
    """
    model.eval()
    N = len(x_fdm)
    X_m, Y_m = np.meshgrid(x_fdm, y_fdm, indexing='ij')
    xy = np.stack([X_m.ravel(), Y_m.ravel()], axis=1)
    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    with torch.no_grad():
        h_pinn = model(xy_t).cpu().numpy().ravel().reshape(N, N)
    # h_fdm is Fortran order; convert to C-order for element-wise alignment
    # Note: h_pinn uses C-order reshape to match xy's C-order ravel
    h_fdm_c = np.ascontiguousarray(h_fdm)
    rmse = np.sqrt(np.mean((h_pinn - h_fdm_c) ** 2))
    return float(rmse)


def _plot_prediction(model, config, epoch, device, output_dir, final=False):
    """Plot current predicted head field"""
    model.eval()

    x_min = config['physics']['domain']['x_min']
    x_max = config['physics']['domain']['x_max']
    y_min = config['physics']['domain']['y_min']
    y_max = config['physics']['domain']['y_max']

    nx, ny = 100, 100
    x_grid = torch.linspace(x_min, x_max, nx)
    y_grid = torch.linspace(y_min, y_max, ny)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')

    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)

    H_pred = model.predict(x_flat, y_flat).reshape(nx, ny)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted head field
    im = axes[0].contourf(X.numpy(), Y.numpy(), H_pred, levels=30, cmap='coolwarm')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title(f'PINN Prediction $h(x,y)$ @ epoch {epoch}')
    plt.colorbar(im, ax=axes[0], label='h (m)')

    # Boundary condition labels
    bc = config['boundary_conditions']
    axes[0].axvline(x_min, color='k', linewidth=2)
    axes[0].axvline(x_max, color='k', linewidth=2)
    axes[0].axhline(y_min, color='k', linewidth=2)
    axes[0].axhline(y_max, color='k', linewidth=2)
    axes[0].text(0.02, 0.5, f'h={bc["h_left"]}', transform=axes[0].transAxes,
                 fontsize=10, fontweight='bold')
    axes[0].text(0.95, 0.5, f'h={bc["h_right"]}', transform=axes[0].transAxes,
                 fontsize=10, fontweight='bold', ha='right')

    # PDE residual distribution (sample points)
    model.train()
    n_eval = 500
    x_eval = torch.rand(n_eval) * (x_max - x_min) + x_min
    y_eval = torch.rand(n_eval) * (y_max - y_min) + y_min
    x_eval = x_eval.to(device).requires_grad_(True)
    y_eval = y_eval.to(device).requires_grad_(True)

    res = model.pde_residual(x_eval, y_eval).detach().cpu().numpy()

    axes[1].scatter(x_eval.detach().cpu().numpy(), y_eval.detach().cpu().numpy(),
                     c=res, cmap='RdBu', s=10, alpha=0.6)
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    axes[1].set_title(f'PDE Residual |residual| (mean={np.abs(res).mean():.2e})')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='|residual|')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    suffix = 'final' if final else f'epoch_{epoch}'
    plt.savefig(os.path.join(output_dir, f'prediction_{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] prediction_{suffix}.png saved")

def _plot_training_curves(history, output_dir):
    """Plot training curves (supports Dirichlet / Neumann separated loss)"""
    epochs = history['epoch']
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].semilogy(epochs, history['loss_total'], label='Total', linewidth=2)
    axes[0].semilogy(epochs, history['loss_pde'], label='PDE', linewidth=1.5, alpha=0.8)
    if 'loss_bc_dirichlet' in history:
        axes[0].semilogy(epochs, history['loss_bc_dirichlet'],
                          label='BC_Dirichlet', linewidth=1.5, alpha=0.8)
        axes[0].semilogy(epochs, history['loss_bc_neumann'],
                          label='BC_Neumann', linewidth=1.5, alpha=0.8)
    else:
        axes[0].semilogy(epochs, history['loss_bc'], label='BC', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves (log scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate curve
    axes[1].semilogy(epochs, history['lr'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)

    # FDM RMSE curve (mid-training evaluation)
    if history.get('rmse_vs_fdm') and history.get('rmse_vs_fdm_epochs'):
        eval_x = history['rmse_vs_fdm_epochs']
        axes[2].plot(eval_x, history['rmse_vs_fdm'], 'o-', color='purple', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RMSE vs FDM')
        axes[2].set_title('PINN vs FDM RMSE (during training)')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.05, color='red', ls='--', label='Target 0.05')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'FDM RMSE\nnot available', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('PINN vs FDM RMSE')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] training_curves.png saved")

def _evaluate_against_analytical(model, config, device, output_dir):
    """Evaluate against analytical solution"""
    model.eval()

    x_min = config['physics']['domain']['x_min']
    x_max = config['physics']['domain']['x_max']
    y_min = config['physics']['domain']['y_min']
    y_max = config['physics']['domain']['y_max']

    n_test = config['validation']['n_test_points']
    x_test = torch.rand(n_test) * (x_max - x_min) + x_min
    y_test = torch.rand(n_test) * (y_max - y_min) + y_min

    # PINN prediction
    h_pred = model.predict(x_test, y_test)

    # Analytical solution: left/right Dirichlet + top/bottom Neumann no-flow homogeneous Laplace solution
    # For K=1 homogeneous field: h(x,y) = 1 - x (1D flow, independent of y, automatically satisfies dh/dy=0)
    h_true = (1 - x_test) * config['boundary_conditions']['h_left'] \
             + x_test * config['boundary_conditions']['h_right']

    # Error metrics
    rmse = np.sqrt(np.mean((h_pred - h_true.numpy()) ** 2))
    mae = np.mean(np.abs(h_pred - h_true.numpy()))
    max_err = np.max(np.abs(h_pred - h_true.numpy()))

    print(f"\nEvaluation results (vs analytical h=1-x):")
    print(f"  RMSE : {rmse:.4e}")
    print(f"  MAE  : {mae:.4e}")
    print(f"  MaxE : {max_err:.4e}")

    # Save results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_err),
        'n_test_points': n_test
    }
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Scatter plot: prediction vs truth
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(h_true.numpy(), h_pred, alpha=0.3, s=5)
    lims = [min(h_true.min(), h_pred.min()), max(h_true.max(), h_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='y=x')
    axes[0].set_xlabel('Analytical $h$')
    axes[0].set_ylabel('PINN Prediction $h$')
    axes[0].set_title(f'PINN vs Analytical (RMSE={rmse:.3e})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution histogram
    errors = h_pred - h_true.numpy()
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution (MAE={mae:.3e})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] evaluation_scatter.png saved")


# ============================================================
# 4. Main entry point
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Stage 3: PINN training framework (3a homogeneous / 3b heterogeneous / 3c inverse)')
    parser.add_argument('--config', type=str,
                        default='configs/stage3a_3b_baseline.yaml',
                        help='Config file path')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cpu / cuda / auto')
    parser.add_argument('--stage', type=str, default='3a',
                        choices=['3a', '3b', '3c'],
                        help='Training stage: 3a=homogeneous validation, 3b=heterogeneous extension, 3c=inverse problem')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best_model.pt in output_dir (skip completed epochs)')
    parser.add_argument('--resume-from', type=int, default=None,
                        help='Resume from specified epoch number (requires corresponding checkpoint)')
    args = parser.parse_args()

    config = load_config(args.config)
    train_pinn(config, device=args.device, stage=args.stage,
               resume=args.resume, resume_from=args.resume_from)
