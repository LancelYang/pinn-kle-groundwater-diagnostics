#!/usr/bin/env python3
"""Diagnose whether fixed h can identify KLE xi.

This isolates the inverse problem from continuous PINN autodiff residuals. The
script keeps a head field fixed and optimizes only KLE coefficients xi against
a discrete div(K grad h)=0 residual on the KLE grid. The fixed h can come from
the FDM truth or from a trained NN sampled onto the KLE grid.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_stage3c_staged import build_model
from paths import resolve_project_path


def load_yaml(path):
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interp_numpy(values, x_grid, y_grid, xq, yq):
    nx = len(x_grid)
    ny = len(y_grid)
    x_norm = (xq[:, None] - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (nx - 1)
    y_norm = (yq[None, :] - y_grid[0]) / (y_grid[-1] - y_grid[0]) * (ny - 1)
    x0 = np.floor(x_norm).astype(int).clip(0, nx - 2)
    y0 = np.floor(y_norm).astype(int).clip(0, ny - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = np.clip(x_norm - x0, 0.0, 1.0)
    wy = np.clip(y_norm - y0, 0.0, 1.0)
    return (
        values[x0, y0] * (1 - wx) * (1 - wy)
        + values[x1, y0] * wx * (1 - wy)
        + values[x0, y1] * (1 - wx) * wy
        + values[x1, y1] * wx * wy
    )


def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-12)


def pde_residual_fixed_h(h, K, dx, dy):
    Kr = harmonic_mean(K[1:-1, 1:-1], K[2:, 1:-1])
    Kl = harmonic_mean(K[:-2, 1:-1], K[1:-1, 1:-1])
    Ku = harmonic_mean(K[1:-1, 1:-1], K[1:-1, 2:])
    Kd = harmonic_mean(K[1:-1, :-2], K[1:-1, 1:-1])

    return (
        (Kr * (h[2:, 1:-1] - h[1:-1, 1:-1])
         - Kl * (h[1:-1, 1:-1] - h[:-2, 1:-1])) / (dx * dx)
        + (Ku * (h[1:-1, 2:] - h[1:-1, 1:-1])
           - Kd * (h[1:-1, 1:-1] - h[1:-1, :-2])) / (dy * dy)
    )


def cosine(a, b):
    an = torch.linalg.norm(a)
    bn = torch.linalg.norm(b)
    if an.item() == 0.0 or bn.item() == 0.0:
        return None
    return float(torch.dot(a, b).item() / (an.item() * bn.item()))


def sample_nn_h_on_grid(config, model_path, x_k, y_k, device):
    model = build_model(config, model_path, device)
    x_mesh, y_mesh = np.meshgrid(x_k, y_k, indexing="ij")
    x_t = torch.tensor(x_mesh.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_mesh.ravel(), dtype=torch.float32, device=device)
    with torch.no_grad():
        h_pred = model.predict(x_t, y_t)
        if isinstance(h_pred, torch.Tensor):
            h = h_pred.detach().cpu().numpy()
        else:
            h = np.asarray(h_pred)
        h = h.reshape(len(x_k), len(y_k))
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3c_fixed_xi_true_h_only.yaml")
    parser.add_argument("--output-dir", default="outputs/stage3c_xi_only_fdm_h")
    parser.add_argument("--h-source", choices=["fdm", "nn"], default="fdm")
    parser.add_argument("--model", default=None,
                        help="Trained Stage 3c model path when --h-source=nn")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--xi-init", choices=["zero", "random"], default="zero")
    parser.add_argument("--xi-init-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_data = np.load(resolve_project_path(cfg["kle"]["k_field_file"]), allow_pickle=True)
    fdm = np.load(resolve_project_path(cfg["inverse"]["fdm_reference_path"]), allow_pickle=True)

    x_k = k_data["x"]
    y_k = k_data["y"]
    h_fdm_on_k = interp_numpy(np.ascontiguousarray(fdm["h"]), fdm["x"], fdm["y"], x_k, y_k)
    if args.h_source == "fdm":
        h_on_k = h_fdm_on_k
    else:
        if not args.model:
            raise ValueError("--model is required when --h-source=nn")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_on_k = sample_nn_h_on_grid(cfg, args.model, x_k, y_k, device)

    h_rmse_vs_fdm_on_k = float(np.sqrt(np.mean((h_on_k - h_fdm_on_k) ** 2)))

    eig = torch.tensor(k_data["eigenvalues"][:cfg["inverse"]["n_modes"]], dtype=torch.float32)
    phi = torch.tensor(k_data["eigenfunctions"][:, :cfg["inverse"]["n_modes"]], dtype=torch.float32)
    xi_true = torch.tensor(k_data["xi"][:cfg["inverse"]["n_modes"]], dtype=torch.float32)
    h = torch.tensor(h_on_k, dtype=torch.float32)
    dx = float(x_k[1] - x_k[0])
    dy = float(y_k[1] - y_k[0])

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    if args.xi_init == "random":
        xi = torch.randn(len(xi_true), generator=gen) * args.xi_init_std
    else:
        xi = torch.zeros(len(xi_true))
    xi = torch.nn.Parameter(xi)

    opt = torch.optim.Adam([xi], lr=args.lr)
    history = {
        "epoch": [],
        "loss": [],
        "xi_corr": [],
        "distance_to_true": [],
        "cos_neg_grad_to_target": [],
    }

    for epoch in range(args.iterations):
        logK = (phi * torch.sqrt(eig)[None, :] * xi[None, :]).sum(dim=1).reshape(h.shape)
        K = torch.exp(torch.clamp(logK, -3.0, 3.0))
        residual = pde_residual_fixed_h(h, K, dx, dy)
        loss = torch.mean(residual ** 2)

        opt.zero_grad()
        raw_grad = torch.autograd.grad(loss, xi, retain_graph=True)[0].detach()
        target = xi_true - xi.detach()
        cos_neg = cosine(-raw_grad, target)
        loss.backward()
        opt.step()

        with torch.no_grad():
            xi_np = xi.detach().numpy()
            xi_true_np = xi_true.numpy()
            corr = float(np.corrcoef(xi_np, xi_true_np)[0, 1]) if len(xi_np) > 1 else float("nan")
            dist = float(torch.linalg.norm(xi_true - xi.detach()).item())

        if epoch % max(1, args.iterations // 20) == 0 or epoch == args.iterations - 1:
            print(
                f"epoch {epoch:5d} | loss {loss.item():.4e} | "
                f"xi_corr {corr:.4f} | dist {dist:.4f} | cos(-g,target) {cos_neg}"
            )
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["xi_corr"].append(corr)
        history["distance_to_true"].append(dist)
        history["cos_neg_grad_to_target"].append(cos_neg)

    with torch.no_grad():
        final_logK = (phi * torch.sqrt(eig)[None, :] * xi[None, :]).sum(dim=1).reshape(h.shape)
        true_logK = k_data["logK"]
        xi_recovered = xi.detach().numpy()
        final_logK_np = final_logK.numpy()
        result = {
            "config": str(resolve_project_path(args.config)),
            "h_source": args.h_source,
            "model": str(resolve_project_path(args.model)) if args.model else None,
            "h_rmse_vs_fdm_on_k": h_rmse_vs_fdm_on_k,
            "iterations": args.iterations,
            "lr": args.lr,
            "xi_init": args.xi_init,
            "final_loss": history["loss"][-1],
            "final_xi_corr": history["xi_corr"][-1],
            "final_distance_to_true": history["distance_to_true"][-1],
            "final_logK_rmse": float(np.sqrt(np.mean((final_logK_np - true_logK) ** 2))),
            "xi_true": xi_true.numpy().tolist(),
            "xi_pred": xi_recovered.tolist(),
            "history": history,
        }

    np.savez(
        out_dir / "recovered_xi_logK.npz",
        xi=xi_recovered,
        xi_true=xi_true.numpy(),
        logK=final_logK_np,
        logK_true=true_logK,
        h_on_k=h_on_k,
        x=x_k,
        y=y_k,
    )

    if args.h_source == "nn" and args.model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, args.model, device)
        with torch.no_grad():
            model.xi.copy_(torch.tensor(xi_recovered, dtype=torch.float32, device=device))
        torch.save(
            model.state_dict(),
            out_dir / "model_with_discrete_xi.pt",
            _use_new_zipfile_serialization=False,
        )

    result_name = "xi_only_fdm_h_results.json" if args.h_source == "fdm" else "xi_only_nn_h_discrete_results.json"
    with open(out_dir / result_name, "w") as f:
        json.dump(result, f, indent=2)

    print(f"saved: {out_dir / result_name}")


if __name__ == "__main__":
    main()
