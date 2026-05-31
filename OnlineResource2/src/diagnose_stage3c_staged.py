#!/usr/bin/env python3
"""Stage 3c staged diagnostic.

Load a trained h network, evaluate its residual quality, then freeze h and
optimize only KLE xi. This tests whether the neural h derivatives are good
enough to recover xi, after the ideal FDM-h xi-only diagnostic has passed.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from paths import resolve_project_path
from train_stage3c import KLEInvertiblePINN, _load_xi_true


def load_config(path):
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_kle_basis(config):
    k_cfg = config["kle"]
    k_field_path = resolve_project_path(k_cfg["k_field_file"])
    data = np.load(k_field_path, allow_pickle=True)
    m_learn = config["inverse"]["n_modes"]
    x_grid = data.get("x", np.linspace(0, k_cfg["Lx"], k_cfg["nx"]))
    y_grid = data.get("y", np.linspace(0, k_cfg["Ly"], k_cfg["ny"]))
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
    return {
        "sqrt_lambda": np.sqrt(data["eigenvalues"][:m_learn]),
        "phi_grid": data["eigenfunctions"][:, :m_learn].T.copy(),
        "x_grid": x_mesh.ravel().astype(np.float32),
        "y_grid": y_mesh.ravel().astype(np.float32),
        "nx": k_cfg["nx"],
        "ny": k_cfg["ny"],
        "Lx": k_cfg["Lx"],
        "Ly": k_cfg["Ly"],
        "M": m_learn,
    }


def build_model(config, model_path, device):
    model = KLEInvertiblePINN(config, prepare_kle_basis(config)).to(device)
    state = torch.load(resolve_project_path(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def cosine(a, b):
    an = torch.linalg.norm(a)
    bn = torch.linalg.norm(b)
    if an.item() == 0.0 or bn.item() == 0.0:
        return None
    return float(torch.dot(a, b).item() / (an.item() * bn.item()))


def evaluate_h_quality(model, config, out_dir, device, grid_n=101):
    fdm = np.load(resolve_project_path(config["inverse"]["fdm_reference_path"]), allow_pickle=True)
    h_fdm = np.ascontiguousarray(fdm["h"])
    x_fdm = fdm["x"]
    y_fdm = fdm["y"]
    x_eval = np.linspace(0.0, 1.0, grid_n)
    y_eval = np.linspace(0.0, 1.0, grid_n)
    x_mesh, y_mesh = np.meshgrid(x_eval, y_eval, indexing="ij")

    # Bilinear sample FDM h onto evaluation grid.
    xi_idx = x_eval[:, None] * (len(x_fdm) - 1)
    yi_idx = y_eval[None, :] * (len(y_fdm) - 1)
    x0 = np.floor(xi_idx).astype(int).clip(0, len(x_fdm) - 2)
    y0 = np.floor(yi_idx).astype(int).clip(0, len(y_fdm) - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xi_idx - x0
    wy = yi_idx - y0
    h_ref = (
        h_fdm[x0, y0] * (1 - wx) * (1 - wy)
        + h_fdm[x1, y0] * wx * (1 - wy)
        + h_fdm[x0, y1] * (1 - wx) * wy
        + h_fdm[x1, y1] * wx * wy
    )

    x_t = torch.tensor(x_mesh.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_mesh.ravel(), dtype=torch.float32, device=device)
    with torch.no_grad():
        h_pred = model.predict(x_t, y_t).reshape(grid_n, grid_n)
    h_err = h_pred - h_ref

    with torch.enable_grad():
        res = model.pde_residual(x_t, y_t).detach().cpu().numpy().reshape(grid_n, grid_n)

    # Neumann boundary derivative diagnostics.
    x_b = torch.linspace(0.0, 1.0, grid_n, device=device)
    y_bottom = torch.zeros(grid_n, device=device)
    y_top = torch.ones(grid_n, device=device)
    with torch.enable_grad():
        bottom_loss = model.bc_neumann(x_b, y_bottom).detach().cpu().numpy()
        top_loss = model.bc_neumann(x_b, y_top).detach().cpu().numpy()
    neumann_abs = np.sqrt(np.concatenate([bottom_loss, top_loss]))

    abs_res = np.abs(res[1:-1, 1:-1])
    metrics = {
        "h_rmse": float(np.sqrt(np.mean(h_err ** 2))),
        "h_mae": float(np.mean(np.abs(h_err))),
        "h_maxe": float(np.max(np.abs(h_err))),
        "h_r2": float(1.0 - np.sum(h_err ** 2) / np.sum((h_ref - h_ref.mean()) ** 2)),
        "pde_residual_mean_abs": float(np.mean(abs_res)),
        "pde_residual_p95_abs": float(np.percentile(abs_res, 95)),
        "pde_residual_max_abs": float(np.max(abs_res)),
        "neumann_dhdy_mean_abs": float(np.mean(neumann_abs)),
        "neumann_dhdy_p95_abs": float(np.percentile(neumann_abs, 95)),
        "neumann_dhdy_max_abs": float(np.max(neumann_abs)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    im0 = axes[0].contourf(x_mesh, y_mesh, h_err, levels=30, cmap="coolwarm")
    axes[0].set_title("h error")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(x_mesh, y_mesh, np.log10(np.abs(res) + 1e-12), levels=30, cmap="magma")
    axes[1].set_title("log10 |PDE residual|")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])
    mid = grid_n // 2
    axes[2].plot(x_eval, h_ref[:, mid], label="FDM")
    axes[2].plot(x_eval, h_pred[:, mid], "--", label="NN h")
    axes[2].set_title("centerline h")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "h_quality_residual_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "h_quality_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def optimize_xi_with_frozen_h(model, config, out_dir, device, iterations, lr, n_pde_points, xi_init):
    xi_true = _load_xi_true(config, model, device)
    if xi_true is None:
        raise ValueError("k_field_file must contain xi for staged diagnostic")

    for p in model.net.parameters():
        p.requires_grad_(False)
    model.xi.requires_grad_(True)
    with torch.no_grad():
        if xi_init == "zero":
            model.xi.zero_()
        elif xi_init == "random":
            model.xi.normal_(0.0, 0.1)
        else:
            raise ValueError(f"unknown xi_init: {xi_init}")

    opt = torch.optim.Adam([model.xi], lr=lr)
    history = {"epoch": [], "loss": [], "xi_corr": [], "distance": [], "cos_neg_grad_to_target": []}

    for epoch in range(iterations):
        x = torch.rand(n_pde_points, device=device)
        y = torch.rand(n_pde_points, device=device)
        loss = torch.mean(model.pde_residual(x, y) ** 2)

        opt.zero_grad()
        raw_grad = torch.autograd.grad(loss, model.xi, retain_graph=True)[0].detach()
        target = xi_true - model.xi.detach()
        cos_neg = cosine(-raw_grad, target)
        loss.backward()
        opt.step()

        with torch.no_grad():
            xi_np = model.xi.detach().cpu().numpy()
            true_np = xi_true.detach().cpu().numpy()
            corr = float(np.corrcoef(xi_np, true_np)[0, 1])
            dist = float(torch.linalg.norm(xi_true - model.xi.detach()).item())
        if epoch % max(1, iterations // 10) == 0 or epoch == iterations - 1:
            print(f"[xi-only NN h] epoch {epoch:5d} | loss {loss.item():.4e} | corr {corr:.4f} | dist {dist:.4f} | cos {cos_neg}")
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["xi_corr"].append(corr)
        history["distance"].append(dist)
        history["cos_neg_grad_to_target"].append(cos_neg)

    data = np.load(resolve_project_path(config["kle"]["k_field_file"]), allow_pickle=True)
    eig = data["eigenvalues"][:len(model.xi)]
    phi = data["eigenfunctions"][:, :len(model.xi)]
    xi_pred = model.xi.detach().cpu().numpy()
    logK_pred = (phi * np.sqrt(eig)[None, :] * xi_pred[None, :]).sum(axis=1).reshape(data["logK"].shape)

    result = {
        "iterations": iterations,
        "lr": lr,
        "n_pde_points": n_pde_points,
        "xi_init": xi_init,
        "final_xi_corr": history["xi_corr"][-1],
        "final_distance_to_true": history["distance"][-1],
        "final_logK_rmse": float(np.sqrt(np.mean((logK_pred - data["logK"]) ** 2))),
        "xi_true": xi_true.detach().cpu().numpy().tolist(),
        "xi_pred": xi_pred.tolist(),
        "history": history,
    }
    with open(out_dir / "freeze_h_optimize_xi_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3c_fixed_xi_true_h_only.yaml")
    parser.add_argument("--model", default="outputs/stage3c_fixed_xi_true_h_only/model_final.pt")
    parser.add_argument("--output-dir", default="outputs/stage3c_staged_freeze_h")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-pde-points", type=int, default=2000)
    parser.add_argument("--xi-init", choices=["zero", "random"], default="zero")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model = build_model(config, args.model, device)

    quality = evaluate_h_quality(model, config, out_dir, device)
    print("h quality:", quality)
    result = optimize_xi_with_frozen_h(
        model, config, out_dir, device,
        iterations=args.iterations,
        lr=args.lr,
        n_pde_points=args.n_pde_points,
        xi_init=args.xi_init,
    )
    print("xi result:", {k: result[k] for k in ["final_xi_corr", "final_distance_to_true", "final_logK_rmse"]})


if __name__ == "__main__":
    main()
