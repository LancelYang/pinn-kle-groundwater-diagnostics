#!/usr/bin/env python3
"""Closed-loop validation for Stage 3c hybrid inversion.

Take xi/logK recovered by the discrete residual stage, reconstruct K, solve the
forward FDM problem, and compare the resulting h field with the true FDM head.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdm_solver import bilinear_interpolate, compute_pde_residual_fdm, solve_steady_flow
from paths import resolve_project_path
from train_stage3c import _bilinear_sample_grid_numpy, generate_synthetic_observations


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def r2_score(pred, ref):
    ss_res = np.sum((pred - ref) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def region_metrics(err):
    n = err.shape[0]
    band = max(2, n // 20)
    masks = {
        "overall": np.ones_like(err, dtype=bool),
        "interior": np.zeros_like(err, dtype=bool),
        "dirichlet_zone": np.zeros_like(err, dtype=bool),
        "neumann_zone": np.zeros_like(err, dtype=bool),
    }
    masks["interior"][band:-band, band:-band] = True
    masks["dirichlet_zone"][:band, :] = True
    masks["dirichlet_zone"][-band:, :] = True
    masks["neumann_zone"][:, :band] = True
    masks["neumann_zone"][:, -band:] = True

    out = {}
    for name, mask in masks.items():
        vals = err[mask]
        out[name] = {
            "rmse": float(np.sqrt(np.mean(vals ** 2))),
            "mae": float(np.mean(np.abs(vals))),
            "max_abs": float(np.max(np.abs(vals))),
        }
    return out


def observation_misfit(h_rec, x_fdm, y_fdm, config_path):
    """Compare recovered closed-loop h against configured observations.

    This is a no-truth selection metric for synthetic workflows: the
    observation values may have been generated from a truth field, but round
    selection only sees the same sparse observations that the inverse problem
    was trained on.
    """
    if config_path is None:
        return None

    import yaml

    with open(resolve_project_path(config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    x_obs, y_obs, h_obs = generate_synthetic_observations(config, device="cpu")
    h_pred_t = _bilinear_sample_grid_numpy(
        h_rec,
        x_fdm,
        y_fdm,
        x_obs,
        y_obs,
    )
    h_pred = h_pred_t.detach().cpu().numpy()
    h_obs_np = h_obs.detach().cpu().numpy()
    err = h_pred - h_obs_np
    return {
        "config": str(resolve_project_path(config_path)),
        "n_obs": int(len(h_obs_np)),
        "rmse": rmse(h_pred, h_obs_np),
        "mae": mae(h_pred, h_obs_np),
        "max_abs": float(np.max(np.abs(err))),
        "noise_std": float(config.get("inverse", {}).get("obs_noise", 0.0)),
        "training_seed": int(config.get("training", {}).get("seed", -1)),
    }


def plot_closed_loop(x, y, K_true, K_rec, h_ref, h_rec, h_true_resolved, out_path):
    Xk, Yk = np.meshgrid(x, y, indexing="ij")
    xh = np.linspace(0.0, 1.0, h_rec.shape[0])
    yh = np.linspace(0.0, 1.0, h_rec.shape[1])
    Xh, Yh = np.meshgrid(xh, yh, indexing="ij")

    h_err = h_rec - h_ref
    h_true_resolve_err = h_true_resolved - h_ref
    vmax_err = max(float(np.max(np.abs(h_err))), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    im = axes[0, 0].contourf(Xk, Yk, K_true, levels=30, cmap="YlOrRd")
    axes[0, 0].set_title("K true")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].contourf(Xk, Yk, K_rec, levels=30, cmap="YlOrRd")
    axes[0, 1].set_title("K recovered")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].contourf(Xk, Yk, np.log(K_rec) - np.log(K_true), levels=30, cmap="coolwarm")
    axes[0, 2].set_title("logK error")
    axes[0, 2].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].contourf(Xh, Yh, h_ref, levels=30, cmap="Spectral_r")
    axes[1, 0].set_title("h true FDM")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].contourf(Xh, Yh, h_rec, levels=30, cmap="Spectral_r")
    axes[1, 1].set_title("h from recovered K")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].contourf(
        Xh, Yh, h_err, levels=30, cmap="coolwarm", vmin=-vmax_err, vmax=vmax_err
    )
    axes[1, 2].set_title("h error recovered - true")
    axes[1, 2].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 2])

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Closed-loop h RMSE={rmse(h_rec, h_ref):.3e}; "
        f"true-resolve drift={rmse(h_true_resolved, h_ref):.3e}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovered", default="outputs/stage3c_xi_only_nn_h_discrete_refined/recovered_xi_logK.npz")
    parser.add_argument("--k-field", default="outputs/stage3_heterogeneous_10mode/K_field.npz")
    parser.add_argument("--fdm-true", default="outputs/stage4_fdm_10mode/h_true_N201.npz")
    parser.add_argument("--output-dir", default="outputs/stage3c_hybrid_closed_loop_refined")
    parser.add_argument("--config", default=None, help="Optional config for sparse observation misfit.")
    parser.add_argument("--N", type=int, default=201)
    args = parser.parse_args()

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = np.load(resolve_project_path(args.recovered), allow_pickle=True)
    k_data = np.load(resolve_project_path(args.k_field), allow_pickle=True)
    fdm_true = np.load(resolve_project_path(args.fdm_true), allow_pickle=True)

    x = rec["x"]
    y = rec["y"]
    logK_rec = rec["logK"]
    K_rec = np.exp(logK_rec)
    logK_true = k_data["logK"]
    K_true = k_data["K"]

    h_ref = np.ascontiguousarray(fdm_true["h"])
    h_rec, x_fdm, y_fdm, conv_rec = solve_steady_flow(K_rec, x, y, args.N)
    h_true_resolved, _, _, conv_true_resolved = solve_steady_flow(K_true, k_data["x"], k_data["y"], args.N)
    if h_ref.shape != h_rec.shape:
        h_ref = bilinear_interpolate(h_ref, fdm_true["x"], fdm_true["y"], x_fdm, y_fdm)

    K_rec_on_fdm = bilinear_interpolate(K_rec, x, y, x_fdm, y_fdm)
    residual_rec = compute_pde_residual_fdm(h_rec, K_rec_on_fdm, 1.0 / (args.N - 1), 1.0 / (args.N - 1))
    residual_abs = np.abs(residual_rec[1:-1, 1:-1])

    h_err = h_rec - h_ref
    h_true_resolve_err = h_true_resolved - h_ref

    xi_true = rec["xi_true"] if "xi_true" in rec.files else k_data["xi"][: len(rec["xi"])]
    xi_rec = rec["xi"]
    xi_corr = float(np.corrcoef(xi_rec, xi_true)[0, 1])

    metrics = {
        "inputs": {
            "recovered": str(resolve_project_path(args.recovered)),
            "k_field": str(resolve_project_path(args.k_field)),
            "fdm_true": str(resolve_project_path(args.fdm_true)),
            "N": args.N,
        },
        "xi": {
            "corr": xi_corr,
            "distance_to_true": float(np.linalg.norm(xi_true - xi_rec)),
        },
        "K": {
            "rmse": rmse(K_rec, K_true),
            "mae": mae(K_rec, K_true),
            "max_abs": float(np.max(np.abs(K_rec - K_true))),
            "logK_rmse": rmse(logK_rec, logK_true),
            "logK_mae": mae(logK_rec, logK_true),
            "logK_max_abs": float(np.max(np.abs(logK_rec - logK_true))),
        },
        "h_closed_loop": {
            "rmse_vs_true_fdm": rmse(h_rec, h_ref),
            "mae_vs_true_fdm": mae(h_rec, h_ref),
            "max_abs_vs_true_fdm": float(np.max(np.abs(h_err))),
            "r2_vs_true_fdm": r2_score(h_rec, h_ref),
            "region_metrics": region_metrics(h_err),
        },
        "observation_misfit": observation_misfit(h_rec, x_fdm, y_fdm, args.config),
        "solver_drift_check": {
            "trueK_resolved_rmse_vs_saved_true_fdm": rmse(h_true_resolved, h_ref),
            "trueK_resolved_mae_vs_saved_true_fdm": mae(h_true_resolve_err, np.zeros_like(h_true_resolve_err)),
            "trueK_resolved_max_abs_vs_saved_true_fdm": float(np.max(np.abs(h_true_resolve_err))),
        },
        "fdm_convergence": {
            "recovered_K": conv_rec,
            "true_K_resolved": conv_true_resolved,
        },
        "recovered_solution_residual": {
            "mean_abs": float(np.mean(residual_abs)),
            "p95_abs": float(np.percentile(residual_abs, 95)),
            "max_abs": float(np.max(residual_abs)),
        },
    }

    np.savez_compressed(
        out_dir / f"h_recovered_N{args.N}.npz",
        h=h_rec,
        h_true=h_ref,
        h_true_resolved=h_true_resolved,
        x=x_fdm,
        y=y_fdm,
        K_rec=K_rec,
        K_true=K_true,
        logK_rec=logK_rec,
        logK_true=logK_true,
        xi_rec=xi_rec,
        xi_true=xi_true,
    )
    np.savez_compressed(
        out_dir / "K_recovered_field.npz",
        K=K_rec,
        logK=logK_rec,
        x=x,
        y=y,
        xi=xi_rec,
        xi_true=xi_true,
    )

    with open(out_dir / "closed_loop_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_closed_loop(
        x,
        y,
        K_true,
        K_rec,
        h_ref,
        h_rec,
        h_true_resolved,
        out_dir / "closed_loop_validation.png",
    )

    print(json.dumps({
        "xi_corr": metrics["xi"]["corr"],
        "logK_rmse": metrics["K"]["logK_rmse"],
        "h_rmse_vs_true_fdm": metrics["h_closed_loop"]["rmse_vs_true_fdm"],
        "h_max_abs_vs_true_fdm": metrics["h_closed_loop"]["max_abs_vs_true_fdm"],
        "solver_drift_rmse": metrics["solver_drift_check"]["trueK_resolved_rmse_vs_saved_true_fdm"],
    }, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
