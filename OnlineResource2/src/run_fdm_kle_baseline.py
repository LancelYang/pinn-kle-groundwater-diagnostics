#!/usr/bin/env python3
"""Direct FDM-KLE baseline for sparse-head Stage 3c cases.

This script answers the most important review-defense question for the current
paper: if we remove the PINN Stage A head approximation entirely and optimize
KLE coefficients directly against sparse FDM head observations, does the same
inverse case still recover?

The workflow intentionally mirrors the hybrid setup:
1. Load a Stage 3c config.
2. Regenerate the same synthetic sparse observations from the saved FDM truth.
3. Optimize KLE coefficients xi by minimizing sparse-head FDM misfit.
4. Reuse the same closed-loop metrics as the hybrid path.
"""

import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import Bounds, minimize

from fdm_solver import bilinear_interpolate, compute_pde_residual_fdm, solve_steady_flow
from paths import resolve_project_path
from train_stage3c import generate_synthetic_observations


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


def load_yaml(path):
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_override_config(config, args):
    cfg = json.loads(json.dumps(config))
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.n_obs_points is not None:
        cfg["inverse"]["n_obs_points"] = args.n_obs_points
    if args.obs_noise is not None:
        cfg["inverse"]["obs_noise"] = args.obs_noise
    return cfg


def sample_grid_numpy(values, x_grid, y_grid, xq, yq):
    nx = len(x_grid)
    ny = len(y_grid)
    x_norm = (xq - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (nx - 1)
    y_norm = (yq - y_grid[0]) / (y_grid[-1] - y_grid[0]) * (ny - 1)

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


def solve_steady_flow_silent(k_grid, x_vec, y_vec, n_tgt):
    with contextlib.redirect_stdout(io.StringIO()):
        return solve_steady_flow(k_grid, x_vec, y_vec, n_tgt)


def reconstruct_logk(xi, k_data, n_modes, clamp_logk):
    phi = k_data["eigenfunctions"][:, :n_modes]
    sqrt_lambda = np.sqrt(k_data["eigenvalues"][:n_modes])
    logk = (phi * sqrt_lambda[None, :] * xi[None, :]).sum(axis=1).reshape(k_data["logK"].shape)
    if clamp_logk is not None:
        logk = np.clip(logk, -clamp_logk, clamp_logk)
    return logk


def make_starts(n_modes, n_random_starts, random_seed, xi_init_std):
    starts = [("zero", np.zeros(n_modes, dtype=float))]
    rng = np.random.default_rng(random_seed)
    for idx in range(n_random_starts):
        xi0 = rng.normal(loc=0.0, scale=xi_init_std, size=n_modes)
        starts.append((f"random_{idx + 1:02d}", xi0.astype(float)))
    return starts


def plot_optimization_summary(start_rows, history_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for start_id in sorted({row["start_id"] for row in history_rows}):
        rows = [row for row in history_rows if row["start_id"] == start_id]
        axes[0].plot(
            [row["eval_index_within_start"] for row in rows],
            [row["obs_rmse"] for row in rows],
            alpha=0.7,
            label=start_id,
        )
    axes[0].set_title("Observation RMSE by evaluation")
    axes[0].set_xlabel("Function evaluation")
    axes[0].set_ylabel("obs_RMSE")
    if len(start_rows) <= 6:
        axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    labels = [row["start_id"] for row in start_rows]
    best_vals = [row["best_obs_rmse"] for row in start_rows]
    axes[1].bar(range(len(labels)), best_vals, color="#4c78a8")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("best obs_RMSE")
    axes[1].set_title("Best sparse-head fit by start")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_candidate(xi, cfg, k_data, x_obs, y_obs, h_obs, solve_n, clamp_logk, xi_prior_lambda):
    n_modes = cfg["inverse"]["n_modes"]
    logk = reconstruct_logk(xi, k_data, n_modes, clamp_logk)
    k_grid = np.exp(logk)
    h_pred, x_pred, y_pred, conv = solve_steady_flow_silent(k_grid, k_data["x"], k_data["y"], solve_n)
    h_obs_pred = sample_grid_numpy(h_pred, x_pred, y_pred, x_obs, y_obs)
    obs_rmse = rmse(h_obs_pred, h_obs)
    obs_mae = mae(h_obs_pred, h_obs)
    prior = float(xi_prior_lambda * np.mean(np.square(xi)))
    objective = float(obs_rmse ** 2 + prior)
    return {
        "objective": objective,
        "obs_rmse": obs_rmse,
        "obs_mae": obs_mae,
        "prior": prior,
        "h_pred": h_pred,
        "x_pred": x_pred,
        "y_pred": y_pred,
        "conv": conv,
        "logK": logk,
        "K": k_grid,
        "h_obs_pred": h_obs_pred,
    }


def run_closed_loop_metrics(best_xi, cfg, k_data, x_obs, y_obs, h_obs, closed_loop_n, clamp_logk):
    logk = reconstruct_logk(best_xi, k_data, cfg["inverse"]["n_modes"], clamp_logk)
    k_rec = np.exp(logk)

    fdm_true = np.load(resolve_project_path(cfg["inverse"]["fdm_reference_path"]), allow_pickle=True)
    h_ref = np.ascontiguousarray(fdm_true["h"])

    h_rec, x_fdm, y_fdm, conv_rec = solve_steady_flow_silent(k_rec, k_data["x"], k_data["y"], closed_loop_n)
    h_true_resolved, _, _, conv_true_resolved = solve_steady_flow_silent(
        k_data["K"],
        k_data["x"],
        k_data["y"],
        closed_loop_n,
    )
    if h_ref.shape != h_rec.shape:
        h_ref = bilinear_interpolate(h_ref, fdm_true["x"], fdm_true["y"], x_fdm, y_fdm)

    k_rec_on_fdm = bilinear_interpolate(k_rec, k_data["x"], k_data["y"], x_fdm, y_fdm)
    residual_rec = compute_pde_residual_fdm(
        h_rec,
        k_rec_on_fdm,
        1.0 / (closed_loop_n - 1),
        1.0 / (closed_loop_n - 1),
    )
    residual_abs = np.abs(residual_rec[1:-1, 1:-1])

    h_err = h_rec - h_ref
    h_true_resolve_err = h_true_resolved - h_ref
    h_obs_closed = sample_grid_numpy(h_rec, x_fdm, y_fdm, x_obs, y_obs)

    xi_true = k_data["xi"][: len(best_xi)]
    xi_corr = float(np.corrcoef(best_xi, xi_true)[0, 1]) if len(best_xi) > 1 else float("nan")

    return {
        "inputs": {
            "fdm_true": str(resolve_project_path(cfg["inverse"]["fdm_reference_path"])),
            "closed_loop_N": closed_loop_n,
        },
        "xi": {
            "corr": xi_corr,
            "distance_to_true": float(np.linalg.norm(xi_true - best_xi)),
        },
        "K": {
            "rmse": rmse(k_rec, k_data["K"]),
            "mae": mae(k_rec, k_data["K"]),
            "max_abs": float(np.max(np.abs(k_rec - k_data["K"]))),
            "logK_rmse": rmse(logk, k_data["logK"]),
            "logK_mae": mae(logk, k_data["logK"]),
            "logK_max_abs": float(np.max(np.abs(logk - k_data["logK"]))),
        },
        "h_closed_loop": {
            "rmse_vs_true_fdm": rmse(h_rec, h_ref),
            "mae_vs_true_fdm": mae(h_rec, h_ref),
            "max_abs_vs_true_fdm": float(np.max(np.abs(h_err))),
            "r2_vs_true_fdm": r2_score(h_rec, h_ref),
            "region_metrics": region_metrics(h_err),
        },
        "observation_misfit": {
            "n_obs": int(len(h_obs)),
            "rmse": rmse(h_obs_closed, h_obs),
            "mae": mae(h_obs_closed, h_obs),
            "max_abs": float(np.max(np.abs(h_obs_closed - h_obs))),
            "noise_std": float(cfg.get("inverse", {}).get("obs_noise", 0.0)),
            "training_seed": int(cfg.get("training", {}).get("seed", -1)),
        },
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
        "artifacts": {
            "h_recovered": h_rec,
            "h_true": h_ref,
            "h_true_resolved": h_true_resolved,
            "x_fdm": x_fdm,
            "y_fdm": y_fdm,
            "K_rec": k_rec,
            "logK_rec": logk,
            "xi_rec": best_xi,
            "xi_true": xi_true,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3c_hybrid_3mode.yaml")
    parser.add_argument("--output-dir", default="outputs/fdm_kle_baseline/3mode_obs200_noise005_seed999")
    parser.add_argument("--seed", type=int, default=None, help="Override config training.seed for observation generation.")
    parser.add_argument("--n-obs-points", type=int, default=None, help="Override config inverse.n_obs_points.")
    parser.add_argument("--obs-noise", type=float, default=None, help="Override config inverse.obs_noise.")
    parser.add_argument("--solve-N", type=int, default=81, help="Grid size used inside sparse-head optimization.")
    parser.add_argument("--closed-loop-N", type=int, default=201, help="Grid size used for final closed-loop validation.")
    parser.add_argument("--method", choices=["Powell", "Nelder-Mead"], default="Powell")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--n-random-starts", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--xi-init-std", type=float, default=0.15)
    parser.add_argument("--xi-prior-lambda", type=float, default=None)
    parser.add_argument("--xi-bound", type=float, default=3.0)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    cfg = maybe_override_config(load_yaml(args.config), args)
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    x_obs_t, y_obs_t, h_obs_t = generate_synthetic_observations(cfg, device="cpu")
    x_obs = x_obs_t.detach().cpu().numpy()
    y_obs = y_obs_t.detach().cpu().numpy()
    h_obs = h_obs_t.detach().cpu().numpy()
    np.savez_compressed(out_dir / "observations.npz", x=x_obs, y=y_obs, h=h_obs)

    k_data = np.load(resolve_project_path(cfg["kle"]["k_field_file"]), allow_pickle=True)
    n_modes = cfg["inverse"]["n_modes"]
    clamp_logk = cfg.get("inverse", {}).get("clamp_logK", 3.0)
    xi_prior_lambda = (
        float(args.xi_prior_lambda)
        if args.xi_prior_lambda is not None
        else float(cfg.get("training", {}).get("weights", {}).get("xi_prior", 0.0))
    )

    starts = make_starts(
        n_modes=n_modes,
        n_random_starts=args.n_random_starts,
        random_seed=args.random_seed,
        xi_init_std=args.xi_init_std,
    )
    bounds = Bounds(
        lb=np.full(n_modes, -args.xi_bound, dtype=float),
        ub=np.full(n_modes, args.xi_bound, dtype=float),
    )

    history_rows = []
    start_rows = []
    eval_counter = 0
    best_result = None
    wall_start = time.time()

    for start_index, (start_id, x0) in enumerate(starts, start=1):
        start_eval_counter = 0
        start_wall = time.time()

        def objective(xi):
            nonlocal eval_counter, start_eval_counter
            eval_counter += 1
            start_eval_counter += 1
            metrics = evaluate_candidate(
                xi=np.asarray(xi, dtype=float),
                cfg=cfg,
                k_data=k_data,
                x_obs=x_obs,
                y_obs=y_obs,
                h_obs=h_obs,
                solve_n=args.solve_N,
                clamp_logk=clamp_logk,
                xi_prior_lambda=xi_prior_lambda,
            )
            history_rows.append({
                "global_eval_index": eval_counter,
                "eval_index_within_start": start_eval_counter,
                "start_id": start_id,
                "objective": metrics["objective"],
                "obs_rmse": metrics["obs_rmse"],
                "obs_mae": metrics["obs_mae"],
                "prior": metrics["prior"],
            })
            return metrics["objective"]

        print(f"[baseline] start {start_index}/{len(starts)}: {start_id} | x0={np.round(x0, 4).tolist()}")
        res = minimize(
            objective,
            x0=x0,
            method=args.method,
            bounds=bounds if args.method == "Powell" else None,
            options={"maxiter": args.maxiter, "disp": False},
        )
        final_metrics = evaluate_candidate(
            xi=np.asarray(res.x, dtype=float),
            cfg=cfg,
            k_data=k_data,
            x_obs=x_obs,
            y_obs=y_obs,
            h_obs=h_obs,
            solve_n=args.solve_N,
            clamp_logk=clamp_logk,
            xi_prior_lambda=xi_prior_lambda,
        )

        row = {
            "start_id": start_id,
            "initial_xi": np.asarray(x0, dtype=float).tolist(),
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
            "nfev": int(getattr(res, "nfev", -1)),
            "runtime_sec": float(time.time() - start_wall),
            "best_objective": float(final_metrics["objective"]),
            "best_obs_rmse": float(final_metrics["obs_rmse"]),
            "best_obs_mae": float(final_metrics["obs_mae"]),
            "xi_pred": np.asarray(res.x, dtype=float).tolist(),
        }
        start_rows.append(row)
        print(
            f"[baseline] {start_id} done | success={row['success']} | "
            f"obs_rmse={row['best_obs_rmse']:.6f} | nfev={row['nfev']} | "
            f"runtime={row['runtime_sec']:.1f}s"
        )

        if best_result is None or row["best_objective"] < best_result["best_objective"]:
            best_result = {
                "start_id": start_id,
                "best_objective": row["best_objective"],
                "best_obs_rmse": row["best_obs_rmse"],
                "xi_pred": np.asarray(res.x, dtype=float),
                "solve_metrics": final_metrics,
            }

    runtime_total = float(time.time() - wall_start)
    closed_loop = run_closed_loop_metrics(
        best_xi=best_result["xi_pred"],
        cfg=cfg,
        k_data=k_data,
        x_obs=x_obs,
        y_obs=y_obs,
        h_obs=h_obs,
        closed_loop_n=args.closed_loop_N,
        clamp_logk=clamp_logk,
    )

    np.savez_compressed(
        out_dir / "recovered_xi_logK.npz",
        xi=closed_loop["artifacts"]["xi_rec"],
        xi_true=closed_loop["artifacts"]["xi_true"],
        logK=closed_loop["artifacts"]["logK_rec"],
        logK_true=k_data["logK"],
        x=k_data["x"],
        y=k_data["y"],
    )
    np.savez_compressed(
        out_dir / "K_recovered_field.npz",
        K=closed_loop["artifacts"]["K_rec"],
        logK=closed_loop["artifacts"]["logK_rec"],
        x=k_data["x"],
        y=k_data["y"],
        xi=closed_loop["artifacts"]["xi_rec"],
        xi_true=closed_loop["artifacts"]["xi_true"],
    )
    np.savez_compressed(
        out_dir / f"h_recovered_N{args.closed_loop_N}.npz",
        h=closed_loop["artifacts"]["h_recovered"],
        h_true=closed_loop["artifacts"]["h_true"],
        h_true_resolved=closed_loop["artifacts"]["h_true_resolved"],
        x=closed_loop["artifacts"]["x_fdm"],
        y=closed_loop["artifacts"]["y_fdm"],
    )

    with open(out_dir / "objective_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_eval_index",
                "eval_index_within_start",
                "start_id",
                "objective",
                "obs_rmse",
                "obs_mae",
                "prior",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    if not args.skip_plot:
        plot_optimization_summary(start_rows, history_rows, out_dir / "optimization_summary.png")

    result = {
        "inputs": {
            "config": str(resolve_project_path(args.config)),
            "k_field_file": str(resolve_project_path(cfg["kle"]["k_field_file"])),
            "fdm_reference_path": str(resolve_project_path(cfg["inverse"]["fdm_reference_path"])),
            "solve_N": args.solve_N,
            "closed_loop_N": args.closed_loop_N,
            "method": args.method,
            "maxiter": args.maxiter,
            "n_random_starts": args.n_random_starts,
            "random_seed": args.random_seed,
            "xi_bound": args.xi_bound,
            "xi_prior_lambda": xi_prior_lambda,
            "clamp_logK": clamp_logk,
        },
        "observation_setup": {
            "n_obs": int(len(h_obs)),
            "obs_noise": float(cfg["inverse"].get("obs_noise", 0.0)),
            "training_seed": int(cfg["training"].get("seed", -1)),
        },
        "runtime_sec": runtime_total,
        "start_results": start_rows,
        "best_start": {
            "start_id": best_result["start_id"],
            "best_objective": float(best_result["best_objective"]),
            "best_obs_rmse_optimization_grid": float(best_result["best_obs_rmse"]),
            "xi_pred": best_result["xi_pred"].tolist(),
        },
        "closed_loop_metrics": {
            key: value
            for key, value in closed_loop.items()
            if key != "artifacts"
        },
    }

    with open(out_dir / "fdm_kle_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "best_start": best_result["start_id"],
        "best_obs_rmse_optimization_grid": best_result["best_obs_rmse"],
        "closed_loop_obs_rmse": closed_loop["observation_misfit"]["rmse"],
        "closed_loop_logK_rmse": closed_loop["K"]["logK_rmse"],
        "closed_loop_h_rmse": closed_loop["h_closed_loop"]["rmse_vs_true_fdm"],
        "runtime_sec": runtime_total,
    }, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
