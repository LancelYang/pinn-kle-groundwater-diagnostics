#!/usr/bin/env python3
"""Finite-difference Jacobian/FIM/Hessian diagnostics for KLE observability.

This is a numerical sensitivity diagnostic, not a new inverse solver. It
estimates the local map

    xi -> h(x_obs, y_obs; xi)

using central finite differences around the true KLE coefficients. It reports
Jacobian singular values, Fisher-information eigenvalues, and the
Gauss-Newton least-squares Hessian spectrum

    H_GN = 2 J^T J.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from paths import resolve_project_path
from run_fdm_kle_baseline import reconstruct_logk, sample_grid_numpy
from fdm_solver import solve_steady_flow
from train_stage3c import generate_synthetic_observations


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    "configs/stage3c_hybrid_3mode.yaml",
    "configs/stage3c_hybrid_5mode.yaml",
    "configs/stage3c_hybrid_10mode.yaml",
]


def load_yaml(path: str | Path) -> dict:
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: dict, n_obs: int, noise: float, seed: int) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["inverse"]["n_obs_points"] = int(n_obs)
    cfg["inverse"]["obs_noise"] = float(noise)
    cfg["training"]["seed"] = int(seed)
    return cfg


def solve_heads_for_xi(xi: np.ndarray, cfg: dict, k_data, x_obs, y_obs, solve_n: int) -> np.ndarray:
    logk = reconstruct_logk(
        xi=xi,
        k_data=k_data,
        n_modes=cfg["inverse"]["n_modes"],
        clamp_logk=cfg["inverse"].get("clamp_logK", 3.0),
    )
    k_grid = np.exp(logk)
    with contextlib.redirect_stdout(io.StringIO()):
        h_pred, x_pred, y_pred, _ = solve_steady_flow(k_grid, k_data["x"], k_data["y"], solve_n)
    return sample_grid_numpy(h_pred, x_pred, y_pred, x_obs, y_obs)


def finite_difference_jacobian(
    cfg: dict,
    solve_n: int,
    fd_step: float,
    reference_xi: np.ndarray | None = None,
    reference_label: str = "true",
) -> tuple[np.ndarray, dict]:
    x_obs_t, y_obs_t, _ = generate_synthetic_observations(cfg, device="cpu")
    x_obs = x_obs_t.detach().cpu().numpy()
    y_obs = y_obs_t.detach().cpu().numpy()
    k_data = np.load(resolve_project_path(cfg["kle"]["k_field_file"]), allow_pickle=True)
    n_modes = int(cfg["inverse"]["n_modes"])
    xi_true = np.asarray(k_data["xi"][:n_modes], dtype=float)
    xi0 = np.asarray(reference_xi, dtype=float) if reference_xi is not None else xi_true
    if xi0.shape != (n_modes,):
        raise ValueError(f"reference_xi shape {xi0.shape} does not match n_modes={n_modes}")

    h0 = solve_heads_for_xi(xi0, cfg, k_data, x_obs, y_obs, solve_n)
    jac = np.zeros((len(x_obs), n_modes), dtype=float)
    for mode in range(n_modes):
        delta = np.zeros(n_modes, dtype=float)
        delta[mode] = fd_step
        h_plus = solve_heads_for_xi(xi0 + delta, cfg, k_data, x_obs, y_obs, solve_n)
        h_minus = solve_heads_for_xi(xi0 - delta, cfg, k_data, x_obs, y_obs, solve_n)
        jac[:, mode] = (h_plus - h_minus) / (2.0 * fd_step)
    meta = {
        "n_obs": int(len(x_obs)),
        "n_modes": n_modes,
        "reference_label": reference_label,
        "xi_reference": xi0.tolist(),
        "xi_true": xi_true.tolist(),
        "xi_distance_to_true": float(np.linalg.norm(xi0 - xi_true)),
        "h0_std": float(np.std(h0)),
        "h0_range": [float(np.min(h0)), float(np.max(h0))],
    }
    return jac, meta


def spectral_metrics(jac: np.ndarray, noise_std: float) -> dict:
    singular_values = np.linalg.svd(jac, compute_uv=False)
    tol = max(jac.shape) * np.finfo(float).eps * singular_values[0] if singular_values.size else 0.0
    numerical_rank = int(np.sum(singular_values > tol))
    rel_tol_rank = int(np.sum(singular_values > singular_values[0] * 1e-6)) if singular_values.size else 0
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0
        else float("inf")
    )
    sigma = max(float(noise_std), 1e-12)
    fim = (jac.T @ jac) / (sigma * sigma)
    fim_eigvals = np.linalg.eigvalsh(fim)
    fim_eigvals = np.sort(np.maximum(fim_eigvals, 0.0))[::-1]
    gn_hessian = 2.0 * (jac.T @ jac)
    gn_eigvals = np.linalg.eigvalsh(gn_hessian)
    gn_eigvals = np.sort(np.maximum(gn_eigvals, 0.0))[::-1]
    fim_condition = (
        float(fim_eigvals[0] / fim_eigvals[-1])
        if fim_eigvals.size and fim_eigvals[-1] > 0
        else float("inf")
    )
    gn_condition = (
        float(gn_eigvals[0] / gn_eigvals[-1])
        if gn_eigvals.size and gn_eigvals[-1] > 0
        else float("inf")
    )
    return {
        "singular_values": singular_values.tolist(),
        "singular_value_log10": np.log10(np.maximum(singular_values, 1e-300)).tolist(),
        "numerical_rank": numerical_rank,
        "relative_rank_1e-6": rel_tol_rank,
        "condition_number": condition_number,
        "fim_eigenvalues": fim_eigvals.tolist(),
        "fim_log10_eigenvalues": np.log10(np.maximum(fim_eigvals, 1e-300)).tolist(),
        "fim_condition_number": fim_condition,
        "gauss_newton_hessian_eigenvalues": gn_eigvals.tolist(),
        "gauss_newton_hessian_log10_eigenvalues": np.log10(np.maximum(gn_eigvals, 1e-300)).tolist(),
        "gauss_newton_hessian_condition_number": gn_condition,
        "noise_std_for_fim": float(noise_std),
        "jacobian_frobenius_norm": float(np.linalg.norm(jac)),
        "column_norms": np.linalg.norm(jac, axis=0).tolist(),
    }


def plot_summary(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    colors = ["#416a8f", "#c6952e", "#bd5a54", "#4f8b63"]
    for idx, row in enumerate(rows):
        sv = np.asarray(row["singular_values"], dtype=float)
        label = f"{row['n_modes']}-mode"
        axes[0].plot(range(1, len(sv) + 1), sv, marker="o", color=colors[idx % len(colors)], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sensitivity direction index")
    axes[0].set_ylabel("Jacobian singular value")
    axes[0].set_title("A. Sparse-head KLE sensitivity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    mode_counts = [row["n_modes"] for row in rows]
    conditions = [row["condition_number"] for row in rows]
    weakest_sv = [min(row["singular_values"]) for row in rows]
    width = 0.34
    x = np.arange(len(rows))
    ax_cond = axes[1]
    ax_weak = ax_cond.twinx()
    bars = ax_cond.bar(x - width / 2, conditions, width, color="#416a8f", label="condition number")
    line = ax_weak.plot(x + width / 2, weakest_sv, marker="o", color="#bd5a54", linewidth=2.0, label="weakest singular value")
    ax_cond.set_xticks(x)
    ax_cond.set_xticklabels([f"{m}-mode" for m in mode_counts])
    ax_cond.set_ylabel("Jacobian condition number", color="#416a8f")
    ax_weak.set_ylabel("weakest singular value", color="#bd5a54")
    ax_weak.set_yscale("log")
    ax_cond.set_title("B. Conditioning summary")
    ax_cond.grid(axis="y", alpha=0.3)
    ax_cond.tick_params(axis="y", labelcolor="#416a8f")
    ax_weak.tick_params(axis="y", labelcolor="#bd5a54")
    for bar, value in zip(bars, conditions):
        ax_cond.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(conditions) * 0.025,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#243142",
            weight="bold",
        )
    handles = [bars, line[0]]
    labels = [handle.get_label() for handle in handles]
    ax_cond.legend(handles, labels, frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case",
            "config",
            "n_modes",
            "n_obs",
            "obs_noise",
            "seed",
            "solve_N",
            "fd_step",
            "numerical_rank",
            "relative_rank_1e-6",
            "condition_number",
            "fim_condition_number",
            "gauss_newton_hessian_condition_number",
            "jacobian_frobenius_norm",
            "singular_values",
            "fim_eigenvalues",
            "gauss_newton_hessian_eigenvalues",
            "column_norms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_note(path: Path, rows: list[dict]) -> None:
    lines = [
        "# KLE Jacobian / FIM / Gauss-Newton Hessian Diagnostic",
        "",
        "This diagnostic estimates the local sparse-head observation Jacobian with",
        "respect to KLE coefficients using central finite differences around the",
        "true coefficient vector. It supports the Computational Geosciences framing",
        "as a numerical identifiability and workflow-diagnostics study.",
        "",
        "The Fisher Information Matrix and Gauss-Newton least-squares Hessian are",
        "derived from the same Jacobian:",
        "",
        "```text",
        "FIM = J^T J / sigma^2",
        "H_GN = 2 J^T J",
        "```",
        "",
        "Therefore FIM and Gauss-Newton Hessian spectra have the same shape as the",
        "squared Jacobian singular spectrum. The main figure shows the Jacobian",
        "spectrum plus a conditioning summary to avoid plotting redundant spectra.",
        "",
        "| Case | n_obs | noise used in FIM | rank | J condition | FIM condition | H_GN condition |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['n_obs']} | {row['obs_noise']:.4g} | "
            f"{row['relative_rank_1e-6']}/{row['n_modes']} | "
            f"{row['condition_number']:.3g} | {row['fim_condition_number']:.3g} | "
            f"{row['gauss_newton_hessian_condition_number']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Full relative rank indicates that the local FDM-KLE inverse map is",
            "  observable for the tested sparse-head layout.",
            "- Large condition numbers indicate practical sensitivity and uncertainty,",
            "  even when the local rank is full.",
            "- These spectra diagnose information content of the direct FDM-KLE problem;",
            "  they do not prove that the PINN-hybrid optimizer can transmit the same",
            "  information through its staged workflow.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=None, help="Config path; may be repeated.")
    parser.add_argument("--output-dir", default="outputs/sensitivity_fim")
    parser.add_argument("--n-obs", type=int, default=200)
    parser.add_argument("--obs-noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--solve-N", type=int, default=81)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = args.config or DEFAULT_CONFIGS

    rows = []
    for config_path in configs:
        cfg = override_config(load_yaml(config_path), args.n_obs, args.obs_noise, args.seed)
        n_modes = int(cfg["inverse"]["n_modes"])
        print(f"[sensitivity] {config_path} | n_modes={n_modes}")
        jac, meta = finite_difference_jacobian(cfg, solve_n=args.solve_N, fd_step=args.fd_step)
        metrics = spectral_metrics(jac, noise_std=args.obs_noise)
        case = f"kle{n_modes}_obs{args.n_obs}_noise{args.obs_noise:g}_seed{args.seed}"
        np.savez_compressed(out_dir / f"{case}_jacobian.npz", jacobian=jac, **meta)
        row = {
            "case": case,
            "config": config_path,
            "n_modes": n_modes,
            "n_obs": args.n_obs,
            "obs_noise": args.obs_noise,
            "seed": args.seed,
            "solve_N": args.solve_N,
            "fd_step": args.fd_step,
            **meta,
            **metrics,
        }
        rows.append(row)

    summary = {
        "diagnostic": "finite_difference_sparse_head_jacobian_fim",
        "n_cases": len(rows),
        "settings": {
            "n_obs": args.n_obs,
            "obs_noise": args.obs_noise,
            "seed": args.seed,
            "solve_N": args.solve_N,
            "fd_step": args.fd_step,
        },
        "rows": rows,
        "limitation": "Local finite-difference sensitivity around true xi; not a global identifiability proof.",
    }
    (out_dir / "sensitivity_fim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "sensitivity_fim_summary.csv", rows)
    write_note(out_dir / "sensitivity_fim_note.md", rows)
    plot_summary(rows, out_dir / "sensitivity_fim_spectrum.png")
    print(json.dumps({"output_dir": str(out_dir), "n_cases": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
