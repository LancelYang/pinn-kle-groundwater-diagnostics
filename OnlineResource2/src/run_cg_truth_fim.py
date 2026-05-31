#!/usr/bin/env python3
"""Run local sparse-head Jacobian/FIM diagnostics across CG compact truths."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_kle_sensitivity_fim import (
    finite_difference_jacobian,
    load_yaml,
    override_config,
    spectral_metrics,
)
from paths import resolve_project_path


DEFAULT_TRUTHS = ["123", "456", "789"]
DEFAULT_DENSITIES = [200]


def parse_csv_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def config_for_case(truth_id: str, n_obs: int) -> Path:
    return resolve_project_path(f"outputs/stage3c_truth_robustness/truth_{truth_id}/config_obs{n_obs}.yaml")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case",
        "truth_id",
        "n_obs",
        "obs_noise",
        "seed",
        "n_modes",
        "relative_rank_1e-6",
        "condition_number",
        "gauss_newton_hessian_condition_number",
        "jacobian_frobenius_norm",
        "weakest_singular_value",
        "strongest_singular_value",
        "h0_std",
        "h0_range_min",
        "h0_range_max",
        "config",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# CG Truth Jacobian/FIM Diagnostic Summary",
        "",
        "Local finite-difference sparse-head sensitivity around true KLE coefficients.",
        "",
        "| truth | n_obs | rank | J condition | H_GN condition | weakest sv | strongest sv |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['truth_id']} | {row['n_obs']} | {row['relative_rank_1e-6']}/{row['n_modes']} | "
            f"{fmt(row['condition_number'])} | {fmt(row['gauss_newton_hessian_condition_number'])} | "
            f"{fmt(row['weakest_singular_value'])} | {fmt(row['strongest_singular_value'])} |"
        )
    lines += [
        "",
        "Interpretation: full local rank supports direct local observability; condition-number differences diagnose practical sensitivity, not global uniqueness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(path: Path, rows: list[dict]) -> None:
    labels = [f"T{row['truth_id']}\nN={row['n_obs']}" for row in rows]
    cond = [row["condition_number"] for row in rows]
    weakest = [row["weakest_singular_value"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    axes[0].bar(labels, cond, color="#416a8f")
    axes[0].set_ylabel("Jacobian condition number")
    axes[0].set_title("A. Local conditioning by truth")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(labels, weakest, color="#bd5a54")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("weakest singular value")
    axes[1].set_title("B. Weakest local sensitivity")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("CG truth-level sparse-head KLE sensitivity", fontsize=13, weight="bold")
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truths", default=",".join(DEFAULT_TRUTHS))
    parser.add_argument("--densities", default=",".join(str(v) for v in DEFAULT_DENSITIES))
    parser.add_argument("--obs-noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--solve-N", type=int, default=81)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs/cg_truth_fim")
    args = parser.parse_args()

    truths = parse_csv_list(args.truths, str)
    densities = parse_csv_list(args.densities, int)
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for truth_id in truths:
        for n_obs in densities:
            config_path = config_for_case(truth_id, n_obs)
            cfg = override_config(load_yaml(config_path), n_obs, args.obs_noise, args.seed)
            n_modes = int(cfg["inverse"]["n_modes"])
            print(f"[cg-truth-fim] truth={truth_id} n_obs={n_obs} n_modes={n_modes}")
            jac, meta = finite_difference_jacobian(cfg, solve_n=args.solve_N, fd_step=args.fd_step)
            metrics = spectral_metrics(jac, noise_std=args.obs_noise)
            case = f"truth_{truth_id}_obs_{n_obs}_noise_{args.obs_noise:g}_seed_{args.seed}"
            singular_values = metrics["singular_values"]
            row = {
                "case": case,
                "truth_id": truth_id,
                "n_obs": n_obs,
                "obs_noise": args.obs_noise,
                "seed": args.seed,
                "n_modes": n_modes,
                "config": str(config_path),
                "relative_rank_1e-6": metrics["relative_rank_1e-6"],
                "condition_number": metrics["condition_number"],
                "gauss_newton_hessian_condition_number": metrics["gauss_newton_hessian_condition_number"],
                "jacobian_frobenius_norm": metrics["jacobian_frobenius_norm"],
                "weakest_singular_value": min(singular_values),
                "strongest_singular_value": max(singular_values),
                "h0_std": meta["h0_std"],
                "h0_range_min": meta["h0_range"][0],
                "h0_range_max": meta["h0_range"][1],
                "singular_values": singular_values,
                "column_norms": metrics["column_norms"],
            }
            rows.append(row)
            (out_dir / f"{case}_jacobian.json").write_text(
                json.dumps({"jacobian": jac.tolist(), "meta": meta, "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

    summary = {
        "diagnostic": "cg_truth_sparse_head_jacobian_fim",
        "settings": {
            "truths": truths,
            "densities": densities,
            "obs_noise": args.obs_noise,
            "seed": args.seed,
            "solve_N": args.solve_N,
            "fd_step": args.fd_step,
        },
        "rows": rows,
        "limitation": "Local finite-difference sensitivity around true xi; not a global identifiability proof.",
    }
    (out_dir / "cg_truth_fim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "cg_truth_fim_summary.csv", rows)
    write_markdown(out_dir / "cg_truth_fim_summary.md", rows)
    make_figure(out_dir / "cg_truth_fim_summary.png", rows)

    cg_table_dir = resolve_project_path("submissions/CG/tables")
    cg_fig_dir = resolve_project_path("submissions/CG/figures")
    cg_table_dir.mkdir(parents=True, exist_ok=True)
    cg_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "cg_truth_fim_summary.csv", cg_table_dir / "cg_truth_fim_summary.csv")
    shutil.copy2(out_dir / "cg_truth_fim_summary.json", cg_table_dir / "cg_truth_fim_summary.json")
    shutil.copy2(out_dir / "cg_truth_fim_summary.md", cg_table_dir / "table_cg_truth_fim_summary.md")
    shutil.copy2(out_dir / "cg_truth_fim_summary.png", cg_fig_dir / "fig_cg_truth_fim_summary.png")
    print(json.dumps({"output_dir": str(out_dir), "n_cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
