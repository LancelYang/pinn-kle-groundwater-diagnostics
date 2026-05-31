#!/usr/bin/env python3
"""Compare local FIM/Jacobian diagnostics at true and hybrid-recovered xi."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_kle_sensitivity_fim import (
    finite_difference_jacobian,
    load_yaml,
    override_config,
    spectral_metrics,
)
from paths import resolve_project_path


DEFAULT_HYBRID_ROOT = "outputs/cg_seed_audit/hybrid"


def parse_csv_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def noise_label(noise: float) -> str:
    label = f"{noise:.6f}".rstrip("0").rstrip(".")
    if label.startswith("0."):
        return "noise_" + label[2:]
    return "noise_" + label.replace(".", "p")


def config_for_case(truth_id: str, n_obs: int) -> Path:
    return resolve_project_path(f"outputs/stage3c_truth_robustness/truth_{truth_id}/config_obs{n_obs}.yaml")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recovered_xi_path(case_dir: Path) -> Path:
    alt = read_json(case_dir / "alternating_summary.json")
    best_dir = Path(alt["best_summary"]["output_dir"])
    return best_dir / "stage_b_discrete" / "recovered_xi_logK.npz"


def fim_row(
    *,
    truth_id: str,
    n_obs: int,
    noise: float,
    seed: int,
    reference_label: str,
    cfg: dict,
    xi_reference: np.ndarray | None,
    solve_n: int,
    fd_step: float,
) -> dict:
    jac, meta = finite_difference_jacobian(
        cfg,
        solve_n=solve_n,
        fd_step=fd_step,
        reference_xi=xi_reference,
        reference_label=reference_label,
    )
    metrics = spectral_metrics(jac, noise_std=noise)
    sv = metrics["singular_values"]
    return {
        "case": f"truth_{truth_id}_obs_{n_obs}_{noise_label(noise)}_seed_{seed}_{reference_label}",
        "truth_id": truth_id,
        "n_obs": n_obs,
        "obs_noise": noise,
        "seed": seed,
        "reference_label": reference_label,
        "n_modes": meta["n_modes"],
        "relative_rank_1e-6": metrics["relative_rank_1e-6"],
        "condition_number": metrics["condition_number"],
        "gauss_newton_hessian_condition_number": metrics["gauss_newton_hessian_condition_number"],
        "jacobian_frobenius_norm": metrics["jacobian_frobenius_norm"],
        "weakest_singular_value": min(sv),
        "strongest_singular_value": max(sv),
        "xi_distance_to_true": meta["xi_distance_to_true"],
        "h0_std": meta["h0_std"],
        "h0_range_min": meta["h0_range"][0],
        "h0_range_max": meta["h0_range"][1],
        "singular_values": sv,
        "column_norms": metrics["column_norms"],
    }


def paired_rows(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        key = (row["truth_id"], row["n_obs"], row["obs_noise"], row["seed"])
        by_key.setdefault(key, {})[row["reference_label"]] = row
    out = []
    for key, pair in sorted(by_key.items()):
        true = pair.get("true")
        recovered = pair.get("hybrid_recovered")
        if not true or not recovered:
            continue
        out.append(
            {
                "truth_id": key[0],
                "n_obs": key[1],
                "obs_noise": key[2],
                "seed": key[3],
                "true_rank": true["relative_rank_1e-6"],
                "recovered_rank": recovered["relative_rank_1e-6"],
                "true_condition_number": true["condition_number"],
                "recovered_condition_number": recovered["condition_number"],
                "condition_ratio_recovered_over_true": recovered["condition_number"] / true["condition_number"],
                "true_weakest_singular_value": true["weakest_singular_value"],
                "recovered_weakest_singular_value": recovered["weakest_singular_value"],
                "weakest_sv_ratio_recovered_over_true": recovered["weakest_singular_value"] / true["weakest_singular_value"],
                "recovered_xi_distance_to_true": recovered["xi_distance_to_true"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, pairs: list[dict]) -> None:
    lines = [
        "# CG Hybrid-Recovered Point Jacobian/FIM Diagnostic",
        "",
        "Paired local sparse-head sensitivity at true xi and at the selected hybrid-recovered xi.",
        "",
        "| truth | noise | seed | true rank | recovered rank | true J cond | recovered J cond | cond ratio | true weakest sv | recovered weakest sv | sv ratio | recovered xi distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['truth_id']} | {row['obs_noise']:g} | {row['seed']} | "
            f"{row['true_rank']} | {row['recovered_rank']} | "
            f"{fmt(row['true_condition_number'])} | {fmt(row['recovered_condition_number'])} | "
            f"{fmt(row['condition_ratio_recovered_over_true'])} | "
            f"{fmt(row['true_weakest_singular_value'])} | {fmt(row['recovered_weakest_singular_value'])} | "
            f"{fmt(row['weakest_sv_ratio_recovered_over_true'])} | "
            f"{fmt(row['recovered_xi_distance_to_true'])} |"
        )
    lines += [
        "",
        "Interpretation: if recovered-point conditioning is comparable to true-point conditioning, the hybrid failure is not explained by moving into a locally unobservable direct-map region.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(path: Path, pairs: list[dict]) -> None:
    labels = [f"n={row['obs_noise']:g}\ns={row['seed']}" for row in pairs]
    true_cond = [row["true_condition_number"] for row in pairs]
    rec_cond = [row["recovered_condition_number"] for row in pairs]
    true_sv = [row["true_weakest_singular_value"] for row in pairs]
    rec_sv = [row["recovered_weakest_singular_value"] for row in pairs]
    x = np.arange(len(pairs))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    axes[0].bar(x - width / 2, true_cond, width, label="true xi", color="#416a8f")
    axes[0].bar(x + width / 2, rec_cond, width, label="hybrid recovered xi", color="#bd5a54")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Jacobian condition number")
    axes[0].set_title("A. Local conditioning")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, true_sv, width, label="true xi", color="#416a8f")
    axes[1].bar(x + width / 2, rec_sv, width, label="hybrid recovered xi", color="#bd5a54")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("weakest singular value")
    axes[1].set_title("B. Weakest local sensitivity")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=False)
    fig.suptitle("Truth 456: true vs hybrid-recovered local sparse-head sensitivity", fontsize=13, weight="bold")
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-id", default="456")
    parser.add_argument("--n-obs", type=int, default=200)
    parser.add_argument("--noises", default="0.005,0.02")
    parser.add_argument("--seeds", default="202,303,404")
    parser.add_argument("--hybrid-root", default=DEFAULT_HYBRID_ROOT)
    parser.add_argument("--solve-N", type=int, default=81)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs/cg_recovered_fim")
    args = parser.parse_args()

    noises = parse_csv_list(args.noises, float)
    seeds = parse_csv_list(args.seeds, int)
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_root = resolve_project_path(args.hybrid_root)
    config_path = config_for_case(args.truth_id, args.n_obs)

    rows = []
    for noise in noises:
        for seed in seeds:
            cfg = override_config(load_yaml(config_path), args.n_obs, noise, seed)
            case_dir = hybrid_root / f"truth_{args.truth_id}" / f"obs_{args.n_obs}" / noise_label(noise) / f"seed_{seed}"
            xi_path = recovered_xi_path(case_dir)
            recovered = np.load(xi_path, allow_pickle=True)["xi"].astype(float)
            print(f"[cg-recovered-fim] truth={args.truth_id} noise={noise:g} seed={seed}")
            rows.append(
                fim_row(
                    truth_id=args.truth_id,
                    n_obs=args.n_obs,
                    noise=noise,
                    seed=seed,
                    reference_label="true",
                    cfg=cfg,
                    xi_reference=None,
                    solve_n=args.solve_N,
                    fd_step=args.fd_step,
                )
            )
            rows.append(
                fim_row(
                    truth_id=args.truth_id,
                    n_obs=args.n_obs,
                    noise=noise,
                    seed=seed,
                    reference_label="hybrid_recovered",
                    cfg=cfg,
                    xi_reference=recovered,
                    solve_n=args.solve_N,
                    fd_step=args.fd_step,
                )
            )

    pairs = paired_rows(rows)
    summary = {
        "diagnostic": "cg_hybrid_recovered_point_sparse_head_jacobian_fim",
        "settings": {
            "truth_id": args.truth_id,
            "n_obs": args.n_obs,
            "noises": noises,
            "seeds": seeds,
            "hybrid_root": str(hybrid_root),
            "solve_N": args.solve_N,
            "fd_step": args.fd_step,
        },
        "rows": rows,
        "paired_rows": pairs,
        "limitation": "Local finite-difference sensitivity at selected points; not a global identifiability proof.",
    }
    (out_dir / "cg_recovered_fim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "cg_recovered_fim_rows.csv", rows)
    write_csv(out_dir / "cg_recovered_fim_pairs.csv", pairs)
    write_markdown(out_dir / "cg_recovered_fim_summary.md", pairs)
    make_figure(out_dir / "cg_recovered_fim_summary.png", pairs)

    cg_table_dir = resolve_project_path("submissions/CG/tables")
    cg_fig_dir = resolve_project_path("submissions/CG/figures")
    cg_table_dir.mkdir(parents=True, exist_ok=True)
    cg_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "cg_recovered_fim_summary.md", cg_table_dir / "table_cg_recovered_fim_summary.md")
    shutil.copy2(out_dir / "cg_recovered_fim_pairs.csv", cg_table_dir / "cg_recovered_fim_pairs.csv")
    shutil.copy2(out_dir / "cg_recovered_fim_summary.json", cg_table_dir / "cg_recovered_fim_summary.json")
    shutil.copy2(out_dir / "cg_recovered_fim_summary.png", cg_fig_dir / "fig_cg_recovered_fim_summary.png")
    print(json.dumps({"output_dir": str(out_dir), "n_pairs": len(pairs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
