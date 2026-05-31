#!/usr/bin/env python3
"""Run matched PINN-hybrid cases for the CG compact matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from analyze_selector_dynamic_range import summarize as summarize_selector
from paths import resolve_project_path
from run_noise_robustness_pilot_grid import bool_from_metrics, noise_label


DEFAULT_TRUTHS = ["123", "456", "789"]
DEFAULT_DENSITIES = [200]
DEFAULT_NOISES = [0.005, 0.01, 0.02]
DEFAULT_SEED = 202
DEFAULT_SEEDS = [DEFAULT_SEED]

SUMMARY_FIELDS = [
    "status",
    "truth_id",
    "n_obs",
    "obs_noise",
    "seed",
    "case_id",
    "output_dir",
    "rounds_completed",
    "selected_round",
    "oracle_best_round",
    "selector_classification",
    "obs_range_to_oracle_range",
    "spearman_obs_vs_oracle_h",
    "oracle_h_rmse_regret",
    "best_h_rmse",
    "best_obs_rmse",
    "best_logK_rmse",
    "best_xi_corr",
    "weak_pass",
    "strong_pass",
    "failure_reason",
]


def parse_csv_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def case_id(truth_id: str, n_obs: int, noise: float, seed: int) -> str:
    return f"truth_{truth_id}_obs_{n_obs}_{noise_label(noise)}_seed_{seed}"


def config_for_case(truth_id: str, n_obs: int) -> Path:
    return resolve_project_path(f"outputs/stage3c_truth_robustness/truth_{truth_id}/config_obs{n_obs}.yaml")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_from_outputs(out_dir: Path, truth_id: str, n_obs: int, noise: float, seed: int, status: str = "ok", failure_reason: str = "") -> dict:
    alt_path = out_dir / "alternating_summary.json"
    selector_path = out_dir / "selector_dynamic_range.json"
    cid = case_id(truth_id, n_obs, noise, seed)
    if not alt_path.exists():
        return {
            "status": "failed",
            "truth_id": truth_id,
            "n_obs": n_obs,
            "obs_noise": noise,
            "seed": seed,
            "case_id": cid,
            "output_dir": str(out_dir),
            "failure_reason": failure_reason or "missing alternating_summary.json",
        }

    alt = read_json(alt_path)
    selector = read_json(selector_path) if selector_path.exists() else summarize_selector(alt_path)
    if not selector_path.exists():
        write_json(selector_path, selector)

    best = alt.get("best_summary") or {}
    stage_c = best.get("stage_c") or {}
    best_h = stage_c.get("h_rmse_vs_true_fdm")
    best_obs = stage_c.get("obs_rmse")
    best_logk = stage_c.get("logK_rmse")
    best_xi_corr = stage_c.get("xi_corr")
    classification = selector.get("selector_classification") or {}

    return {
        "status": status,
        "truth_id": truth_id,
        "n_obs": n_obs,
        "obs_noise": noise,
        "seed": seed,
        "case_id": cid,
        "output_dir": str(out_dir),
        "rounds_completed": alt.get("rounds_completed"),
        "selected_round": alt.get("best_round"),
        "oracle_best_round": selector.get("oracle_best_round"),
        "selector_classification": classification.get("class"),
        "obs_range_to_oracle_range": selector.get("obs_range_to_oracle_range"),
        "spearman_obs_vs_oracle_h": selector.get("spearman_obs_vs_oracle_h"),
        "oracle_h_rmse_regret": selector.get("oracle_h_rmse_regret"),
        "best_h_rmse": best_h,
        "best_obs_rmse": best_obs,
        "best_logK_rmse": best_logk,
        "best_xi_corr": best_xi_corr,
        "weak_pass": bool_from_metrics(best_h, best_logk, strong=False),
        "strong_pass": bool_from_metrics(best_h, best_logk, strong=True),
        "failure_reason": failure_reason,
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]

    def grouped(key: str, label: str) -> list[dict]:
        groups = []
        for value in sorted({row[key] for row in ok}):
            subset = [row for row in ok if row[key] == value]
            groups.append(
                {
                    label: value,
                    "n_cases": len(subset),
                    "hard_selector_count": sum(row.get("selector_classification") == "hard_pass" for row in subset),
                    "weak_pass_count": sum(row.get("weak_pass") is True for row in subset),
                    "strong_pass_count": sum(row.get("strong_pass") is True for row in subset),
                    "median_h_rmse": median([row["best_h_rmse"] for row in subset if row.get("best_h_rmse") is not None]),
                    "median_logK_rmse": median([row["best_logK_rmse"] for row in subset if row.get("best_logK_rmse") is not None]),
                    "median_selector_regret": median([row["oracle_h_rmse_regret"] for row in subset if row.get("oracle_h_rmse_regret") is not None]),
                }
            )
        return groups

    return {
        "n_cases": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "hard_selector_count": sum(row.get("selector_classification") == "hard_pass" for row in ok),
        "weak_pass_count": sum(row.get("weak_pass") is True for row in ok),
        "strong_pass_count": sum(row.get("strong_pass") is True for row in ok),
        "by_noise": grouped("obs_noise", "obs_noise"),
        "by_density": grouped("n_obs", "n_obs"),
        "by_truth": grouped("truth_id", "truth_id"),
    }


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


def write_markdown(path: Path, summary: dict) -> None:
    agg = summary["aggregate"]
    lines = [
        "# CG Compact PINN-Hybrid Matrix Summary",
        "",
        "This table summarizes the matched PINN-hybrid half of the compact CG matrix.",
        "",
        "## Overall",
        "",
        "| cases | ok | failed | hard selector | weak-pass | strong-pass |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {agg['n_cases']} | {agg['n_ok']} | {agg['n_failed']} | {agg['hard_selector_count']} | {agg['weak_pass_count']} | {agg['strong_pass_count']} |",
        "",
        "## By Noise",
        "",
        "| noise | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_noise"]:
        lines.append(
            f"| {row['obs_noise']:g} | {row['n_cases']} | {row['hard_selector_count']} | "
            f"{row['weak_pass_count']} | {row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | "
            f"{fmt(row['median_logK_rmse'])} | {fmt(row['median_selector_regret'])} |"
        )
    lines += [
        "",
        "## By Truth",
        "",
        "| truth | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_truth"]:
        lines.append(
            f"| {row['truth_id']} | {row['n_cases']} | {row['hard_selector_count']} | "
            f"{row['weak_pass_count']} | {row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | "
            f"{fmt(row['median_logK_rmse'])} | {fmt(row['median_selector_regret'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_command(args, out_dir: Path, config_path: Path, n_obs: int, noise: float, seed: int) -> list[str]:
    return [
        sys.executable,
        str(resolve_project_path("src/train_stage3c_hybrid.py")),
        "--config",
        str(config_path),
        "--output-dir",
        str(out_dir),
        "--alt-rounds",
        str(args.alt_rounds),
        "--alt-selection-metric",
        "observation_rmse",
        "--alt-stop-patience",
        str(args.alt_stop_patience),
        "--seed",
        str(seed),
        "--n-obs-points",
        str(n_obs),
        "--obs-noise",
        str(noise),
        "--stage-a-iters",
        str(args.stage_a_iters),
        "--stage-a-cont-iters",
        str(args.stage_a_cont_iters),
        "--stage-a-cont-pde-end",
        str(args.stage_a_cont_pde_end),
        "--stage-b-iters",
        str(args.stage_b_iters),
        "--n-pde-points",
        str(args.n_pde_points),
        "--n-bc-points",
        str(args.n_bc_points),
        "--closed-loop-N",
        str(args.closed_loop_N),
        "--print-every",
        str(args.print_every),
        "--plot-every",
        str(args.plot_every),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truths", default=",".join(DEFAULT_TRUTHS))
    parser.add_argument("--densities", default=",".join(str(v) for v in DEFAULT_DENSITIES))
    parser.add_argument("--noises", default=",".join(str(v) for v in DEFAULT_NOISES))
    parser.add_argument("--seed", type=int, default=None, help="Single seed shortcut.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides --seed when provided.")
    parser.add_argument("--output-root", default="outputs/cg_compact_matrix/hybrid")
    parser.add_argument("--table-prefix", default="cg_compact_hybrid")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--alt-rounds", type=int, default=5)
    parser.add_argument("--alt-stop-patience", type=int, default=0)
    parser.add_argument("--stage-a-iters", type=int, default=500)
    parser.add_argument("--stage-a-cont-iters", type=int, default=500)
    parser.add_argument("--stage-a-cont-pde-end", type=float, default=0.01)
    parser.add_argument("--stage-b-iters", type=int, default=1000)
    parser.add_argument("--n-pde-points", type=int, default=1000)
    parser.add_argument("--n-bc-points", type=int, default=200)
    parser.add_argument("--closed-loop-N", type=int, default=101)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--plot-every", type=int, default=100000)
    args = parser.parse_args()

    truths = parse_csv_list(args.truths, str)
    densities = parse_csv_list(args.densities, int)
    noises = parse_csv_list(args.noises, float)
    seeds = parse_csv_list(args.seeds, int) if args.seeds else [args.seed if args.seed is not None else DEFAULT_SEED]
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    project_root = resolve_project_path(".")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "truths": truths,
        "densities": densities,
        "noises": noises,
        "seeds": seeds,
        "output_root": str(output_root),
        "settings": {
            "alt_rounds": args.alt_rounds,
            "alt_selection_metric": "observation_rmse",
            "alt_stop_patience": args.alt_stop_patience,
            "stage_a_iters": args.stage_a_iters,
            "stage_a_cont_iters": args.stage_a_cont_iters,
            "stage_a_cont_pde_end": args.stage_a_cont_pde_end,
            "stage_b_iters": args.stage_b_iters,
            "n_pde_points": args.n_pde_points,
            "n_bc_points": args.n_bc_points,
            "closed_loop_N": args.closed_loop_N,
        },
    }
    write_json(output_root / "compact_hybrid_manifest.json", manifest)

    rows = []
    cases = [(truth, n_obs, noise, seed) for truth in truths for n_obs in densities for noise in noises for seed in seeds]
    for index, (truth_id, n_obs, noise, seed) in enumerate(cases, start=1):
        cid = case_id(truth_id, n_obs, noise, seed)
        out_dir = output_root / f"truth_{truth_id}" / f"obs_{n_obs}" / noise_label(noise) / f"seed_{seed}"
        config_path = config_for_case(truth_id, n_obs)
        alt_path = out_dir / "alternating_summary.json"
        print(f"[{index}/{len(cases)}] {cid}")

        if alt_path.exists() and not args.rerun_existing:
            print("  existing alternating_summary.json found; summarizing")
            rows.append(row_from_outputs(out_dir, truth_id, n_obs, noise, seed))
            continue

        cmd = train_command(args, out_dir, config_path, n_obs, noise, seed)
        write_json(out_dir / "runner_train_command.json", {"command": cmd, "dry_run": args.dry_run})
        if args.dry_run:
            rows.append(
                {
                    "status": "dry_run",
                    "truth_id": truth_id,
                    "n_obs": n_obs,
                    "obs_noise": noise,
                    "seed": seed,
                    "case_id": cid,
                    "output_dir": str(out_dir),
                    "failure_reason": "",
                }
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "runner_train_stdout.log").open("w", encoding="utf-8") as stdout, (
            out_dir / "runner_train_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(cmd, cwd=str(project_root), stdout=stdout, stderr=stderr, check=False)

        if completed.returncode != 0:
            reason = f"train_stage3c_hybrid failed with return code {completed.returncode}"
            print(f"  {reason}")
            rows.append(row_from_outputs(out_dir, truth_id, n_obs, noise, seed, status="failed", failure_reason=reason))
            continue

        selector = summarize_selector(alt_path)
        write_json(out_dir / "selector_dynamic_range.json", selector)
        row = row_from_outputs(out_dir, truth_id, n_obs, noise, seed)
        rows.append(row)
        print(
            "  class={selector_classification} weak={weak_pass} strong={strong_pass} "
            "h={best_h_rmse:.4g} logK={best_logK_rmse:.4g} xi_corr={best_xi_corr:.4g}".format(**row)
        )

    summary = {"manifest": manifest, "aggregate": aggregate(rows), "rows": rows}
    write_json(output_root / "compact_hybrid_summary.json", summary)
    write_csv(output_root / "compact_hybrid_summary.csv", rows)
    write_markdown(output_root / "compact_hybrid_summary.md", summary)

    cg_table_dir = resolve_project_path("submissions/CG/tables")
    cg_table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_root / "compact_hybrid_summary.csv", cg_table_dir / f"{args.table_prefix}_summary.csv")
    shutil.copy2(output_root / "compact_hybrid_summary.json", cg_table_dir / f"{args.table_prefix}_summary.json")
    shutil.copy2(output_root / "compact_hybrid_summary.md", cg_table_dir / f"table_{args.table_prefix}_summary.md")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
