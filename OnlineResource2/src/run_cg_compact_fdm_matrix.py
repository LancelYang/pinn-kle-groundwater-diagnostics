#!/usr/bin/env python3
"""Run the direct FDM-KLE half of the CG compact matrix.

This is the first, cheaper half of the CG compact matrix plan. It establishes
the direct reduced-space recoverability reference before matched PINN-hybrid
runs are expanded.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from paths import resolve_project_path


DEFAULT_TRUTHS = ["123", "456", "789"]
DEFAULT_DENSITIES = [100, 200]
DEFAULT_NOISES = [0.005, 0.01, 0.02]
DEFAULT_SEED = 202

SUMMARY_FIELDS = [
    "status",
    "truth_id",
    "n_obs",
    "obs_noise",
    "seed",
    "case_id",
    "output_dir",
    "best_start",
    "nfev",
    "runtime_sec",
    "obs_rmse",
    "h_rmse",
    "logK_rmse",
    "xi_corr",
    "xi_distance",
    "weak_pass",
    "strong_pass",
    "failure_reason",
]


def noise_label(noise: float) -> str:
    label = f"{noise:.6f}".rstrip("0").rstrip(".")
    if label.startswith("0."):
        return "noise_" + label[2:]
    return "noise_" + label.replace(".", "p")


def case_id(truth_id: str, n_obs: int, noise: float, seed: int) -> str:
    return f"truth_{truth_id}_obs_{n_obs}_{noise_label(noise)}_seed_{seed}"


def parse_csv_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def config_for_case(truth_id: str, n_obs: int) -> Path:
    return resolve_project_path(f"outputs/stage3c_truth_robustness/truth_{truth_id}/config_obs{n_obs}.yaml")


def result_path(out_dir: Path) -> Path:
    return out_dir / "fdm_kle_baseline_results.json"


def as_bool_from_metrics(h_rmse: float | None, logk_rmse: float | None, strong: bool = False) -> bool | None:
    if h_rmse is None or logk_rmse is None:
        return None
    if strong:
        return bool(logk_rmse < 0.10 and h_rmse < 0.003)
    return bool(logk_rmse < 0.15 and h_rmse < 0.006)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_result(out_dir: Path, truth_id: str, n_obs: int, noise: float, seed: int, status: str = "ok", failure_reason: str = "") -> dict:
    path = result_path(out_dir)
    cid = case_id(truth_id, n_obs, noise, seed)
    if not path.exists():
        return {
            "status": "failed",
            "truth_id": truth_id,
            "n_obs": n_obs,
            "obs_noise": noise,
            "seed": seed,
            "case_id": cid,
            "output_dir": str(out_dir),
            "failure_reason": failure_reason or "missing fdm_kle_baseline_results.json",
        }

    data = read_json(path)
    best_start = data.get("best_start") or {}
    closed = data.get("closed_loop_metrics") or {}
    obs = closed.get("observation_misfit") or {}
    h = closed.get("h_closed_loop") or {}
    k = closed.get("K") or {}
    xi = closed.get("xi") or {}
    h_rmse = h.get("rmse_vs_true_fdm")
    logk_rmse = k.get("logK_rmse")

    return {
        "status": status,
        "truth_id": truth_id,
        "n_obs": n_obs,
        "obs_noise": noise,
        "seed": seed,
        "case_id": cid,
        "output_dir": str(out_dir),
        "best_start": best_start.get("start_id"),
        "nfev": sum((row.get("nfev") or 0) for row in data.get("start_results", [])),
        "runtime_sec": data.get("runtime_sec"),
        "obs_rmse": obs.get("rmse"),
        "h_rmse": h_rmse,
        "logK_rmse": logk_rmse,
        "xi_corr": xi.get("corr"),
        "xi_distance": xi.get("distance_to_true"),
        "weak_pass": as_bool_from_metrics(h_rmse, logk_rmse, strong=False),
        "strong_pass": as_bool_from_metrics(h_rmse, logk_rmse, strong=True),
        "failure_reason": failure_reason,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    by_noise = []
    for noise in sorted({float(row["obs_noise"]) for row in ok}):
        subset = [row for row in ok if float(row["obs_noise"]) == noise]
        by_noise.append(
            {
                "obs_noise": noise,
                "n_cases": len(subset),
                "weak_pass_count": sum(row.get("weak_pass") is True for row in subset),
                "strong_pass_count": sum(row.get("strong_pass") is True for row in subset),
                "median_logK_rmse": median([row["logK_rmse"] for row in subset if row.get("logK_rmse") is not None]),
                "median_h_rmse": median([row["h_rmse"] for row in subset if row.get("h_rmse") is not None]),
            }
        )
    by_density = []
    for n_obs in sorted({int(row["n_obs"]) for row in ok}):
        subset = [row for row in ok if int(row["n_obs"]) == n_obs]
        by_density.append(
            {
                "n_obs": n_obs,
                "n_cases": len(subset),
                "weak_pass_count": sum(row.get("weak_pass") is True for row in subset),
                "strong_pass_count": sum(row.get("strong_pass") is True for row in subset),
                "median_logK_rmse": median([row["logK_rmse"] for row in subset if row.get("logK_rmse") is not None]),
                "median_h_rmse": median([row["h_rmse"] for row in subset if row.get("h_rmse") is not None]),
            }
        )
    by_truth = []
    for truth_id in sorted({str(row["truth_id"]) for row in ok}):
        subset = [row for row in ok if str(row["truth_id"]) == truth_id]
        by_truth.append(
            {
                "truth_id": truth_id,
                "n_cases": len(subset),
                "weak_pass_count": sum(row.get("weak_pass") is True for row in subset),
                "strong_pass_count": sum(row.get("strong_pass") is True for row in subset),
                "median_logK_rmse": median([row["logK_rmse"] for row in subset if row.get("logK_rmse") is not None]),
                "median_h_rmse": median([row["h_rmse"] for row in subset if row.get("h_rmse") is not None]),
            }
        )
    return {
        "n_cases": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "weak_pass_count": sum(row.get("weak_pass") is True for row in ok),
        "strong_pass_count": sum(row.get("strong_pass") is True for row in ok),
        "by_noise": by_noise,
        "by_density": by_density,
        "by_truth": by_truth,
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


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
        "# CG Compact Direct FDM-KLE Matrix Summary",
        "",
        "This table summarizes the direct FDM-KLE half of the compact CG matrix.",
        "It establishes the direct reduced-space recoverability reference before matched PINN-hybrid expansion.",
        "",
        "## Overall",
        "",
        "| cases | ok | failed | weak-pass | strong-pass |",
        "|---:|---:|---:|---:|---:|",
        f"| {agg['n_cases']} | {agg['n_ok']} | {agg['n_failed']} | {agg['weak_pass_count']} | {agg['strong_pass_count']} |",
        "",
        "## By Noise",
        "",
        "| noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_noise"]:
        lines.append(
            f"| {row['obs_noise']:g} | {row['n_cases']} | {row['weak_pass_count']} | "
            f"{row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | {fmt(row['median_logK_rmse'])} |"
        )
    lines += [
        "",
        "## By Observation Density",
        "",
        "| n_obs | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_density"]:
        lines.append(
            f"| {row['n_obs']} | {row['n_cases']} | {row['weak_pass_count']} | "
            f"{row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | {fmt(row['median_logK_rmse'])} |"
        )
    lines += [
        "",
        "## By Truth",
        "",
        "| truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg["by_truth"]:
        lines.append(
            f"| {row['truth_id']} | {row['n_cases']} | {row['weak_pass_count']} | "
            f"{row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | {fmt(row['median_logK_rmse'])} |"
        )
    lines += [
        "",
        "Interpretation: direct FDM-KLE remains robust across the compact truth/noise/density matrix.",
        "Matched PINN-hybrid runs should be compared against this reference, not against a single-truth baseline.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truths", default=",".join(DEFAULT_TRUTHS))
    parser.add_argument("--densities", default=",".join(str(v) for v in DEFAULT_DENSITIES))
    parser.add_argument("--noises", default=",".join(str(v) for v in DEFAULT_NOISES))
    parser.add_argument("--seed", type=int, default=None, help="Single seed shortcut.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides --seed when provided.")
    parser.add_argument("--output-root", default="outputs/cg_compact_matrix/fdm_kle")
    parser.add_argument("--table-prefix", default="cg_compact_fdm")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solve-N", type=int, default=81)
    parser.add_argument("--closed-loop-N", type=int, default=201)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--n-random-starts", type=int, default=2)
    parser.add_argument("--skip-plot", action="store_true")
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
            "solve_N": args.solve_N,
            "closed_loop_N": args.closed_loop_N,
            "maxiter": args.maxiter,
            "n_random_starts": args.n_random_starts,
        },
    }
    write_json(output_root / "compact_fdm_manifest.json", manifest)

    rows = []
    cases = [(truth, n_obs, noise, seed) for truth in truths for n_obs in densities for noise in noises for seed in seeds]
    for index, (truth_id, n_obs, noise, seed) in enumerate(cases, start=1):
        cid = case_id(truth_id, n_obs, noise, seed)
        out_dir = output_root / f"truth_{truth_id}" / f"obs_{n_obs}" / noise_label(noise) / f"seed_{seed}"
        config_path = config_for_case(truth_id, n_obs)
        print(f"[{index}/{len(cases)}] {cid}")

        if result_path(out_dir).exists() and not args.rerun_existing:
            print("  existing result found; summarizing")
            rows.append(row_from_result(out_dir, truth_id, n_obs, noise, seed))
            continue

        cmd = [
            sys.executable,
            str(resolve_project_path("src/run_fdm_kle_baseline.py")),
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--n-obs-points",
            str(n_obs),
            "--obs-noise",
            str(noise),
            "--solve-N",
            str(args.solve_N),
            "--closed-loop-N",
            str(args.closed_loop_N),
            "--maxiter",
            str(args.maxiter),
            "--n-random-starts",
            str(args.n_random_starts),
        ]
        if args.skip_plot:
            cmd.append("--skip-plot")

        write_json(out_dir / "runner_command.json", {"command": cmd, "dry_run": args.dry_run})
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
        with (out_dir / "runner_stdout.log").open("w", encoding="utf-8") as stdout, (
            out_dir / "runner_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(cmd, cwd=str(project_root), stdout=stdout, stderr=stderr, check=False)

        if completed.returncode != 0:
            reason = f"run_fdm_kle_baseline failed with return code {completed.returncode}"
            print(f"  {reason}")
            rows.append(row_from_result(out_dir, truth_id, n_obs, noise, seed, status="failed", failure_reason=reason))
            continue

        row = row_from_result(out_dir, truth_id, n_obs, noise, seed)
        rows.append(row)
        print(
            "  weak={weak_pass} strong={strong_pass} h={h_rmse:.4g} logK={logK_rmse:.4g} xi_corr={xi_corr:.4g}".format(
                **row
            )
        )

    summary = {"manifest": manifest, "aggregate": aggregate(rows), "rows": rows}
    write_json(output_root / "compact_fdm_summary.json", summary)
    write_csv(output_root / "compact_fdm_summary.csv", rows)
    write_markdown(output_root / "compact_fdm_summary.md", summary)

    cg_table_dir = resolve_project_path("submissions/CG/tables")
    cg_table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_root / "compact_fdm_summary.csv", cg_table_dir / f"{args.table_prefix}_summary.csv")
    shutil.copy2(output_root / "compact_fdm_summary.json", cg_table_dir / f"{args.table_prefix}_summary.json")
    shutil.copy2(output_root / "compact_fdm_summary.md", cg_table_dir / f"table_{args.table_prefix}_summary.md")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
