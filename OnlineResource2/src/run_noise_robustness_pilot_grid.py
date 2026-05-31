#!/usr/bin/env python3
"""Run and summarize a small P0 noise-robustness pilot grid.

This script intentionally runs a compact pilot before the full review-hardening
matrix. Each run uses observation-RMSE round selection and keeps oracle metrics
only as diagnostics for selector auditing.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from analyze_selector_dynamic_range import summarize as summarize_selector
from paths import resolve_project_path


DEFAULT_CASES = [
    (200, 0.005, 202),
    (200, 0.005, 303),
    (200, 0.05, 202),
    (200, 0.05, 303),
    (100, 0.05, 202),
    (100, 0.05, 303),
]


SUMMARY_FIELDS = [
    "status",
    "n_obs",
    "obs_noise",
    "seed",
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


def parse_case(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be n_obs:noise:seed")
    try:
        return int(parts[0]), float(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be n_obs:noise:seed") from exc


def noise_label(noise):
    if noise == 0:
        return "noise_0"
    label = f"{noise:.6f}".rstrip("0").rstrip(".")
    if label.startswith("0."):
        return "noise_" + label[2:]
    return "noise_" + label.replace(".", "p")


def output_dir_for_case(root, n_obs, noise, seed):
    return root / "3mode" / f"obs_{n_obs}" / noise_label(noise) / f"seed_{seed}"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bool_from_metrics(h_rmse, logk_rmse, strong=False):
    if h_rmse is None or logk_rmse is None:
        return None
    if strong:
        return bool(logk_rmse < 0.10 and h_rmse < 0.003)
    return bool(logk_rmse < 0.15 and h_rmse < 0.006)


def row_from_outputs(out_dir, n_obs, noise, seed, status="ok", failure_reason=""):
    alt_path = out_dir / "alternating_summary.json"
    selector_path = out_dir / "selector_dynamic_range.json"
    if not alt_path.exists():
        return {
            "status": "failed",
            "n_obs": n_obs,
            "obs_noise": noise,
            "seed": seed,
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
        "n_obs": n_obs,
        "obs_noise": noise,
        "seed": seed,
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(cmd, cwd, stdout_path, stderr_path, dry_run=False):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(str(part) for part in cmd)
    if dry_run:
        return {"returncode": 0, "command": command_text, "dry_run": True}
    with open(stdout_path, "w", encoding="utf-8") as stdout, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "returncode": completed.returncode,
        "command": command_text,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def train_command(args, out_dir, n_obs, noise, seed):
    return [
        sys.executable,
        str(resolve_project_path("src/train_stage3c_hybrid.py")),
        "--config",
        str(resolve_project_path(args.config)),
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


def aggregate(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    return {
        "n_cases": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "selector_classes": {
            label: sum(1 for row in ok_rows if row.get("selector_classification") == label)
            for label in ["hard_pass", "weak_pass", "fail", "insufficient_data"]
        },
        "selector_unknown_count": sum(
            1 for row in ok_rows if row.get("selector_classification") is None
        ),
        "weak_pass_count": sum(1 for row in ok_rows if row.get("weak_pass") is True),
        "strong_pass_count": sum(1 for row in ok_rows if row.get("strong_pass") is True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3c_hybrid_3mode.yaml")
    parser.add_argument("--output-root", default="outputs/noise_robustness_pilot")
    parser.add_argument("--case", action="append", type=parse_case, default=None)
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

    root = resolve_project_path(args.output_root)
    project_root = resolve_project_path(".")
    cases = args.case if args.case else DEFAULT_CASES
    rows = []

    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve_project_path(args.config)),
        "output_root": str(root),
        "cases": [
            {"n_obs": n_obs, "obs_noise": noise, "seed": seed}
            for n_obs, noise, seed in cases
        ],
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
    write_json(root / "pilot_manifest.json", manifest)

    for index, (n_obs, noise, seed) in enumerate(cases, start=1):
        out_dir = output_dir_for_case(root, n_obs, noise, seed)
        alt_path = out_dir / "alternating_summary.json"
        print(
            f"[{index}/{len(cases)}] n_obs={n_obs} noise={noise:g} seed={seed} -> {out_dir}"
        )
        if alt_path.exists() and not args.rerun_existing:
            print("  existing alternating_summary.json found; summarizing")
            rows.append(row_from_outputs(out_dir, n_obs, noise, seed))
            continue

        cmd = train_command(args, out_dir, n_obs, noise, seed)
        run_info = run_command(
            cmd,
            project_root,
            out_dir / "runner_train_stdout.log",
            out_dir / "runner_train_stderr.log",
            dry_run=args.dry_run,
        )
        write_json(out_dir / "runner_train_command.json", run_info)
        if args.dry_run:
            rows.append(
                {
                    "status": "dry_run",
                    "n_obs": n_obs,
                    "obs_noise": noise,
                    "seed": seed,
                    "output_dir": str(out_dir),
                    "failure_reason": "",
                }
            )
            continue
        if run_info["returncode"] != 0:
            reason = f"training command failed with return code {run_info['returncode']}"
            print(f"  {reason}")
            rows.append(row_from_outputs(out_dir, n_obs, noise, seed, status="failed", failure_reason=reason))
            continue

        selector = summarize_selector(alt_path)
        write_json(out_dir / "selector_dynamic_range.json", selector)
        row = row_from_outputs(out_dir, n_obs, noise, seed)
        rows.append(row)
        print(
            "  class={selector_classification} selected={selected_round} "
            "oracle={oracle_best_round} h={best_h_rmse:.6g} logK={best_logK_rmse:.6g}".format(
                **row
            )
        )

    result = {
        "manifest": manifest,
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    write_json(root / "pilot_summary.json", result)
    write_csv(root / "pilot_summary.csv", rows)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
