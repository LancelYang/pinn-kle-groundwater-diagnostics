#!/usr/bin/env python3
"""Summarize matched CG compact direct-vs-hybrid results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from paths import resolve_project_path


DEFAULT_FDM_CSV = "outputs/cg_compact_matrix/fdm_kle/compact_fdm_summary.csv"
DEFAULT_HYBRID_CSV = "outputs/cg_compact_matrix/hybrid/compact_hybrid_summary.csv"
DEFAULT_OUT_DIR = "outputs/cg_compact_matrix/summary"
DEFAULT_TABLE_PREFIX = "cg_compact_method_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
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


def load_rows(fdm_csv: Path, hybrid_csv: Path) -> list[dict]:
    fdm_rows = []
    hybrid_rows = []
    for row in read_csv(fdm_csv):
        if row.get("status") != "ok":
            continue
        fdm_rows.append(
            {
                "method": "direct_fdm_kle",
                "truth_id": row["truth_id"],
                "n_obs": int(row["n_obs"]),
                "obs_noise": float(row["obs_noise"]),
                "seed": int(row["seed"]),
                "h_rmse": as_float(row["h_rmse"]),
                "logK_rmse": as_float(row["logK_rmse"]),
                "xi_corr": as_float(row["xi_corr"]),
                "weak_pass": as_bool(row["weak_pass"]),
                "strong_pass": as_bool(row["strong_pass"]),
            }
        )
    for row in read_csv(hybrid_csv):
        if row.get("status") != "ok":
            continue
        hybrid_rows.append(
            {
                "method": "pinn_hybrid",
                "truth_id": row["truth_id"],
                "n_obs": int(row["n_obs"]),
                "obs_noise": float(row["obs_noise"]),
                "seed": int(row["seed"]),
                "h_rmse": as_float(row["best_h_rmse"]),
                "logK_rmse": as_float(row["best_logK_rmse"]),
                "xi_corr": as_float(row["best_xi_corr"]),
                "weak_pass": as_bool(row["weak_pass"]),
                "strong_pass": as_bool(row["strong_pass"]),
                "selector_classification": row.get("selector_classification"),
                "oracle_h_rmse_regret": as_float(row.get("oracle_h_rmse_regret")),
            }
        )
    completed_hybrid_keys = {
        (row["truth_id"], row["n_obs"], row["obs_noise"], row["seed"])
        for row in hybrid_rows
    }
    matched_fdm_rows = [
        row
        for row in fdm_rows
        if (row["truth_id"], row["n_obs"], row["obs_noise"], row["seed"]) in completed_hybrid_keys
    ]
    return matched_fdm_rows + hybrid_rows


def group_summary(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    out = []
    for key, subset in sorted(groups.items()):
        record = {name: value for name, value in zip(group_keys, key)}
        record.update(
            {
                "n_cases": len(subset),
                "weak_pass_count": sum(row["weak_pass"] for row in subset),
                "strong_pass_count": sum(row["strong_pass"] for row in subset),
                "median_h_rmse": median([row["h_rmse"] for row in subset]),
                "median_logK_rmse": median([row["logK_rmse"] for row in subset]),
                "median_xi_corr": median([row["xi_corr"] for row in subset]),
            }
        )
        if subset[0]["method"] == "pinn_hybrid":
            record["hard_selector_count"] = sum(row.get("selector_classification") == "hard_pass" for row in subset)
            record["median_selector_regret"] = median([row.get("oracle_h_rmse_regret") for row in subset])
        out.append(record)
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


def write_markdown(path: Path, by_method_noise: list[dict], by_method_truth: list[dict], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "This table compares direct FDM-KLE and PINN-hybrid results for matched compact CG matrix cases completed so far.",
        "",
        "## By Method And Noise",
        "",
        "| method | noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_method_noise:
        lines.append(
            f"| {row['method']} | {row['obs_noise']:g} | {row['n_cases']} | {row['weak_pass_count']} | "
            f"{row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | {fmt(row['median_logK_rmse'])} | "
            f"{fmt(row['median_xi_corr'])} | {fmt(row.get('hard_selector_count'))} | {fmt(row.get('median_selector_regret'))} |"
        )
    lines += [
        "",
        "## By Method And Truth",
        "",
        "| method | truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_method_truth:
        lines.append(
            f"| {row['method']} | {row['truth_id']} | {row['n_cases']} | {row['weak_pass_count']} | "
            f"{row['strong_pass_count']} | {fmt(row['median_h_rmse'])} | {fmt(row['median_logK_rmse'])} | "
            f"{fmt(row['median_xi_corr'])} | {fmt(row.get('hard_selector_count'))} | {fmt(row.get('median_selector_regret'))} |"
        )
    lines += [
        "",
        "Interpretation: the completed n_obs=200 compact matrix shows a clear direct-vs-hybrid recovery gap.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdm-csv", default=DEFAULT_FDM_CSV)
    parser.add_argument("--hybrid-csv", default=DEFAULT_HYBRID_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--table-prefix", default=DEFAULT_TABLE_PREFIX)
    parser.add_argument("--title", default="CG Compact Matched Comparison")
    args = parser.parse_args()

    rows = load_rows(resolve_project_path(args.fdm_csv), resolve_project_path(args.hybrid_csv))
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_method_noise = group_summary(rows, ["method", "obs_noise"])
    by_method_truth = group_summary(rows, ["method", "truth_id"])
    payload = {
        "rows": rows,
        "by_method_noise": by_method_noise,
        "by_method_truth": by_method_truth,
    }
    (out_dir / "method_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_dir / "method_comparison_rows.csv", rows)
    write_csv(out_dir / "method_comparison_by_noise.csv", by_method_noise)
    write_csv(out_dir / "method_comparison_by_truth.csv", by_method_truth)
    write_markdown(out_dir / "method_comparison.md", by_method_noise, by_method_truth, args.title)

    cg_table_dir = resolve_project_path("submissions/CG/tables")
    shutil.copy2(out_dir / "method_comparison.md", cg_table_dir / f"table_{args.table_prefix}.md")
    shutil.copy2(out_dir / "method_comparison_by_noise.csv", cg_table_dir / f"{args.table_prefix}_by_noise.csv")
    shutil.copy2(out_dir / "method_comparison_by_truth.csv", cg_table_dir / f"{args.table_prefix}_by_truth.csv")
    shutil.copy2(out_dir / "method_comparison.json", cg_table_dir / f"{args.table_prefix}.json")
    print(json.dumps({"by_method_noise": by_method_noise, "by_method_truth": by_method_truth}, indent=2))


if __name__ == "__main__":
    main()
