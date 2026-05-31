#!/usr/bin/env python3
"""Summarize no-truth selector dynamic range for alternating runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from paths import resolve_project_path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def paired_round_values(rows, key_a, key_b):
    pairs = []
    for row in rows:
        a = row.get(key_a)
        b = row.get(key_b)
        if a is None or b is None:
            continue
        a = float(a)
        b = float(b)
        if np.isfinite(a) and np.isfinite(b):
            pairs.append((int(row.get("round", len(pairs) + 1)), a, b))
    return pairs


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return None
    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def best_round(rows, metric):
    candidates = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            candidates.append((value, int(row.get("round", len(candidates) + 1)), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def classify_selector(range_ratio, rho, regret):
    if range_ratio is None or regret is None:
        return {
            "class": "insufficient_data",
            "reason": "missing range ratio or regret",
        }
    if range_ratio >= 0.5 and (rho is None or rho >= 0.8) and regret <= 0.002:
        return {
            "class": "hard_pass",
            "reason": "range ratio >= 0.5, rank agreement acceptable, and regret <= 0.002",
        }
    if range_ratio >= 0.25 and regret <= 0.003:
        return {
            "class": "weak_pass",
            "reason": "range ratio >= 0.25 and regret <= 0.003",
        }
    return {
        "class": "fail",
        "reason": "range ratio < 0.25 or regret > 0.003",
    }


def describe(values):
    if values.size == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "range": None,
            "mean": None,
            "std": None,
        }
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def summarize(path):
    data = load_json(path)
    rows = data.get("rounds", [])
    obs = describe(finite_values(rows, "stage_c_obs_rmse"))
    oracle = describe(finite_values(rows, "stage_c_h_rmse"))
    stage_b = describe(finite_values(rows, "stage_b_final_loss"))
    ratio = None
    if obs["range"] is not None and oracle["range"] not in (None, 0.0):
        ratio = float(obs["range"] / oracle["range"])

    pairs = paired_round_values(rows, "stage_c_obs_rmse", "stage_c_h_rmse")
    rho = None if not pairs else spearman(
        np.asarray([p[1] for p in pairs], dtype=float),
        np.asarray([p[2] for p in pairs], dtype=float),
    )
    obs_best = best_round(rows, "stage_c_obs_rmse")
    oracle_best = best_round(rows, "stage_c_h_rmse")
    regret = None
    if obs_best is not None and oracle_best is not None:
        regret = float(obs_best["stage_c_h_rmse"] - oracle_best["stage_c_h_rmse"])
    classification = classify_selector(ratio, rho, regret)

    return {
        "summary_path": str(path),
        "selection_metric": data.get("selection_metric"),
        "rounds_completed": data.get("rounds_completed"),
        "best_round": data.get("best_round"),
        "stop_reason": data.get("early_stop", {}).get("stop_reason"),
        "stage_c_obs_rmse": obs,
        "oracle_stage_c_h_rmse": oracle,
        "stage_b_final_loss": stage_b,
        "obs_range_to_oracle_range": ratio,
        "spearman_obs_vs_oracle_h": rho,
        "observation_selected_round": None if obs_best is None else obs_best.get("round"),
        "oracle_best_round": None if oracle_best is None else oracle_best.get("round"),
        "oracle_h_rmse_regret": regret,
        "selector_classification": classification,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alternating_summary")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = resolve_project_path(args.alternating_summary)
    result = summarize(path)

    if args.output:
        out = resolve_project_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
