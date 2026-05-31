#!/usr/bin/env python3
"""Formal Stage 3c hybrid workflow.

Stage A trains/refines h with xi fixed. Stage B samples the trained h onto the
KLE grid and recovers xi with a finite-difference residual. Stage C optionally
runs the closed-loop FDM validation for the recovered K field.
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter

from diagnose_xi_identifiability import cosine, pde_residual_fixed_h
from diagnose_stage3c_staged import evaluate_h_quality
from paths import resolve_project_path
from train_stage3c import (
    generate_synthetic_observations,
    sample_boundary_points,
    sample_pde_points,
    train_stage3c,
)


def load_yaml(path):
    with open(resolve_project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def sample_model_h_on_k_grid(model, x_k, y_k, device):
    x_mesh, y_mesh = np.meshgrid(x_k, y_k, indexing="ij")
    x_t = torch.tensor(x_mesh.ravel(), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_mesh.ravel(), dtype=torch.float32, device=device)
    h = model.predict(x_t, y_t)
    return np.asarray(h).reshape(len(x_k), len(y_k))


def interp_numpy(values, x_grid, y_grid, xq, yq):
    nx = len(x_grid)
    ny = len(y_grid)
    x_norm = (xq[:, None] - x_grid[0]) / (x_grid[-1] - x_grid[0]) * (nx - 1)
    y_norm = (yq[None, :] - y_grid[0]) / (y_grid[-1] - y_grid[0]) * (ny - 1)
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


def configure_stage_a(base_cfg, args, stage_a_dir, xi_init=None, xi_init_file=None):
    cfg = copy.deepcopy(base_cfg)
    cfg["logging"]["output_dir"] = str(stage_a_dir)
    cfg["logging"]["print_every"] = args.print_every
    cfg["logging"]["plot_every"] = args.plot_every

    xi_init = args.xi_init if xi_init is None else xi_init
    xi_init_file = args.xi_init_file if xi_init_file is None else xi_init_file

    cfg["inverse"]["xi_init"] = xi_init
    cfg["inverse"]["freeze_xi"] = True
    cfg["inverse"]["xi_init_std"] = args.xi_init_std
    cfg["inverse"]["xi_init_seed"] = args.seed
    if args.n_obs_points is not None:
        cfg["inverse"]["n_obs_points"] = args.n_obs_points
    if args.obs_noise is not None:
        cfg["inverse"]["obs_noise"] = args.obs_noise
    if xi_init == "true_fraction":
        cfg["inverse"]["xi_init_fraction"] = args.xi_init_fraction
    if xi_init == "file":
        cfg["inverse"]["xi_init_file"] = xi_init_file

    cfg["training"]["seed"] = args.seed
    cfg["training"]["phase1_iterations"] = 0
    cfg["training"]["phase2_iterations"] = args.stage_a_iters
    cfg["training"]["lr_phase2"] = args.stage_a_lr
    cfg["training"]["n_pde_points"] = args.n_pde_points
    cfg["training"]["n_bc_points"] = args.n_bc_points
    cfg["training"]["scheduler_patience"] = max(args.stage_a_iters, 1)
    cfg["training"]["weights"]["pde"] = args.stage_a_pde_weight
    cfg["training"]["weights"]["data"] = args.stage_a_data_weight
    cfg.setdefault("diagnostics", {})["gradient_check"] = False
    return cfg


def optimize_xi_discrete(model, config, out_dir, args, device):
    k_data = np.load(resolve_project_path(config["kle"]["k_field_file"]), allow_pickle=True)
    fdm = np.load(resolve_project_path(config["inverse"]["fdm_reference_path"]), allow_pickle=True)

    x_k = k_data["x"]
    y_k = k_data["y"]
    h_on_k = sample_model_h_on_k_grid(model, x_k, y_k, device)
    if args.h_smooth_sigma > 0.0:
        h_on_k = gaussian_filter(h_on_k, sigma=args.h_smooth_sigma, mode="nearest")
        h_on_k[0, :] = 1.0
        h_on_k[-1, :] = 0.0
    h_fdm_on_k = interp_numpy(np.ascontiguousarray(fdm["h"]), fdm["x"], fdm["y"], x_k, y_k)
    h_rmse_vs_fdm_on_k = float(np.sqrt(np.mean((h_on_k - h_fdm_on_k) ** 2)))

    n_modes = config["inverse"]["n_modes"]
    eig = torch.tensor(k_data["eigenvalues"][:n_modes], dtype=torch.float32, device=device)
    phi = torch.tensor(k_data["eigenfunctions"][:, :n_modes], dtype=torch.float32, device=device)
    xi_true = torch.tensor(k_data["xi"][:n_modes], dtype=torch.float32, device=device)
    h = torch.tensor(h_on_k, dtype=torch.float32, device=device)
    dx = float(x_k[1] - x_k[0])
    dy = float(y_k[1] - y_k[0])

    if args.stage_b_init == "current":
        xi_start = model.xi.detach().clone().to(device)
    elif args.stage_b_init == "zero":
        xi_start = torch.zeros(n_modes, dtype=torch.float32, device=device)
    elif args.stage_b_init == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        xi_start = torch.randn(n_modes, generator=gen, device=device) * args.xi_init_std
    else:
        raise ValueError(f"unknown stage_b_init: {args.stage_b_init}")

    xi = torch.nn.Parameter(xi_start)
    opt = torch.optim.Adam([xi], lr=args.stage_b_lr)
    history = {"epoch": [], "loss": [], "xi_corr": [], "distance_to_true": [], "cos_neg_grad_to_target": []}

    for epoch in range(args.stage_b_iters):
        logK = (phi * torch.sqrt(eig)[None, :] * xi[None, :]).sum(dim=1).reshape(h.shape)
        K = torch.exp(torch.clamp(logK, -3.0, 3.0))
        residual = pde_residual_fixed_h(h, K, dx, dy)
        loss = torch.mean(residual ** 2)

        opt.zero_grad()
        raw_grad = torch.autograd.grad(loss, xi, retain_graph=True)[0].detach()
        target = xi_true - xi.detach()
        cos_neg = cosine(-raw_grad, target)
        loss.backward()
        opt.step()

        with torch.no_grad():
            xi_np = xi.detach().cpu().numpy()
            xi_true_np = xi_true.detach().cpu().numpy()
            corr = float(np.corrcoef(xi_np, xi_true_np)[0, 1])
            dist = float(torch.linalg.norm(xi_true - xi.detach()).item())
        if epoch % max(1, args.stage_b_iters // 10) == 0 or epoch == args.stage_b_iters - 1:
            print(
                f"[Stage B] epoch {epoch:5d} | loss {loss.item():.4e} | "
                f"xi_corr {corr:.4f} | dist {dist:.4f} | cos {cos_neg}"
            )
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["xi_corr"].append(corr)
        history["distance_to_true"].append(dist)
        history["cos_neg_grad_to_target"].append(cos_neg)

    with torch.no_grad():
        final_logK = (phi * torch.sqrt(eig)[None, :] * xi[None, :]).sum(dim=1).reshape(h.shape)
        xi_rec = xi.detach().cpu().numpy()
        xi_true_np = xi_true.detach().cpu().numpy()
        logK_np = final_logK.detach().cpu().numpy()
        true_logK = k_data["logK"]
        result = {
            "stage_b_init": args.stage_b_init,
            "h_smooth_sigma": args.h_smooth_sigma,
            "h_rmse_vs_fdm_on_k": h_rmse_vs_fdm_on_k,
            "iterations": args.stage_b_iters,
            "lr": args.stage_b_lr,
            "final_loss": history["loss"][-1],
            "final_xi_corr": history["xi_corr"][-1],
            "final_distance_to_true": history["distance_to_true"][-1],
            "final_logK_rmse": float(np.sqrt(np.mean((logK_np - true_logK) ** 2))),
            "xi_true": xi_true_np.tolist(),
            "xi_pred": xi_rec.tolist(),
            "history": history,
        }

    np.savez(
        out_dir / "recovered_xi_logK.npz",
        xi=xi_rec,
        xi_true=xi_true_np,
        logK=logK_np,
        logK_true=true_logK,
        h_on_k=h_on_k,
        x=x_k,
        y=y_k,
    )
    with torch.no_grad():
        model.xi.copy_(torch.tensor(xi_rec, dtype=torch.float32, device=device))
    torch.save(model.state_dict(), out_dir / "model_with_discrete_xi.pt", _use_new_zipfile_serialization=False)
    with open(out_dir / "stage_b_discrete_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_stage_a_continuation(model, config, out_dir, args, device):
    if args.stage_a_cont_iters <= 0:
        return None

    print("\n=== Stage A continuation: scheduled PDE refinement ===")
    for p in model.net.parameters():
        p.requires_grad_(True)
    model.xi.requires_grad_(False)
    model.train()

    x_obs, y_obs, h_obs = generate_synthetic_observations(config, device)
    n_pde = config["training"]["n_pde_points"]
    n_bc_per_side = config["training"]["n_bc_points"] // 4
    weights = config["training"]["weights"]
    w_bc_d = weights.get("bc_dirichlet", weights.get("bc", 1.0))
    w_bc_n = weights.get("bc_neumann", weights.get("bc", 1.0))
    w_data = args.stage_a_cont_data_weight

    opt = torch.optim.Adam(
        [p for p in model.net.parameters() if p.requires_grad],
        lr=args.stage_a_cont_lr,
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    history = {
        "epoch": [],
        "pde_weight": [],
        "loss_total": [],
        "loss_pde": [],
        "loss_bc_dirichlet": [],
        "loss_bc_neumann": [],
        "loss_data": [],
    }

    for epoch in range(args.stage_a_cont_iters):
        if args.stage_a_cont_iters == 1:
            alpha = 1.0
        else:
            alpha = epoch / (args.stage_a_cont_iters - 1)
        w_pde = args.stage_a_cont_pde_start + alpha * (
            args.stage_a_cont_pde_end - args.stage_a_cont_pde_start
        )

        x_pde, y_pde = sample_pde_points(config, n_pde, device)
        bc_points = sample_boundary_points(config, n_bc_per_side, device)

        loss_pde = torch.mean(model.pde_residual(x_pde, y_pde) ** 2)
        loss_bc_d = torch.mean(
            model.bc_dirichlet(
                bc_points["dirichlet"]["x"],
                bc_points["dirichlet"]["y"],
                bc_points["dirichlet"]["h"],
            )
        )
        loss_bc_n = torch.mean(
            model.bc_neumann(
                bc_points["neumann"]["x"],
                bc_points["neumann"]["y"],
            )
        )
        loss_data = torch.mean(model.data_loss(x_obs, y_obs, h_obs))
        loss = w_pde * loss_pde + w_bc_d * loss_bc_d + w_bc_n * loss_bc_n + w_data * loss_data

        opt.zero_grad()
        loss.backward()
        clip = config["training"].get("gradient_clip")
        if clip:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), clip)
        opt.step()

        if epoch % max(1, args.stage_a_cont_iters // 10) == 0 or epoch == args.stage_a_cont_iters - 1:
            print(
                f"[Stage A cont] epoch {epoch:5d} | w_pde {w_pde:.3e} | "
                f"loss {loss.item():.4e} | PDE {loss_pde.item():.2e} | "
                f"data {loss_data.item():.2e}"
            )
        history["epoch"].append(epoch)
        history["pde_weight"].append(float(w_pde))
        history["loss_total"].append(float(loss.item()))
        history["loss_pde"].append(float(loss_pde.item()))
        history["loss_bc_dirichlet"].append(float(loss_bc_d.item()))
        history["loss_bc_neumann"].append(float(loss_bc_n.item()))
        history["loss_data"].append(float(loss_data.item()))

    torch.save(
        model.state_dict(),
        out_dir / "model_after_stage_a_continuation.pt",
        _use_new_zipfile_serialization=False,
    )
    with open(out_dir / "stage_a_continuation_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def run_closed_loop(out_dir, config, args):
    closed_loop_dir = out_dir / "stage_c_closed_loop"
    cmd = [
        sys.executable,
        str(resolve_project_path("src/validate_stage3c_hybrid_closed_loop.py")),
        "--recovered",
        str(out_dir / "stage_b_discrete" / "recovered_xi_logK.npz"),
        "--k-field",
        str(resolve_project_path(config["kle"]["k_field_file"])),
        "--fdm-true",
        str(resolve_project_path(config["inverse"]["fdm_reference_path"])),
        "--output-dir",
        str(closed_loop_dir),
        "--config",
        str(out_dir / "stage_a_config.yaml"),
        "--N",
        str(args.closed_loop_N),
    ]
    subprocess.run(cmd, check=True)
    metrics_path = closed_loop_dir / "closed_loop_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_round(args, config_path, out_dir, xi_init, xi_init_file, cont_history, h_quality, stage_b, closed_loop, round_index=None):
    summary = {
        "config": str(resolve_project_path(config_path)),
        "output_dir": str(out_dir),
        "round": round_index,
        "xi_init": xi_init,
        "xi_init_file": None if xi_init_file is None else str(resolve_project_path(xi_init_file)),
        "stage_b_init": args.stage_b_init,
        "stage_a_continuation": {
            "enabled": cont_history is not None,
            "iterations": args.stage_a_cont_iters,
            "pde_start": args.stage_a_cont_pde_start,
            "pde_end": args.stage_a_cont_pde_end,
        },
        "stage_a": h_quality,
        "stage_b": {
            "h_rmse_vs_fdm_on_k": stage_b["h_rmse_vs_fdm_on_k"],
            "final_xi_corr": stage_b["final_xi_corr"],
            "final_distance_to_true": stage_b["final_distance_to_true"],
            "final_logK_rmse": stage_b["final_logK_rmse"],
            "final_loss": stage_b["final_loss"],
        },
        "stage_c": {
            "h_rmse_vs_true_fdm": closed_loop["h_closed_loop"]["rmse_vs_true_fdm"],
            "h_max_abs_vs_true_fdm": closed_loop["h_closed_loop"]["max_abs_vs_true_fdm"],
            "h_r2_vs_true_fdm": closed_loop["h_closed_loop"]["r2_vs_true_fdm"],
            "obs_rmse": None
            if closed_loop.get("observation_misfit") is None
            else closed_loop["observation_misfit"]["rmse"],
            "logK_rmse": closed_loop["K"]["logK_rmse"],
            "xi_corr": closed_loop["xi"]["corr"],
        },
    }
    return summary


def selection_score(summary, metric):
    if metric == "oracle_h_rmse":
        return summary["stage_c"]["h_rmse_vs_true_fdm"]
    if metric == "observation_rmse":
        value = summary["stage_c"].get("obs_rmse")
        if value is None:
            raise ValueError("observation_rmse requested but closed-loop observation_misfit is missing")
        return value
    if metric == "stage_b_loss":
        return summary["stage_b"]["final_loss"]
    raise ValueError(f"unknown alternating selection metric: {metric}")


def run_hybrid_round(base_cfg, args, round_dir, device, xi_init, xi_init_file=None, round_index=None):
    round_dir.mkdir(parents=True, exist_ok=True)
    stage_a_dir = round_dir / "stage_a_h"
    stage_b_dir = round_dir / "stage_b_discrete"
    stage_b_dir.mkdir(parents=True, exist_ok=True)

    stage_a_cfg = configure_stage_a(base_cfg, args, stage_a_dir, xi_init=xi_init, xi_init_file=xi_init_file)
    save_yaml(stage_a_cfg, round_dir / "stage_a_config.yaml")

    label = "" if round_index is None else f" round {round_index}"
    print(f"\n=== Stage A{label}: fixed-xi h training ===")
    model, _ = train_stage3c(stage_a_cfg)
    cont_history = run_stage_a_continuation(model, stage_a_cfg, stage_a_dir, args, device)

    quality_dir = round_dir / "stage_a_quality"
    quality_dir.mkdir(exist_ok=True)
    h_quality = evaluate_h_quality(model, stage_a_cfg, quality_dir, device)
    with open(round_dir / "stage_a_h_quality.json", "w", encoding="utf-8") as f:
        json.dump(h_quality, f, indent=2)

    print(f"\n=== Stage B{label}: discrete xi recovery ===")
    stage_b = optimize_xi_discrete(model, stage_a_cfg, stage_b_dir, args, device)

    print(f"\n=== Stage C{label}: closed-loop FDM validation ===")
    closed_loop = run_closed_loop(round_dir, stage_a_cfg, args)

    summary = summarize_round(
        args,
        args.config,
        round_dir,
        xi_init,
        xi_init_file,
        cont_history,
        h_quality,
        stage_b,
        closed_loop,
        round_index=round_index,
    )
    with open(round_dir / "hybrid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def write_best_round_artifacts(out_dir, best_summary):
    best_dir = out_dir / "best_round"
    best_dir.mkdir(parents=True, exist_ok=True)
    source_dir = resolve_project_path(best_summary["output_dir"])
    artifacts = [
        ("hybrid_summary.json", "hybrid_summary.json"),
        ("stage_b_discrete/recovered_xi_logK.npz", "recovered_xi_logK.npz"),
        ("stage_b_discrete/stage_b_discrete_results.json", "stage_b_discrete_results.json"),
        ("stage_c_closed_loop/closed_loop_metrics.json", "closed_loop_metrics.json"),
        ("stage_c_closed_loop/closed_loop_validation.png", "closed_loop_validation.png"),
    ]
    for rel_src, rel_dst in artifacts:
        src = source_dir / rel_src
        if src.exists():
            shutil.copy2(src, best_dir / rel_dst)
    with open(best_dir / "best_round_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)


def run_alternating(base_cfg, args, out_dir, device):
    rounds = []
    best_summary = None
    best_score = None
    prev_score = None
    worsen_count = 0
    stop_reason = "completed"
    next_xi_init = args.xi_init
    next_xi_file = args.xi_init_file

    for round_index in range(1, args.alt_rounds + 1):
        round_dir = out_dir / f"round_{round_index:02d}"
        summary = run_hybrid_round(
            base_cfg,
            args,
            round_dir,
            device,
            xi_init=next_xi_init,
            xi_init_file=next_xi_file,
            round_index=round_index,
        )
        score = selection_score(summary, args.alt_selection_metric)
        improved = best_score is None or score < best_score - args.alt_min_delta
        if improved:
            best_summary = summary
            best_score = score

        round_record = {
            "round": round_index,
            "output_dir": summary["output_dir"],
            "xi_init": summary["xi_init"],
            "xi_init_file": summary["xi_init_file"],
            "stage_a_h_rmse": summary["stage_a"]["h_rmse"],
            "stage_b_final_loss": summary["stage_b"]["final_loss"],
            "stage_b_xi_corr": summary["stage_b"]["final_xi_corr"],
            "stage_b_logK_rmse": summary["stage_b"]["final_logK_rmse"],
            "selection_score": score,
            "stage_c_h_rmse": summary["stage_c"]["h_rmse_vs_true_fdm"],
            "stage_c_obs_rmse": summary["stage_c"].get("obs_rmse"),
            "stage_c_h_r2": summary["stage_c"]["h_r2_vs_true_fdm"],
            "is_best_so_far": improved,
        }
        rounds.append(round_record)

        next_xi_init = "file"
        next_xi_file = str(round_dir / "stage_b_discrete" / "recovered_xi_logK.npz")

        if prev_score is not None and args.alt_stop_patience > 0:
            threshold = args.alt_min_delta + args.alt_worsen_rel * max(abs(prev_score), 1e-12)
            if score > prev_score + threshold:
                worsen_count += 1
            else:
                worsen_count = 0
            if worsen_count >= args.alt_stop_patience:
                stop_reason = (
                    f"early_stop: round {round_index} {args.alt_selection_metric} "
                    f"{score:.6g} worsened from previous {prev_score:.6g}"
                )
                break
        prev_score = score

    result = {
        "config": str(resolve_project_path(args.config)),
        "output_dir": str(out_dir),
        "rounds_requested": args.alt_rounds,
        "rounds_completed": len(rounds),
        "selection_metric": args.alt_selection_metric,
        "early_stop": {
            "patience": args.alt_stop_patience,
            "min_delta": args.alt_min_delta,
            "worsen_rel": args.alt_worsen_rel,
            "stop_reason": stop_reason,
        },
        "best_round": None if best_summary is None else best_summary["round"],
        "best_output_dir": None if best_summary is None else best_summary["output_dir"],
        "best_selection_score": best_score,
        "best_stage_c_h_rmse": None
        if best_summary is None
        else best_summary["stage_c"]["h_rmse_vs_true_fdm"],
        "best_stage_c_obs_rmse": None
        if best_summary is None
        else best_summary["stage_c"].get("obs_rmse"),
        "best_summary": best_summary,
        "rounds": rounds,
    }
    with open(out_dir / "alternating_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if best_summary is not None:
        write_best_round_artifacts(out_dir, best_summary)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3c_fixed_xi_true_h_only.yaml")
    parser.add_argument("--output-dir", default="outputs/stage3c_hybrid_zero_smoke")
    parser.add_argument("--xi-init", choices=["zero", "random", "true_fraction", "file"], default="zero")
    parser.add_argument("--xi-init-file", default=None)
    parser.add_argument("--xi-init-std", type=float, default=0.1)
    parser.add_argument("--xi-init-fraction", type=float, default=1.0)
    parser.add_argument("--stage-b-init", choices=["current", "zero", "random"], default="current")
    parser.add_argument("--n-obs-points", type=int, default=None)
    parser.add_argument("--obs-noise", type=float, default=None)
    parser.add_argument("--stage-a-iters", type=int, default=500)
    parser.add_argument("--stage-a-lr", type=float, default=5e-4)
    parser.add_argument("--stage-a-pde-weight", type=float, default=1.0)
    parser.add_argument("--stage-a-data-weight", type=float, default=100.0)
    parser.add_argument("--stage-a-cont-iters", type=int, default=0)
    parser.add_argument("--stage-a-cont-lr", type=float, default=3e-4)
    parser.add_argument("--stage-a-cont-pde-start", type=float, default=0.0)
    parser.add_argument("--stage-a-cont-pde-end", type=float, default=1.0)
    parser.add_argument("--stage-a-cont-data-weight", type=float, default=100.0)
    parser.add_argument("--stage-b-iters", type=int, default=1000)
    parser.add_argument("--stage-b-lr", type=float, default=1e-2)
    parser.add_argument("--h-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--n-pde-points", type=int, default=2000)
    parser.add_argument("--n-bc-points", type=int, default=400)
    parser.add_argument("--closed-loop-N", type=int, default=201)
    parser.add_argument("--alt-rounds", type=int, default=1)
    parser.add_argument(
        "--alt-selection-metric",
        choices=["oracle_h_rmse", "observation_rmse", "stage_b_loss"],
        default="oracle_h_rmse",
        help=(
            "Metric used to select the best alternating round. "
            "Use observation_rmse for no-truth synthetic selection; "
            "oracle_h_rmse is diagnostic only."
        ),
    )
    parser.add_argument("--alt-stop-patience", type=int, default=1)
    parser.add_argument("--alt-min-delta", type=float, default=0.0)
    parser.add_argument("--alt-worsen-rel", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--plot-every", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(args.config)
    if args.alt_rounds > 1:
        run_alternating(base_cfg, args, out_dir, device)
    else:
        run_hybrid_round(
            base_cfg,
            args,
            out_dir,
            device,
            xi_init=args.xi_init,
            xi_init_file=args.xi_init_file,
            round_index=None,
        )


if __name__ == "__main__":
    main()
