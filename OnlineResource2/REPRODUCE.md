# Reproduction and Audit Notes

This package is designed for **numerical audit** of the manuscript's reported
values, not as a self-contained full-reproduction archive. Summary tables in
`tables/` contain all manuscript-reported numbers in machine-readable form.

## What Can Be Audited Directly From This Package

- Manuscript summary fractions and medians from the CSV/JSON files in
  `tables/`.
- Compact direct-vs-hybrid comparisons:
  `tables/cg_compact_method_comparison*.csv/json`.
- Three-seed audit comparisons:
  `tables/cg_seed_audit_method_comparison*.csv/json`.
- Expanded 8-truth FDM baseline:
  `tables/fdm_8truth_summary.csv/json`.
- Truth-point FIM density sweep:
  `tables/fim_density_sweep_summary.csv/json`.
- Recovered-point FIM audit:
  `tables/cg_recovered_fim_combined_summary.json` and
  `tables/cg_recovered_fim_combined_pairs.csv`.

## Aggregate Map

The manuscript uses different case sets for different audit questions. The
same denominator should not be assumed across all results.

| Aggregate | Cases | Main reported values |
|---|---:|---|
| Pilot matched aggregate | 18 | FDM-KLE 15/18 weak-pass; PINN-hybrid 1/18 weak-pass |
| Selector-vs-recovery pilot | 18 | Selector 18/18 hard-pass; recovery 1/18 weak-pass |
| Compact matched matrix | 9 | FDM-KLE 9/9 pass; PINN-hybrid 2/9 weak-pass |
| Three-seed audit | 18 | FDM-KLE 18/18 weak-pass; PINN-hybrid 3/18 weak-pass; selector 17/18 hard-pass |
| Expanded FDM baseline | 24 | FDM-KLE 24/24 pass across eight truths |
| FIM density sweep | 12 | 12/12 full-rank Jacobians |
| Recovered-point FIM | 12 | 12/12 full-rank recovered-point Jacobians |

## Core Modules

The `src/` directory includes the core modules and diagnostic scripts used by
the reported experiments:

| Script | Evidence Layer | Purpose |
|---|---|---|
| `kle.py` | — | KLE parameterization, eigendecomposition, field generation |
| `fdm_solver.py` | E3 | 2D steady-state finite-difference solver |
| `train.py` | — | PINN training framework (Adam + L-BFGS) |
| `train_stage3c.py` | — | Staged inverse training (Phase 1 → Phase 2) |
| `train_stage3c_hybrid.py` | E2, E4 | Multi-round A/B hybrid workflow |
| `diagnose_xi_identifiability.py` | E1 | Gradient-bottleneck diagnostic |
| `analyze_selector_dynamic_range.py` | E2 | Round-selection audit |
| `diagnose_stage3c_staged.py` | E4 | Stage-coupling diagnostic |
| `analyze_kle_sensitivity_fim.py` | E5 | Jacobian/FIM local observability |
| `validate_stage3c_hybrid_closed_loop.py` | — | Closed-loop validation |
| `generate_heterogeneous_field.py` | — | Truth-field generation utility |
| `paths.py` | — | Cross-platform path resolver |

These modules can be inspected for algorithmic logic and invoked individually
with a YAML configuration file (see `configs/`).

## Full Reproduction

Full reproduction of the experiment matrix requires the raw project output tree
(~1.2 GB), which is not included in this compact review package. The final
public DOI archive (Zenodo, upon manuscript acceptance) will include:

- raw truth fields and FDM reference arrays;
- observation files and recovered coefficient arrays;
- per-run `runner_command.json` files;
- full output directories for the compact matrix, seed audit, FIM density
  sweep, recovered-point FIM audit, and expanded FDM baseline;
- orchestration scripts (`run_cg_*.py`, `summarize_*.py`, `Makefile`).
