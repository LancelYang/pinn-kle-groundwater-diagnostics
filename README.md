# PINN-KLE Groundwater Inverse Diagnostics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)]()

Companion repository for:

> **"Separating linearized observability, direct recoverability, and workflow limitations in physics-informed neural network inversion"**
>
> Fei Yang — *Computational Geosciences* (submitted)

## Overview

This repository contains the manuscript, supplementary materials, and computational framework for a controlled numerical diagnostic study of PINN-KLE conductivity inversion under steady two-dimensional groundwater flow. The study systematically separates four concerns—local information content, gradient transfer, round-selection quality, and workflow-limited recovery—and demonstrates that hybrid workflow failures are workflow-limited rather than information-limited.

### Key Findings

| Evidence | Finding |
|---|---|
| E1: Gradient bottleneck | Head-observation loss has no direct gradient pathway to KLE coefficients |
| E2: Selector ≠ Recovery | No-truth selector identifies best round in 17/18 cases, yet only 3/18 weak-pass recovery |
| E3: FDM-KLE baseline | Direct FDM-KLE baseline weak-passes 24/24 cases across 8 truth realizations |
| E4: Budget defense | Stage-B-led degradation with A/B coupling |
| E5: Local observability | Full-rank local observability at hybrid-recovered parameters (cond-number ratio 0.919–1.010) |

## Repository Structure

```
├── manuscript/                        # Submitted manuscript (LaTeX)
│   ├── manuscript_cg.tex              # Main manuscript
│   ├── references.bib                 # Bibliography
│   ├── sn-jnl.cls                     # Springer journal class
│   ├── sn-mathphys.bst               # Springer bibliography style
│   └── figures/                       # Manuscript figures (Fig1–4)
├── OnlineResource1/                   # Online Resource 1 (method supplement)
│   ├── OnlineResource1.tex
│   └── OnlineResource1.pdf
├── OnlineResource2/                   # Online Resource 2 (data & code package)
│   ├── README.md                      # Package description
│   ├── REPRODUCE.md                   # Reproduction and audit guide
│   ├── requirements.txt               # Python dependencies
│   ├── src/                           # Core modules & diagnostic scripts
│   ├── configs/                       # YAML experiment configurations
│   ├── tables/                        # CSV/JSON/Markdown summary tables
│   └── figures/                       # Diagnostic figures
├── cover_letter.md                    # Submission cover letter
├── requirements.txt                    # Root-level Python dependencies
└── LICENSE                            # MIT License
```

## Quick Start

### Installation

```bash
git clone https://github.com/LancelYang/pinn-kle-groundwater-diagnostics.git
cd pinn-kle-groundwater-diagnostics
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### Inspect Summary Tables

All manuscript-reported values are available as machine-readable files:

```bash
# Compact matched comparison (FDM vs hybrid)
cat OnlineResource2/tables/cg_compact_method_comparison.csv

# Three-seed audit
cat OnlineResource2/tables/cg_seed_audit_method_comparison.csv

# Expanded FDM baseline (8 truths, 24 cases)
cat OnlineResource2/tables/fdm_8truth_summary.csv

# FIM density sweep
cat OnlineResource2/tables/fim_density_sweep_summary.csv

# Recovered-point FIM audit
cat OnlineResource2/tables/cg_recovered_fim_combined_pairs.csv
```

### Run a PINN-Hybrid Experiment

The core training modules can be invoked directly with a YAML configuration:

```bash
cd OnlineResource2/src
python train_stage3c_hybrid.py --config ../configs/stage3c_hybrid_3mode.yaml
```

This runs a single PINN-hybrid round. Full matrix experiments require the
raw truth-field data (see Data Availability below).

## Core Modules

| Module | Purpose |
|---|---|
| `kle.py` | Karhunen-Loève expansion: covariance construction, eigendecomposition, truncated field generation |
| `fdm_solver.py` | 2D steady-state finite-difference solver for Darcy flow with sparse direct solve |
| `train.py` | PINN training framework (Adam + L-BFGS, PDE/BC/IC losses, K-field modes) |
| `train_stage3c.py` | Staged inverse training: Phase 1 (fixed ξ) → Phase 2 (joint MLP + ξ) |
| `train_stage3c_hybrid.py` | Multi-round alternating Stage A (h) / Stage B (ξ) workflow |
| `diagnose_xi_identifiability.py` | Gradient-bottleneck diagnostic (E1) |
| `analyze_selector_dynamic_range.py` | Round-selection audit (E2) |
| `diagnose_stage3c_staged.py` | Stage-coupling diagnostic (E4) |
| `analyze_kle_sensitivity_fim.py` | Jacobian/FIM local observability (E5) |
| `validate_stage3c_hybrid_closed_loop.py` | Closed-loop validation of hybrid workflow |
| `paths.py` | Cross-platform path resolver for Windows/macOS compatibility |

## Diagnostic Framework

The five-layer diagnostic framework:

1. **Gradient audit** — Component-wise gradient from head-observation loss to KLE coefficients
2. **Round-selection audit** — No-truth selector vs. conductivity recovery rate
3. **Direct FDM-KLE baseline** — Deterministic least-squares inversion as matched comparison
4. **Degradation driver analysis** — Stage-B-led budget and A/B coupling diagnostics
5. **Local observability (Jacobian/FIM)** — Full-rank checks, condition numbers, density sweeps

## Data Availability

- **Summary tables** (CSV/JSON) are included in `OnlineResource2/tables/` and contain all manuscript-reported numerical values
- **Raw experiment outputs** (~1.2 GB, including truth fields, FDM reference solutions, per-run training artifacts) are excluded from this repository due to size. They will be archived at Zenodo upon manuscript acceptance and linked in the data availability statement
- **Observation data** is synthetic (generated by the FDM solver), not field data
- **Reproduction scripts** that orchestrate the full experiment matrix are included in the Zenodo archive alongside the raw data

## Requirements

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.21 | Numerical arrays, linear algebra |
| scipy | ≥1.7 | Sparse solvers, eigendecomposition |
| matplotlib | ≥3.5 | Figure generation |
| pyyaml | ≥5.0 | Configuration parsing |
| torch | ≥1.12 | Neural network training |

Tested on Python 3.9–3.13, macOS and Windows 10/11.

## Citation

If you use this code, please cite:

```bibtex
@article{yang2026separating,
  title   = {Separating linearized observability, direct recoverability,
             and workflow limitations in physics-informed neural network inversion},
  author  = {Yang, Fei},
  journal = {Computational Geosciences},
  year    = {2026},
  note    = {Submitted}
}
```

## License

- **Code**: [MIT License](LICENSE)
- **Data & summary tables**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Contact

Fei Yang — yangf@cigem.cn

China Institute of Geo-Environment Monitoring, China Geological Survey, Beijing 100081, China
