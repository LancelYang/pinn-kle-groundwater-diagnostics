# Reproduction and Audit Notes

This compact review package is designed for numerical audit, not as the final
public archive of every raw training artifact. It includes summary tables,
core runner scripts, dependency files, and the main configuration files needed
to inspect how the reported manuscript values were produced.

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

## Source Scripts Included

The `src/` directory includes the runner and dependency modules used by the
reported compact experiments and diagnostics, including the FDM solver,
FDM-KLE baseline, PINN-hybrid runner, selector audit, and Jacobian/FIM tools.
These scripts retain the original project-relative path behavior through
`paths.py`.

## Full Reruns

Full reruns require the raw project output tree and synthetic truth files,
which are not fully duplicated in this compact review package because of size.
The final public DOI archive should include:

- raw truth fields and FDM reference arrays;
- observation files and recovered coefficient arrays;
- per-run `runner_command.json` files;
- full output directories for the compact matrix, seed audit, FIM density
  sweep, recovered-point FIM audit, and expanded FDM baseline.

Representative project-level commands used to generate the main audit layers:

```bash
make cg-compact-fdm
make cg-compact-hybrid-obs200
make cg-compact-compare
make cg-compact-seed-fdm
make cg-compact-seed-hybrid
make cg-seed-audit-compare
make cg-truth-fim
make cg-recovered-fim
```

Some extended diagnostics were run as one-off scripted jobs after the Makefile
targets above; their machine-readable outputs are included in `tables/`.
