# Online Resource 2: Review Data and Code Summary Package

**Article title:** Separating linearized observability, direct recoverability, and workflow limitations in physics-informed neural network inversion

**Journal:** Computational Geosciences

**Author:** Fei Yang

**Affiliation:** China Institute of Geo-Environment Monitoring, China Geological Survey, Minzu University South Road, Beijing 100081, China

**Corresponding author:** Fei Yang, yangf@cigem.cn

## Contents

This review package contains the machine-readable summary data and diagnostic
scripts needed to audit the numerical values reported in the manuscript. It is
intended as a compact review package, not as the final public archive of all raw
training outputs.

- `REPRODUCE.md`: audit scope, aggregate map, and representative project-level
  commands.
- `tables/`: CSV, JSON, and Markdown summary tables used by the manuscript,
  including `numerical_aggregate_map.md`.
- `figures/`: PNG files corresponding to the four main diagnostic figure
  sources.
- `src/`: Core scripts for the CG compact matrices, FDM-KLE baseline comparison,
  Jacobian/FIM diagnostics, recovered-point diagnostics, and summary generation.
- `configs/`: Main YAML configuration files used by the PINN-hybrid and
  Jacobian/FIM diagnostics.
- `requirements.txt` and `requirements-macos-lock.txt`: dependency files used in
  the reported macOS review environment.

## Notes

Truth labels in the manuscript and supplementary tables are anonymized display
labels. Internal output directories and machine-readable files may retain the
original random-seed identifiers used to generate the synthetic fields. These
internal identifiers are preserved to maintain reproducibility and avoid
renaming source data after analysis.

The raw NumPy arrays for truth realizations, synthetic observation sets, and FDM
reference solutions are retained in the project output tree. A public archive
containing this review package and the raw numerical arrays will be deposited in
Zenodo or HydroShare before publication if the manuscript is accepted.

## Review Use

The package is intended to support numerical checking of the submitted tables,
figures, and reported summary statistics. The included scripts are sufficient
to inspect the computational workflow and rerun selected diagnostics when the
project output tree is available. Some large raw training artifacts are not
duplicated in this compact review package; they will be included in the public
archive or supplied to reviewers on request if needed.
