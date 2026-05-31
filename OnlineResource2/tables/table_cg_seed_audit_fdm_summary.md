# CG Compact Direct FDM-KLE Matrix Summary

This table summarizes the direct FDM-KLE half of the compact CG matrix.
It establishes the direct reduced-space recoverability reference before matched PINN-hybrid expansion.

## Overall

| cases | ok | failed | weak-pass | strong-pass |
|---:|---:|---:|---:|---:|
| 18 | 18 | 0 | 18 | 15 |

## By Noise

| noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 9 | 9 | 9 | 7.337e-04 | 0.0171 |
| 0.02 | 9 | 9 | 6 | 0.0025 | 0.0692 |

## By Observation Density

| n_obs | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---:|---:|---:|---:|---:|---:|
| 200 | 18 | 18 | 15 | 0.0015 | 0.0343 |

## By Truth

| truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---|---:|---:|---:|---:|---:|
| 123 | 6 | 6 | 5 | 0.0016 | 0.0396 |
| 456 | 6 | 6 | 5 | 0.0015 | 0.0364 |
| 789 | 6 | 6 | 5 | 0.0016 | 0.0333 |

Interpretation: direct FDM-KLE remains robust across the compact truth/noise/density matrix.
Matched PINN-hybrid runs should be compared against this reference, not against a single-truth baseline.
