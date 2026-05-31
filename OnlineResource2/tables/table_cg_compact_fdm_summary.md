# CG Compact Direct FDM-KLE Matrix Summary

This table summarizes the direct FDM-KLE half of the compact CG matrix.
It establishes the direct reduced-space recoverability reference before matched PINN-hybrid expansion.

## Overall

| cases | ok | failed | weak-pass | strong-pass |
|---:|---:|---:|---:|---:|
| 18 | 18 | 0 | 18 | 18 |

## By Noise

| noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 6 | 6 | 6 | 7.121e-04 | 0.0157 |
| 0.01 | 6 | 6 | 6 | 0.0012 | 0.0310 |
| 0.02 | 6 | 6 | 6 | 0.0021 | 0.0621 |

## By Observation Density

| n_obs | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---:|---:|---:|---:|---:|---:|
| 100 | 9 | 9 | 9 | 7.502e-04 | 0.0181 |
| 200 | 9 | 9 | 9 | 0.0013 | 0.0372 |

## By Truth

| truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE |
|---|---:|---:|---:|---:|---:|
| 123 | 6 | 6 | 6 | 8.920e-04 | 0.0264 |
| 456 | 6 | 6 | 6 | 0.0012 | 0.0310 |
| 789 | 6 | 6 | 6 | 9.483e-04 | 0.0273 |

Interpretation: direct FDM-KLE remains robust across the compact truth/noise/density matrix.
Matched PINN-hybrid runs should be compared against this reference, not against a single-truth baseline.
