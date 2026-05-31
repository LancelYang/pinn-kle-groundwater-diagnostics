# CG Seed-Audit Matched Comparison

This table compares direct FDM-KLE and PINN-hybrid results for matched compact CG matrix cases completed so far.

## By Method And Noise

| method | noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fdm_kle | 0.005 | 9 | 9 | 9 | 7.337e-04 | 0.0171 | 0.9998 | NA | NA |
| direct_fdm_kle | 0.02 | 9 | 9 | 6 | 0.0025 | 0.0692 | 0.9975 | NA | NA |
| pinn_hybrid | 0.005 | 9 | 3 | 0 | 0.0202 | 0.5895 | 0.4282 | 8 | 0 |
| pinn_hybrid | 0.02 | 9 | 0 | 0 | 0.0331 | 0.7599 | 0.3844 | 9 | 0 |

## By Method And Truth

| method | truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fdm_kle | 123 | 6 | 6 | 5 | 0.0016 | 0.0396 | 0.9993 | NA | NA |
| direct_fdm_kle | 456 | 6 | 6 | 5 | 0.0015 | 0.0364 | 0.9989 | NA | NA |
| direct_fdm_kle | 789 | 6 | 6 | 5 | 0.0016 | 0.0333 | 0.9987 | NA | NA |
| pinn_hybrid | 123 | 6 | 3 | 0 | 0.0076 | 0.2337 | 0.9903 | 5 | 0 |
| pinn_hybrid | 456 | 6 | 0 | 0 | 0.0583 | 1.3743 | 0.3976 | 6 | 0 |
| pinn_hybrid | 789 | 6 | 0 | 0 | 0.0241 | 0.6782 | 0.1047 | 6 | 0 |

Interpretation: the completed n_obs=200 compact matrix shows a clear direct-vs-hybrid recovery gap.
