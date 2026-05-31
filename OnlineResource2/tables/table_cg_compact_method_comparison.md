# CG Compact Matched Comparison

This table compares direct FDM-KLE and PINN-hybrid results for matched compact CG matrix cases completed so far.

## By Method And Noise

| method | noise | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fdm_kle | 0.005 | 3 | 3 | 3 | 7.337e-04 | 0.0184 | 0.9999 | NA | NA |
| direct_fdm_kle | 0.01 | 3 | 3 | 3 | 0.0013 | 0.0372 | 0.9994 | NA | NA |
| direct_fdm_kle | 0.02 | 3 | 3 | 3 | 0.0025 | 0.0750 | 0.9976 | NA | NA |
| pinn_hybrid | 0.005 | 3 | 1 | 0 | 0.0202 | 0.5895 | 0.4282 | 3 | 0 |
| pinn_hybrid | 0.01 | 3 | 1 | 0 | 0.0207 | 0.6259 | 0.4380 | 3 | 0 |
| pinn_hybrid | 0.02 | 3 | 0 | 0 | 0.0338 | 0.7767 | 0.3844 | 3 | 0 |

## By Method And Truth

| method | truth | cases | weak-pass | strong-pass | median h RMSE | median logK RMSE | median xi corr | hard selector | median selector regret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct_fdm_kle | 123 | 3 | 3 | 3 | 0.0013 | 0.0406 | 0.9993 | NA | NA |
| direct_fdm_kle | 456 | 3 | 3 | 3 | 0.0014 | 0.0344 | 0.9995 | NA | NA |
| direct_fdm_kle | 789 | 3 | 3 | 3 | 0.0012 | 0.0372 | 0.9994 | NA | NA |
| pinn_hybrid | 123 | 3 | 2 | 0 | 0.0054 | 0.1497 | 0.9997 | 3 | 0 |
| pinn_hybrid | 456 | 3 | 0 | 0 | 0.0556 | 1.3470 | 0.4282 | 3 | 0 |
| pinn_hybrid | 789 | 3 | 0 | 0 | 0.0207 | 0.6259 | 0.1591 | 3 | 0 |

Interpretation: the completed n_obs=200 compact matrix shows a clear direct-vs-hybrid recovery gap.
