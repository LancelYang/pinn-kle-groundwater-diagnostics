# CG Compact PINN-Hybrid Matrix Summary

This table summarizes the matched PINN-hybrid half of the compact CG matrix.

## Overall

| cases | ok | failed | hard selector | weak-pass | strong-pass |
|---:|---:|---:|---:|---:|---:|
| 9 | 9 | 0 | 9 | 2 | 0 |

## By Noise

| noise | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 3 | 3 | 1 | 0 | 0.0202 | 0.5895 | 0 |
| 0.01 | 3 | 3 | 1 | 0 | 0.0207 | 0.6259 | 0 |
| 0.02 | 3 | 3 | 0 | 0 | 0.0338 | 0.7767 | 0 |

## By Truth

| truth | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | 3 | 3 | 2 | 0 | 0.0054 | 0.1497 | 0 |
| 456 | 3 | 3 | 0 | 0 | 0.0556 | 1.3470 | 0 |
| 789 | 3 | 3 | 0 | 0 | 0.0207 | 0.6259 | 0 |
