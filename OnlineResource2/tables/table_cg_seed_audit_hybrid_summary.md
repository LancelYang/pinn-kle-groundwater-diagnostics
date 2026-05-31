# CG Compact PINN-Hybrid Matrix Summary

This table summarizes the matched PINN-hybrid half of the compact CG matrix.

## Overall

| cases | ok | failed | hard selector | weak-pass | strong-pass |
|---:|---:|---:|---:|---:|---:|
| 18 | 18 | 0 | 17 | 3 | 0 |

## By Noise

| noise | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 9 | 8 | 3 | 0 | 0.0202 | 0.5895 | 0 |
| 0.02 | 9 | 9 | 0 | 0 | 0.0331 | 0.7599 | 0 |

## By Truth

| truth | cases | hard selector | weak-pass | strong-pass | median h RMSE | median logK RMSE | median selector regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | 6 | 5 | 3 | 0 | 0.0076 | 0.2337 | 0 |
| 456 | 6 | 6 | 0 | 0 | 0.0583 | 1.3743 | 0 |
| 789 | 6 | 6 | 0 | 0 | 0.0241 | 0.6782 | 0 |
