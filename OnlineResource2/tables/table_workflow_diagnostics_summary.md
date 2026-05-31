# Workflow Diagnostics Summary

This file aggregates existing post-processing evidence for the CG submission path.
No new PINN training or FDM inversion is run by the workflow-diagnostics script.

## 1. Gradient-Transfer Audit

| Metric | Value |
|---|---:|
| initial data-loss xi-gradient norm | 0 |
| after-Stage-A data-loss xi-gradient norm | 0 |
| initial PDE xi-gradient norm | 4.836e+03 |
| after-Stage-A PDE xi-gradient norm | 1.675 |
| initial PDE cosine to truth direction | 0.200 |
| after-Stage-A PDE cosine to truth direction | 0.309 |
| first actual xi-update cosine to truth direction | 0.056 |

Interpretation: the sparse head data term has zero direct coefficient gradient in the standard PINN-KLE formulation; coefficient updates are mediated by the physics residual and its alignment with the true coefficient direction is weak.

## 2. No-Truth Round Selection Versus Recovery

| noise | cases | hard selector | weak recovery | median regret | median hybrid logK RMSE |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 2 | 2 | 1 | 0 | 0.153 |
| 0.01 | 4 | 4 | 0 | 0 | 0.178 |
| 0.015 | 4 | 4 | 0 | 0 | 0.236 |
| 0.02 | 4 | 4 | 0 | 0 | 0.318 |
| 0.05 | 4 | 4 | 0 | 0 | 1.133 |

Overall, 18/18 cases are hard-pass selector cases, but only 1/18 pass the conductivity-recovery criterion.
This separates choosing the best available alternating round from obtaining a useful conductivity estimate.

## 3. Direct FDM-KLE Versus PINN-Hybrid

| noise | cases | FDM weak-pass | hybrid weak-pass | median FDM logK RMSE | median hybrid logK RMSE | median hybrid/FDM ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 2 | 2 | 1 | 0.016 | 0.153 | 10.743 |
| 0.01 | 4 | 4 | 0 | 0.032 | 0.178 | 6.372 |
| 0.015 | 4 | 4 | 0 | 0.048 | 0.236 | 5.195 |
| 0.02 | 4 | 4 | 0 | 0.064 | 0.318 | 4.915 |
| 0.05 | 4 | 1 | 0 | 0.164 | 1.133 | 5.442 |

Matched direct-baseline cases: FDM-KLE weak-pass 15/18; PINN-hybrid weak-pass 1/18.

## 4. Alternating-Round Degradation

| Metric | Value |
|---|---:|
| alternating runs analyzed | 11 |
| round-to-round transitions | 38 |
| worsening transitions | 9 |
| worsening transitions with Stage-B logK rebound | 9 |
| worsening transitions with Stage-A h rebound | 6 |
| corr(delta Stage B, delta Stage C) on worsening transitions | 0.784 |

Interpretation: late-round degradation is Stage-B-led with frequent Stage-A/B coupling; it is not adequately explained as only a head-fitting failure.

## 5. Mode Bridge

| modes | hybrid logK RMSE | perfect-h logK RMSE | workflow gap | classification |
|---:|---:|---:|---:|---|
| 3 | 0.125 | 1.722e-04 | 0.125 | workflow damage visible but still weak-pass |
| 5 | 1.219 | 7.416e-04 | 1.219 | workflow-limited failure |
| 10 | 1.426 | 0.001 | 1.424 | workflow-limited failure |

Takeaway: The hybrid-to-fixed-h gap is already visible at 3 modes but still tolerable; the workflow breakdown becomes fatal by 5 modes and remains dominant at 10 modes.
