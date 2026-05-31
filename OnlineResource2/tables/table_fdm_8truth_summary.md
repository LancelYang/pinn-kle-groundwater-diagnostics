# Table S1: Expanded FDM-KLE baseline results (8 truths × 3 noise levels)

All results at $n_{\mathrm{obs}} = 200$, seed $202$. Weak-pass criterion: $\log K$ RMSE $< 0.15$, $h$ RMSE $< 0.006$.

| Truth | $\sigma_{\log K}$ | $\sigma=0.005$ | $\sigma=0.01$ | $\sigma=0.02$ |
|------:|:---:|:---:|:---:|:---:|
| 123 | 0.36 | PASS (0.0203 / 7.3e-4) | PASS (0.0413 / 1.3e-3) | PASS (0.0810 / 2.5e-3) |
| 456 | 0.71 | PASS (0.0172 / 8.7e-4) | PASS (0.0343 / 1.4e-3) | PASS (0.0691 / 2.5e-3) |
| 789 | 0.47 | PASS (0.0184 / 6.9e-4) | PASS (0.0370 / 1.2e-3) | PASS (0.0753 / 2.3e-3) |
| 321 | 0.40 | PASS (0.0160 / 4.6e-4) | PASS (0.0322 / 8.7e-4) | PASS (0.0639 / 1.8e-3) |
| 555 | 0.17 | PASS (0.0200 / 5.4e-4) | PASS (0.0402 / 1.1e-3) | PASS (0.0801 / 2.2e-3) |
| 777 | 0.34 | PASS (0.0170 / 4.5e-4) | PASS (0.0344 / 9.1e-4) | PASS (0.0686 / 1.9e-3) |
| 999 | 0.56 | PASS (0.0175 / 4.7e-4) | PASS (0.0353 / 9.2e-4) | PASS (0.0696 / 1.9e-3) |
| **1357** | **0.80** | PASS (0.0230 / 5.5e-4) | PASS (0.0450 / 1.1e-3) | **PASS (0.0892 / 2.2e-3)** |

Median $\log K$ RMSE by noise: $0.0180$ at $\sigma=0.005$, $0.0360$ at $\sigma=0.01$, $0.0723$ at $\sigma=0.02$.

Cell format: $\log K$ RMSE / $h$ RMSE (closed-loop N=201). All 24 cases satisfy both weak-pass ($\log K<0.15$, $h<0.006$) and strong-pass ($\log K<0.10$, $h<0.003$) criteria.
