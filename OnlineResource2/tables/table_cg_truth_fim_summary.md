# CG Truth Jacobian/FIM Diagnostic Summary

Local finite-difference sparse-head sensitivity around true KLE coefficients.

| truth | n_obs | rank | J condition | H_GN condition | weakest sv | strongest sv |
|---|---:|---:|---:|---:|---:|---:|
| 123 | 200 | 3/3 | 10.8185 | 117.0403 | 0.0920 | 0.9948 |
| 456 | 200 | 3/3 | 7.5323 | 56.7355 | 0.1052 | 0.7926 |
| 789 | 200 | 3/3 | 11.4414 | 130.9062 | 0.0928 | 1.0614 |

Interpretation: full local rank supports direct local observability; condition-number differences diagnose practical sensitivity, not global uniqueness.
