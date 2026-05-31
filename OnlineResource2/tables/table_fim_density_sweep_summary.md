# CG Truth Jacobian/FIM Diagnostic Summary

Local finite-difference sparse-head sensitivity around true KLE coefficients.

| truth | n_obs | rank | J condition | H_GN condition | weakest sv | strongest sv |
|---|---:|---:|---:|---:|---:|---:|
| 123 | 50 | 3/3 | 9.1183 | 83.1437 | 0.0562 | 0.5127 |
| 123 | 100 | 3/3 | 9.9693 | 99.3879 | 0.0681 | 0.6790 |
| 123 | 200 | 3/3 | 10.8185 | 117.0403 | 0.0920 | 0.9948 |
| 123 | 400 | 3/3 | 10.9527 | 119.9607 | 0.1326 | 1.4528 |
| 456 | 50 | 3/3 | 6.4194 | 41.2086 | 0.0638 | 0.4094 |
| 456 | 100 | 3/3 | 6.8691 | 47.1846 | 0.0802 | 0.5511 |
| 456 | 200 | 3/3 | 7.5323 | 56.7355 | 0.1052 | 0.7926 |
| 456 | 400 | 3/3 | 7.8256 | 61.2400 | 0.1481 | 1.1587 |
| 789 | 50 | 3/3 | 10.3144 | 106.3872 | 0.0537 | 0.5541 |
| 789 | 100 | 3/3 | 10.3704 | 107.5450 | 0.0701 | 0.7266 |
| 789 | 200 | 3/3 | 11.4414 | 130.9062 | 0.0928 | 1.0614 |
| 789 | 400 | 3/3 | 11.7508 | 138.0816 | 0.1320 | 1.5507 |

Interpretation: full local rank supports direct local observability; condition-number differences diagnose practical sensitivity, not global uniqueness.
