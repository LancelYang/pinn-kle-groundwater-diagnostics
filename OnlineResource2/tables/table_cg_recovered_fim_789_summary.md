# CG Hybrid-Recovered Point Jacobian/FIM Diagnostic

Paired local sparse-head sensitivity at true xi and at the selected hybrid-recovered xi.

| truth | noise | seed | true rank | recovered rank | true J cond | recovered J cond | cond ratio | true weakest sv | recovered weakest sv | sv ratio | recovered xi distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 789 | 0.005 | 202 | 3 | 3 | 11.4414 | 10.8361 | 0.9471 | 0.0928 | 0.0929 | 1.0016 | 2.0157 |
| 789 | 0.005 | 303 | 3 | 3 | 10.9690 | 10.1629 | 0.9265 | 0.0963 | 0.0984 | 1.0223 | 2.0287 |
| 789 | 0.005 | 404 | 3 | 3 | 10.7764 | 10.3260 | 0.9582 | 0.0995 | 0.1006 | 1.0114 | 1.5384 |
| 789 | 0.02 | 202 | 3 | 3 | 11.4414 | 10.9692 | 0.9587 | 0.0928 | 0.0911 | 0.9820 | 2.5657 |
| 789 | 0.02 | 303 | 3 | 3 | 10.9690 | 10.2598 | 0.9353 | 0.0963 | 0.0972 | 1.0094 | 2.5560 |
| 789 | 0.02 | 404 | 3 | 3 | 10.7764 | 9.9077 | 0.9194 | 0.0995 | 0.0988 | 0.9935 | 2.3889 |

Interpretation: if recovered-point conditioning is comparable to true-point conditioning, the hybrid failure is not explained by moving into a locally unobservable direct-map region.
