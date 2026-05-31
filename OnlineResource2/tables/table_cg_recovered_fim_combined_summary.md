# CG Hybrid-Recovered Point Jacobian/FIM Diagnostic (Combined)

Paired local sparse-head sensitivity at true xi and at the selected hybrid-recovered xi.
Two hard-failure truths (456 and 789), each with 6 seed-audit cases.

| truth | noise | seed | true rank | recovered rank | true J cond | recovered J cond | cond ratio | true weakest sv | recovered weakest sv | sv ratio | recovered xi distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 456 | 0.005 | 202 | 3 | 3 | 7.5323 | 7.3791 | 0.9797 | 0.1052 | 0.0995 | 0.9461 | 4.1311 |
| 456 | 0.005 | 303 | 3 | 3 | 7.2915 | 6.8142 | 0.9345 | 0.1097 | 0.1070 | 0.9754 | 4.1781 |
| 456 | 0.005 | 404 | 3 | 3 | 6.9937 | 6.9363 | 0.9918 | 0.1151 | 0.1075 | 0.9338 | 4.4441 |
| 456 | 0.02 | 202 | 3 | 3 | 7.5323 | 7.6056 | 1.0097 | 0.1052 | 0.0984 | 0.9352 | 4.2652 |
| 456 | 0.02 | 303 | 3 | 3 | 7.2915 | 6.8981 | 0.9460 | 0.1097 | 0.1053 | 0.9601 | 4.1599 |
| 456 | 0.02 | 404 | 3 | 3 | 6.9937 | 6.9235 | 0.9900 | 0.1151 | 0.1071 | 0.9301 | 4.5856 |
| 789 | 0.005 | 202 | 3 | 3 | 11.4414 | 10.8361 | 0.9471 | 0.0928 | 0.0929 | 1.0016 | 2.0157 |
| 789 | 0.005 | 303 | 3 | 3 | 10.9690 | 10.1629 | 0.9265 | 0.0963 | 0.0984 | 1.0223 | 2.0287 |
| 789 | 0.005 | 404 | 3 | 3 | 10.7764 | 10.3260 | 0.9582 | 0.0995 | 0.1006 | 1.0114 | 1.5384 |
| 789 | 0.02 | 202 | 3 | 3 | 11.4414 | 10.9692 | 0.9587 | 0.0928 | 0.0911 | 0.9820 | 2.5657 |
| 789 | 0.02 | 303 | 3 | 3 | 10.9690 | 10.2598 | 0.9353 | 0.0963 | 0.0972 | 1.0094 | 2.5560 |
| 789 | 0.02 | 404 | 3 | 3 | 10.7764 | 9.9077 | 0.9194 | 0.0995 | 0.0988 | 0.9935 | 2.3889 |

### Summary statistics (12 paired cases)

- Condition-number ratio (recovered/true): min 0.9194, median 0.9527, max 1.0097
- Weakest-SV ratio (recovered/true): min 0.9301, median 0.9787, max 1.0223
- Recovered xi distance to true: min 1.5384, median 3.3484, max 4.5856
- All Jacobians (true and recovered): rank 3/3

Interpretation: across both hard-failure truths, recovered-point conditioning is comparable to true-point conditioning. The hybrid failure is not explained by moving into a locally unobservable direct-map region.
