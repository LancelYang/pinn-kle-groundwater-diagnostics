# CG Hybrid-Recovered Point Jacobian/FIM Diagnostic

Paired local sparse-head sensitivity at true xi and at the selected hybrid-recovered xi.

| truth | noise | seed | true rank | recovered rank | true J cond | recovered J cond | cond ratio | true weakest sv | recovered weakest sv | sv ratio | recovered xi distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 456 | 0.005 | 202 | 3 | 3 | 7.5323 | 7.3791 | 0.9797 | 0.1052 | 0.0995 | 0.9461 | 4.1311 |
| 456 | 0.005 | 303 | 3 | 3 | 7.2915 | 6.8142 | 0.9345 | 0.1097 | 0.1070 | 0.9754 | 4.1781 |
| 456 | 0.005 | 404 | 3 | 3 | 6.9937 | 6.9363 | 0.9918 | 0.1151 | 0.1075 | 0.9338 | 4.4441 |
| 456 | 0.02 | 202 | 3 | 3 | 7.5323 | 7.6056 | 1.0097 | 0.1052 | 0.0984 | 0.9352 | 4.2652 |
| 456 | 0.02 | 303 | 3 | 3 | 7.2915 | 6.8981 | 0.9460 | 0.1097 | 0.1053 | 0.9601 | 4.1599 |
| 456 | 0.02 | 404 | 3 | 3 | 6.9937 | 6.9235 | 0.9900 | 0.1151 | 0.1071 | 0.9301 | 4.5856 |

Interpretation: if recovered-point conditioning is comparable to true-point conditioning, the hybrid failure is not explained by moving into a locally unobservable direct-map region.
