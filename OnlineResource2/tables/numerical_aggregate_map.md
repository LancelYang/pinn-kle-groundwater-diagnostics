# Numerical Aggregate Map

This note prevents the main numerical aggregates from being read as a single
uniform ensemble. The manuscript uses several deliberately different case
sets, each answering a different audit question.

| Label | Manuscript role | Case set | Key result | Interpretation boundary |
|---|---|---|---|---|
| Pilot aggregate | Early workflow-diagnostics aggregate | 18 medium-budget matched cases spanning five noise levels | Direct FDM-KLE weak-passes 15/18; PINN-hybrid weak-passes 1/18 | Used to establish the initial direct-vs-hybrid gap; not the final multi-truth ensemble. |
| Selector-vs-recovery pilot | No-truth round-selection audit | 18 medium-budget pilot cases | 18/18 hard-pass selector cases; 1/18 conductivity weak-pass | Shows that selecting the best available round is not equivalent to conductivity recovery. |
| Compact matched matrix | Controlled three-truth, one-seed comparison | 3 truths x 3 noises at n_obs=200 | Direct FDM-KLE passes 9/9; PINN-hybrid weak-passes 2/9 | Separates direct recoverability from workflow recovery under a compact matched design. |
| Three-seed audit | Immediate single-seed check | 3 truths x 2 noises x 3 seeds at n_obs=200 | Direct FDM-KLE weak-passes 18/18; PINN-hybrid weak-passes 3/18; selector hard-passes 17/18 | Removes the simplest single-seed explanation; does not establish full seed-averaged statistics. |
| Expanded FDM baseline | Direct reduced-map recoverability check | 8 truths x 3 noises at n_obs=200, one observation seed | Direct FDM-KLE passes 24/24 | Strengthens the direct recoverability reference; five added truths are FDM-only and not matched with hybrid. |
| Truth-point FIM density sweep | Local observability under observation-density changes | 3 truths x 4 densities at sigma=0.005 | 12/12 Jacobians full rank | Tests local rank/conditioning only; does not prove global uniqueness or optimization success. |
| Recovered-point FIM audit | Local observability at failed hybrid outputs | 2 hard-failure truths x 2 noises x 3 seeds | 12/12 recovered-point Jacobians full rank; condition-ratio range 0.919-1.010 | Shows failed hybrid points are not locally unobservable under the direct map; does not identify the full optimization trajectory. |

Recommended citation in text: when using a fraction such as 17/18, 18/18,
3/18, 15/18, or 24/24, name the aggregate explicitly.
