# KLE Jacobian / FIM / Gauss-Newton Hessian Diagnostic

This diagnostic estimates the local sparse-head observation Jacobian with
respect to KLE coefficients using central finite differences around the
true coefficient vector. It supports the Computational Geosciences framing
as a numerical identifiability and workflow-diagnostics study.

The Fisher Information Matrix and Gauss-Newton least-squares Hessian are
derived from the same Jacobian:

```text
FIM = J^T J / sigma^2
H_GN = 2 J^T J
```

Therefore FIM and Gauss-Newton Hessian spectra have the same shape as the
squared Jacobian singular spectrum. The main figure shows the Jacobian
spectrum plus a conditioning summary to avoid plotting redundant spectra.

| Case | n_obs | noise used in FIM | rank | J condition | FIM condition | H_GN condition |
|---|---:|---:|---:|---:|---:|---:|
| kle3_obs200_noise0.005_seed999 | 200 | 0.005 | 3/3 | 10.6 | 112 | 112 |
| kle5_obs200_noise0.005_seed999 | 200 | 0.005 | 5/5 | 35.6 | 1.27e+03 | 1.27e+03 |
| kle10_obs200_noise0.005_seed999 | 200 | 0.005 | 10/10 | 54.3 | 2.95e+03 | 2.95e+03 |

Interpretation:

- Full relative rank indicates that the local FDM-KLE inverse map is
  observable for the tested sparse-head layout.
- Large condition numbers indicate practical sensitivity and uncertainty,
  even when the local rank is full.
- These spectra diagnose information content of the direct FDM-KLE problem;
  they do not prove that the PINN-hybrid optimizer can transmit the same
  information through its staged workflow.
