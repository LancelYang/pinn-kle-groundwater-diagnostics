# Cover Letter

**Manuscript:** Separating linearized observability, direct recoverability, and workflow limitations in physics-informed neural network inversion

**Journal:** Computational Geosciences

---

Dear Editor,

I submit the manuscript titled *"Separating linearized observability, direct recoverability, and workflow limitations in physics-informed neural network inversion"* for consideration for publication in *Computational Geosciences*.

## Summary

Physics-informed neural networks (PINNs) are increasingly applied to groundwater inverse problems, but an important evaluation gap persists between good hydraulic-head fitting and reliable hydraulic-conductivity recovery. This study presents a controlled numerical diagnostic framework that systematically separates four concerns—local information content, gradient transfer, round-selection quality, and workflow-limited recovery—using a PINN-KLE inversion of two-dimensional steady groundwater flow as the test case.

The central finding is that the staged PINN-hybrid workflow can lose coefficient information that remains recoverable by a direct deterministic FDM-KLE baseline. Across an expanded set of eight truth realizations (logK standard deviation 0.17–0.80), the direct FDM-KLE baseline passes all 24 reported cases, whereas the PINN-hybrid workflow achieves weak-pass conductivity recovery in only 3 of 18 matched seed-audit cases. Local Jacobian/Fisher-information diagnostics indicate that these failures do not arise from loss of local direct-map observability in the tested reduced map: sparse-head Jacobians remain full rank across observation densities from 50 to 400 points, and local conditioning at hybrid-recovered coefficients remains comparable to true-point conditioning (condition-number ratio 0.919–1.010 across twelve paired cases). The failure is therefore best described as workflow-limited relative to the matched direct baseline, rather than as absence of low-dimensional information in the sparse-head observations.

## Relevance to Computational Geosciences

I believe this manuscript is well-suited to *Computational Geosciences* for several reasons:

1. **Methodological contribution to PINN inverse modeling.** PINN-based subsurface inversion is an active area in computational geoscience, and the diagnostic framework proposed here addresses a recurring concern raised in recent literature: the distinction between forward approximation quality and inverse parameter recovery.

2. **Computationally reproducible diagnostics.** All diagnostics (gradient audit, round-selection audit, matched direct baseline, Jacobian/FIM sensitivity) are script-driven and designed for computational reproducibility, consistent with the journal's emphasis on verifiable computational experiments.

3. **Bridges hydrogeology and scientific machine learning.** The study connects classical hydrogeological concepts (parameter identifiability, sensitivity analysis, deterministic least-squares inversion) with modern PINN methodology while keeping the claims limited to a controlled synthetic benchmark.

## Key Contributions

1. A five-layer diagnostic framework that separates gradient transfer, round selection, direct recoverability, degradation drivers, and mode bridges.
2. Evidence that the head-observation loss in staged PINN-KLE workflows provides no direct gradient pathway to KLE coefficients.
3. An expanded FDM-KLE baseline (24/24 passes across eight truths) demonstrating that sparse-head information exists but is not preserved by the staged PINN-hybrid workflow.
4. Recovered-point local observability analysis showing that hybrid failure does not correspond to loss of direct-map observability at the recovered parameters.
5. A practical recommendation: PINN groundwater inverse studies should report head fitting, local observability, deterministic baseline recoverability, and closed-loop conductivity recovery as separate diagnostic components.

## Declarations

- This manuscript is original, has not been published previously, and is not under consideration elsewhere.
- The author has approved the manuscript and agrees with its submission to *Computational Geosciences*.
- The author declares no competing interests.
- Machine-readable summary data and diagnostic scripts are supplied as Online Resource 2 for review. The raw numerical arrays are available from the corresponding author upon reasonable request during review, and the full review data package plus raw arrays will be deposited in Zenodo or HydroShare before publication if the manuscript is accepted.
- An AI-assisted coding and editing tool was used for debugging, workflow organization, and manuscript copy-editing. The author reviewed all generated material and takes full responsibility for the submitted work.

---

Thank you for considering my manuscript. I look forward to your response.

Sincerely,

Fei Yang — corresponding author  
yangf@cigem.cn  

China Institute of Geo-Environment Monitoring  
Beijing 100081, China
