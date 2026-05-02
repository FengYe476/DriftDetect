# SMAD Pre-check 3: Methodology Validation

**Date:** 2026-05-01  
**Verdict:** "Posterior slow ≠ drift" is a REAL FINDING, not a methodological artifact.

## Background

Month 4 Pre-check 1 found that `U_posterior`, estimated using PCA plus lag-1 autocorrelation, had very low subspace overlap with the empirical drift subspace `U_drift`. Across ranks `r = {3, 5, 10}`, posterior/drift overlap was only about 0.13-0.28.

Pre-check 3 was designed to determine whether this failure was caused by a methodological artifact in the approximate slow-feature estimation method, or whether it reflected a genuine property of DreamerV3's latent dynamics. The check therefore replaced the approximate slow-basis method with proper linear Slow Feature Analysis (SFA), visually inspected the resulting bases, and tested whether the drift-basis result depended on the trim window used for drift PCA.

## Pre-check 3a: SFA Basis Comparison

Pre-check 3a re-estimated three rank-`r` bases from the Cheetah v2 rollouts:

- `U_sfa`: proper SFA basis, where slow features minimize first-derivative variance after PCA whitening.
- `U_posterior`: Month 4 posterior basis, using PCA directions ranked by within-seed lag-1 autocorrelation.
- `U_drift`: empirical drift basis, using PCA on `imagined_deter - true_deter` over the trimmed drift window.

The key result is that even proper SFA produces a basis with near-zero overlap to the drift directions. Therefore, the mismatch is not caused by the PCA+autocorrelation approximation. It is a fundamental property of the learned latent dynamics.

| r | SFA/posterior | SFA/drift | posterior/drift |
|---:|---:|---:|---:|
| 3 | 0.0033 | 0.0166 | 0.1320 |
| 5 | 0.1013 | 0.0610 | 0.1671 |
| 10 | 0.1368 | 0.0905 | 0.2829 |

The posterior/drift values reproduce the Month 4 result, while SFA/drift is even lower. This rules out the hypothesis that the original `U_posterior` failure was merely an artifact of using lag-1 autocorrelation as a proxy for slowness.

## Pre-check 3b: Visual Sanity

Pre-check 3b visually inspected the rank-3 basis directions on seed 0. SFA directions were projected against the true latent trajectory, posterior directions were projected against the true latent trajectory, and drift directions were projected against both true and imagined latent trajectories.

The derivative-variance sanity check confirms that SFA directions are genuinely slow, while the posterior directions selected by PCA+autocorrelation are not slow in the SFA sense.

| basis | dir0 | dir1 | dir2 |
|---|---:|---:|---:|
| SFA | 0.003806 | 0.004144 | 0.005023 |
| Posterior | 1.709023 | 1.575134 | 0.159083 |
| Drift | 0.176068 | 0.280383 | 0.143980 |

SFA derivative variance is approximately 0.004. Posterior derivative variance is roughly 1.6 for its first two directions, about 400x larger than SFA. This confirms that PCA+autocorrelation does not select slow features in the strict SFA sense. Drift directions are intermediate: they are not as slow as SFA directions, but they are also not as fast as the dominant posterior directions.

The visual projections further show that drift directions capture the divergence between true and imagined latent trajectories.

Reference figures:

- `results/figures/smad_precheck_3b_visual.pdf`
- `results/figures/smad_precheck_3b_derivatives.pdf`

## Pre-check 3c: Trim Window Sensitivity

Pre-check 3c tested whether the low posterior/drift and SFA/drift overlaps were artifacts of the original drift trim window `[25, 175]`. `U_sfa` and `U_posterior` were estimated once from the full true posterior deter trajectories. `U_drift` was re-estimated separately for three drift PCA windows: `[10, 190]`, `[25, 175]`, and `[50, 150]`, all inclusive.

The posterior/drift overlaps remain stable across trim windows:

| trim window | r=3 | r=5 | r=10 |
|---|---:|---:|---:|
| [10,190] | 0.1279 | 0.1590 | 0.2883 |
| [25,175] | 0.1320 | 0.1671 | 0.2829 |
| [50,150] | 0.1267 | 0.1428 | 0.2729 |

The SFA/drift overlaps are also stable and remain very low:

| trim window | r=3 | r=5 | r=10 |
|---|---:|---:|---:|
| [10,190] | 0.0198 | 0.0503 | 0.1031 |
| [25,175] | 0.0166 | 0.0610 | 0.0905 |
| [50,150] | 0.0107 | 0.0625 | 0.0854 |

The maximum observed deviation is approximately 0.03. This establishes that the low-overlap result is not a boundary artifact of the original trim window.

## Interpretation

1. Drift directions are temporally medium-speed. Their derivative variance is approximately 0.14-0.28, neither as slow as the SFA features nor as fast as the leading posterior directions.
2. The drift subspace is a distinct object from both the environment's slow subspace and the high-variance autoregressive posterior subspace.
3. This supports the hypothesis that RSSM drift accumulates along model-specific error attractors that are misaligned with the true environment's slow dynamics.
4. Finding A, the dc_trend frequency dominance result, is not contradicted. Drift can be temporally slow-growing, with low-frequency dominated MSE, while still growing in spatial directions that are not the environment's slow-feature directions.
5. This distinction between temporal frequency structure and spatial direction structure is itself a paper-worthy insight.

## Impact on SMAD Design

1. `U_posterior` is definitively ruled out as the SMAD basis. Both the original PCA+autocorrelation version and the proper SFA version fail to align with drift.
2. `U_drift` remains the correct choice for Phase 2, but its circularity concern, raised by the advisor, must be addressed by Pre-check 4 with cross-validation.
3. Theorem 1 motivation must be revised. We cannot claim that "posterior slow features are the protected subspace." Instead, the correct framing is: "under linear-Gaussian assumptions, the true slow subspace is identifiable; empirically, the drift subspace serves as a practical proxy."

## Basis Decision for Phase 2

**Decision:** Use `U_drift` (`r=10`) as the SMAD basis for Phase 2.

**Rationale:** Only `U_drift` aligns with the actual drift phenomenon. SFA and posterior bases are scientifically informative but operationally irrelevant for drift mitigation.

**Remaining risk:** Circularity, because `U_drift` is estimated from the same rollouts used to evaluate damping. This will be addressed by Pre-check 4.
