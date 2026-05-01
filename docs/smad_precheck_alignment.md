# SMAD Pre-check 1: U_posterior vs Drift Subspace Alignment

Date: 2026-05-01

Status: COMPLETE

## Method

This pre-check tests whether the posterior slow subspace of the DreamerV3 RSSM
deterministic state aligns with the empirical imagination drift subspace.

The analysis used the 20 Cheetah Run v2 rollout files:

```text
results/rollouts/cheetah_v3_seed{0..19}_v2.npz
```

Each file contains:

```text
true_latent:     (200, 1536)
imagined_latent: (200, 1536)
```

DreamerV3 `get_feat()` is:

```text
concat([stoch(1024), deter(512)])
```

Only the deterministic RSSM state was analyzed:

```text
true_deter = true_latent[:, 1024:]
imag_deter = imagined_latent[:, 1024:]
```

The posterior slow-mode basis was estimated from the true posterior deter
trajectories:

1. Pool `true_deter` across all seeds and timesteps.
2. Fit PCA to the pooled matrix of shape `(4000, 512)`.
3. Retain the number of PCs needed to explain 95% variance.
4. Project each seed trajectory onto the retained PCs.
5. Compute lag-1 autocorrelation for each retained PC within each seed.
6. Average lag-1 autocorrelation across seeds.
7. Rank PCs by average lag-1 autocorrelation.
8. Define `U_posterior` from the top `r` slow PCs.

The drift subspace was estimated from raw deterministic-state drift vectors:

```text
Delta_t = imag_deter_t - true_deter_t
```

To avoid endpoint effects, the first 25 and last 25 timesteps were excluded.
This leaves:

```text
20 seeds * 150 steps = 3000 drift vectors
```

PCA was fit to this `(3000, 512)` drift matrix. The top `r` drift PCs define
`U_drift`.

For each rank `r`, two quantities were computed.

Subspace overlap:

```text
overlap = ||U_posterior.T @ U_drift||_F^2 / r
```

Variance projection ratio:

```text
sum ||U_posterior.T @ Delta_t||^2 / sum ||Delta_t||^2
```

The decision thresholds were:

```text
overlap > 0.7         STRONG GO
0.4 <= overlap <= 0.7 WEAK GO
overlap < 0.4         NO GO with U_posterior
```

## Results

Posterior PCA retained 38 PCs to explain 95.04% of posterior deter variance.

The pooled drift matrix had shape `(3000, 512)`.

Main rank results:

| Rank r | Overlap | Drift variance in U_posterior | Verdict |
|---:|---:|---:|---|
| 3 | 0.1323 | 0.0614 | NO GO with U_posterior |
| 5 | 0.1675 | 0.1286 | NO GO with U_posterior |
| 10 | 0.2827 | 0.1904 | NO GO with U_posterior |

Rank-sensitivity curve:

| Rank r | Overlap | Drift variance in U_posterior |
|---:|---:|---:|
| 1 | 0.0414 | 0.0207 |
| 2 | 0.0995 | 0.0338 |
| 3 | 0.1323 | 0.0614 |
| 5 | 0.1675 | 0.1286 |
| 10 | 0.2827 | 0.1904 |
| 15 | 0.3357 | 0.2256 |
| 20 | 0.3785 | 0.2586 |

The overlap increases with rank, but even at `r=20` it remains below the
`0.4` weak-go threshold.

The default SMAD rank `r=3` is especially weak:

```text
overlap = 0.1323
variance projection ratio = 0.0614
```

So the top-3 posterior slow modes capture only about 6.1% of raw drift energy,
and their subspace has low overlap with the top-3 empirical drift directions.

Generated artifacts are `results/tables/smad_alignment_check.json` and
`results/figures/smad_alignment_*.pdf`.

## Decision

Decision: NO GO with `U_posterior` as the fixed SMAD basis.

The pre-check does not support the Month 5 assumption that posterior slow modes
are a good proxy for the drift direction subspace. All requested ranks
`r in {3, 5, 10}` fall below the `0.4` threshold:

```text
r=3:  overlap 0.1323
r=5:  overlap 0.1675
r=10: overlap 0.2827
```

The conclusion is robust across the plotted sensitivity ranks
`r in {1, 2, 3, 5, 10, 15, 20}`. The alignment improves with rank, but not
enough to justify using `U_posterior` directly.

## Implications for Month 5

SMAD Phase 2 should not anchor the intervention to posterior slow modes alone.

The safer intervention design is:

```text
Use U_drift directly as the SMAD basis.
```

This means the basis should be estimated from empirical imagination drift
vectors, not from posterior temporal autocorrelation. A `U_drift`-anchored
Phase 2 would target the directions where the trained model actually drifts.

If the intervention still uses `U_posterior` for theoretical reasons, it should
be treated as a high-risk ablation rather than the primary Month 5 path.
Alignment should then be monitored during training, and failure to improve
drift along `U_drift` should trigger a pivot.

Practical Month 5 design changes:

1. Estimate `U_drift` from a fixed calibration set before SMAD training.
2. Penalize drift projected onto `U_drift`, not onto posterior slow PCs.
3. Keep a secondary monitor for `U_posterior` to test whether slow-mode
   structure emerges during intervention.
4. Report both `U_drift` and `U_posterior` metrics to avoid overfitting the
   analysis to one basis.

## Limitations

This pre-check used only Cheetah Run.

The result may differ for Cartpole Swingup or for future DreamerV4 image-token
rollouts.

The analysis is linear. It tests subspace overlap in the 512-dimensional deter
state and does not capture curved manifolds or state-dependent drift bases.

The drift basis was estimated from existing open-loop rollouts from one trained
checkpoint. It should be re-estimated if the checkpoint, rollout policy, or
imagination start distribution changes.

The first and last 25 timesteps were excluded to reduce boundary effects. That
choice is defensible for the 200-step diagnostic horizon, but it is still a
fixed analysis choice.

The requested `docs/PROPOSAL_ADDENDUM.md` file was not present in this checkout.
The implementation follows the SMAD definitions provided in the task prompt and
the existing DriftDetect latent format.
