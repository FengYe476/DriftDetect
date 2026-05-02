# SMAD Pre-check 4: Anti-Tautology Cross-Validation

**Date:** 2026-05-01  
**Verdict:** PASS. Post-hoc damping effect generalizes to held-out rollouts with transfer ratio 0.97-0.98.

## Motivation

Month 4 Pre-check 2 estimated `U_drift` from all 20 Cheetah rollouts and evaluated post-hoc damping on those same 20 rollouts. This creates a circularity concern: if we extract the top PCA directions of observed drift and then subtract drift along those same directions, measured drift should mechanically decrease.

The advisor flagged this as a possible false positive with four specific concerns:

1. **Mathematical circularity:** `U_drift` may only work because it was estimated and evaluated on the same samples.
2. **Total MSE dropping more than dc_trend MSE:** this could indicate that the intervention is simply removing broad error energy, not specifically targeting the drift mechanism.
3. **eta=0.50 effectively halving the imagination update:** such a large damping coefficient may disrupt legitimate recurrent dynamics in a real RSSM intervention.
4. **No eval_return test:** post-hoc damping cannot determine whether the agent's policy performance would survive the intervention.

Pre-check 4 addresses the first two concerns directly and clarifies which concerns must remain Phase 2 questions.

## Method

We ran 5 random 10/10 splits of rollout seeds `0-19`. For each split, `numpy.random.RandomState(split_idx)` was used for reproducibility.

For each split:

1. Estimate `U_drift` on the estimation set only, using rank `r=10` PCA on drift vectors `imagined_deter - true_deter` in the inclusive trim window `[25,175]`.
2. Apply post-hoc damping on both the estimation and validation sets:

   `imagined_damped[t] = imagined_deter[t] - eta * U_drift @ (U_drift.T @ drift_t)`

3. Evaluate `eta = 0.20` and `eta = 0.50`.
4. Run the full 5-band frequency decomposition independently on each set. The PCA basis for frequency decomposition was fit only on that set's true latents, never on all 20 seeds.
5. Compare in-sample and held-out damping reductions for dc_trend MSE and total band MSE.

## Results

SMAD Pre-check 4: Cross-validation damping test (`r=10`)

`eta=0.20`

| Split | Est dc_trend% | Est total% | Val dc_trend% | Val total% |
|---:|---:|---:|---:|---:|
| 0 | 27.19 | 26.80 | 27.09 | 27.11 |
| 1 | 25.24 | 26.28 | 26.33 | 26.25 |
| 2 | 23.27 | 25.27 | 21.66 | 24.47 |
| 3 | 28.00 | 27.46 | 27.85 | 27.14 |
| 4 | 23.81 | 25.67 | 20.97 | 23.94 |
| Mean | 25.50 | 26.30 | 24.78 | 25.78 |

`eta=0.50`

| Split | Est dc_trend% | Est total% | Val dc_trend% | Val total% |
|---:|---:|---:|---:|---:|
| 0 | 56.17 | 55.59 | 55.96 | 56.18 |
| 1 | 49.88 | 53.50 | 53.17 | 54.13 |
| 2 | 48.52 | 52.67 | 45.04 | 50.88 |
| 3 | 55.61 | 56.00 | 56.82 | 56.15 |
| 4 | 48.99 | 53.22 | 42.37 | 49.24 |
| Mean | 51.83 | 54.20 | 50.67 | 53.32 |

Transfer ratio (`Val/Est`):

| eta | dc_trend | total |
|---:|---:|---:|
| 0.20 | 0.972 | 0.981 |
| 0.50 | 0.978 | 0.984 |

Key findings:

1. Transfer ratios are 0.972 for dc_trend and 0.981 for total at `eta=0.20`, and 0.978/0.984 at `eta=0.50`.
2. Validation dc_trend reduction at `eta=0.20` has mean 24.78%, with range 21.0%-27.9% across splits.
3. Validation dc_trend reduction at `eta=0.50` has mean 50.67%, with range 42.4%-56.8% across splits.
4. Total MSE reduction is very close to dc_trend reduction, within 1-3 percentage points, indicating minimal spillover to other bands.
5. Cross-split variance is modest, confirming that the drift subspace is a stable property of the RSSM checkpoint, not an artifact of specific rollout noise.

## Resolution of Advisor Concerns

1. **Circularity:** Resolved. The near-perfect transfer ratio shows that `U_drift` estimated from 10 seeds works equally well on 10 unseen seeds. The drift subspace is a checkpoint-level property, not a per-rollout artifact.
2. **Total MSE > dc_trend reduction:** The cross-validated numbers show total and dc_trend reductions are nearly equal: 25.78% total vs 24.78% dc_trend at `eta=0.20`. The Month 4 gap, -50.99% vs -47.75%, was slightly inflated by in-sample evaluation, but the qualitative pattern holds.
3. **eta=0.50 halving the update:** This remains a valid concern for the real RSSM intervention. Post-hoc damping does not re-run the recurrent dynamics, so it cannot detect whether damping disrupts legitimate slow dynamics. This must be tested in Phase 2 with eval_return monitoring.
4. **No eval_return test:** Correct. Post-hoc damping cannot test eval_return because it does not interact with the policy. Phase 2 training is required for this.

## Impact on Phase 2

- `U_drift` (`r=10`) is validated as a non-circular basis for SMAD.
- `eta=0.20` is confirmed as the conservative default, with 24.78% dc_trend reduction on held-out data.
- The remaining open questions, whether damping transfers to real RSSM recurrence and whether eval_return is preserved, require Phase 2 training.
