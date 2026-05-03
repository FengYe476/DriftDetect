# B1 Verdict and Month 7 Plan

## B1 Verdict

STALENESS is the primary cause of SMAD's training-time effectiveness gap.

The corrected B1 analysis used eta=0 rollouts from the SMAD-trained checkpoint,
so damping was disabled during rollout collection. This measures the natural
drift structure learned by the SMAD-trained model.

## Evidence

Corrected overlap between the baseline drift basis and the post-SMAD drift basis:

| Rank | overlap |
|---:|---:|
| r=3 | 0.0053 |
| r=5 | 0.0130 |
| r=10 | 0.0224 |

These values are near-orthogonal. The original fixed basis is essentially stale
after SMAD training.

Post-hoc damping on the corrected eta=0 rollouts:

| Basis | dc_trend reduction |
|---|---:|
| Re-estimated post-SMAD basis | 22.6% |
| Original baseline basis | 0.9% |

The re-estimated basis exceeds the `20%` threshold, while the old basis has
almost no effect. This explains the gap between post-hoc effectiveness and
training-time fixed-basis SMAD.

## Month 7 Recommendation

Proceed with an Adaptive-SMAD prototype.

Initial design:

- Task: Cheetah Run.
- First test: single seed, 500k steps.
- Basis: `U_drift`, re-estimated during training.
- Re-estimation frequency: start with every `50k` steps.
- Faster fallback cadence: every `25k` steps if early diagnostics show rapid
  basis drift.
- EMA smoothing: alpha `0.7` default.
- Primary metric: dc_trend MSE reduction.
- Secondary metrics: eval return, total latent MSE, basis overlap across
  re-estimation points, collapse warnings.

Success criterion:

```text
Adaptive-SMAD J_slow / dc_trend reduction >= 20%
```

This restores the post-hoc intervention threshold that fixed-basis SMAD missed
during training.

## Fallback

If Adaptive-SMAD achieves less than `10%` J_slow reduction, treat basis
staleness as only part of the failure mode and explore adaptation-aware methods.

Candidate fallback directions:

- Penalize drift directly rather than only damping a subspace.
- Add an anti-compensation regularizer on orthogonal directions.
- Use an ensemble or replay buffer of recent drift bases instead of a single
  current basis.
- Couple basis updates to validation-rollout drift diagnostics rather than a
  fixed training-step schedule.

## Risk

Even with re-estimation, the model may adapt faster than the re-estimation
frequency. The extreme corrected overlaps, `0.005-0.022`, imply that the drift
subspace can move dramatically during training.

Month 7 should therefore monitor basis overlap at every re-estimation point. If
successive bases are already near-orthogonal at a `50k` cadence, rerun with a
`25k` cadence before abandoning the Adaptive-SMAD idea.
