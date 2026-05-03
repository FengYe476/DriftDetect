# Mismatch Finding Evidence: Posterior Slow Features Are Not Drift Directions

## 1. Finding Statement

Posterior-estimated slow subspaces and empirical drift subspaces are misaligned
in trained autoregressive world models. This misalignment explains why SMAD with
the `U_posterior` basis failed in Pre-check 1 while the `U_drift` basis partially
succeeded.

## 2. Evidence Table

Posterior/drift overlap is computed as:

```text
trace(A A^T B B^T) / r
```

where `A` and `B` are rank-`r` orthonormal basis matrices. Lower values indicate
stronger mismatch.

| Setting | r=3 | r=5 | r=10 | dc_trend% | Notes |
|---|---:|---:|---:|---:|---|
| V3 Cheetah | 0.13 | 0.17 | 0.28 | 46.7% | Strong mismatch |
| V3 Cartpole | 0.06 | 0.08 | 0.12 | 97.0% | Strongest mismatch |
| V4 Cheetah | 0.64 | 0.41 | 0.43 | 63.9% | Partial mismatch |
| Toy simple | 0.61 +/- 0.16 | 0.70 +/- 0.11 | 0.80 +/- 0.01 | 19.1% | Near-aligned |
| Toy medium | 0.41 +/- 0.14 | 0.67 +/- 0.14 | 0.85 +/- 0.05 | 43.1% | Partial mismatch |
| Toy hard | 0.38 +/- 0.04 | 0.50 +/- 0.10 | 0.74 +/- 0.18 | 38.3% | Moderate mismatch |

Sources:

- V3 Cheetah: `results/tables/smad_alignment_check.json`
- V3 Cartpole: `results/tables/cartpole_mismatch_check.json`
- V4 Cheetah: `results/tables/v4_mismatch_check.json`
- Toy simple/medium/hard: `results/tables/toy_mismatch_complexity.json`

## 3. Interpretation

V3 models show strong mismatch. V3 Cheetah has posterior/drift overlap below
`0.3` at ranks `3`, `5`, and `10`, and V3 Cartpole is even lower, with overlap
between `0.06` and `0.12`.

V4 Cheetah shows partial mismatch. Its rank-3 overlap is higher at `0.64`, but
the rank-5 and rank-10 values are `0.41` and `0.43`. This is not an
unambiguous strong-mismatch result, but it is also not clean posterior/drift
alignment.

The toy models show complexity-dependent mismatch. The simple setting is
near-aligned, especially at rank `10`. The hard setting has lower rank-3 and
rank-5 overlap and therefore more mismatch. This suggests that mismatch is not
universal in all dynamical systems; it emerges more strongly in trained
real-model settings and in more complex toy settings.

SFA slow features are also misaligned with empirical drift directions in the
real-model settings:

| Setting | r=3 SFA/drift | r=5 SFA/drift | r=10 SFA/drift |
|---|---:|---:|---:|
| V3 Cheetah | 0.017 | 0.061 | 0.090 |
| V3 Cartpole | 0.047 | 0.072 | 0.092 |
| V4 Cheetah | 0.008 | 0.007 | 0.067 |

This matters because it shows that the mismatch is not merely an artifact of
the PCA plus autocorrelation posterior-basis procedure. Independently estimated
slow features also fail to recover the drift subspace in the real-model cases.

The mismatch is therefore not universal, but it is present in all trained
real-model settings tested so far.

## 4. Connection to SMAD Results

Pre-check 1 failed with `U_posterior` because the posterior slow-feature basis
targets slow variation in posterior latent trajectories, not the empirical
directions along which open-loop imagination drifts. In V3 Cheetah, the
posterior/drift overlap was only `0.13` to `0.28` across ranks `3`, `5`, and
`10`, so damping the posterior slow basis was not expected to target the actual
drift direction well.

Post-hoc SMAD with `U_drift` partially succeeded because it directly targeted
the empirical drift directions. This is the clearest operational consequence of
the mismatch: the basis that looks theoretically natural from posterior slow
features is not the basis that best controls imagination drift.

B1 correction with eta=0 rollouts confirmed that drift directions shift
substantially during SMAD training. The overlap between baseline and post-SMAD
drift subspaces is `0.005` to `0.022` across ranks `3`, `5`, and `10`, which is
near-orthogonal. Post-hoc damping with the re-estimated basis reduces dc_trend
by `22.6%`, exceeding the `20%` threshold that fixed-basis SMAD failed to meet
during training (`4.8%`). This confirms that basis staleness, not model
adaptation, is the primary cause of the training-time effectiveness gap.

## 5. Paper Narrative

The mismatch between posterior slow features and drift directions is a key
challenge for world-model intervention. It explains why naive slow-feature
regularization can fail and motivates estimating drift directions directly from
open-loop rollout errors.

The paper narrative should distinguish two objects:

- posterior slow subspaces: directions that vary slowly under posterior
  inference on observed trajectories
- empirical drift subspaces: directions where imagined trajectories diverge
  from posterior trajectories

SMAD's empirical success depends on the second object. The theory explains why
slow subspaces are identifiable under idealized assumptions, but the experiments
show that identifying posterior slow features is not sufficient when the goal is
to damp learned-model drift.

This finding motivates drift-direction estimation as a practical bridge between
the slow-subspace theory and intervention in nonlinear trained world models.

## 6. Limitations

B1 correction with eta=0 rollouts has been completed and confirms the
staleness interpretation.

The V4 result is mixed. Rank-3 overlap is relatively high, while rank-5 and
rank-10 overlaps are moderate. This supports partial mismatch, not a clean
strong-mismatch claim.

Toy models are much simpler than DreamerV3 and V4. The toy sweep is useful for
showing complexity dependence, but it should not be treated as a one-to-one
mechanistic explanation of real-model mismatch.

The evidence currently covers V3 Cheetah, V3 Cartpole, V4 Cheetah, and three
toy complexity settings. More tasks and architectures would be needed to claim
that posterior/drift mismatch is universal across autoregressive world models.
