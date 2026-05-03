# Adaptive-SMAD Design Document

## 1. Motivation from B1

The B1 staleness check shows that the fixed `U_drift` basis used in Phase 2
becomes stale during SMAD training. The corrected B1 analysis measured the
natural post-training drift structure of the SMAD-trained model by collecting
rollouts with damping disabled during inference.

The overlap between the original baseline drift basis and the post-SMAD drift
basis was near zero: `0.005-0.022` across ranks `r in {3, 5, 10}`. This means
the post-SMAD drift subspace is nearly orthogonal to the original basis.

Post-hoc damping on corrected eta=0 rollouts confirms that the issue is basis
staleness:

| Basis | dc_trend reduction |
|---|---:|
| Re-estimated post-SMAD basis | 22.6% |
| Original baseline basis | 0.9% |

The re-estimated basis recovers a `22.6%` `dc_trend` reduction, clearing the
original `20%` intervention threshold. The old fixed basis achieves only
`0.9%` on the same rollouts. This explains the gap between post-hoc
effectiveness (`24.8%`) and training-time fixed-basis effectiveness (`4.8%`):
the RSSM's drift directions move during training, so the fixed basis no longer
targets the active drift subspace.

Adaptive-SMAD addresses this failure mode by periodically re-estimating
`U_drift` during training so the damping basis tracks the evolving drift
geometry.

## 2. Core Design Decisions

### Re-estimation Frequency

Default cadence: re-estimate `U_drift` every `50k` training steps. For a
`500k`-step training run, this gives 10 re-estimations.

Fallback cadence: switch to every `25k` steps if early diagnostics show rapid
basis drift.

Rationale: each re-estimation is expected to take approximately 30 seconds,
covering `5-10` fresh imagination rollouts plus PCA. At a `50k` cadence, total
overhead is about 5 minutes over a 10-hour training run, which is small enough
to justify the stronger diagnostic and intervention value.

### No EMA Smoothing

Adaptive-SMAD should directly replace `U` at each re-estimation point.

Rationale: B1 showed baseline/post-SMAD overlap of only `0.005-0.022`, meaning
old and new bases can be nearly orthogonal. Under that geometry, an EMA update
with `alpha=0.7` would still be dominated by the fresh basis after
orthonormalization, while adding implementation complexity and another
hyperparameter. Direct replacement is simpler, easier to test, and better
matched to the observed staleness failure.

### Re-estimation Data Source

At each re-estimation step, collect `5-10` fresh imagination rollouts from the
current model checkpoint. Compute drift vectors from the difference between
imagined and true latent trajectories:

```text
drift_t = imagined_latent_t - true_latent_t
```

Then run PCA on the deterministic-state drift vectors and use the top `r`
directions as the new `U_drift`.

Do not use training imagination data for basis re-estimation. Training
imagination quality varies over time and can introduce tight coupling between
the basis estimator and the current minibatch optimization path. Fresh
diagnostic rollouts provide a cleaner estimate of the current model's drift
subspace.

### Activation Schedule gamma(s)

Use the same activation schedule as fixed-SMAD:

```text
s0 = 50k
s1 = 100k
gamma(s) = 0 for s < s0
gamma(s) = (s - s0) / (s1 - s0) for s0 <= s < s1
gamma(s) = 1 for s >= s1
```

The first re-estimation happens at `s0=50k`, coinciding with damping
activation. Before `s0`, the run should behave like the baseline identity path
with no active damping.

### Damping Parameters

Use the same conservative defaults as Phase 2 SMAD:

```text
eta = 0.20
r = 10
```

Rationale: Phase 2 showed that this setting preserved `eval_return` and avoided
representation collapse. Adaptive-SMAD changes only the freshness of the basis,
not the default damping strength or rank.

### U Persistence

Save `U_drift` to the checkpoint at each re-estimation point.

Rationale: persisted bases are required for resume capability and for post-hoc
analysis of the basis trajectory. The saved trajectory should allow later
analysis of `overlap(U_t, U_{t-1})`, `overlap(U_t, U_baseline)`, and the timing
of any rapid basis rotations.

## 3. Architecture

### `src/smad/adaptive_smad/re_estimation.py`

Owns periodic `U_drift` re-estimation.

Responsibilities:

- Take the current model checkpoint or live model object.
- Collect `N` fresh imagination rollouts, with `N` defaulting to `5-10`.
- Extract true and imagined latent trajectories.
- Compute deterministic-state drift vectors, matching the existing
  `src/smad/U_estimation.py` convention.
- Run PCA over the drift vectors.
- Return the top `r` orthonormal directions as `U_drift`.

This module should reuse the existing `U_estimation` conventions where possible:
`deter` starts at latent index `1024`, has dimension `512`, and the output basis
has shape `(512, r)`.

### `src/smad/adaptive_smad/scheduler.py`

Manages the adaptive schedule.

Responsibilities:

- Track the current training step.
- Trigger re-estimation at the configured frequency.
- Apply the existing `gamma(s)` activation schedule from
  `src/smad/anchor_loss.py`.
- Coordinate with the existing `img_step_patch` mechanism.
- Persist each refreshed `U_drift` for checkpoint resume and analysis.
- Emit monitoring metrics at every re-estimation point.

### Training Integration

The training loop calls:

```python
scheduler.maybe_update(step, model)
```

at each training step.

If `step` is not a re-estimation point, `maybe_update` is a no-op except for
returning the current activation value and state. If `step` is a re-estimation
point, the scheduler collects fresh rollouts, estimates a new `U_drift`, builds
the corresponding projector `P = U @ U.T`, and updates the active SMAD damping
state in place.

The existing `img_step_patch` currently closes over a fixed projector `P`.
Adaptive-SMAD should therefore either:

- patch `img_step` with a mutable holder for `P`, updated in place by the
  scheduler, or
- install a new patch when `U` changes and restore the previous patch cleanly.

The mutable-holder approach is preferable because it minimizes patch churn and
makes the identity test easier to reason about.

## 4. Identity Test Specification

The critical correctness test is that Adaptive-SMAD reduces exactly to
fixed-basis SMAD when re-estimation is disabled.

Configuration:

- Set `re_estimation_freq = infinity`, meaning never re-estimate after
  initialization.
- Use the same initial `U_drift`.
- Use the same `eta`, `r`, and `gamma(s)` schedule.
- Use the same random seeds, minibatches, environment inputs, and rollout
  settings.
- Run both fixed-basis SMAD and Adaptive-SMAD for 100 training steps.

Assertion:

```text
All model parameters, losses, and rollout outputs are bitwise identical.
```

This test verifies that the adaptive scheduler introduces no behavioral change
unless a re-estimation event actually occurs.

## 5. Monitoring Metrics

At each re-estimation point, log:

| Metric | Purpose |
|---|---|
| `overlap(U_t, U_{t-1})` | Measures basis change speed across adjacent updates. |
| `overlap(U_t, U_baseline)` | Measures cumulative basis drift from the initial baseline basis. |
| `J_slow` on a quick diagnostic rollout | Checks whether the refreshed basis is improving the target frequency band. |
| Re-estimation wall-clock time | Tracks overhead and confirms the cadence remains practical. |

Use the same overlap convention as the existing SMAD analyses:

```text
overlap(U_a, U_b) = ||U_a.T @ U_b||_F^2 / r
```

## 6. Risk and Fallback

If `overlap(U_t, U_{t-1})` stays below `0.1` at the `50k` cadence, switch to a
`25k` cadence. Persist the decision in the run metadata so downstream analysis
can distinguish the original schedule from the accelerated one.

If `eval_return` drops by more than `10%` relative to the baseline, abort the
run and diagnose before continuing. Phase 2 fixed-basis SMAD was safe, so a
large return drop would indicate that adaptive updates are disturbing useful
dynamics or shifting the policy training distribution too aggressively.

If `J_slow` improvement remains below `10%` even with Adaptive-SMAD, treat basis
staleness as insufficient to explain the full training-time gap. In that case,
investigate adaptation-aware methods, such as direct drift penalties,
anti-compensation regularizers, or ensembles of recent drift bases.
