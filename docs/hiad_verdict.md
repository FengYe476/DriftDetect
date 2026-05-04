# HIAD Verdict - Month 8

## Executive Summary

HIAD as originally designed did not clear the Month 8 intervention threshold.
The self-estimated `Delta z_t` PCA basis produced useful but insufficient slow
drift reduction: `J_slow` improved by `14.2%`, just below the `15%` marginal
threshold, while `J_total` worsened by `22.6%`. This is a negative result for
the original self-estimating HIAD method.

The investigation still produced a positive mechanism result. Inference-time
recurrent damping works when the basis is high quality. Post-hoc recurrent
damping with the diagnostic `U_drift` basis achieved `J_slow -26.3%` and
`J_total -20.7%` at `eta=0.30`, with raw latent MSE reduced by `27.7%`. There
was no error redistribution in this setting.

The method pivot is therefore:

```text
from: self-estimated update-basis HIAD
to:   diagnostic-derived basis + inference-time recurrent damping
```

The viable contribution is no longer that `Delta z_t` PCA can recover the
drift basis during inference. The viable contribution is that a drift basis
estimated from diagnostics can be used at inference time to reduce slow
imagination drift without retraining the RSSM.

## 1. Sanity Check Results

The HIAD sanity check tested whether PCA on raw RSSM update vectors aligns with
the known diagnostic basis `results/smad/U_drift_cheetah_r10.npy`.

| Window | Mean overlap | Std | Min | Max |
|---:|---:|---:|---:|---:|
| `25` | `0.145` | `0.025` | `0.075` | `0.223` |
| `50` | `0.178` | `0.031` | `0.115` | `0.262` |
| `100` | `0.238` | `0.025` | `0.191` | `0.281` |
| `200` / full | `0.208` | `0.025` | `0.177` | `0.274` |

The default HIAD window, `window_size=25`, had mean overlap `0.145`. The best
tested-window mean was `0.238` at `window_size=100`. All overlap means remained
below the `0.3` pivot threshold from `docs/hiad_design.md`, giving an early
PIVOT signal.

We proceeded to the first HIAD run anyway because overlap is only a geometric
proxy. The functional question was whether the imperfect basis could still
reduce `J_slow` enough to justify a sweep.

Source: `results/tables/hiad_sanity_check.json`.

## 2. HIAD First Run Results

The first HIAD rollout evaluation used `eta=0.20`, `r=10`,
`re_est_freq=25`, and `basis_mode=initial_basis`. The output-space frequency
evaluation used the same five-band Butterworth pipeline as the existing SMAD
evaluations over trim window `[25, 175]`.

| Metric | Baseline | HIAD | Change |
|---|---:|---:|---:|
| `J_slow` / `dc_trend` | `3.939` | `3.379` | `-14.2%` |
| `J_total` | `10.431` | `12.788` | `+22.6%` |
| `very_low` | `0.324` | `0.523` | `+61.1%` |
| `low` | `2.516` | `4.229` | `+68.1%` |
| `mid` | `2.255` | `3.141` | `+39.3%` |
| `high` | `1.397` | `1.516` | `+8.5%` |

The slow-band reduction was real but too small, and the total error increase
was unacceptable. HIAD with a `Delta z_t` PCA basis is therefore not viable as
the Month 8 intervention.

Source: `results/tables/hiad_first_run.json`.

## 3. Post-Hoc Recurrent Damping Results

The post-hoc damping test used the diagnostic `U_drift` basis instead of the
self-estimated HIAD basis. It evaluated deterministic latent space only,
slicing `true_latent[:, 1024:1536]` and `imagined_latent[:, 1024:1536]`. The
frequency pipeline fit one shared PCA on pooled `true_deter`, retained `38`
components at `95.0%` explained variance, and filtered latent errors into the
same five frequency bands.

Two modes were tested:

```text
non_recurrent: damp each original transition independently
recurrent:     chain damped states using original model deltas
```

The recurrent mode is a linear approximation: it uses
`imagined_deter[t+1] - imagined_deter[t]` from the saved rollout rather than
re-running `img_step` from the damped state. Even with that caveat, it directly
tests whether recurrent accumulation is the right intervention geometry.

| eta | Mode | J_slow change | J_total change | Raw MSE change |
|---:|---|---:|---:|---:|
| `0.20` | non-recurrent | `-0.0%` | `-1.1%` | `-1.1%` |
| `0.20` | recurrent | `-19.8%` | `-15.2%` | `-19.8%` |
| `0.30` | non-recurrent | `-0.0%` | `-1.5%` | `-1.6%` |
| `0.30` | recurrent | `-26.3%` | `-20.7%` | `-27.7%` |
| `0.50` | non-recurrent | `-0.0%` | `-2.2%` | `-2.3%` |
| `0.50` | recurrent | `-32.4%` | `-27.5%` | `-39.6%` |

No error redistribution appeared in recurrent mode. At `eta=0.30`, all
frequency bands improved: `dc_trend -26.3%`, `very_low -22.1%`, `low -17.9%`,
`mid -10.0%`, and `high -0.8%`. At `eta=0.50`, the same pattern held with
larger reductions.

The key conclusion is that recurrent damping is the effective mechanism, but
basis quality is the critical factor. Non-recurrent damping barely moves
`J_slow`; recurrent damping with the right basis clears the `25%` success
threshold.

Source: `results/tables/posthoc_recurrent_U_drift.json`.

## 4. Root Cause Analysis

The original HIAD proxy fails because raw transition updates and cumulative
drift are not the same object.

`Delta z_t = img_step(z_t, a_t) - z_t` captures where the model moves in one
transition. True drift is the accumulated difference between posterior and
imagined latent trajectories. The sanity check showed that the subspace of
large transition updates only weakly overlaps the diagnostic drift subspace:
about `0.15` at the default window and at most about `0.24` as a tested-window
mean.

This mismatch explains the first-run behavior. The `Delta z_t` PCA basis has
some overlap with true drift, so `dc_trend` improves by `14.2%`. But it also
projects onto non-drift directions, so damping redistributes error into
`very_low`, `low`, and `mid` bands.

The diagnostic `U_drift` basis is estimated from true-vs-imagined posterior
drift and aligns with the actual accumulated error direction. When recurrent
damping uses that basis, the intervention reduces slow drift and total error
together.

## 5. Revised Method: Diagnostic-Derived Inference Damping

The viable Month 8 method is diagnostic-derived inference damping:

```text
1. Collect diagnostic rollouts from the trained model.
2. Estimate U from true-vs-imagined deterministic latent drift.
3. Apply recurrent damping during inference or post-hoc latent rollout:
   Delta z_damped = Delta z - eta * U U^T Delta z
4. Chain the damped deterministic states through the imagination horizon.
```

This method has zero retraining cost and adds no model parameters. It does
require diagnostic rollouts, so it is not as self-contained as the original
HIAD proposal, but the empirical evidence is much stronger.

Ongoing validation is the controlled RunPod baseline plus anchor-only
Adaptive-SMAD run. That run is intended to produce a fresh `current_U` basis
estimated during training without applying transition damping. The next test is
whether this anchor-only `current_U` improves over the static Month 5
`U_drift` basis under the same inference-time recurrent damping procedure.

## 6. Impact on Paper Strategy

The original HIAD design should be archived as a negative result. It is useful
because it shows that not every model-internal low-dimensional signal is a
drift signal. This strengthens the paper's intervention-limits narrative and
connects directly to the posterior/drift mismatch finding.

Inference-time recurrent damping with a diagnostic basis is the viable method
contribution. It keeps the strongest part of HIAD - direct rollout-time
control - while replacing the weak self-estimation signal with a basis tied to
true drift.

If the anchor-only `current_U` basis outperforms static `U_drift`, the paper
gets a stronger adaptive-basis story. If it matches `U_drift`, the result is
still publishable with a simpler method: diagnostic basis estimation followed
by inference-time recurrent damping.

## 7. Verdict

| Question | Verdict |
|---|---|
| Does `Delta z_t` PCA recover the true drift geometry? | No. Mean overlaps stayed below `0.3`. |
| Is original HIAD viable as a self-estimating inference method? | No. `J_slow -14.2%`, `J_total +22.6%`. |
| Does inference-time recurrent damping work with a high-quality basis? | Yes. `J_slow -26.3%`, `J_total -20.7%` at `eta=0.30`. |
| What should Month 8 pursue next? | Diagnostic-derived or Adaptive-SMAD-derived basis plus recurrent inference damping. |
