# HIAD Design Document

## 1. Motivation

HIAD shifts SMAD from training-time regularization to inference-time rollout
control. The motivation is the gap between what damping can do post-hoc and
what training-time methods have achieved so far.

The fixed-basis SMAD Phase 2 result was safe but below the intervention
threshold: training-time damping reduced `J_slow` by only `4.8%`. The same
basic damping idea was much stronger when applied post-hoc to held-out
rollouts, where the pre-check eta sweep showed `24.8%` `dc_trend` reduction.
B1 then showed that a re-estimated post-SMAD basis recovers `22.6%` post-hoc
`dc_trend` reduction on corrected eta=0 rollouts, while the old baseline basis
recovers only `0.9%`.

Adaptive-SMAD addressed basis staleness by refreshing `U_drift` during
training, but the controlled Month 7 comparison should be treated as an
approximately `10%` `J_slow` reduction. The larger Month 7 cross-pod number is
invalid for planning because it compared runs across different pod
environments and dependency versions.

The working interpretation is:

```text
post-hoc damping works because it directly edits the imagined transition
training-time damping is weaker because the RSSM and actor can adapt around it
inference-time damping avoids that adaptation path
```

HIAD therefore keeps the part that works - direct damping of RSSM imagination -
and moves adaptation into the basis estimator rather than the trained model.
The goal is a zero-training-cost method that updates its damping basis during a
single open-loop rollout.

## 2. Self-Consistency Drift Definition

The tempting definition of self-consistency drift is:

```text
drift_t = z_t^imag - img_step(z_{t-1}^imag, a_{t-1})
```

For a vanilla open-loop rollout, this quantity is exactly zero by construction.
The imagined latent `z_t^imag` is produced by applying `img_step` to
`z_{t-1}^imag` with the previous action. Without access to posterior or
ground-truth latents, vanilla self-consistency cannot reveal true drift.

HIAD's actual re-estimation signal is PCA on a sliding buffer of recent raw
RSSM update vectors, each captured before the current step's damping is
applied:

```text
Delta z_t = img_step(z_t, a_t) - z_t
U_t = top-r PCA components of {Delta z_{t-W}, ..., Delta z_{t-1}}
```

Note: from the second re-estimation window onward, the input state `z_t` is
itself the product of earlier damped transitions. The update vectors are
therefore "pre-current-damping" rather than "fully undamped." This feedback
loop is discussed further in Section 7.

Each `Delta z_t` captures the direction and magnitude of one RSSM transition
update in deterministic latent space. If imagination drift concentrates in a
low-dimensional subspace, the principal components of recent update vectors
should align with that active transition subspace. This is a proxy, not true
drift, because it uses model-internal motion rather than
`imagined_latent - posterior_latent`.

This mirrors the convention in `src/smad/U_estimation.py`.
`estimate_U_from_drift()` runs PCA on pooled deterministic drift vectors with
shape `(N, 512)` and returns an orthonormal `(512, r)` basis. HIAD keeps the
same PCA interface and deterministic-state dimensionality, but substitutes the
raw update buffer because posterior latents are unavailable during inference.

## 3. Algorithm

HIAD runs inside a single imagination rollout. It maintains a current basis
`U`, a projector `P = U U^T`, and a sliding window of recent raw RSSM update
vectors.

Default mode uses the pre-computed Cheetah basis:
`results/smad/U_drift_cheetah_r10.npy`.

```text
inputs:
  rssm, initial_state, actions[0:H]
  initial_basis path
  eta, r, re_est_freq, window_size

initialize:
  U = load_U(initial_basis)
  P = U U^T
  buffer = empty sliding window with max length window_size
  z = initial_state

for t in 0 .. H-1:
  z_raw_next = original_img_step(z, actions[t])
  Delta z_t = z_raw_next.deter - z.deter
  # Note: z.deter may already be damped from earlier steps.
  # Delta z_t is pre-current-step-damping, not fully undamped.

  z_next.deter = z.deter + Delta z_t - eta * P * Delta z_t
  z_next.stoch and stats = recompute from z_next.deter

  append Delta z_t to buffer
  z = z_next

  if (t + 1) % re_est_freq == 0 and len(buffer) >= r:
    U = top-r PCA components of buffer
    P = U U^T
```

In code, the damping step should preserve the same structure as
`MutableImgStepPatch` in `src/smad/img_step_patch.py`: call the original
`img_step` to obtain the raw prior, compute the deterministic update, damp the
projected update, then recompute stochastic statistics from the damped
deterministic state.

### self_bootstrap mode

`self_bootstrap` removes the need for a pre-computed `U_drift` file. Instead
of loading `results/smad/U_drift_cheetah_r10.npy`, HIAD runs the first
`re_est_freq` steps with `eta=0`, collects the initial `Delta z_t` buffer, runs
PCA on that buffer, and uses the result as the first active `U`. After that,
the rollout proceeds normally with damping and periodic re-estimation.

```text
self_bootstrap mode:

initialize:
  U = None
  P = zero matrix
  buffer = empty sliding window
  z = initial_state

for t in 0 .. H-1:
  z_raw_next = original_img_step(z, actions[t])
  Delta z_t = z_raw_next.deter - z.deter

  if U is not None:
    z_next.deter = z.deter + Delta z_t - eta * P * Delta z_t
  else:
    z_next.deter = z_raw_next.deter   # no damping during bootstrap

  z_next.stoch and stats = recompute from z_next.deter
  append Delta z_t to buffer
  z = z_next

  if (t + 1) % re_est_freq == 0 and len(buffer) >= r:
    U = top-r PCA components of buffer
    P = U U^T
```

This mode matters for the paper because it makes HIAD a fully self-contained
inference-time method. It requires no diagnostic rollout archive and no
separate true-vs-imagined basis estimation pipeline.

The trade-off is that the first window is undamped. With the default
`re_est_freq=25` and `horizon=200`, the first `12.5%` of the rollout is
unprotected. The Week 2 sweep should therefore compare `initial_basis` and
`self_bootstrap` directly rather than assuming the self-contained mode is free.

## 4. Relationship to Existing Code

HIAD should reuse the runtime damping path in `src/smad/img_step_patch.py`.
`MutableImgStepPatch` already owns an updateable projection matrix and exposes
`update_U(U)`, which is the right interface for replacing the basis after each
PCA refresh. The new HIAD logic should add update-buffer collection and
rollout-step cadence around that patch, not replace the damping formula.

HIAD should reuse the PCA convention from `src/smad/U_estimation.py`.
`estimate_U_from_drift()` already validates the `(N, 512)` input shape,
centers the pooled vectors, runs SVD, and returns an orthonormal `(512, r)`
basis. HIAD can treat its recent `Delta z_t` buffer as the PCA pool.

HIAD differs from `AdaptiveReEstimator` in
`src/smad/adaptive_smad/re_estimation.py`. Adaptive-SMAD collects fresh
rollouts from the current training model and estimates drift from
`imagined_latent - true_latent`. HIAD is inference-only: it has no training
loop, no environment posterior targets, and no true latent trajectory. Its
basis update is therefore driven by the recent raw RSSM update buffer.

HIAD is closer to `AdaptiveSMADScheduler` in
`src/smad/adaptive_smad/scheduler.py` at the control-flow level. Both methods
replace `U` directly, track basis history, and can report subspace overlap.
The difference is cadence: Adaptive-SMAD schedules by training step, while HIAD
schedules by imagination step inside a single rollout.

`src/diagnostics/extract_rollout.py` shows the open-loop flow that HIAD would
instrument. During the imagination window, the script advances
`imag_latent = agent._wm.dynamics.img_step(imag_latent, action_tensor,
sample=False)`. HIAD should hook this transition point so it can record the raw
deterministic update before damping and then pass the damped latent to the
decoder.

## 5. Configuration Parameters

| Parameter | Default | Sweep / Options | Purpose |
|---|---:|---|---|
| `re_est_freq` | `25` | `{10, 25, 50}` | Number of imagination steps between PCA basis refreshes. |
| `r` | `10` | `{5, 10, 20}` | Rank of the damping basis. |
| `eta` | `0.20` | `{0.10, 0.20, 0.30}` | Damping strength applied to projected RSSM updates. |
| `initial_basis` | `results/smad/U_drift_cheetah_r10.npy` | path | Initial `U` for `basis_mode=initial_basis`. |
| `basis_mode` | `initial_basis` | `{initial_basis, self_bootstrap}` | Whether to load a pre-computed basis or estimate the first basis from the rollout itself. |
| `window_size` | `25` | match `re_est_freq` for v1 | Number of recent raw update vectors used for PCA. |
| `min_buffer_size` | `r` | `{r, 2*r}` | Minimum number of Delta z vectors required before the first PCA. Guards against rank-deficient SVD. |
| `horizon` | `200` | fixed for first pass | Open-loop imagination horizon used in the Month 8 HIAD screen. |

In `basis_mode=initial_basis`, HIAD starts damping at step 0 using
`initial_basis`. This gives the strongest early-horizon protection but depends
on a pre-analysis artifact.

In `basis_mode=self_bootstrap`, HIAD starts with `eta=0` for the first
`re_est_freq` steps, estimates `U` from the initial update buffer, then enables
normal damping and periodic re-estimation. This mode is the stronger paper
claim because it is self-contained at inference time. The cost is that, with
the default `25`-step bootstrap window in a `200`-step rollout, `12.5%` of the
rollout is undamped.

## 6. Sanity Check Protocol

The Week 1 Day 5 sanity check tests whether the HIAD update-buffer proxy
recovers the known fixed-basis drift geometry.

Procedure:

```text
1. Using the baseline Cheetah checkpoint (`results/checkpoints/cheetah_run_500k.pt`),
   run 20 undamped open-loop rollouts (reuse existing seeds 0-19 if available,
   or generate fresh ones).
2. Collect the HIAD Delta z_t buffer over the target window.
3. Estimate U_hiad with the same rank as U_drift_cheetah_r10.npy.
4. Compute overlap with the fixed true-vs-imagined basis:

   overlap = ||U_fixed.T @ U_hiad||_F^2 / r
```

The fixed basis `U_fixed` is loaded from
`results/smad/U_drift_cheetah_r10.npy`, which was estimated from
true-vs-imagined posterior drift in the Month 5 pre-check pipeline.

Interpretation:

| Overlap | Verdict | Action |
|---:|---|---|
| `> 0.7` | Proxy recovers the fixed-basis drift geometry. | Proceed to the HIAD sweep. |
| `0.3-0.7` | Mixed signal. | Run eta/r sensitivity before scaling. |
| `< 0.3` | Proxy is likely mismatched to true drift. | Avoid spending on scale-up and pivot. |

This check is not a performance result. It only tests whether raw update PCA is
geometrically close enough to the known `U_drift` basis to justify a controlled
HIAD evaluation.

## 7. Caveats

Self-consistency drift is not true drift. True drift requires posterior or
ground-truth latent trajectories, while HIAD only sees the model's own open-loop
updates. The method should therefore be described as an inference-time proxy,
not as a replacement for true-vs-imagined diagnostics.

HIAD can only damp directions represented in its recent update buffer. If the
harmful slow drift direction is not active in the local window, short-window
PCA may miss it.

HIAD may reinforce the basis it is already damping. Once damping changes the
rollout trajectory, later update buffers are drawn from the damped trajectory's
state distribution. This is still useful for inference-time control, but it is
not an unbiased estimate of the baseline model's natural drift subspace.

Initial basis quality matters in `initial_basis` mode. A stale or task-mismatched
basis can waste early damping steps before the first re-estimation. Conversely,
`self_bootstrap` avoids pre-analysis but leaves the first window unprotected.

Short windows can have high variance. With `window_size=25` and `r=10`, the PCA
pool is small relative to the `512`-dimensional deterministic state. The W1
sanity check and W2 sweep should monitor overlap stability across seeds.

## 8. Expected Outcomes

Best case: HIAD reduces `J_slow` by at least `25%`, clearing both the `22.6%`
corrected post-SMAD new-basis post-hoc baseline and the earlier `24.8%`
post-hoc damping screen. This would support HIAD as a zero-training-cost
intervention that preserves the strong part of SMAD while avoiding training
adaptation.

Acceptable case: HIAD reduces `J_slow` by `15-25%`. This would still be useful
as a cheap inference-time method, especially if `self_bootstrap` performs close
to `initial_basis`.

Failure case: HIAD reduces `J_slow` by less than `15%`. Treat this as evidence
that raw update PCA is not a strong enough proxy for true slow drift, then pivot
to Path A: the mechanism paper centered on drift diagnosis, mismatch, basis
staleness, and the limits of intervention.
