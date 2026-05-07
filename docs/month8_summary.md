# Month 8 Summary (2026-09)

## Executive Summary

Month 8 began with a plan centered on HIAD, an inference-time damping method
intended to suppress slow drift directions in Dreamer world-model rollouts. The
actual trajectory diverged early. A self-consistency sanity check showed that
the PCA directions of `Delta z_t` had overlap below `0.3` with the learned
`U_drift` basis, which made the intended damping target unreliable. HIAD was
therefore abandoned as a primary intervention rather than taken into the
planned configuration sweep.

The work then shifted from inference-time modification to training-time
mechanism analysis. Anchor loss and Adaptive-SMAD experiments showed that local
penalties can reduce drift on a selected subspace while leaving total latent
drift unchanged or slightly worse. DriftHead experiments further showed that
drift is predictable, but recurrently applying corrections at inference caused
out-of-distribution divergence. These results redirected the project toward
losses that alter the RSSM during training instead of modifying imagined states
after training.

The strongest Month 8 result to date is the moment matching method. On Cheetah
V3 at `500k` environment steps with `seed=42`, moment matching reduced latent
`J_total` from `0.173` to `0.018`, a `-89.6%` change, while peak `eval_return`
changed from `718.5` to `703.2`, a `-2.1%` decrease. The per-step drift
increment also fell from `0.0062` to `0.0009` (`-84.8%`), supporting the
interpretation that the cumulative reduction is driven by a reduction in the
one-step error source rather than by metric redistribution.

The method is principled in the limited but concrete sense that it closes all
known escape routes observed during Month 8: direction redistribution,
frequency redistribution, and noise variance increase. Cross-architecture
validation on V4 and cross-task validation remain pending, and the current
result is still single-seed. The velocity MSE trade-off, `+9.9%` in the first
moment matching run, also remains an active limitation.

## Key Result: Moment Matching Method

### Method Design

Moment matching adds a batch-level penalty on the one-step RSSM prediction
error:

```text
L_moment = beta_mean * ||E_batch[epsilon_t]||^2
         + beta_var  * ||Var_batch[epsilon_t]||^2

epsilon_t = f(post_t, a_t) - post_{t+1}
```

The mean term penalizes systematic one-step bias across all latent dimensions,
which addresses direction redistribution. The loss is single-step rather than
horizon-aggregated, so it removes the explicit time axis that allowed frequency
redistribution in earlier methods. The variance term explicitly penalizes
growth in one-step error variance, closing the noise-variance escape route
observed in the mean-only true bias loss experiment.

The first successful configuration used `beta_mean=1.0`, `beta_var=0.1`,
`n_starts=8`, a `50k` environment-step warmup, and `batch=32`. The batch size
and core baseline training settings were unchanged from the controlled
DreamerV3 baseline. The implementation is in `scripts/train_moment_match.py`
with configuration in `configs/month8_moment_match.yaml`.

The loss is integrated into `total_model_loss` before the single backward pass,
matching the anchor-loss pattern in `scripts/train_adaptive_smad.py` around
lines `817-831`. This avoids the optimizer conflict that affected the first
bias-loss implementation, where a separate backward pass introduced invalid
gradient interactions with the model optimizer.

### Quantitative Results (Cheetah V3, 500k steps, seed=42)

| Metric | Baseline | Moment Match | Change |
|--------|---------:|-------------:|-------:|
| Peak eval_return | 718.5 | 703.2 | -2.1% |
| Latent J_total (trim 25:175) | 0.173 | 0.018 | -89.6% |
| Per-step drift increment | 0.0062 | 0.0009 | -84.8% |
| Step 0->1 drift accumulation | 0.041 | 0.00016 | -99.6% |
| Position MSE | 0.9100 | 0.8200 | -9.6% |
| Velocity MSE | 19.13 | 21.02 | +9.9% |
| Early band drift (25-75) | 0.163 | 0.018 | -88.9% |
| Mid band drift (75-125) | 0.176 | 0.018 | -89.7% |
| Late band drift (125-175) | 0.180 | 0.018 | -90.2% |

Several checks support the interpretation that the `-89.6%` `J_total`
reduction is not a measurement artifact. The per-step increment reduction
(`-84.8%`) is consistent with the cumulative reduction, and the rollouts were
extracted from fresh actor episodes rather than reused training batches. The
early, mid, and late time bands improved by similar amounts, so the result does
not appear to be driven by moving drift from one part of the imagined horizon to
another.

## Method Comparison Table

| Method | eval_return Change | J_total Change | Failure Mode |
|--------|-------------------:|---------------:|--------------|
| Baseline DreamerV3 | reference | reference | Reference condition |
| Anchor loss (`beta=0.1`) | ~0% | +4.1% | Direction redistribution |
| Adaptive-SMAD | ~0% | -10% (`J_slow`) | Basis chasing, conservation |
| HIAD recurrent | not evaluated | ~+45% | OOD divergence |
| DriftHead recurrent | not evaluated | ~+74000% | OOD divergence |
| DriftHead non-recurrent (post-hoc) | not evaluated | -42.9% | Works post-hoc, but does not change RSSM |
| DriftHead v2 (actor uses corrected feat) | -16% (`618.4`) | not measured | Distribution shift |
| Bias loss (`||drift increment||^2`) | unclear | killed early | Optimizer conflict bug |
| True bias loss (mean only) | -49% (`~370` estimated) | killed early | Noise variance escape (`var: 0->27.5`) |
| **Moment matching** | **-2.1% (`703.2`)** | **-89.6%** | **None confirmed yet** |

## Methodological Journey

The first week focused on HIAD and the assumptions needed for inference-time
damping. The core sanity check compared `Delta z_t` PCA directions against the
`U_drift` basis and found overlap below `0.3`. This made the target of damping
ambiguous: the learned drift subspace did not align with the directions that
would actually be modified during inference. On that basis, HIAD was treated as
fundamentally compromised and the planned `54`-configuration sweep was skipped.

The next phase characterized anchor loss and Adaptive-SMAD under a fair-step
controlled comparison. These experiments confirmed that apparent progress on a
selected direction can coexist with no improvement in total drift. Anchor loss
reduced drift on the baseline `U` direction, but drift increased in other
directions and net `J_total` rose by `+4.1%`. This established drift
conservation as a central failure mechanism rather than a minor implementation
issue.

DriftHead experiments then tested whether drift is predictable enough to
correct. The head's loss converged to approximately `0.1`, and its gate grew
from `0` to `1.8`, indicating learnable structure in the drift signal. A
non-recurrent post-hoc application reduced `J_total` by `-42.9%`, but it did
not change the trained RSSM. Recurrent application of the correction triggered
out-of-distribution divergence, matching the broader HIAD result that latent
state modification at inference destabilizes imagined rollouts.

The project then moved to training-time bias losses. The first bias-loss
variant penalized `||drift_{t+1} - drift_t||^2` but was stopped early after an
implementation bug was identified: the loss used a separate backward pass,
creating an optimizer conflict. The true bias-loss variant penalized
`||E[epsilon]||^2` and was integrated correctly into the model loss, but the
RSSM escaped by increasing noise variance. The diagnostic metric
`mm_noise_var` grew from `0` to `27.5` over `30k` steps, and the run was stopped
at `85k`.

Moment matching added the missing variance penalty to the mean-bias penalty.
With both terms active, the RSSM no longer had the known direction, frequency,
or noise-variance routes available in the observed experiments. The resulting
Cheetah V3 run produced the largest total-drift reduction in the project so far
while preserving task performance within `2.1%` of the controlled baseline.

## Conservation Phenomenon Documentation

The Month 8 conservation result, Finding D, is the key negative result behind
the pivot to global training-time penalties. Anchor loss reduced `J_slow` by
`-79.5%` on the baseline `U` direction, but drift on other directions increased
by `+486%`. The net `J_total` change was `+4.1%`, so the intervention did not
reduce total latent drift despite appearing successful on the targeted metric.

The slow-subspace fraction also shifted sharply, from `11.8%` in the baseline
to `66.7%` under anchor loss. This does not indicate a clean reduction in drift;
it indicates that the metric's decomposition changed under the intervention.
The result motivated moment matching's global penalty on the one-step
prediction error rather than another local penalty on a selected subspace or
frequency band.

## Compute Budget

| Experiment | GPU-hours |
|------------|----------:|
| Controlled baseline (500k) | ~5.5 |
| Anchor-only training (500k) | ~5.5 |
| DriftHead v1 training (500k) | ~5.8 |
| DriftHead v2 training (500k) | ~5.8 |
| CorrectionHead attempts (3x failed) | ~3.0 |
| Multi-horizon (killed early) | ~2.0 |
| Bias loss (killed at 165k) | ~2.5 |
| True bias loss (killed at 85k) | ~1.5 |
| Moment matching v1 (500k) | ~5.8 |
| Moment matching v2 microtune (in progress) | ~5.8 |
| Rollout extraction (multiple) | ~1.0 |
| Inference experiments | ~0.5 |
| **Total spent** | **~44.7 GPU-h** |

Remaining Month 8 budget is approximately `63` GPU-hours. The projected
remaining budget for Months 9-12 is also approximately `63` GPU-hours after
Month 8, assuming the in-progress moment matching v2 run is counted in Month 8.

## Files Added/Modified

Recent GitHub commits include:

| Commit | Content |
|--------|---------|
| `d294a70` | Training metrics and rollout manifest |
| `303b857` | Training scripts for DriftHead, CorrectionHead, bias loss, true bias loss, and moment matching |
| Additional commits | Config updates, `.gitignore` updates, and `src/` fixes |

Key Month 8 files include:

| Path | Role |
|------|------|
| `scripts/train_moment_match.py` | Moment matching training launcher |
| `scripts/train_drift_head.py` | DriftHead v1 training |
| `scripts/train_drift_head_v2.py` | DriftHead v2 training with actor using corrected features |
| `scripts/train_true_bias_loss.py` | Mean-only true bias-loss training |
| `scripts/train_bias_loss.py` | Initial bias-loss training variant |
| `scripts/train_multi_horizon.py` | Multi-horizon training attempt |
| `scripts/extract_with_drift_head.py` | DriftHead rollout extraction |
| `scripts/batch_extract_moment_match.py` | Moment matching rollout extraction |
| `configs/month8_anchor_only.yaml` | Anchor-only controlled config |
| `configs/month8_bias_loss.yaml` | Initial bias-loss config |
| `configs/month8_correction.yaml` | CorrectionHead config |
| `configs/month8_correction_disabled.yaml` | Disabled-correction control config |
| `configs/month8_drift_head.yaml` | DriftHead config |
| `configs/month8_moment_match.yaml` | Moment matching config |
| `configs/month8_multi_horizon.yaml` | Multi-horizon config |
| `configs/month8_true_bias_loss.yaml` | True bias-loss config |
| `results/month8_*/metrics.jsonl` | Month 8 training curves |
| `results/month8_controlled/rollouts_moment_match/` | Moment matching rollouts and manifest |

## Plan vs Actual

Original plan items completed in modified form:

- [x] HIAD design and implementation, followed by abandonment after the sanity
  check failed.
- [x] Controlled baseline training.
- [x] Anchor-only training.
- [x] HIAD sanity check, which provided the pivot signal.
- [x] Conservation phenomenon documentation.

Original plan items skipped:

- [ ] HIAD configuration sweep (`54` configs), abandoned with HIAD.
- [ ] H1-H4 hypothesis tests, superseded by the drift conservation mechanism as
  a unifying explanation.

New items completed that were not in the original plan:

- [x] DriftHead architecture and training.
- [x] DriftHead non-recurrent post-hoc evaluation.
- [x] Bias-loss exploration across multiple variants.
- [x] Moment matching method development.
- [x] Moment matching evaluation with `-89.6%` `J_total`.

## Outstanding Month 8 Work (TBD)

### TBD: Moment Matching v2 Microtune Result

`beta_var` was reduced from `0.1` to `0.05` to address the velocity MSE
increase of `+9.9%` in the first moment matching run. The expected outcome is
improved velocity MSE with a smaller but still substantial `J_total` reduction,
approximately `-70%`. Fill this section when training completes.

### TBD: Cartpole Validation

Cross-task validation on Cartpole should use the best moment matching
configuration after the v2 microtune decision. Estimated GPU need is
approximately `0.5` GPU-hours. Fill this section when complete.

### TBD: V4 Cross-Architecture Validation

Moment matching still needs to be applied to the V4 transformer architecture.
Estimated GPU need is approximately `5-10` GPU-hours. Fill this section when
complete.

### TBD: Theorem 1 Walkthrough Verdict

Three outcomes remain possible. Outcome A is that the proof goes through and
supports a strong theory section. Outcome B is that the proof requires
additional assumptions but remains publishable with adjustments. Outcome C is
that a fundamental gap is found and the claim should be downgraded to a
conjecture. Fill this section after the walkthrough.

### TBD: AI Reviewer Round 1 Feedback

Fill this section after Round 1.

### TBD: Section 5 (Method) Draft

Fill this section after writing.

### TBD: Tier Decision v2

Based on the moment matching result of `-89.6%` `J_total`, tier C+ (`75th`
percentile NeurIPS 2027) is now plausible. The final decision remains pending
V4 validation and the Section 5 draft.

## Risk Assessment for Month 9

The first high-priority risk is that V4 transformer validation may fail. If
moment matching does not transfer beyond DreamerV3, the cross-architecture
claim weakens substantially. This should be tested early in Month 9 rather than
left as a late validation item.

The second risk is that the velocity MSE increase may persist after tuning.
The first moment matching run improved position MSE by `-9.6%` but worsened
velocity MSE by `+9.9%`. If this trade-off remains, the paper should frame it
explicitly rather than present moment matching as uniformly improving all
observation-space metrics.

The third risk is reviewer skepticism about the scale of the `J_total`
reduction. A `-89.6%` reduction is large enough that the methodology will need
clear checks against redistribution, data reuse, and metric artifacts. The
current evidence is encouraging but not sufficient without multi-seed and
cross-task replication.

Mitigation priorities are to run V4 validation early in Month 9, run multi-seed
Cheetah validation beyond the current `seed=42`, and add per-frequency-band
analysis showing that the reduction does not hide drift in another band.

## Key Discoveries

1. Drift conservation appears to be a fundamental limitation of local
   interventions. Anchor loss and SMAD-style penalties can reduce a selected
   drift component while redistributing drift elsewhere.
2. Inference-time latent modification appears unstable in these settings. HIAD,
   recurrent DriftHead, and related variants triggered out-of-distribution
   divergence when corrections were fed back into the recurrent rollout.
3. Drift is predictable. DriftHead reached loss near `0.1`, and non-recurrent
   post-hoc correction reduced `J_total` by `-42.9%`, showing that the signal is
   structured even if recurrent correction is unstable.
4. Mean-only bias correction is insufficient. True bias loss closed the mean
   route but allowed an increase in one-step error variance, with
   `mm_noise_var` growing from `0` to `27.5`.
5. Moment matching closes the observed direction, frequency, and noise escape
   routes by penalizing both the batch mean and batch variance of one-step RSSM
   prediction error.
6. One-step improvement accumulates into total-horizon improvement. The
   `-84.8%` reduction in per-step drift increment is consistent with the
   `-89.6%` cumulative `J_total` reduction.

## Last Updated

2026-05-06 23:26:09 EDT
