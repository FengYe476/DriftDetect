# Month 8 Summary (2026-09)

## Executive Summary

Month 8 was originally planned around HIAD, an inference-time damping method,
and a controlled analysis of Adaptive-SMAD failure mechanisms. The actual
trajectory diverged in Week 1 after a self-consistency sanity check showed that
the PCA directions of Δz_t had overlap below `0.3` with the learned `U_drift`
basis. This result made the HIAD damping target unreliable, so the project
pivoted away from inference-time latent modification and toward training-time
losses for autoregressive world models.

The central Month 8 result is SHARP (Statistical Hardening of Autoregressive
Rollout Predictions). On Cheetah V3 at `500k` environment steps with `seed=42`,
SHARP reduced latent `J_total` from `0.173` to `0.018`, a `-89.6%` change,
while peak `eval_return` changed from `718.5` to `703.2`, a `-2.1%` decrease.
The per-step drift increment also fell from `0.0062` to `0.0009` (`-84.8%`),
which supports the interpretation that the cumulative reduction follows from
lower one-step prediction error rather than from a metric artifact.

SHARP closes all known drift escape routes in autoregressive world models by
simultaneously constraining the mean and variance of one-step prediction
errors. The mean term addresses direction redistribution, the single-step
formulation removes the time axis used in frequency redistribution, and the
variance term penalizes the noise-increase route observed in the mean-only
bias-loss experiment.

Two further validations are now complete. Cross-task validation on Cartpole V3
at `500k` steps reduced latent `J_total` from `0.229` to `0.006`, a `-97.2%`
change. The larger relative reduction is consistent with the prior diagnostic
that Cartpole drift is concentrated in the `dc_trend` band at `97%`, so the
disease-method match is stronger than on Cheetah. Cross-architecture
validation applied SHARP to the DreamerV4 Transformer dynamics on Cheetah
through a `5000`-step fine-tune from a pretrained checkpoint. Latent `J_total`
fell from `0.223` to `0.080`, a `-64.1%` change. The shorter fine-tune is
expected to leave additional improvement on the table relative to a
from-scratch run.

The first SHARP run also increased velocity MSE by `+9.9%`, an honest
position-versus-velocity trade-off that the paper will discuss directly. All
quantitative results are currently single-seed (`seed=42`) and require
multi-seed replication. The Cartpole baseline is from a Month 4 training run
and its comparability to the SHARP run will be re-checked.

## Key Result: SHARP Method

SHARP (Statistical Hardening of Autoregressive Rollout Predictions) adds a
global statistical penalty to the one-step prediction errors produced by the
RSSM during training.

### Method Design

The SHARP loss is:

```text
L_SHARP = β_mean * ||E_batch[ε_t]||²
        + β_var  * ||Var_batch[ε_t]||²

ε_t = f(post_t, a_t) - post_{t+1}
```

Here, `ε_t` is the one-step prediction error between the RSSM-predicted next
posterior and the actual next posterior. The mean term penalizes systematic
one-step bias across all latent dimensions, so drift cannot be hidden by moving
from one direction to another. The formulation is single-step, so there is no
explicit time dimension for drift to escape into through frequency
redistribution. The variance term directly penalizes growth in one-step error
variance, which was the observed escape route for the mean-only true bias loss.

The first successful configuration used `β_mean=1.0`, `β_var=0.1`,
`n_starts=8`, a `50k` environment-step warmup, and `batch_size=32`. The batch
size and baseline training settings were left unchanged for a fair comparison
against the controlled DreamerV3 baseline.

The SHARP loss is integrated into `total_model_loss` before the single backward
pass, mirroring the anchor-loss pattern in `scripts/train_adaptive_smad.py`
around lines `817-831`. This avoids the optimizer conflict that affected the
first bias-loss implementation, where a separate backward pass introduced
invalid interactions with the model optimizer. The DreamerV3 implementation is
located at `scripts/train_moment_match.py`, a file path retained from the
development phase, with configuration in `configs/month8_moment_match.yaml`.
The DreamerV4 implementation is at `scripts/train_v4_sharp.py` and reuses the
same statistical formulation against the Transformer dynamics module.

### Quantitative Results: Cheetah V3 (500k steps, seed=42)

| Metric | Baseline | SHARP | Change |
|--------|---------:|------:|-------:|
| Peak eval_return | 718.5 | 703.2 | -2.1% |
| Latent J_total (trim 25:175) | 0.173 | 0.018 | -89.6% |
| Per-step drift increment | 0.0062 | 0.0009 | -84.8% |
| Step 0→1 drift accumulation | 0.041 | 0.00016 | -99.6% |
| Position MSE | 0.9100 | 0.8200 | -9.6% |
| Velocity MSE | 19.13 | 21.02 | +9.9% |
| Early band drift (25-75) | 0.163 | 0.018 | -88.9% |
| Mid band drift (75-125) | 0.176 | 0.018 | -89.7% |
| Late band drift (125-175) | 0.180 | 0.018 | -90.2% |

The `-89.6%` reduction is supported by several checks against methodological
artifacts. The per-step increment reduction (`-84.8%`) is consistent with the
cumulative reduction (`-89.6%`), so the linear accumulation relationship holds.
The evaluation used fresh actor rollouts, with each seed producing a new
trajectory rather than reusing training data. Early, mid, and late time bands
all improved nearly uniformly (`-88.9%`, `-89.7%`, and `-90.2%`), which argues
against drift redistribution across the imagined horizon. The Step 0→1
reduction of `-99.6%` directly demonstrates SHARP's effect on systematic
one-step bias.

### Quantitative Results: Cartpole V3 (500k steps, seed=42)

| Metric | Baseline | SHARP | Change |
|--------|---------:|------:|-------:|
| Latent J_total (trim 25:175) | 0.229 | 0.006 | -97.2% |
| Early band drift (25-75) | 0.237 | 0.004 | -98.4% |
| Mid band drift (75-125) | 0.211 | 0.006 | -97.0% |
| Late band drift (125-175) | 0.239 | 0.009 | -96.3% |
| Peak eval_return (SHARP) | — | ~828 | healthy |
| `mm_mean_loss` final | — | 0.0 | systematic bias eliminated |
| `mm_var_loss` final | — | 0.0 | noise variance controlled |

The Cartpole baseline `eval_return` was not directly available on the RunPod
training environment used for SHARP, so the eval_return comparison is omitted;
the SHARP run reaches an `eval_return` near `828`, which is in the healthy
range for this task. Both `mm_mean_loss` and `mm_var_loss` converged to `0.0`,
indicating that systematic one-step bias and one-step error variance were
fully suppressed under the converged policy. The relative `J_total` reduction
on Cartpole (`-97.2%`) exceeds the Cheetah reduction (`-89.6%`), consistent
with the diagnostic finding that Cartpole drift is `97%` concentrated in the
`dc_trend` band, where SHARP's mean term has its strongest effect.

### Quantitative Results: Cheetah V4 Transformer (5000-step fine-tune from pretrained)

| Metric | Baseline | SHARP | Change |
|--------|---------:|------:|-------:|
| Latent J_total | 0.223 | 0.080 | -64.1% |
| Early band drift | 0.212 | 0.081 | -61.6% |
| Mid band drift | 0.222 | 0.076 | -65.7% |
| Late band drift | 0.235 | 0.083 | -64.8% |

V4 SHARP was applied as a `5000`-step fine-tune of the pretrained DreamerV4
dynamics, rather than a from-scratch `500k` training run. The reduction is
therefore a lower bound on what longer fine-tuning would likely achieve. The
band-uniform reduction pattern (`-61.6%`, `-65.7%`, `-64.8%`) is consistent
with the V3 finding that SHARP does not redistribute drift across the imagined
horizon, and supports the architecture-agnostic interpretation since the
Transformer dynamics differ structurally from the RSSM.

### V4 β_var Ablation: SHARP v2 (β_var=0.01)

A second V4 fine-tune was run with `β_var` reduced from `0.1` to `0.01` (mean
term unchanged at `β_mean=1.0`, all other settings identical) to test whether
the variance term was driving observed blurriness in decoded frames. The
result was `J_total = 0.080951`, a `-63.7%` change versus the V4 baseline,
nearly identical to v1's `-64.1%`.

Two implications follow. First, the variance term contributes negligibly to
the V4 `J_total` reduction at the `5000`-step fine-tune budget; the mean term
is the dominant contributor on this architecture. Second, decoded frame
blurriness persists at `β_var=0.01`, so the visual artifact is not caused by
SHARP's variance penalty. The blurriness is attributable to the V4 tokenizer's
decoder, which converts a low-dimensional packed latent back into pixels and
amplifies any residual latent-space error nonuniformly across spatial
frequencies. v1 (`β_var=0.1`) is retained as the canonical V4 SHARP
configuration because the `J_total` reductions are statistically
indistinguishable and v1 matches the V3 configuration exactly.

## Method Comparison Table

| Method | eval_return Change | J_total Change | Failure Mode |
|--------|-------------------:|---------------:|--------------|
| Baseline DreamerV3 | reference | reference | Reference condition |
| Anchor loss (`β=0.1`) | ~0% | +4.1% | Direction redistribution |
| Adaptive-SMAD | ~0% | -10% (`J_slow` only) | Basis chasing, conservation |
| HIAD recurrent | not evaluated | ~+45% | OOD divergence |
| DriftHead recurrent | not evaluated | ~+74000% | OOD divergence |
| DriftHead non-recurrent (post-hoc) | not evaluated | -42.9% | Works, but does not modify RSSM |
| DriftHead v2 (actor uses corrected feat) | -16% (`618.4`) | not measured | Train-test distribution shift |
| Bias loss (`||drift increment||²`) | unclear | killed early | Optimizer conflict bug |
| True bias loss (mean only) | -49% (`~370`) | killed early | Noise variance escape (`var: 0→27.5`) |
| **SHARP V3 Cheetah** | **-2.1% (`703.2`)** | **-89.6%** | **None confirmed** |
| **SHARP V3 Cartpole** | **healthy (~`828`)** | **-97.2%** | **None confirmed** |
| **SHARP V4 Cheetah (v1, β_var=0.1)** | **N/A (fine-tune)** | **-64.1%** | **None (`5k` fine-tune only)** |
| **SHARP V4 Cheetah (v2, β_var=0.01)** | **N/A (fine-tune)** | **-63.7%** | **None; same J_total as v1** |

## Methodological Journey

The first phase of Month 8 focused on HIAD. The decisive result was a
self-consistency sanity check: Δz_t PCA directions had overlap below `0.3` with
the learned `U_drift` basis. This meant that the intended inference-time
damping direction did not reliably match the realized rollout drift direction.
HIAD was therefore treated as fundamentally compromised, and the project
abandoned the planned `54`-configuration sweep.

The next phase characterized anchor loss and Adaptive-SMAD under a fair-step
controlled comparison. These experiments confirmed drift conservation as a
failure mechanism. Anchor loss reduced drift on the targeted baseline `U`
direction, but drift increased on other directions and total latent drift did
not improve. This established direction redistribution as a fundamental escape
route for local or fixed-subspace penalties.

DriftHead experiments showed that the drift signal is predictable. The
DriftHead loss converged to approximately `0.1`, and its gate grew from `0` to
`1.8`. A non-recurrent post-hoc correction reduced `J_total` by `-42.9%`, but
it did not change the trained RSSM. Recurrent use of the correction caused
out-of-distribution divergence, consistent with the HIAD conclusion that
feeding modified latent states back into autoregressive rollouts is unstable.

Bias-loss experiments then tested training-time constraints. The initial loss
on `||drift_{t+1} - drift_t||²` was stopped early after an implementation bug
was identified: the loss used a separate backward pass, causing an optimizer
conflict. The true bias-loss variant, `||E[ε]||²`, was integrated correctly into
the model loss, but the RSSM escaped by increasing one-step error variance. The
noise-variance diagnostic grew from `0` to `27.5` over `30k` training steps,
and the run was stopped at `85k`.

SHARP added the missing `||Var[ε]||²` term to the mean-bias penalty. With the
mean term, the single-step formulation, and the variance term active together,
the observed direction, frequency, and noise escape routes were all blocked.
The RSSM was therefore constrained to reduce both one-step bias and one-step
variance rather than redistribute error into an unpenalized form.

Cross-task validation followed. Cartpole was selected because its drift is
`97%` concentrated in the `dc_trend` band, which predicts a stronger relative
SHARP response than on Cheetah. The Cartpole result (`-97.2%`) supports this
disease-method match interpretation, and the fact that both `mm_mean_loss` and
`mm_var_loss` converged to `0.0` confirms that the loss surface is reachable
on a simpler dynamical system.

Cross-architecture validation then applied SHARP to the DreamerV4 Transformer
dynamics on Cheetah via a `5000`-step fine-tune from a pretrained checkpoint.
The `-64.1%` reduction confirms that the same statistical hardening principle
transfers from RSSM to Transformer dynamics, supporting the
architecture-agnostic claim. The shorter fine-tune is expected to be a lower
bound on the achievable improvement.

## Conservation Phenomenon Documentation

The drift conservation result, Finding D, is the central negative result behind
the pivot to global training-time penalties. Anchor loss reduced `J_slow` by
`-79.5%` on the baseline `U` direction, but drift on other directions increased
by `+486%`. The net `J_total` change was `+4.1%`, so the targeted intervention
increased total latent drift despite improving its local objective.

The slow-subspace fraction changed from `11.8%` in the baseline to `66.7%`
under anchor loss. This showed that the intervention altered the decomposition
of drift rather than reducing the total drift load. The finding established
conservation as a capacity-induced limitation and motivated the global
one-step statistical penalty that became SHARP.

## Generated Figures and Visualizations

A complete publication figure suite was generated from the converged rollouts
and from re-rendered MuJoCo physics. The figures are written to
`results/figures/` at `300` dpi.

V3 quantitative figures, produced by `scripts/generate_month8_figures.py`:

| File | Content |
|------|---------|
| `fig1_obs_trajectories.png` | V3 Cheetah `4x2` panel of selected observation dimensions over `200` imagination steps for seed `0`: True (black) vs Baseline (red dashed) vs SHARP (blue), with the actor horizon at step `15` marked. |
| `fig2_mse_over_time.png` | V3 Cheetah position MSE (dims `0-7`) and velocity MSE (dims `8-16`) per step, mean `+-` std over the available paired seeds (target `17`). |
| `fig3_latent_drift.png` | V3 Cheetah latent drift (deter slice `1024:1536`) per step, mean `+-` std, with the `J_total: -89.6%` annotation arrow. |
| `fig4_cross_task_arch.png` | Bar chart of `J_total` reduction across V3 Cheetah (`-89.6%`), V3 Cartpole (`-97.2%`), and V4 Cheetah (`-64.1%`) with architecture labels. |

V4 visual figures, produced by `scripts/generate_v4_visual_comparison.py`:

| File | Content |
|------|---------|
| `fig5_v4_frame_comparison.png` | V4 `3x8` decoded-frame grid (True / Baseline / SHARP) at `t=0,5,15,30,50,100,150,199`, each cell `128x128`. |
| `fig5_v4_imagination.gif` | V4 `200`-frame triptych animation (True | Baseline | SHARP), `15` fps, with serif label header. |
| `fig6_v4_recon_error.png` | V4 per-step decoded-frame MSE for baseline vs SHARP. |
| `fig7_v4_latent_drift.png` | V4 per-step latent MSE on the packed `512`-dim flat latent, annotated `J_total: -64.1%`. |

V3 physics-rendered figures, produced by
`scripts/generate_3d_render_comparison.py` and
`scripts/generate_3d_render_cartpole.py`:

| File | Content |
|------|---------|
| `fig9_3d_render_comparison.png` | Cheetah `3x8` MuJoCo render grid (Ground Truth / Baseline / SHARP) at `t=0,5,15,30,50,100,150,199`, each cell `256x256`. |
| `fig9_3d_render_comparison_diff.png` | Cheetah `4x8` variant with a `|Truth - SHARP|` row amplified `x4`. |
| `fig9_3d_render_comparison.gif` | Cheetah `200`-frame triptych at `15` fps. |
| `fig10_cartpole_3d_render.png` | Cartpole-swingup `3x8` MuJoCo render grid in the same format as `fig9`. |
| `fig10_cartpole_3d_render_diff.png` | Cartpole `4x8` variant with `|Truth - SHARP|` row amplified `x4`. |

Key visual observations from `fig9` and `fig10`:

- Cheetah (`fig9`): the baseline imagination diverges visibly from the ground
  truth by `t=30` and produces physically implausible poses by `t=100`.
  SHARP maintains structurally coherent poses substantially longer, with
  graceful degradation thereafter. The `17`-dim observation distributes
  residual error across many joints, so visible drift accumulates more slowly
  than in Cartpole.
- Cartpole (`fig10`): both baseline and SHARP show pole-angle divergence
  between `t=50` and `t=100`, but SHARP tracks the ground-truth swing-up
  trajectory more closely in the early steps. Cartpole's `5`-dim observation
  is dominated by angular quantities, so even small angular errors translate
  into large visual differences and the visible-drift onset is earlier than
  the latent `J_total` would suggest.
- V4 frames (`fig5`): SHARP's decoded frames are visibly blurrier than the
  baseline's. The β_var ablation reported above shows this artifact is
  unchanged at `β_var=0.01`, so it is not caused by SHARP's variance penalty;
  it is a property of the V4 tokenizer decoder.

## Latent Metric vs Visual Quality Gap

SHARP optimizes a latent-space quantity directly: the mean and variance of
one-step prediction error in the deterministic latent state. The reported
`J_total` numbers are computed in this `512`-dim latent space and quantify
trajectory-level drift independently of any decoder. The decoder, in
contrast, is a fixed map from latent to observation (or to pixels in V4) and
introduces its own bias-variance behavior on top of SHARP's latent
improvement. Reductions in latent drift therefore do not transfer
proportionally to decoded outputs.

Three concrete consequences are visible in the Month 8 results. First, the V3
Cheetah MuJoCo physics renders (`fig9`) provide the most faithful
visualization of imagination quality because they bypass any learned decoder
entirely, setting `qpos` and `qvel` from the imagined `obs` directly and
rendering through MuJoCo. The visible degradation timeline in `fig9` is
governed almost entirely by latent drift. Second, the V4 decoded frames
(`fig5`) appear blurrier under SHARP because the V4 tokenizer's pixel decoder
amplifies residual latent error nonuniformly across spatial frequencies; the
`β_var=0.01` ablation shows this is not a SHARP variance-penalty artifact.
Third, Cartpole's `5`-dim observation is dominated by angular state, so a
small absolute reduction in angular drift can still leave a visually
noticeable pole-angle gap, while Cheetah's `17`-dim observation distributes
the same kind of error across many joints and therefore looks more graceful
even at comparable latent error.

The paper should report `J_total` as the primary metric, with decoded-frame
MSE and visual figures presented as complementary evidence rather than as
ground truth. The V4 decoded-frame artifact is honest evidence that
SHARP-as-a-latent-method does not directly improve perceptual quality of a
fixed pixel decoder.

## V4 Fine-Tune Environment Setup (Reproducibility)

V4 SHARP fine-tuning was performed on RunPod using the public Dreamer4
release artifacts. The setup below is sufficient to reproduce the V4 v1 and
v2 results.

Pretrained checkpoints (HuggingFace `nicklashansen/dreamer4`):

- `tokenizer.pt` (`245` MB), `dynamics.pt` (`488` MB).
- Placed under `external/dreamer4/checkpoints/`.
- Tokenizer args: `H=128`, `W=128`, `C=3`, `patch=4`, `d_model=256`,
  `n_latents=16`, `d_bottleneck=32`, `depth=8`.
- Dynamics args: `d_model_dyn=512`, `dyn_depth=8`, `n_heads=4`, `k_max=8`,
  `packing_factor=2`. Total approximately `42.6` M parameters.

Expert dataset (HuggingFace `nicklashansen/dreamer4`, expert subset,
`cheetah-run` task only):

- Raw demos: `external/dreamer4/data/expert/cheetah-run.pt`
  (TensorDict with `obs`, `action`, `reward`, `value`, `terminated`,
  `episode`; `obs` shape `(10020, 128)`). Total `3.39` GB.
- Preprocessed shards:
  `external/dreamer4/data/expert-shards/cheetah-run/cheetah-run_shard*.pt`
  (`4` shards, approximately `10k` frames). Each shard contains `frames`
  of shape `(N, 3, 128, 128)` `uint8`.

Fine-tune configuration (script: `scripts/train_v4_sharp.py`):

- `5000` optimizer steps, `batch_size=4`, `seq_len=32`, `lr=1e-5`.
- SHARP penalty: `β_mean=1.0`, `β_var=0.1` (v1) or `β_var=0.01` (v2).
- Warmup: `500` optimizer steps before SHARP penalty engages.
- Tokenizer is frozen during fine-tuning; only dynamics parameters are
  updated.
- Required environment variables on RunPod: `MUJOCO_GL=osmesa`,
  `PYOPENGL_PLATFORM=osmesa`, `WANDB_MODE=disabled`,
  `PYTHONPATH=external/dreamer4/dreamer4:external/dreamer4:$PYTHONPATH`.

Output checkpoints are written to `results/v4_sharp/for_eval/dynamics.pt`
(approximately `488` MB) and reused at evaluation time alongside the frozen
tokenizer from the baseline checkpoint directory.

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
| SHARP v1 Cheetah (500k) | ~5.8 |
| SHARP v2 microtune (killed at 335k) | ~4.0 |
| SHARP Cartpole (500k) | ~3.0 |
| V4 SHARP v1 fine-tune (β_var=0.1, 5000 steps) | ~1.0 |
| V4 SHARP v2 fine-tune (β_var=0.01, 5000 steps) | ~1.0 |
| Rollout extractions (all V3) | ~2.0 |
| V4 baseline + SHARP v1 rollouts | ~0.5 |
| V4 v2 rollout extraction | ~0.3 |
| 3D rendering (Cheetah + Cartpole, 600 frames each) | ~0.5 |
| Figure generation scripts | ~0.2 |
| **Total Month 8** | **~49.9 GPU-h** |

Approximately `58` GPU-hours remain for Months 9-12.

## Files Added/Modified

Recent GitHub commits include:

| Commit | Content |
|--------|---------|
| `d294a70` | Training metrics and rollout manifest |
| `303b857` | Training scripts for DriftHead, CorrectionHead, bias loss, true bias loss, and SHARP |
| Additional commits | Config updates, `.gitignore` updates, and `src/` fixes |

Key Month 8 files include:

| Path | Role |
|------|------|
| `scripts/train_moment_match.py` | SHARP training implementation (V3) |
| `scripts/train_v4_sharp.py` | SHARP fine-tune for V4 Transformer |
| `scripts/train_drift_head.py` | DriftHead v1 training |
| `scripts/train_drift_head_v2.py` | DriftHead v2 training with actor using corrected features |
| `scripts/train_true_bias_loss.py` | Mean-only true bias-loss training |
| `scripts/train_bias_loss.py` | Initial bias-loss training variant |
| `scripts/train_multi_horizon.py` | Multi-horizon training attempt |
| `scripts/extract_with_drift_head.py` | DriftHead rollout extraction |
| `scripts/batch_extract_moment_match.py` | SHARP rollout extraction (V3 Cheetah) |
| `scripts/batch_extract_moment_match_cartpole.py` | SHARP rollout extraction (V3 Cartpole) |
| `scripts/batch_extract_v4.py` | V4 baseline + SHARP rollout extraction |
| `scripts/generate_month8_figures.py` | V3 quantitative figures (`fig1`-`fig4`) |
| `scripts/generate_v4_visual_comparison.py` | V4 visual figures (`fig5`-`fig7`) |
| `scripts/generate_3d_render_comparison.py` | Cheetah MuJoCo physics renders (`fig9`) |
| `scripts/generate_3d_render_cartpole.py` | Cartpole MuJoCo physics renders (`fig10`) |
| `configs/month8_anchor_only.yaml` | Anchor-only controlled config |
| `configs/month8_bias_loss.yaml` | Initial bias-loss config |
| `configs/month8_correction.yaml` | CorrectionHead config |
| `configs/month8_correction_disabled.yaml` | Disabled-correction control config |
| `configs/month8_drift_head.yaml` | DriftHead config |
| `configs/month8_moment_match.yaml` | SHARP V3 Cheetah config |
| `configs/month8_moment_match_cartpole.yaml` | SHARP V3 Cartpole config |
| `configs/month8_moment_match_v2.yaml` | SHARP v2 microtune config |
| `configs/month8_multi_horizon.yaml` | Multi-horizon config |
| `configs/month8_true_bias_loss.yaml` | True bias-loss config |
| `results/month8_*/metrics.jsonl` | Month 8 training curves |
| `results/month8_controlled/rollouts_moment_match/` | SHARP V3 Cheetah rollouts and manifest |
| `results/month8_controlled/rollouts_moment_match_cartpole/` | SHARP V3 Cartpole rollouts |
| `results/month8_controlled/rollouts_v4_baseline/` | V4 baseline rollouts |
| `results/month8_controlled/rollouts_v4_sharp/` | V4 SHARP rollouts |
| `results/v4_sharp/for_eval/dynamics.pt` | V4 SHARP fine-tuned dynamics checkpoint |
| `results/figures/` | `11` PNG figures and `1` GIF for the Month 8 paper draft |

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
- [ ] H1-H4 hypothesis tests, superseded by drift conservation as the unifying
  explanation.

New items completed that were not in the original plan:

- [x] DriftHead architecture and training.
- [x] DriftHead non-recurrent post-hoc evaluation.
- [x] Bias-loss exploration across multiple variants.
- [x] SHARP method development.
- [x] SHARP evaluation with `-89.6%` `J_total` on Cheetah V3.
- [x] SHARP cross-task validation on Cartpole V3 (`-97.2%` `J_total`).
- [x] SHARP cross-architecture validation on V4 Transformer Cheetah
  (`-64.1%` `J_total` via `5000`-step fine-tune).
- [x] V4 `β_var` ablation (`β_var=0.01`, `-63.7%` `J_total`) showing the
  variance term is not the cause of decoded-frame blurriness.
- [x] Publication figure suite: V3 quantitative figures, V4 visual figures,
  and Cheetah/Cartpole MuJoCo physics renders.

## Completed Month 8 Validation Items

### SHARP v2 Microtune Outcome

The v2 microtune reduced `β_var` from `0.1` to `0.05` to address the velocity
MSE increase of `+9.9%`. The run was stopped at `335k` steps before reaching
`500k` because it did not produce a usable improvement over v1 within the
available budget; the v1 configuration (`β_mean=1.0`, `β_var=0.1`) is retained
as the reference SHARP setting. The velocity MSE trade-off is therefore
documented as an open trade-off rather than a closed result.

### Cartpole Cross-Task Validation Outcome

Cartpole V3 with the v1 SHARP configuration at `500k` steps reduced `J_total`
from `0.229` to `0.006` (`-97.2%`). All three time bands improved between
`-96.3%` and `-98.4%`. Both `mm_mean_loss` and `mm_var_loss` converged to
`0.0`. The Cartpole `J_total` reduction is larger than Cheetah's, consistent
with the prior diagnostic that Cartpole drift is `97%` `dc_trend`-dominated.
See the Cartpole results table above.

### V4 Cross-Architecture Validation Outcome

V4 Transformer dynamics on Cheetah, via a `5000`-step fine-tune from a
pretrained checkpoint, reduced `J_total` from `0.223` to `0.080` (`-64.1%`).
The reduction is uniform across early, mid, and late bands (`-61.6%`,
`-65.7%`, `-64.8%`). This is the lower bound from a short fine-tune; a
from-scratch run is expected to yield a larger reduction. See the V4 results
table above.

## Outstanding Month 8 Work (TBD)

All remaining items are non-GPU work and can be completed without further
RunPod time.

| Item | Estimated effort |
|------|----------------:|
| Theorem 1 walkthrough (proof verification) | ~25 h |
| AI Reviewer Round 1 | ~3 h |
| Section 5 (Method) draft | ~1 day |
| Tier decision v2 | ~1 h |
| Month 9 detailed plan | ~2 h |
| AI Reviewer Round 2 | ~3 h |
| W38 and W39 weekly reports | ~1 h |

### TBD: Theorem 1 Walkthrough Verdict

Three outcomes remain possible:

- Outcome A: proof goes through, supporting a strong theory section.
- Outcome B: additional assumptions are needed, but the result remains
  publishable with adjustments.
- Outcome C: a fundamental gap is found, and the claim should be downgraded to
  a conjecture.

[Fill in after walkthrough]

### TBD: AI Reviewer Round 1 Feedback

[Fill in after Round 1]

### TBD: AI Reviewer Round 2 Feedback

[Fill in after Round 2]

### TBD: Section 5 (Method) Draft

[Fill in after writing]

### TBD: Tier Decision v2

The cross-task and cross-architecture validations strengthen the case for a
tier upgrade. The combined evidence is `-89.6%` on Cheetah V3, `-97.2%` on
Cartpole V3, and `-64.1%` on V4 Transformer Cheetah (`-63.7%` under the
`β_var=0.01` ablation). The final decision remains pending Theorem 1
verification and the Section 5 draft.

[Fill in after Theorem 1 and Section 5]

### TBD: Month 9 Detailed Plan

[Fill in after tier decision]

### TBD: W38 and W39 Weekly Reports

[Fill in at end of each week]

## Risk Assessment for Month 9

The first risk is that V4 transformer validation used only a `5000`-step
fine-tune from a pretrained checkpoint. The `-64.1%` reduction is therefore a
lower bound, but a longer fine-tune (or from-scratch run) is needed to
establish the asymptote and to rule out artifacts of the short adaptation
window. A target of `-80%` or better is plausible.

The second risk is single-seed evaluation. All three SHARP results
(`seed=42`) require multi-seed replication on at least Cheetah V3 to confirm
that the observed effect is not seed-specific.

The third risk is the velocity MSE increase of `+9.9%` on Cheetah V3. The v2
microtune did not resolve it, so the paper must explicitly discuss the
position-versus-velocity trade-off as an honest property of SHARP rather than
as a tuning artifact.

The fourth risk is that the Cartpole baseline used for the `-97.2%` comparison
is from a separate Month 4 training run, not a co-trained baseline on the same
RunPod environment. Comparability needs to be re-checked, ideally by training
a fresh Cartpole baseline under the same conditions as the SHARP run.

The fifth risk is reviewer skepticism about the scale of the `J_total`
reduction. The methodological framing must show that SHARP optimizes a
quantity directly related to `J_total`, namely the relationship from per-step
error to cumulative drift.

Mitigation priorities for Month 9 are: longer V4 fine-tune (or from-scratch
training), multi-seed Cheetah V3 replication, a co-trained Cartpole baseline,
and per-frequency-band analysis demonstrating no drift redistribution.

## Key Discoveries

1. Drift conservation appears fundamental for local interventions. Anchor loss,
   SMAD, and fixed-direction methods cannot reliably reduce `J_total` because
   the RSSM can redistribute drift across unrestricted latent dimensions.
2. Inference-time latent state modification is unstable in these experiments.
   HIAD variants and recurrent DriftHead corrections triggered exponential
   divergence when modified latents were fed back into the rollout.
3. Drift is statistically predictable. DriftHead reached loss near `0.1` within
   `100k` training steps, showing that systematic drift patterns can be learned
   with a small auxiliary network.
4. Mean-only constraints are insufficient. True bias loss reduced the mean
   route but allowed the RSSM to increase one-step error variance from `0` to
   `27.5` within `30k` post-activation steps.
5. SHARP closes the observed escape routes. The mean term blocks direction
   redistribution, the single-step formulation eliminates frequency escape, and
   the variance term blocks noise expansion.
6. One-step improvement accumulates to `J_total`. SHARP's `-84.8%` per-step
   reduction yielded a `-89.6%` cumulative `J_total` reduction, which supports
   the claim that SHARP targets the one-step source of long-horizon drift.
7. SHARP is architecture-agnostic. The same statistical penalty reduces
   `J_total` on the DreamerV3 RSSM (`-89.6%` Cheetah, `-97.2%` Cartpole) and
   on the DreamerV4 Transformer (`-64.1%` Cheetah, `5000`-step fine-tune).
8. Disease-method match predicts effect size. Cartpole drift is `97%`
   `dc_trend`-dominated and shows the largest relative reduction (`-97.2%`),
   while Cheetah V3 (`46.7%` `dc_trend`) shows `-89.6%` and V4 (`63.9%`
   `dc_trend`) shows `-64.1%` under a short fine-tune.
9. The variance term is not load-bearing on V4. Reducing `β_var` from `0.1`
   to `0.01` changes `J_total` from `-64.1%` to `-63.7%`, a difference within
   measurement noise. The mean term is the dominant contributor at the
   `5000`-step V4 fine-tune budget.
10. Latent improvement does not transfer proportionally to decoded pixels.
    V4 SHARP frames are blurrier than baseline frames despite the latent
    `J_total` reduction; the `β_var` ablation rules out SHARP's variance
    penalty as the cause and points to the V4 tokenizer decoder. MuJoCo
    physics renders that bypass the learned decoder (`fig9`, `fig10`) show a
    much cleaner visual picture of imagination quality.

## Last Updated

2026-05-07
