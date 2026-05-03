# B1 SMAD Staleness Check

## Setup

The corrected B1 check evaluates the SMAD-trained Cheetah Run checkpoint with
damping disabled during inference. This is the required test because the goal is
to measure the model's natural post-training drift structure, not the drift
created while the SMAD patch is actively damping the transition.

Run collection on RunPod with the Phase 2 SMAD checkpoint:

```bash
python scripts/run_b1_post_smad_drift.py \
  --mode collect \
  --checkpoint results/smad_phase2_v2/smad_eta020/checkpoints/final.pt \
  --device cuda \
  --eta_inference 0.0 \
  --n_seeds 20 \
  --output_dir results/b1_post_smad_rollouts_eta0/
```

Then run analysis:

```bash
python scripts/run_b1_post_smad_drift.py \
  --mode analyze \
  --rollout_dir results/b1_post_smad_rollouts_eta0/ \
  --baseline_U results/smad/U_drift_cheetah_r10.npy \
  --output results/tables/b1_staleness_check_corrected.json
```

## Checkpoint Details

Phase 2 SMAD training used Cheetah Run with `eta=0.20`, rank `r=10`, seed `42`,
and the fixed drift basis `results/smad/U_drift_cheetah_r10.npy`.

The corrected collection loads the SMAD-trained checkpoint weights but runs
inference with `eta_inference=0.0`. The SMAD patch is therefore disabled during
rollout collection, exposing the natural drift directions of the SMAD-trained
model.

Corrected output:

- `results/b1_post_smad_rollouts_eta0/`
- `results/tables/b1_staleness_check_corrected.json`

## Results

| Metric | Original eta=0.20 rollouts | Corrected eta=0 rollouts |
|---|---:|---:|
| baseline/post-SMAD overlap r=3 | 0.0068 | 0.0053 |
| baseline/post-SMAD overlap r=5 | 0.0116 | 0.0130 |
| baseline/post-SMAD overlap r=10 | 0.0242 | 0.0224 |
| Post-hoc damping with NEW basis: dc_trend reduction | 18.8% | 22.6% |
| Post-hoc damping with OLD basis: dc_trend reduction | 1.0% | 0.9% |

The corrected baseline/post-SMAD drift overlap is near zero at all tested ranks.
The post-SMAD drift subspace is therefore nearly orthogonal to the original
baseline drift basis.

Post-hoc damping with the re-estimated post-SMAD basis reduces dc_trend MSE by
`22.6%`, exceeding the original `20%` intervention threshold. The old baseline
basis reduces dc_trend by only `0.9%` on the same corrected eta=0 rollouts.

## Correction History

The original B1 analysis used existing rollouts from
`results/smad_phase2/smad_eta020/rollouts/`. Those rollouts were collected with
eta=0.20, meaning damping was active during collection. This contaminated the
drift-direction estimate because the measured rollouts reflected the damped
transition, not the natural behavior of the SMAD-trained model.

The correction collected fresh rollouts from the SMAD-trained checkpoint with
`eta_inference=0.0`. The corrected values are nearly identical to the original
contaminated values:

- rank-3 overlap: `0.0068` original vs `0.0053` corrected
- rank-5 overlap: `0.0116` original vs `0.0130` corrected
- rank-10 overlap: `0.0242` original vs `0.0224` corrected
- new-basis dc_trend reduction: `18.8%` original vs `22.6%` corrected
- old-basis dc_trend reduction: `1.0%` original vs `0.9%` corrected

This validates the original directional conclusion. Damping contamination was
not a significant factor.

## Staleness vs Adaptation Verdict

Final verdict: **STALENESS confirmed. Adaptive-SMAD should work.**

The rank-10 baseline/post-SMAD overlap is `0.0224`, far below the staleness
threshold. The re-estimated basis recovers `22.6%` dc_trend reduction, while the
old basis recovers only `0.9%`. This means the training-time effectiveness gap
is primarily caused by basis staleness rather than model adaptation.

## Implications for Month 7

Adaptive-SMAD is now the recommended next intervention experiment. The first
prototype should periodically re-estimate `U_drift` during training so the
damping basis tracks the evolving drift subspace.

The corrected B1 result suggests the basis can become stale extremely quickly:
baseline and post-SMAD drift directions are almost orthogonal after training.
Month 7 should therefore start with a conservative re-estimation schedule and
monitor whether the model adapts faster than the basis update cadence.
