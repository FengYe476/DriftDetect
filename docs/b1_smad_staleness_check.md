# B1 SMAD Staleness Check

## Setup

Run on RunPod with the Phase 2 SMAD checkpoint when available:

```bash
python scripts/run_b1_post_smad_drift.py \
  --mode collect \
  --checkpoint /path/to/smad_eta020/latest.pt \
  --device cuda
```

This collects 20 eta=0 inference rollouts into `results/b1_post_smad_rollouts/`, estimates a post-SMAD drift basis, compares it with `results/smad/U_drift_cheetah_r10.npy`, and writes `results/tables/b1_staleness_check.json`.

If the checkpoint is not available locally, the script can analyze the existing Phase 2 rollout artifacts:

```bash
python scripts/run_b1_post_smad_drift.py --mode analyze-existing
```

## Checkpoint Details

Phase 2 SMAD training used Cheetah Run with `eta=0.20`, `r=10`, seed `42`, and the fixed drift basis `results/smad/U_drift_cheetah_r10.npy`.

The training config points to `results/smad_phase2/smad_eta020/`, but this local checkout may contain only metrics and rollout artifacts, not the actual `.pt` checkpoint. If the checkpoint is missing, run the collection mode on RunPod and pass the checkpoint explicitly with `--checkpoint`.

The collection path applies the SMAD `img_step` patch with `eta=0.0` so the checkpoint is evaluated without damping at inference time.

## Results

Local run status: the SMAD `.pt` checkpoint was not present in this checkout, so the script analyzed the existing Phase 2 rollout artifacts in `results/smad_phase2/smad_eta020/rollouts/`.

| Metric | Value |
|---|---:|
| baseline/post_smad overlap r=3 | 0.0068 |
| baseline/post_smad overlap r=5 | 0.0116 |
| baseline/post_smad overlap r=10 | 0.0242 |
| Post-hoc damping with NEW basis: dc_trend reduction | 18.7967% |
| Post-hoc damping with OLD basis: dc_trend reduction | 0.9967% |

## Staleness vs Adaptation Verdict

Use the rank-10 baseline/post-SMAD overlap and the dc_trend reduction gap:

- `overlap < 0.5` and new-basis post-hoc reduction exceeds old-basis reduction: staleness is primary.
- `overlap > 0.7` and new-basis and old-basis reductions are approximately equal: model adaptation is primary.
- Otherwise: mixed evidence.

Local rollout-artifact verdict: **staleness is primary**. The baseline and post-SMAD drift bases are nearly orthogonal at rank 10, and the re-estimated basis recovers much more post-hoc dc_trend reduction than the old baseline basis.

## Implications for Month 7

If staleness is primary, Adaptive-SMAD is a plausible next experiment because re-estimating the basis during training should recover post-hoc effectiveness.

If adaptation is primary, Adaptive-SMAD is less likely to help; future work should focus on interventions the RSSM cannot easily compensate for, or on objectives that directly penalize low-frequency drift rather than damping a fixed subspace.
