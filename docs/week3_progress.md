# Week 3 Progress Report

Date: 2026-04-29

## Overview

Week 3 was originally planned for adapter design work, but the week was reallocated to DreamerV3 checkpoint training. This change became necessary after a full checkpoint search confirmed that no public DreamerV3 checkpoints were available for the DeepMind Control Suite tasks needed by DriftDetect. Training our own baseline became the critical path for the rest of Month 1.

## Plan Adjustment

| Item | Original Plan | Actual Work | Reason |
|---|---|---|---|
| Week 3 focus | Adapter design | Train DreamerV3 from scratch | No pretrained checkpoints available |

This was a strategic reordering, not a scope failure. Without a usable checkpoint, adapter work and frequency-domain diagnostics could not proceed on a meaningful trained model.

## Training Setup

- Platform: RunPod cloud GPU
- Hardware: NVIDIA RTX 4090 (24GB)
- Cost: $0.70/hour, total ~$6.65
- Total time: ~9.5 hours
- Task: `dmc_cheetah_run`
- Training steps: 505k (target 500k)

## Hyperparameter Optimization

| Parameter | Default | Our Config | Impact |
|---|---|---|---|
| `batch_size` | 16 | 32 | 2x |
| `envs` | 4 | 8 | 2x |
| GPU utilization | ~26% | ~76% | +50% |
| Speed | ~9.7 steps/sec | ~13.1 steps/sec | +35% |

The optimization strategy was intentionally aggressive because a single successful checkpoint was more valuable than preserving the exact reference settings from the upstream implementation.

## Performance Results

- `eval_return` timeline:
  - Step 25k: 45
  - Step 175k: 503
  - Step 325k: 652
  - Step 365k: 727
  - Step 485k: 791 (peak)
  - Step 495k: 753
- `train_return` summary:
  - Average: ~718 near the end of training
  - Maximum: 774.9 at Step 504k
- Comparison to DreamerV3 paper:
  - DriftDetect training exceeded the reported upper bound of the paper's 650-750 range for this setting

## Key Decisions

- Single task vs multi-task:
  - Chose a single `dmc_cheetah_run` baseline instead of splitting effort across Cheetah, Cartpole, and Reacher
  - This reduced cost and secured one strong checkpoint quickly
- RunPod vs local M5 Pro:
  - Chose RunPod because local MPS feasibility was marginal and would have slowed Month 1 substantially
  - The cloud run provided much better throughput and predictable completion
- Aggressive hyperparameter optimization:
  - Increased `batch_size` and `envs` to improve GPU utilization
  - This materially improved speed without destabilizing the run

## Issues Encountered

- SSH connection timeouts:
  - Resolved by reconnecting to the RunPod instance
  - No training restart was required
- Exploration-period instability at Step 225k-265k:
  - Observed a temporary performance dip during the exploration-heavy portion of training
  - The run self-recovered without intervention

## Output Artifacts

- Checkpoint:
  - `results/checkpoints/cheetah_run_500k.pt`
  - Large binary artifact kept outside normal Git text tracking
- Metadata:
  - `results/checkpoints/manifest.json`
  - `docs/training_record_cheetah_500k.md`

## Next Steps (Week 4)

1. Verify checkpoint integrity and loading behavior.
2. Add or finalize the trained-checkpoint adapter path for rollout extraction.
3. Run batch rollouts across multiple seeds.
4. Start DreamerV4 reconstruction and comparison setup.
5. Prepare Month 1 baseline release materials.

## Lessons Learned

- Public checkpoint availability should be treated as a first-order project dependency, not a background assumption.
- Small hyperparameter changes can dramatically improve cloud cost efficiency when GPU utilization is initially low.
- A single high-quality baseline checkpoint is more valuable early in the project than spreading effort across multiple partially complete tasks.
- Cloud training was worth the cost because it unblocked the rest of the research timeline and preserved Month 1 momentum.
