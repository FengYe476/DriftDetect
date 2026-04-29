# Month 1 Summary - DriftDetect

## Executive Summary

Month 1 set up the DriftDetect baseline infrastructure and produced the first scientifically useful DreamerV3 rollout dataset. The original plan assumed public checkpoints might be available, but we instead trained a high-performing Cheetah Run checkpoint and upgraded the rollout pipeline around it. By the end of Week 4, DriftDetect has a checkpoint, adapter interface, DreamerV3 adapter, 20 rollout files, and an initial sanity notebook confirming open-loop drift. Month 2 is ready to focus on frequency-domain error decomposition rather than infrastructure recovery.

## Deliverables Status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Infrastructure | Complete | Conda envs, external repos, docs, and project layout are in place |
| extract_rollout.py | Complete | Upgraded in Week 4 for trained checkpoints |
| WorldModelAdapter | Complete | Architecture-agnostic interface in `src/models/adapter.py` |
| DreamerV3Adapter | Complete | Adapter implementation in `src/models/dreamerv3_adapter.py` |
| Cheetah checkpoint | Complete | Peak eval return 791.1 |
| 20 rollouts | Complete | Cheetah Run seeds 0-19 |
| V4 recon report | Complete | See `docs/v4_recon.md` |
| month1_summary.md | Complete | This file |

## Key Metrics

- Checkpoint performance: 791.1 peak eval return, exceeding the reported DreamerV3 Cheetah Run range of 650-750.
- Training cost: about $6.65, from 9.5 hours on a RunPod RTX 4090.
- Rollouts generated: 20 Cheetah Run rollouts, seeds 0-19.
- Rollout shape: `true_obs` and `imagined_obs` are `(200, 17)`, `actions` is `(200, 6)`, and `rewards` is `(200,)`.
- L2 drift confirmed: seed 0 grows from 3.57 at step 0 to 13.81 at step 100, with max 19.82 at step 118.
- Batch drift confirmed: all 20 seeds have last-10-step mean L2 greater than first-10-step mean L2, and all have max L2 above 10.

## Plan vs Actual

| Aspect | Original Plan | Actual |
|--------|--------------|--------|
| Tasks (Month 1) | 3 tasks | 1 task, Cheetah only |
| Checkpoint source | Public, hoped | Self-trained |
| Training platform | Local M5 Pro | RunPod RTX 4090 |
| Week 3 task | Adapter design | Checkpoint training |
| Rollout count | 60, across 3 tasks x 20 seeds | 20, Cheetah Run seeds 0-19 |

## Major Decisions

- Narrowed Month 1 to Cheetah Run because no public DreamerV3 DMC checkpoints were available and one strong baseline was more valuable than three partial ones.
- Trained the checkpoint on RunPod because local MPS training risked consuming the schedule with uncertain throughput.
- Increased training throughput with larger batch and environment counts, which produced a strong checkpoint at low cost.
- Kept rollouts out of Git and committed only `results/rollouts/manifest.json`, preserving reproducibility without storing binary artifacts in source control.
- Introduced `WorldModelAdapter` and `DreamerV3Adapter` so Month 2 diagnostics can operate through a stable architecture-agnostic interface.
- Deferred DreamerV4 implementation because the inspected repo is CUDA/image/tokenizer oriented and substantially heavier than the current V3 proprio pipeline.

## Risk Assessment for Month 2

The main Month 2 risk is that frequency-domain decomposition may expose preprocessing or alignment assumptions that were invisible in time-domain sanity checks. The mitigation is to begin Month 2 with small, deterministic checks against the existing 20 rollouts before scaling analysis.

Another risk is over-interpreting one task. Cheetah Run is now a solid baseline, but results should be framed as a single-task proof of method until additional tasks are trained or sourced.

The V4 comparison remains higher risk because the current Dreamer4 implementation is not API-compatible with V3, requires CUDA, and is image-token based. The mitigation is to keep V4 as a Month 3/4 adapter target instead of a Month 2 dependency.

## Month 2 Readiness

- [x] Checkpoint ready
- [x] 20 rollouts ready
- [x] Adapter interface ready
- [ ] `freq_decompose.py`: NOT STARTED, Month 2 Week 1
- [ ] `error_curves.py`: NOT STARTED, Month 2 Week 2

## Final Git Tag

Month 1 baseline release tag:

`v0.1-month1-baseline`
