# DreamerV3 Cartpole Swingup Training Record

Completed on 2026-04-30 for the Month 2 Week 4 Cartpole diagnostic milestone.

## Training Configuration
- Task: `dmc_cartpole_swingup`
- Steps: 500k
- Batch size: `32`
- Parallel environments: `8`
- GPU: `RTX 4090 (RunPod)`
- Training time: ~10 hours
- Cost: ~$7

## Performance Results

### eval_return Timeline
- Full training log is not available locally.
- Peak observed during monitoring: ~854.
- Stable observed range during late training: ~820-850.

### Final Assessment
- Peak performance: ~854.
- Stable performance: ~820-850 range.
- Status: Ready for Month 2 Week 4 rollout extraction and frequency diagnostics.

## Checkpoint Information
- Filename: `cartpole_swingup_500k.pt`
- Location: `results/checkpoints/cartpole_swingup_500k.pt`
- Size: 218 MB
- Date: 2026-04-30
- Status: Ready

## Notes
Training used the same DreamerV3 hyperparameter profile as the Cheetah run
(`batch_size=32`, `envs=8`) on RunPod RTX 4090. The checkpoint is ready for
latent-aware v2 rollout extraction and cross-task comparison against Cheetah.
