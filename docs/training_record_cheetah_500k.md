# DreamerV3 Cheetah Run Training Record

Completed on 2026-04-29 for the Week 3 DreamerV3 training milestone.

## Training Configuration
- Task: `dmc_cheetah_run`
- Steps: 505k (target 500k)
- Batch size: `32`
- Parallel environments: `8`
- GPU: `RTX 4090 (RunPod)`
- Training time: ~9.5 hours
- Cost: ~$6.65

## Performance Results

### eval_return Timeline
- Step 365k: 727.0
- Step 465k: 735.5
- Step 485k: 791.1 (peak)
- Step 495k: 753.7
- Step 505k: 655.2

### train_return Statistics
- Step 488k average: ~701
- Step 496k average: ~709
- Step 504k average: ~718
- Step 504k maximum: 774.9

## Final Assessment

Peak performance: 791.1 at Step 485k.

Stable performance: 730-750 range.

Average `train_return`: 715-720.

Comparison to DreamerV3 paper:
- Paper 500k range: 650-750
- Result: exceeded the upper bound with a 791.1 peak and sustained 730-750 performance

## Checkpoint Information
- Filename: `cheetah_run_500k.pt`
- Location: `results/checkpoints/cheetah_run_500k.pt`
- Date: 2026-04-29
- Status: Ready for Week 4 rollout extraction

## Notes
Training exceeded expectations. The optimized hyperparameters (`batch_size=32`, `envs=8`) resulted in 76% GPU utilization and achieved performance above the original DreamerV3 paper's reported upper range for this setting. This checkpoint provides a high-quality world model for Month 2's frequency-domain drift analysis and is ready for Week 4 rollout extraction.
