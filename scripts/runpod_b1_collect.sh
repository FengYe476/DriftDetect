#!/bin/bash
# B1 Correction: collect eta=0 rollouts from SMAD checkpoint
# Run this on RunPod with RTX 4090

set -e

# Activate environment
conda activate env-dreamerv3

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Locate checkpoint (update path if needed)
CKPT_PATH="results/smad_phase2/smad_eta020/latest.pt"

# Collect 20 seeds of rollouts with NO damping
python scripts/run_b1_post_smad_drift.py \
    --mode collect \
    --checkpoint "$CKPT_PATH" \
    --device cuda \
    --eta_inference 0.0 \
    --n_seeds 20 \
    --output_dir results/b1_post_smad_rollouts_eta0/

# Run analysis on the new eta=0 rollouts
python scripts/run_b1_post_smad_drift.py \
    --mode analyze \
    --rollout_dir results/b1_post_smad_rollouts_eta0/ \
    --baseline_U results/smad/U_drift_cheetah_r10.npy \
    --output results/tables/b1_staleness_check_corrected.json

echo "B1 correction complete. Check results/tables/b1_staleness_check_corrected.json"
