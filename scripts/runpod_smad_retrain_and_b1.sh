#!/bin/bash
# Retrain Month 5 SMAD Phase 2 and run corrected B1 staleness check.
# Run from the DriftDetect repo root on RunPod with an RTX 4090:
#   bash scripts/runpod_smad_retrain_and_b1.sh

set -Eeuo pipefail

CURRENT_PHASE="initialization"
COMPLETED_PHASES=()
BASELINE_RETRAINED=0

on_error() {
  local exit_code=$?
  echo ""
  echo "ERROR: Phase failed: ${CURRENT_PHASE}"
  echo "Exit code: ${exit_code}"
  if [ "${#COMPLETED_PHASES[@]}" -gt 0 ]; then
    echo "Completed phases:"
    printf '  - %s\n' "${COMPLETED_PHASES[@]}"
  else
    echo "Completed phases: none"
  fi
  echo ""
  echo "Partial outputs currently present:"
  list_outputs || true
  echo ""
  print_download_checklist || true
  exit "${exit_code}"
}
trap on_error ERR

mark_done() {
  COMPLETED_PHASES+=("$1")
}

phase() {
  CURRENT_PHASE="$1"
  echo ""
  echo "=============================================================================="
  echo "PHASE: ${CURRENT_PHASE}"
  echo "=============================================================================="
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="dmc_cheetah_run"
SEED=42
STEPS=500000
BATCH_SIZE=32
ENVS=8
SMAD_ETA="0.20"
SMAD_R=10
U_BASIS="results/smad/U_drift_cheetah_r10.npy"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_COLLECT="${DEVICE_COLLECT:-cuda}"
N_SEEDS=20

SMAD_RUN_NAME="smad_eta020_v2"
BASELINE_RUN_NAME="baseline_v2"
SMAD_TRAIN_DIR="results/smad_phase2/${SMAD_RUN_NAME}"
BASELINE_TRAIN_DIR="results/smad_phase2/${BASELINE_RUN_NAME}"

SMAD_V2_DIR="results/smad_phase2_v2/smad_eta020"
BASELINE_V2_DIR="results/smad_phase2_v2/baseline"
SMAD_CKPT_DIR="${SMAD_V2_DIR}/checkpoints"
BASELINE_CKPT_DIR="${BASELINE_V2_DIR}/checkpoints"
SMAD_FINAL="${SMAD_CKPT_DIR}/final.pt"
SMAD_LATEST="${SMAD_CKPT_DIR}/latest.pt"
BASELINE_FINAL="${BASELINE_CKPT_DIR}/final.pt"
BASELINE_LATEST="${BASELINE_CKPT_DIR}/latest.pt"

SMAD_ROLLOUT_DIR="${SMAD_V2_DIR}/rollouts"
BASELINE_ROLLOUT_DIR="${BASELINE_V2_DIR}/rollouts"
B1_ETA0_ROLLOUT_DIR="results/b1_post_smad_rollouts_eta0"
B1_CORRECTED_JSON="results/tables/b1_staleness_check_corrected.json"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_RECOLLECT="${FORCE_RECOLLECT:-0}"

run_cmd() {
  echo "+ $*"
  "$@"
}

train_smad() {
  if [ "${FORCE_RETRAIN}" != "1" ] && [ -s "${SMAD_FINAL}" ]; then
    echo "SMAD final checkpoint already exists; skipping retrain: ${SMAD_FINAL}"
    verify_checkpoint "${SMAD_FINAL}"
    return
  fi

  run_cmd python scripts/train_smad_phase2.py \
    --run_name "${SMAD_RUN_NAME}" \
    --smad_eta "${SMAD_ETA}" \
    --smad_r "${SMAD_R}" \
    --smad_U_path "${U_BASIS}" \
    --configs dmc_proprio \
    --task "${TASK}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --envs "${ENVS}" \
    --seed "${SEED}" \
    --device "${DEVICE_TRAIN}"

  copy_training_checkpoint "${SMAD_TRAIN_DIR}" "${SMAD_CKPT_DIR}"
  verify_checkpoint "${SMAD_FINAL}"
}

train_baseline_if_needed() {
  local existing
  existing="$(find_existing_baseline_checkpoint || true)"
  if [ -n "${existing}" ]; then
    echo "Found existing baseline checkpoint: ${existing}"
    mkdir -p "${BASELINE_CKPT_DIR}"
    cp -f "${existing}" "${BASELINE_FINAL}"
    cp -f "${existing}" "${BASELINE_LATEST}"
    verify_checkpoint "${BASELINE_FINAL}"
    BASELINE_RETRAINED=0
    return
  fi

  BASELINE_RETRAINED=1
  if [ "${FORCE_RETRAIN}" != "1" ] && [ -s "${BASELINE_FINAL}" ]; then
    echo "Baseline final checkpoint already exists; skipping retrain: ${BASELINE_FINAL}"
    verify_checkpoint "${BASELINE_FINAL}"
    return
  fi

  run_cmd python scripts/train_smad_phase2.py \
    --run_name "${BASELINE_RUN_NAME}" \
    --smad_eta 0.0 \
    --configs dmc_proprio \
    --task "${TASK}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --envs "${ENVS}" \
    --seed "${SEED}" \
    --device "${DEVICE_TRAIN}"

  copy_training_checkpoint "${BASELINE_TRAIN_DIR}" "${BASELINE_CKPT_DIR}"
  verify_checkpoint "${BASELINE_FINAL}"
}

find_existing_baseline_checkpoint() {
  local candidates=(
    "results/smad_phase2_v2/baseline/checkpoints/final.pt"
    "results/smad_phase2_v2/baseline/checkpoints/latest.pt"
    "results/smad_phase2/baseline/checkpoints/final.pt"
    "results/smad_phase2/baseline/checkpoints/latest.pt"
    "results/smad_phase2/baseline/latest.pt"
    "results/smad_phase2/baseline/final.pt"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -s "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

copy_training_checkpoint() {
  local train_dir="$1"
  local ckpt_dir="$2"
  local source="${train_dir}/latest.pt"
  if [ ! -s "${source}" ]; then
    echo "Could not find ${source}; searching ${train_dir} for checkpoint files."
    source="$(find "${train_dir}" -type f \( -name '*.pt' -o -name '*.pth' \) -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)"
  fi
  if [ -z "${source}" ] || [ ! -s "${source}" ]; then
    echo "No checkpoint found after training in ${train_dir}."
    return 1
  fi
  mkdir -p "${ckpt_dir}"
  cp -f "${source}" "${ckpt_dir}/final.pt"
  cp -f "${source}" "${ckpt_dir}/latest.pt"
  echo "Copied checkpoint ${source} -> ${ckpt_dir}/final.pt and latest.pt"
}

verify_checkpoint() {
  local path="$1"
  if [ ! -s "${path}" ]; then
    echo "Missing checkpoint: ${path}"
    return 1
  fi
  python - "$path" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
size_mb = path.stat().st_size / (1024 * 1024)
print(f"{path}: {size_mb:.1f} MB")
if size_mb < 20:
    raise SystemExit(f"Checkpoint is suspiciously small: {size_mb:.1f} MB")
if size_mb > 500:
    print(f"Warning: checkpoint is larger than expected: {size_mb:.1f} MB")
PY
}

collect_smad_eta020_rollouts() {
  local force_args=()
  if [ "${FORCE_RECOLLECT}" = "1" ]; then
    force_args+=(--force-recollect)
  fi
  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode collect \
    --checkpoint "${SMAD_FINAL}" \
    --device "${DEVICE_COLLECT}" \
    --eta_inference "${SMAD_ETA}" \
    --n_seeds "${N_SEEDS}" \
    --output_dir "${SMAD_ROLLOUT_DIR}" \
    --baseline_U "${U_BASIS}" \
    "${force_args[@]}"
}

collect_b1_eta0_rollouts() {
  local force_args=()
  if [ "${FORCE_RECOLLECT}" = "1" ]; then
    force_args+=(--force-recollect)
  fi
  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode collect \
    --checkpoint "${SMAD_FINAL}" \
    --device "${DEVICE_COLLECT}" \
    --eta_inference 0.0 \
    --n_seeds "${N_SEEDS}" \
    --output_dir "${B1_ETA0_ROLLOUT_DIR}" \
    "${force_args[@]}"
}

run_b1_analysis() {
  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode analyze \
    --rollout_dir "${B1_ETA0_ROLLOUT_DIR}" \
    --baseline_U "${U_BASIS}" \
    --output "${B1_CORRECTED_JSON}"
}

collect_baseline_rollouts() {
  local force_args=()
  if [ "${FORCE_RECOLLECT}" = "1" ]; then
    force_args+=(--force-recollect)
  fi
  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode collect \
    --checkpoint "${BASELINE_FINAL}" \
    --device "${DEVICE_COLLECT}" \
    --eta_inference 0.0 \
    --n_seeds "${N_SEEDS}" \
    --output_dir "${BASELINE_ROLLOUT_DIR}" \
    "${force_args[@]}"
}

count_npz() {
  local dir="$1"
  if [ -d "${dir}" ]; then
    find "${dir}" -maxdepth 1 -name '*.npz' | wc -l | tr -d ' '
  else
    echo "0"
  fi
}

list_outputs() {
  echo "Checkpoints:"
  find results/smad_phase2_v2 -type f \( -name '*.pt' -o -name '*.pth' \) -exec ls -lh {} \; 2>/dev/null || true
  echo ""
  echo "Rollout directories:"
  for dir in "${SMAD_ROLLOUT_DIR}" "${B1_ETA0_ROLLOUT_DIR}" "${BASELINE_ROLLOUT_DIR}"; do
    echo "  ${dir}: $(count_npz "${dir}") npz files"
  done
  echo ""
  if [ -f "${B1_CORRECTED_JSON}" ]; then
    echo "B1 corrected result:"
    ls -lh "${B1_CORRECTED_JSON}"
  fi
}

print_b1_result() {
  if [ ! -f "${B1_CORRECTED_JSON}" ]; then
    echo "B1 corrected result not found: ${B1_CORRECTED_JSON}"
    return
  fi
  python - "${B1_CORRECTED_JSON}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
print("B1 corrected side-by-side table:")
for row in data.get("side_by_side_comparison", []):
    print(
        f"  {row['metric']}: original={row['original_eta020_rollouts']} "
        f"corrected={row['corrected_eta0_rollouts']}"
    )
print("Verdict:", data.get("interpretation", {}).get("verdict"))
PY
}

print_download_checklist() {
  echo "Download checklist:"
  echo "  - results/smad_phase2_v2/smad_eta020/checkpoints/final.pt"
  echo "  - results/smad_phase2_v2/smad_eta020/checkpoints/latest.pt"
  echo "  - results/smad_phase2_v2/smad_eta020/rollouts/*.npz"
  echo "  - results/b1_post_smad_rollouts_eta0/*.npz"
  echo "  - results/tables/b1_staleness_check_corrected.json"
  if [ "${BASELINE_RETRAINED}" = "1" ]; then
    echo "  - results/smad_phase2_v2/baseline/checkpoints/final.pt"
    echo "  - results/smad_phase2_v2/baseline/rollouts/*.npz"
  else
    echo "  - baseline checkpoint/rollouts were reused if already available"
  fi
}

phase "Environment setup"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate env-dreamerv3
else
  echo "conda not found on PATH."
  exit 1
fi
python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this RunPod script.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
if [ ! -f "${U_BASIS}" ]; then
  echo "Missing U_drift basis: ${U_BASIS}"
  exit 1
fi
mark_done "${CURRENT_PHASE}"

phase "Phase 1: Retrain SMAD eta=0.20"
train_smad
mark_done "${CURRENT_PHASE}"

phase "Phase 1b: Collect SMAD eta=0.20 evaluation rollouts"
collect_smad_eta020_rollouts
mark_done "${CURRENT_PHASE}"

phase "Phase 2: Collect B1 eta=0 natural-drift rollouts"
collect_b1_eta0_rollouts
mark_done "${CURRENT_PHASE}"

phase "Phase 2b: Analyze corrected B1 staleness check"
run_b1_analysis
mark_done "${CURRENT_PHASE}"

phase "Phase 3: Baseline checkpoint and rollout set"
train_baseline_if_needed
collect_baseline_rollouts
mark_done "${CURRENT_PHASE}"

phase "Phase 4: Verification and download checklist"
list_outputs
print_b1_result
echo ""
print_download_checklist
mark_done "${CURRENT_PHASE}"

echo ""
echo "All phases completed successfully."
