#!/bin/bash
# Adaptive-SMAD v1 RunPod launcher.
#
# Usage: bash scripts/runpod_adaptive_smad_v1.sh
# Expected runtime: ~20-21 GPU-hours on RTX 4090
# Expected cost: ~$14-15
#
# Timing estimates:
#   Baseline training: ~10 GPU-hours
#   Adaptive-SMAD training: ~10 GPU-hours + ~5 min re-estimation overhead
#   Rollout extraction: ~10 min per run
#   Total: ~20-21 GPU-hours

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

run_cmd() {
  echo "+ $*"
  "$@"
}

run_nohup_and_wait() {
  local label="$1"
  local log_path="$2"
  shift 2
  mkdir -p "$(dirname "${log_path}")"
  echo "+ nohup $* > ${log_path} 2>&1 &"
  nohup "$@" > "${log_path}" 2>&1 &
  local pid=$!
  echo "${label} PID: ${pid}"
  echo "${label} log: ${log_path}"
  set +e
  wait "${pid}"
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "${label} failed with exit code ${status}."
    echo "Last 80 log lines from ${log_path}:"
    tail -80 "${log_path}" || true
    return "${status}"
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK="dmc_cheetah_run"
SEED=42
STEPS=500000
BATCH_SIZE=32
ENVS=8
SMAD_ETA="0.20"
SMAD_R=10
U_BASIS="results/smad/U_drift_cheetah_r10.npy"
CONFIG_PATH="configs/adaptive_smad_v1.yaml"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_COLLECT="${DEVICE_COLLECT:-cuda}"
N_SEEDS=20

ADAPTIVE_ROOT="results/adaptive_smad_v1"
ADAPTIVE_LOG="${ADAPTIVE_ROOT}/train.log"
ADAPTIVE_LATEST="${ADAPTIVE_ROOT}/latest.pt"
ADAPTIVE_CKPT_DIR="${ADAPTIVE_ROOT}/checkpoints"
ADAPTIVE_FINAL="${ADAPTIVE_CKPT_DIR}/final.pt"
ADAPTIVE_REEST_LOG="results/tables/adaptive_smad_v1_reest_log.json"

BASELINE_DIR="${ADAPTIVE_ROOT}/baseline"
PHASE2_BASELINE_DIR="results/smad_phase2_v2/baseline"
BASELINE_RUN_NAME="adaptive_smad_v1_baseline"
BASELINE_TRAIN_DIR="results/smad_phase2/${BASELINE_RUN_NAME}"
BASELINE_TRAIN_LOG="${ADAPTIVE_ROOT}/baseline_train.log"

ADAPTIVE_ROLLOUT_ETA020="${ADAPTIVE_ROOT}/rollouts_adaptive_eta020"
ADAPTIVE_ROLLOUT_ETA000="${ADAPTIVE_ROOT}/rollouts_adaptive_eta000"
BASELINE_ROLLOUT_DIR="${ADAPTIVE_ROOT}/rollouts_baseline"

EVAL_VIEW_ROOT="${ADAPTIVE_ROOT}/eval_views"
BASELINE_VIEW_DIR="${EVAL_VIEW_ROOT}/baseline"
ADAPTIVE_ETA020_VIEW_DIR="${EVAL_VIEW_ROOT}/adaptive_eta020"
ADAPTIVE_ETA000_VIEW_DIR="${EVAL_VIEW_ROOT}/adaptive_eta000"

PRIMARY_METRICS_JSON="results/tables/adaptive_smad_v1_metrics_eta020.json"
DIAGNOSTIC_METRICS_JSON="results/tables/adaptive_smad_v1_metrics_eta000.json"
FINAL_METRICS_JSON="results/tables/adaptive_smad_v1_metrics.json"
B1_ADAPTIVE_JSON="results/tables/adaptive_smad_v1_b1_eta000.json"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_RECOLLECT="${FORCE_RECOLLECT:-0}"
# Baseline reuse modes: symlink (default), copy, retrain.
BASELINE_REUSE_MODE="${BASELINE_REUSE_MODE:-symlink}"

find_checkpoint() {
  local dir="$1"
  local candidates=(
    "${dir}/checkpoints/final.pt"
    "${dir}/checkpoints/latest.pt"
    "${dir}/latest.pt"
    "${dir}/final.pt"
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

copy_training_checkpoint() {
  local train_dir="$1"
  local target_dir="$2"
  local source="${train_dir}/latest.pt"
  if [ ! -s "${source}" ]; then
    echo "Could not find ${source}; searching ${train_dir} for checkpoint files."
    source="$(find "${train_dir}" -type f \( -name '*.pt' -o -name '*.pth' \) -print0 | xargs -0 ls -t 2>/dev/null | head -1 || true)"
  fi
  if [ -z "${source}" ] || [ ! -s "${source}" ]; then
    echo "No checkpoint found in ${train_dir}."
    return 1
  fi
  mkdir -p "${target_dir}/checkpoints"
  cp -f "${source}" "${target_dir}/latest.pt"
  cp -f "${source}" "${target_dir}/checkpoints/final.pt"
  cp -f "${source}" "${target_dir}/checkpoints/latest.pt"
  if [ -f "${train_dir}/metrics.jsonl" ]; then
    cp -f "${train_dir}/metrics.jsonl" "${target_dir}/metrics.jsonl"
  fi
  echo "Copied checkpoint ${source} -> ${target_dir}/checkpoints/final.pt and latest.pt"
}

train_baseline() {
  BASELINE_RETRAINED=1
  run_nohup_and_wait "Baseline training" "${BASELINE_TRAIN_LOG}" \
    python scripts/train_smad_phase2.py \
      --run_name "${BASELINE_RUN_NAME}" \
      --smad_eta 0.0 \
      --configs dmc_proprio \
      --task "${TASK}" \
      --steps "${STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --envs "${ENVS}" \
      --seed "${SEED}" \
      --device "${DEVICE_TRAIN}"

  copy_training_checkpoint "${BASELINE_TRAIN_DIR}" "${BASELINE_DIR}"
  verify_checkpoint "$(find_checkpoint "${BASELINE_DIR}")"
}

prepare_baseline() {
  local existing
  existing="$(find_checkpoint "${BASELINE_DIR}" 2>/dev/null || true)"
  if [ -n "${existing}" ] && [ "${FORCE_RETRAIN}" != "1" ]; then
    echo "Adaptive-SMAD baseline checkpoint already exists: ${existing}"
    verify_checkpoint "${existing}"
    return
  fi

  if [ "${FORCE_RETRAIN}" = "1" ]; then
    echo "FORCE_RETRAIN=1; retraining baseline."
    train_baseline
    return
  fi

  if [ -d "${PHASE2_BASELINE_DIR}" ]; then
    local phase2_ckpt
    phase2_ckpt="$(find_checkpoint "${PHASE2_BASELINE_DIR}" 2>/dev/null || true)"
    if [ -z "${phase2_ckpt}" ]; then
      echo "Found ${PHASE2_BASELINE_DIR}, but no checkpoint was found inside it."
      return 1
    fi
    mkdir -p "${ADAPTIVE_ROOT}"
    case "${BASELINE_REUSE_MODE}" in
      symlink)
        if [ ! -e "${BASELINE_DIR}" ] && [ ! -L "${BASELINE_DIR}" ]; then
          ln -s "../smad_phase2_v2/baseline" "${BASELINE_DIR}"
          echo "Symlinked ${BASELINE_DIR} -> ../smad_phase2_v2/baseline"
        else
          echo "${BASELINE_DIR} already exists; leaving it in place."
        fi
        ;;
      copy)
        if [ ! -e "${BASELINE_DIR}" ]; then
          cp -a "${PHASE2_BASELINE_DIR}" "${BASELINE_DIR}"
          echo "Copied ${PHASE2_BASELINE_DIR} -> ${BASELINE_DIR}"
        else
          echo "${BASELINE_DIR} already exists; leaving it in place."
        fi
        ;;
      retrain)
        train_baseline
        return
        ;;
      *)
        echo "Unknown BASELINE_REUSE_MODE=${BASELINE_REUSE_MODE}; expected symlink, copy, or retrain."
        return 1
        ;;
    esac
    verify_checkpoint "$(find_checkpoint "${BASELINE_DIR}")"
    return
  fi

  echo "No Phase 2 baseline directory found at ${PHASE2_BASELINE_DIR}; retraining baseline."
  train_baseline
}

train_adaptive_if_needed() {
  if [ "${FORCE_RETRAIN}" != "1" ] && [ -s "${ADAPTIVE_FINAL}" ]; then
    echo "Adaptive-SMAD final checkpoint already exists; skipping training: ${ADAPTIVE_FINAL}"
    verify_checkpoint "${ADAPTIVE_FINAL}"
    return
  fi

  mkdir -p "${ADAPTIVE_ROOT}"
  run_nohup_and_wait "Adaptive-SMAD training" "${ADAPTIVE_LOG}" \
    python scripts/train_adaptive_smad.py --config "${CONFIG_PATH}"

  expose_adaptive_checkpoint
}

expose_adaptive_checkpoint() {
  if [ ! -s "${ADAPTIVE_LATEST}" ]; then
    echo "Adaptive training did not produce ${ADAPTIVE_LATEST}."
    return 1
  fi
  mkdir -p "${ADAPTIVE_CKPT_DIR}"
  cp -f "${ADAPTIVE_LATEST}" "${ADAPTIVE_FINAL}"
  cp -f "${ADAPTIVE_LATEST}" "${ADAPTIVE_CKPT_DIR}/latest.pt"
  echo "Exposed Adaptive-SMAD final checkpoint at ${ADAPTIVE_FINAL}"
  verify_checkpoint "${ADAPTIVE_FINAL}"
}

collect_rollouts() {
  local checkpoint="$1"
  local output_dir="$2"
  local eta="$3"
  shift 3
  local extra_args=("$@")
  local force_args=()
  if [ "${FORCE_RECOLLECT}" = "1" ]; then
    force_args+=(--force-recollect)
  fi
  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode collect \
    --checkpoint "${checkpoint}" \
    --device "${DEVICE_COLLECT}" \
    --eta_inference "${eta}" \
    --n_seeds "${N_SEEDS}" \
    --output_dir "${output_dir}" \
    --rank "${SMAD_R}" \
    "${extra_args[@]}" \
    "${force_args[@]}"
  verify_rollout_count "${output_dir}" "${N_SEEDS}"
}

collect_all_rollouts() {
  local baseline_ckpt
  baseline_ckpt="$(find_checkpoint "${BASELINE_DIR}")"

  collect_rollouts "${baseline_ckpt}" "${BASELINE_ROLLOUT_DIR}" 0.0
  collect_rollouts "${ADAPTIVE_FINAL}" "${ADAPTIVE_ROLLOUT_ETA020}" "${SMAD_ETA}" \
    --adaptive-u-from-checkpoint
  collect_rollouts "${ADAPTIVE_FINAL}" "${ADAPTIVE_ROLLOUT_ETA000}" 0.0
}

verify_rollout_count() {
  local dir="$1"
  local expected="$2"
  local count
  count="$(count_npz "${dir}")"
  echo "${dir}: ${count}/${expected} rollout files"
  if [ "${count}" -lt "${expected}" ]; then
    echo "Expected at least ${expected} rollout files in ${dir}, found ${count}."
    return 1
  fi
}

link_path() {
  local source="$1"
  local target="$2"
  mkdir -p "$(dirname "${target}")"
  if [ -L "${target}" ]; then
    local current
    current="$(readlink "${target}")"
    if [ "${current}" = "${source}" ]; then
      return
    fi
    rm "${target}"
  fi
  if [ -e "${target}" ]; then
    echo "${target} exists and is not a symlink; leaving it in place."
    return
  fi
  ln -s "${source}" "${target}"
}

prepare_eval_view() {
  local view_dir="$1"
  local checkpoint="$2"
  local rollout_dir="$3"
  local metrics_source="${4:-}"
  mkdir -p "${view_dir}"
  link_path "${REPO_ROOT}/${checkpoint}" "${view_dir}/latest.pt"
  link_path "${REPO_ROOT}/${rollout_dir}" "${view_dir}/rollouts"
  if [ -n "${metrics_source}" ] && [ -f "${metrics_source}" ]; then
    link_path "${REPO_ROOT}/${metrics_source}" "${view_dir}/metrics.jsonl"
  fi
}

run_metrics() {
  local baseline_ckpt
  baseline_ckpt="$(find_checkpoint "${BASELINE_DIR}")"

  prepare_eval_view "${BASELINE_VIEW_DIR}" "${baseline_ckpt}" "${BASELINE_ROLLOUT_DIR}" "${BASELINE_DIR}/metrics.jsonl"
  prepare_eval_view "${ADAPTIVE_ETA020_VIEW_DIR}" "${ADAPTIVE_FINAL}" "${ADAPTIVE_ROLLOUT_ETA020}" "${ADAPTIVE_ROOT}/metrics.jsonl"
  prepare_eval_view "${ADAPTIVE_ETA000_VIEW_DIR}" "${ADAPTIVE_FINAL}" "${ADAPTIVE_ROLLOUT_ETA000}" "${ADAPTIVE_ROOT}/metrics.jsonl"

  run_cmd python scripts/run_smad_eval.py \
    --baseline_dir "${BASELINE_VIEW_DIR}" \
    --smad_dir "${ADAPTIVE_ETA020_VIEW_DIR}" \
    --n_seeds "${N_SEEDS}" \
    --horizon 200 \
    --trim 25 175 \
    --output_json "${PRIMARY_METRICS_JSON}" \
    --smad_eta "${SMAD_ETA}" \
    --smad_U_path "${U_BASIS}"

  run_cmd python scripts/run_smad_eval.py \
    --baseline_dir "${BASELINE_VIEW_DIR}" \
    --smad_dir "${ADAPTIVE_ETA000_VIEW_DIR}" \
    --n_seeds "${N_SEEDS}" \
    --horizon 200 \
    --trim 25 175 \
    --output_json "${DIAGNOSTIC_METRICS_JSON}" \
    --smad_eta 0.0 \
    --smad_U_path "${U_BASIS}"

  run_cmd python scripts/run_b1_post_smad_drift.py \
    --mode analyze \
    --rollout_dir "${ADAPTIVE_ROLLOUT_ETA000}" \
    --baseline_U "${U_BASIS}" \
    --output "${B1_ADAPTIVE_JSON}"

  merge_metrics
}

merge_metrics() {
  mkdir -p "$(dirname "${FINAL_METRICS_JSON}")"
  python - \
    "${PRIMARY_METRICS_JSON}" \
    "${DIAGNOSTIC_METRICS_JSON}" \
    "${B1_ADAPTIVE_JSON}" \
    "${FINAL_METRICS_JSON}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

primary_path, diagnostic_path, b1_path, output_path = map(Path, sys.argv[1:])

def load(path):
    return json.loads(path.read_text()) if path.exists() else None

result = {
    "analysis": "adaptive_smad_v1_eval",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "primary_eta020": load(primary_path),
    "diagnostic_eta000": load(diagnostic_path),
    "b1_eta000": load(b1_path),
    "source_files": {
        "primary_eta020": str(primary_path),
        "diagnostic_eta000": str(diagnostic_path),
        "b1_eta000": str(b1_path),
    },
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(f"Saved merged metrics: {output_path}")
PY
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
  find "${ADAPTIVE_ROOT}" -maxdepth 3 -type f \( -name '*.pt' -o -name '*.pth' \) -exec ls -lh {} \; 2>/dev/null || true
  if [ -L "${BASELINE_DIR}" ]; then
    echo "Baseline symlink:"
    ls -l "${BASELINE_DIR}"
  fi
  echo ""
  echo "Rollout directories:"
  for dir in "${BASELINE_ROLLOUT_DIR}" "${ADAPTIVE_ROLLOUT_ETA020}" "${ADAPTIVE_ROLLOUT_ETA000}"; do
    echo "  ${dir}: $(count_npz "${dir}") npz files"
  done
  echo ""
  echo "Metrics:"
  for path in "${FINAL_METRICS_JSON}" "${PRIMARY_METRICS_JSON}" "${DIAGNOSTIC_METRICS_JSON}" "${B1_ADAPTIVE_JSON}" "${ADAPTIVE_REEST_LOG}"; do
    if [ -f "${path}" ]; then
      ls -lh "${path}"
    fi
  done
}

print_download_checklist() {
  echo "Download checklist:"
  echo "  - ${ADAPTIVE_FINAL}"
  echo "  - ${ADAPTIVE_LOG}"
  echo "  - ${ADAPTIVE_REEST_LOG}"
  echo "  - ${ADAPTIVE_ROLLOUT_ETA020}/*.npz"
  echo "  - ${ADAPTIVE_ROLLOUT_ETA000}/*.npz"
  echo "  - ${BASELINE_ROLLOUT_DIR}/*.npz"
  echo "  - ${FINAL_METRICS_JSON}"
  echo "  - ${B1_ADAPTIVE_JSON}"
  if [ "${BASELINE_RETRAINED}" = "1" ]; then
    echo "  - ${BASELINE_DIR}/checkpoints/final.pt"
  else
    echo "  - baseline checkpoint was reused via ${BASELINE_REUSE_MODE}"
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
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Missing Adaptive-SMAD config: ${CONFIG_PATH}"
  exit 1
fi
mark_done "${CURRENT_PHASE}"

phase "Phase 1: Baseline checkpoint"
prepare_baseline
mark_done "${CURRENT_PHASE}"

phase "Phase 2: Adaptive-SMAD training"
train_adaptive_if_needed
mark_done "${CURRENT_PHASE}"

phase "Phase 3: Rollout extraction"
collect_all_rollouts
mark_done "${CURRENT_PHASE}"

phase "Phase 4: Frequency metrics and B1 diagnostic"
run_metrics
mark_done "${CURRENT_PHASE}"

phase "Phase 5: Verification and download checklist"
list_outputs
echo ""
print_download_checklist
mark_done "${CURRENT_PHASE}"

echo ""
echo "Adaptive-SMAD v1 RunPod launcher completed successfully."
