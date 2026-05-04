#!/usr/bin/env bash
set -Eeuo pipefail

# Month 8 Controlled Training
# Pod: RunPod RTX 4090, standard PyTorch image
# Expected: ~20-24 GPU-hours, ~$14-17
# Purpose: controlled baseline + anchor-only Adaptive-SMAD on the same pod.

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_URL="${REPO_URL:-https://github.com/FengYe476/DriftDetect.git}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${WORKSPACE}/DriftDetect"

TASK="dmc_cheetah_run"
SEED=42
STEPS=500000
BATCH_SIZE=32
ENVS=8
N_SEEDS=20
TOTAL_STEPS=1000
IMAGINATION_START=200
HORIZON=200
SMAD_R=10
HIAD_ETA=0.20
HIAD_RE_EST_FREQ=25
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_COLLECT="${DEVICE_COLLECT:-cuda}"

CONTROL_ROOT="results/month8_controlled"
ENV_SNAPSHOT_DIR="${CONTROL_ROOT}/env_snapshot"
CONTROL_LOG_DIR="${CONTROL_ROOT}/logs"
BASELINE_DIR="${CONTROL_ROOT}/baseline"
ANCHOR_DIR="${CONTROL_ROOT}/anchor_only"
BASELINE_RUN_NAME="month8_baseline"
BASELINE_PHASE2_LINK="results/smad_phase2/${BASELINE_RUN_NAME}"
ANCHOR_RUN_NAME="month8_controlled/anchor_only"
ANCHOR_CONFIG="configs/month8_anchor_only.yaml"
U_BASIS="results/smad/U_drift_cheetah_r10.npy"
CURRENT_U="${CONTROL_ROOT}/current_U_anchor_only.npy"
REEST_LOG_LINK="results/tables/adaptive_smad_v1_reest_log.json"
REEST_LOG_TARGET="../month8_controlled/anchor_only/adaptive_smad_v1_reest_log.json"

ROLLOUT_BASELINE_DIR="${CONTROL_ROOT}/rollouts_baseline"
ROLLOUT_ANCHOR_ETA0_DIR="${CONTROL_ROOT}/rollouts_anchor_eta0"
ROLLOUT_ANCHOR_HIAD_DIR="${CONTROL_ROOT}/rollouts_anchor_hiad"

CURRENT_PHASE="initialization"
COMPLETED_PHASES=()

on_error() {
  local exit_code=$?
  echo ""
  echo "ERROR: Phase failed: ${CURRENT_PHASE}"
  echo "Exit code: ${exit_code}"
  if [ "${#COMPLETED_PHASES[@]}" -gt 0 ]; then
    echo "Completed phases:"
    printf "  - %s\n" "${COMPLETED_PHASES[@]}"
  else
    echo "Completed phases: none"
  fi
  echo ""
  list_outputs || true
  exit "${exit_code}"
}
trap on_error ERR

phase() {
  CURRENT_PHASE="$1"
  echo ""
  echo "=============================================================================="
  echo "PHASE: ${CURRENT_PHASE}"
  echo "START: $(date)"
  echo "=============================================================================="
}

mark_done() {
  COMPLETED_PHASES+=("$1")
  echo "DONE: $1 at $(date)"
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
    echo "Last 100 log lines from ${log_path}:"
    tail -100 "${log_path}" || true
    return "${status}"
  fi
}

ensure_repo() {
  mkdir -p "${WORKSPACE}"
  cd "${WORKSPACE}"
  if [ ! -d "${REPO_DIR}/.git" ]; then
    run_cmd git clone "${REPO_URL}" DriftDetect
  else
    cd "${REPO_DIR}"
    run_cmd git pull --ff-only
  fi
  cd "${REPO_DIR}"
}

prepare_dirs() {
  mkdir -p \
    "${ENV_SNAPSHOT_DIR}" \
    "${CONTROL_LOG_DIR}" \
    "${BASELINE_DIR}" \
    "${ANCHOR_DIR}" \
    "${ROLLOUT_BASELINE_DIR}" \
    "${ROLLOUT_ANCHOR_ETA0_DIR}" \
    "${ROLLOUT_ANCHOR_HIAD_DIR}" \
    results/smad_phase2 \
    results/tables
}

prepare_baseline_logdir_link() {
  local target="../month8_controlled/baseline"
  if [ -L "${BASELINE_PHASE2_LINK}" ]; then
    local current
    current="$(readlink "${BASELINE_PHASE2_LINK}")"
    if [ "${current}" != "${target}" ]; then
      echo "${BASELINE_PHASE2_LINK} points to ${current}, expected ${target}."
      return 1
    fi
    return
  fi
  if [ -e "${BASELINE_PHASE2_LINK}" ]; then
    echo "${BASELINE_PHASE2_LINK} already exists and is not a symlink."
    echo "Move it aside or set a fresh workspace before running this controlled script."
    return 1
  fi
  ln -s "${target}" "${BASELINE_PHASE2_LINK}"
}

prepare_reest_log_link() {
  if [ -L "${REEST_LOG_LINK}" ]; then
    ln -sfn "${REEST_LOG_TARGET}" "${REEST_LOG_LINK}"
    return
  fi
  if [ -e "${REEST_LOG_LINK}" ]; then
    echo "Warning: ${REEST_LOG_LINK} exists and is not a symlink."
    echo "Adaptive re-estimation log will be written there by train_adaptive_smad.py."
    return
  fi
  ln -s "${REEST_LOG_TARGET}" "${REEST_LOG_LINK}"
}

snapshot_environment() {
  python -m pip freeze > "${ENV_SNAPSHOT_DIR}/pip_freeze.txt"
  nvidia-smi > "${ENV_SNAPSHOT_DIR}/nvidia_smi.txt"
  python - <<'PY' > "results/month8_controlled/env_snapshot/torch_info.txt"
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
}

verify_prereqs() {
  if [ ! -f "${U_BASIS}" ]; then
    echo "Missing U basis: ${U_BASIS}"
    return 1
  fi
  if [ ! -f "${ANCHOR_CONFIG}" ]; then
    echo "Missing Month 8 adaptive config: ${ANCHOR_CONFIG}"
    return 1
  fi
  python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Month 8 controlled training.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
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
PY
}

train_baseline() {
  echo "=== Starting controlled baseline at $(date) ==="
  prepare_baseline_logdir_link
  run_nohup_and_wait "Controlled baseline training" "${CONTROL_LOG_DIR}/baseline_train.log" \
    python scripts/train_smad_phase2.py \
      --run_name "${BASELINE_RUN_NAME}" \
      --smad_eta 0.0 \
      --configs dmc_proprio \
      --task "${TASK}" \
      --steps "${STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --envs "${ENVS}" \
      --seed "${SEED}" \
      --device "${DEVICE_TRAIN}" \
      --compile False
  verify_checkpoint "${BASELINE_DIR}/latest.pt"
  echo "=== Baseline done at $(date) ==="
}

train_anchor_only() {
  echo "=== Starting anchor-only Adaptive-SMAD at $(date) ==="
  echo "Anchor loss enabled via ${ANCHOR_CONFIG}; eta=0 keeps transition damping off while re-estimating U."
  prepare_reest_log_link
  run_nohup_and_wait "Anchor-only Adaptive-SMAD training" "${CONTROL_LOG_DIR}/anchor_only_train.log" \
    python scripts/train_adaptive_smad.py \
      --config "${ANCHOR_CONFIG}" \
      --run_name "${ANCHOR_RUN_NAME}" \
      --device "${DEVICE_TRAIN}"
  verify_checkpoint "${ANCHOR_DIR}/latest.pt"
  if [ -f "${REEST_LOG_LINK}" ] && [ ! -f "${ANCHOR_DIR}/adaptive_smad_v1_reest_log.json" ]; then
    cp -f "${REEST_LOG_LINK}" "${ANCHOR_DIR}/adaptive_smad_v1_reest_log.json"
  fi
  echo "=== Anchor-only Adaptive-SMAD done at $(date) ==="
}

export_current_u() {
  echo "=== Exporting current_U at $(date) ==="
  python - <<'PY'
import numpy as np
import torch

ckpt = torch.load(
    "results/month8_controlled/anchor_only/latest.pt",
    map_location="cpu",
    weights_only=False,
)
state = ckpt["adaptive_smad_scheduler_state"]
U = np.array(state["current_U"], dtype=np.float64)
np.save("results/month8_controlled/current_U_anchor_only.npy", U)
print(f"Shape: {U.shape}")
print(f"Max |U^T U - I|: {abs(U.T @ U - np.eye(U.shape[1])).max():.6f}")
print(f"U_history entries: {len(state.get('U_history', []))}")
PY
  test -s "${CURRENT_U}"
}

collect_rollout_set() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$3"
  shift 3
  local extra_args=("$@")
  mkdir -p "${output_dir}"
  echo "=== Collecting ${label} rollouts at $(date) ==="
  for seed in $(seq 0 $((N_SEEDS - 1))); do
    local output="${output_dir}/cheetah_seed${seed}.npz"
    echo "[${label}] seed ${seed}/${N_SEEDS}: ${output}"
    python src/diagnostics/extract_rollout.py \
      --checkpoint "${checkpoint}" \
      --task "${TASK}" \
      --seed "${seed}" \
      --total_steps "${TOTAL_STEPS}" \
      --imagination_start "${IMAGINATION_START}" \
      --horizon "${HORIZON}" \
      --device "${DEVICE_COLLECT}" \
      --output "${output}" \
      "${extra_args[@]}"
  done
  verify_rollout_count "${output_dir}" "${N_SEEDS}"
}

collect_rollouts() {
  echo "=== Collecting rollouts at $(date) ==="
  collect_rollout_set \
    "baseline_eta0" \
    "${BASELINE_DIR}/latest.pt" \
    "${ROLLOUT_BASELINE_DIR}"

  collect_rollout_set \
    "anchor_eta0" \
    "${ANCHOR_DIR}/latest.pt" \
    "${ROLLOUT_ANCHOR_ETA0_DIR}"

  collect_rollout_set \
    "anchor_hiad_eta020" \
    "${ANCHOR_DIR}/latest.pt" \
    "${ROLLOUT_ANCHOR_HIAD_DIR}" \
    --use_hiad \
    --hiad_eta "${HIAD_ETA}" \
    --hiad_r "${SMAD_R}" \
    --hiad_re_est_freq "${HIAD_RE_EST_FREQ}" \
    --hiad_basis_mode initial_basis \
    --hiad_initial_basis "${CURRENT_U}"
}

verify_rollout_count() {
  local dir="$1"
  local expected="$2"
  local count
  count="$(find "${dir}" -maxdepth 1 -name '*.npz' | wc -l | tr -d ' ')"
  echo "${dir}: ${count}/${expected} rollout files"
  if [ "${count}" -lt "${expected}" ]; then
    echo "Expected at least ${expected} rollout files in ${dir}, found ${count}."
    return 1
  fi
}

checksums_and_summary() {
  echo "=== Computing checksums at $(date) ==="
  sha256sum "${BASELINE_DIR}/latest.pt" | tee "${CONTROL_ROOT}/baseline_latest.sha256"
  sha256sum "${ANCHOR_DIR}/latest.pt" | tee "${CONTROL_ROOT}/anchor_only_latest.sha256"
  sha256sum "${CURRENT_U}" | tee "${CONTROL_ROOT}/current_U_anchor_only.sha256"

  echo "=== Training summary at $(date) ==="
  python - <<'PY'
import json

for name in ["baseline", "anchor_only"]:
    path = f"results/month8_controlled/{name}/metrics.jsonl"
    try:
        with open(path) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        eval_records = [record for record in records if "eval_return" in record]
        if eval_records:
            last = eval_records[-1]
            print(
                f"{name}: final eval_return = {last.get('eval_return', 'N/A')} "
                f"at step {last.get('step', 'N/A')}"
            )
        elif records:
            print(f"{name}: no eval_return found; last record keys = {sorted(records[-1])}")
        else:
            print(f"{name}: metrics file is empty")
    except Exception as exc:
        print(f"{name}: could not read metrics: {exc}")
PY
}

list_outputs() {
  echo "Month 8 controlled outputs:"
  find "${CONTROL_ROOT}" -maxdepth 3 \( -type f -o -type l \) 2>/dev/null | sort || true
}

phase "Phase 0: Environment setup"
apt-get update
apt-get install -y libosmesa6-dev
ensure_repo
prepare_dirs
python -m pip install dm-control mujoco ruamel.yaml scipy
snapshot_environment
verify_prereqs
mark_done "${CURRENT_PHASE}"

phase "Phase 1: Controlled baseline"
train_baseline
mark_done "${CURRENT_PHASE}"

phase "Phase 2: Anchor-only Adaptive-SMAD"
train_anchor_only
mark_done "${CURRENT_PHASE}"

phase "Phase 3: Export current_U"
export_current_u
mark_done "${CURRENT_PHASE}"

phase "Phase 4: Collect rollouts"
collect_rollouts
mark_done "${CURRENT_PHASE}"

phase "Phase 5: Checksums and summary"
checksums_and_summary
list_outputs
mark_done "${CURRENT_PHASE}"

echo "=== All Month 8 controlled training complete at $(date) ==="
