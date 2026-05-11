#!/usr/bin/env bash
set -Eeuo pipefail

# R2-Dreamer final restart launcher for RunPod RTX 4090.
#
# Usage:
#   bash scripts/runpod_r2_restart.sh
#   MODE=baseline bash scripts/runpod_r2_restart.sh
#   MODE=sharp bash scripts/runpod_r2_restart.sh

MODE="${MODE:-both}"  # one of: baseline, sharp, both

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_URL="${REPO_URL:-https://github.com/FengYe476/DriftDetect.git}"
R2_URL="${R2_URL:-https://github.com/NM512/r2dreamer.git}"
R2_REF="${R2_REF:-}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/DriftDetect}"

TASK="${TASK:-dmc_cheetah_run}"
SEED="${SEED:-42}"
STEPS="${STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ENVS="${ENVS:-16}"
N_SEEDS="${N_SEEDS:-20}"
HORIZON="${HORIZON:-200}"
IMAGINATION_START="${IMAGINATION_START:-200}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_EXTRACT="${DEVICE_EXTRACT:-cuda:0}"
STORAGE_DEVICE="${STORAGE_DEVICE:-}"

BASELINE_RUN_NAME="r2_final/baseline"
SHARP_RUN_NAME="r2_final/sharp_v2"
BASELINE_DIR="results/${BASELINE_RUN_NAME}"
SHARP_DIR="results/${SHARP_RUN_NAME}"
BASELINE_CKPT="${BASELINE_DIR}/latest.pt"
SHARP_CKPT="${SHARP_DIR}/latest.pt"
R2_FINAL_ROOT="results/r2_final"
ROLLOUT_OUTDIR="${R2_FINAL_ROOT}/rollouts"
TABLES_DIR="results/tables"
COMPARISON_JSON="${TABLES_DIR}/r2_comparison.json"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${R2_FINAL_ROOT}/logs/${RUN_ID}}"

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

run_logged() {
  local label="$1"
  local log_path="$2"
  shift 2
  mkdir -p "$(dirname "${log_path}")"
  echo "+ $* > ${log_path} 2>&1"
  set +e
  "$@" > "${log_path}" 2>&1
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
    run_cmd git clone "${REPO_URL}" "${REPO_DIR}"
  else
    cd "${REPO_DIR}"
    run_cmd git pull --ff-only
  fi
  cd "${REPO_DIR}"
}

ensure_r2_repo() {
  mkdir -p external
  if [ ! -d external/r2dreamer/.git ]; then
    run_cmd git clone "${R2_URL}" external/r2dreamer
  else
    run_cmd git -C external/r2dreamer fetch --all --prune
    if [ -z "${R2_REF}" ]; then
      local current_branch
      current_branch="$(git -C external/r2dreamer symbolic-ref --short -q HEAD || true)"
      if [ -z "${current_branch}" ]; then
        run_cmd git -C external/r2dreamer checkout main
      fi
      run_cmd git -C external/r2dreamer pull --ff-only
    fi
  fi
  if [ -n "${R2_REF}" ]; then
    run_cmd git -C external/r2dreamer checkout "${R2_REF}"
  fi
}

install_dependencies() {
  if command -v apt-get >/dev/null 2>&1; then
    run_cmd apt-get update
    run_cmd apt-get install -y libosmesa6-dev
  fi
  run_cmd python -m pip install --upgrade pip
  run_cmd python -m pip install -r requirements.txt
  run_cmd python -m pip install -r external/r2dreamer/requirements.txt
}

prepare_dirs() {
  mkdir -p \
    "${BASELINE_DIR}" \
    "${SHARP_DIR}" \
    "${ROLLOUT_OUTDIR}" \
    "${TABLES_DIR}" \
    "${LOG_DIR}"
}

verify_source() {
  python -m py_compile scripts/train_r2_sharp.py scripts/extract_r2_rollouts.py
}

verify_checkpoint() {
  local path="$1"
  python - "${path}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
module_path = pathlib.Path("scripts/train_r2_sharp.py").resolve()
spec = importlib.util.spec_from_file_location("train_r2_sharp_verify", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
ok = module.verify_checkpoint(path)
raise SystemExit(0 if ok else 1)
PY
}

train_baseline() {
  phase "R2 baseline training"
  local cmd=(
    python scripts/train_r2_sharp.py
    --run_name "${BASELINE_RUN_NAME}"
    --task "${TASK}"
    --total_steps "${STEPS}"
    --seed "${SEED}"
    --device "${DEVICE_TRAIN}"
    --envs "${ENVS}"
    --batch_size "${BATCH_SIZE}"
    --disable_sharp
  )
  if [ -n "${STORAGE_DEVICE}" ]; then
    cmd+=(--storage_device "${STORAGE_DEVICE}")
  fi
  run_nohup_and_wait \
    "R2 baseline training" \
    "${LOG_DIR}/baseline_train_${RUN_ID}.log" \
    "${cmd[@]}"
  verify_checkpoint "${BASELINE_CKPT}"
  mark_done "${CURRENT_PHASE}"
}

train_sharp() {
  phase "R2 SHARP v2 training"
  local cmd=(
    python scripts/train_r2_sharp.py
    --run_name "${SHARP_RUN_NAME}"
    --task "${TASK}"
    --total_steps "${STEPS}"
    --seed "${SEED}"
    --device "${DEVICE_TRAIN}"
    --envs "${ENVS}"
    --batch_size "${BATCH_SIZE}"
    --beta_mean 0.1
    --beta_var 0.01
  )
  if [ -n "${STORAGE_DEVICE}" ]; then
    cmd+=(--storage_device "${STORAGE_DEVICE}")
  fi
  run_nohup_and_wait \
    "R2 SHARP v2 training" \
    "${LOG_DIR}/sharp_v2_train_${RUN_ID}.log" \
    "${cmd[@]}"
  verify_checkpoint "${SHARP_CKPT}"
  mark_done "${CURRENT_PHASE}"
}

extract_rollouts_if_ready() {
  phase "R2 rollout extraction and comparison"
  if [ ! -s "${BASELINE_CKPT}" ] || [ ! -s "${SHARP_CKPT}" ]; then
    echo "Skipping extraction: both checkpoints are required."
    echo "Baseline: ${BASELINE_CKPT}"
    echo "SHARP v2: ${SHARP_CKPT}"
    mark_done "${CURRENT_PHASE}"
    return 0
  fi

  verify_checkpoint "${BASELINE_CKPT}"
  verify_checkpoint "${SHARP_CKPT}"
  run_logged \
    "R2 rollout extraction" \
    "${LOG_DIR}/extract_rollouts_${RUN_ID}.log" \
    python scripts/extract_r2_rollouts.py \
      --baseline_ckpt "${BASELINE_CKPT}" \
      --sharp_ckpt "${SHARP_CKPT}" \
      --task "${TASK}" \
      --device "${DEVICE_EXTRACT}" \
      --outdir "${ROLLOUT_OUTDIR}" \
      --num_seeds "${N_SEEDS}" \
      --imagination_start "${IMAGINATION_START}" \
      --horizon "${HORIZON}"

  cp -f "${ROLLOUT_OUTDIR}/r2_drift_comparison.json" "${COMPARISON_JSON}"
  test -s "${COMPARISON_JSON}"
  echo "Saved comparison: ${COMPARISON_JSON}"
  mark_done "${CURRENT_PHASE}"
}

list_outputs() {
  echo "R2 final outputs:"
  find "${R2_FINAL_ROOT}" "${TABLES_DIR}" -maxdepth 4 \( -type f -o -type l \) 2>/dev/null | sort || true
}

phase "Environment setup"
ensure_repo
ensure_r2_repo
install_dependencies
prepare_dirs
verify_source
mark_done "${CURRENT_PHASE}"

case "${MODE}" in
  baseline)
    train_baseline
    ;;
  sharp)
    train_sharp
    ;;
  both)
    train_baseline
    train_sharp
    ;;
  *)
    echo "Invalid MODE='${MODE}'. Expected one of: baseline, sharp, both."
    exit 2
    ;;
esac

extract_rollouts_if_ready
list_outputs
echo "=== R2 restart workflow complete at $(date) ==="
