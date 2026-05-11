#!/usr/bin/env bash
set -Eeuo pipefail

# RunPod launcher for V3 Jacobian-regularization baselines.
#
# Usage:
#   bash scripts/runpod_jacobian_reg.sh 43
#   bash scripts/runpod_jacobian_reg.sh 43 cheetah
#   bash scripts/runpod_jacobian_reg.sh 43 cartpole

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash scripts/runpod_jacobian_reg.sh SEED [cheetah|cartpole]"
  exit 2
fi

SEED="$1"
TASK_CHOICE="${2:-cheetah}"
if ! [[ "${SEED}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer, got '${SEED}'."
  exit 2
fi

case "${TASK_CHOICE}" in
  cheetah)
    TASK_SHORT="cheetah"
    DMC_TASK="dmc_cheetah_run"
    BASE_CONFIG_PATH="configs/month8_moment_match.yaml"
    ;;
  cartpole)
    TASK_SHORT="cartpole"
    DMC_TASK="dmc_cartpole_swingup"
    BASE_CONFIG_PATH="configs/month8_moment_match_cartpole.yaml"
    ;;
  *)
    echo "TASK must be 'cheetah' or 'cartpole', got '${TASK_CHOICE}'."
    exit 2
    ;;
esac

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
DEFAULT_REPO_DIR="${WORKSPACE}/DriftDetect"
if [ -d "${SCRIPT_REPO_ROOT}/.git" ]; then
  DEFAULT_REPO_DIR="${SCRIPT_REPO_ROOT}"
fi

REPO_URL="${REPO_URL:-https://github.com/FengYe476/DriftDetect.git}"
DREAMERV3_URL="${DREAMERV3_URL:-https://github.com/NM512/dreamerv3-torch.git}"
DREAMERV3_REF="${DREAMERV3_REF:-6ef8646d807cd10ce0c88e10a7e943211e7fc44c}"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"

LAMBDA_JAC="${LAMBDA_JAC:-0.01}"
N_JAC_STARTS="${N_JAC_STARTS:-8}"
N_JAC_PROJECTIONS="${N_JAC_PROJECTIONS:-1}"
WARMUP_ENV_STEPS="${WARMUP_ENV_STEPS:-50000}"
N_ROLLOUT_SEEDS="${N_ROLLOUT_SEEDS:-20}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-1000}"
IMAGINATION_START="${IMAGINATION_START:-200}"
HORIZON="${HORIZON:-200}"
DEVICE_TRAIN="${DEVICE_TRAIN:-cuda:0}"
DEVICE_EXTRACT="${DEVICE_EXTRACT:-cuda:0}"

ROOT="results/jacobian_reg"
RUN_NAME="jacobian_reg/${TASK_SHORT}_jacobian_reg_seed${SEED}"
RUN_DIR="results/${RUN_NAME}"
CKPT="${RUN_DIR}/latest.pt"
CONFIG_DIR="${ROOT}/configs/${TASK_SHORT}_seed${SEED}"
CONFIG_PATH="${CONFIG_DIR}/jacobian_reg.yaml"
LOG_DIR="${ROOT}/logs/${TASK_SHORT}_seed${SEED}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

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
    echo "Last 120 log lines from ${log_path}:"
    tail -120 "${log_path}" || true
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
    echo "Last 120 log lines from ${log_path}:"
    tail -120 "${log_path}" || true
    return "${status}"
  fi
}

ensure_repo() {
  mkdir -p "$(dirname "${REPO_DIR}")"
  if [ ! -d "${REPO_DIR}/.git" ]; then
    run_cmd git clone "${REPO_URL}" "${REPO_DIR}"
  else
    cd "${REPO_DIR}"
    run_cmd git pull --ff-only
  fi
  cd "${REPO_DIR}"
}

ensure_dreamerv3_repo() {
  mkdir -p external
  if [ ! -d external/dreamerv3-torch/.git ]; then
    run_cmd git clone "${DREAMERV3_URL}" external/dreamerv3-torch
  else
    run_cmd git -C external/dreamerv3-torch fetch --all --prune
  fi
  run_cmd git -C external/dreamerv3-torch checkout "${DREAMERV3_REF}"
}

patch_dmc_size() {
  python - <<'PY'
from pathlib import Path

path = Path("external/dreamerv3-torch/envs/dmc.py")
text = path.read_text()
old = "        self._size = size\n"
new = "        self._size = tuple(size)\n"
if new in text:
    print("dmc.py tuple(size) patch already present.")
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print("Applied dmc.py tuple(size) patch.")
else:
    raise SystemExit("Could not find dmc.py self._size assignment to patch.")
PY
}

install_dependencies() {
  if command -v apt-get >/dev/null 2>&1; then
    run_cmd apt-get update
    run_cmd apt-get install -y libosmesa6-dev
  fi
  run_cmd python -m pip install --upgrade pip
  run_cmd python -m pip install -r requirements.txt
  run_cmd python -m pip install dm-control mujoco ruamel.yaml scipy matplotlib moviepy einops protobuf==3.20.0 gym==0.26.2 opencv-python tensorboard
}

verify_runtime() {
  python - <<'PY'
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not sys.version.startswith("3.12."):
    raise SystemExit("Expected Python 3.12 on the RunPod image.")
if not torch.__version__.startswith("2.8."):
    raise SystemExit("Expected PyTorch 2.8.x on the RunPod image.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Jacobian-reg training.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
}

prepare_dirs() {
  mkdir -p "${RUN_DIR}" "${CONFIG_DIR}" "${LOG_DIR}" results/tables
}

generate_config() {
  python - \
    "${SEED}" \
    "${BASE_CONFIG_PATH}" \
    "${CONFIG_PATH}" \
    "${LAMBDA_JAC}" \
    "${N_JAC_STARTS}" \
    "${N_JAC_PROJECTIONS}" \
    "${WARMUP_ENV_STEPS}" <<'PY'
import sys
from pathlib import Path

from ruamel.yaml import YAML

seed = int(sys.argv[1])
base_config = Path(sys.argv[2])
output_path = Path(sys.argv[3])
lambda_jac = float(sys.argv[4])
n_starts = int(sys.argv[5])
n_projections = int(sys.argv[6])
warmup_env_steps = int(sys.argv[7])

yaml = YAML()
yaml.default_flow_style = False
config = yaml.load(base_config.read_text())
config["seed"] = seed
config.pop("moment_match", None)
config["jacobian_reg"] = {
    "lambda_jac": lambda_jac,
    "n_starts": n_starts,
    "n_projections": n_projections,
    "warmup_env_steps": warmup_env_steps,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as handle:
    yaml.dump(config, handle)
PY
}

print_config() {
  echo ""
  echo "Task: ${TASK_SHORT} (${DMC_TASK})"
  echo "Base config: ${BASE_CONFIG_PATH}"
  echo "Jacobian config: ${CONFIG_PATH}"
  echo "Run dir: ${RUN_DIR}"
  echo "Generated config:"
  sed -n '1,120p' "${CONFIG_PATH}"
}

verify_checkpoint() {
  local path="$1"
  python - "${path}" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
try:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("checkpoint file is missing or empty")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    for key in ("agent_state_dict", "optims_state_dict"):
        if key not in checkpoint:
            raise ValueError(f"missing required key: {key}")
    state = checkpoint["agent_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("agent_state_dict must be a non-empty dict")
    optims = checkpoint["optims_state_dict"]
    if not isinstance(optims, dict) or not optims:
        raise ValueError("optims_state_dict must be a non-empty dict")
    print(
        f"CHECKPOINT VERIFY PASS: {path} ({size_mb:.1f} MB), "
        f"agent_state_dict: {len(state)} keys, optims_state_dict: {len(optims)} keys",
        flush=True,
    )
except Exception as exc:
    print(f"CHECKPOINT VERIFY FAIL: {path} ({size_mb:.1f} MB): {exc}", flush=True)
    raise SystemExit(1)
PY
}

train_jacobian_reg() {
  phase "${TASK_SHORT} Jacobian-reg training seed ${SEED}"
  run_nohup_and_wait \
    "V3 ${TASK_SHORT} Jacobian-reg seed ${SEED}" \
    "${LOG_DIR}/train_${RUN_ID}.log" \
    python scripts/train_jacobian_reg.py \
      --config "${CONFIG_PATH}" \
      --run_name "${RUN_NAME}" \
      --device "${DEVICE_TRAIN}" \
      --lambda_jac "${LAMBDA_JAC}" \
      --n_jac_starts "${N_JAC_STARTS}" \
      --n_jac_projections "${N_JAC_PROJECTIONS}" \
      --warmup_env_steps "${WARMUP_ENV_STEPS}"
  verify_checkpoint "${CKPT}"
  mark_done "${CURRENT_PHASE}"
}

extract_rollouts() {
  phase "${TASK_SHORT} Jacobian-reg rollout extraction seed ${SEED}"
  run_logged \
    "${TASK_SHORT} Jacobian-reg rollout extraction seed ${SEED}" \
    "${LOG_DIR}/extract_${RUN_ID}.log" \
    python scripts/batch_extract_v3_multiseed.py \
      --checkpoint "${CKPT}" \
      --output_dir "${RUN_DIR}/rollouts" \
      --task "${DMC_TASK}" \
      --num_seeds "${N_ROLLOUT_SEEDS}" \
      --total_steps "${TOTAL_ROLLOUT_STEPS}" \
      --imagination_start "${IMAGINATION_START}" \
      --horizon "${HORIZON}" \
      --device "${DEVICE_EXTRACT}" \
      --manifest_name "manifest_${TASK_SHORT}_jacobian_reg_seed${SEED}.json"
  mark_done "${CURRENT_PHASE}"
}

list_outputs() {
  echo "Jacobian-reg outputs for ${TASK_SHORT} seed ${SEED}:"
  find "${ROOT}" -maxdepth 4 \( -type f -o -type l \) 2>/dev/null | sort || true
}

phase "Environment setup"
ensure_repo
ensure_dreamerv3_repo
patch_dmc_size
verify_runtime
install_dependencies
verify_runtime
prepare_dirs
generate_config
print_config
mark_done "${CURRENT_PHASE}"

train_jacobian_reg
extract_rollouts
list_outputs
echo "=== V3 ${TASK_SHORT} Jacobian-reg seed ${SEED} complete at $(date) ==="
