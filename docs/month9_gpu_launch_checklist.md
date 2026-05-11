# Month 9 GPU Launch Checklist

This checklist launches all Month 9 training runs on RunPod. All commands assume
the repo is available at `/workspace/DriftDetect` unless `REPO_DIR` is set.

## Preflight On Every Pod

```bash
cd /workspace
git clone https://github.com/FengYe476/DriftDetect.git || true
cd /workspace/DriftDetect
git pull --ff-only

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Phase 1: R2-Dreamer Restart

Priority: HIGH. Estimated runtime: about `12` GPU-hours.

Pod 1:

```bash
cd /workspace/DriftDetect
MODE=both bash scripts/runpod_r2_restart.sh
```

Expected runtime is about `6h` baseline plus `6h` SHARP v2. R2 risk is high:
Month 8 eval swung from `170` to `748`, so monitor logs for instability.

Verify:

```bash
cd /workspace/DriftDetect
test -s results/r2_final/baseline/latest.pt
test -s results/r2_final/sharp_v2/latest.pt
test -s results/tables/r2_comparison.json
ls -lh results/r2_final/baseline/latest.pt results/r2_final/sharp_v2/latest.pt
find results/r2_final -path '*rollouts*' -name '*.npz' | wc -l
python -m json.tool results/tables/r2_comparison.json | head -80
```

Success criteria:

- Baseline and SHARP v2 checkpoints exist and were verified by the launcher.
- Rollouts are extracted for both checkpoints.
- `results/tables/r2_comparison.json` exists and reports `J_total`.

## Phase 2: V3 Cheetah Multi-Seed SHARP

Priority: CRITICAL. Estimated runtime: about `60` GPU-hours.

Each pod runs same-seed baseline then SHARP sequentially, about `10h` per run
and `20h` per pod.

Pod 2:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 43 cheetah
```

Pod 3:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 44 cheetah
```

Pod 4:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 45 cheetah
```

Verify after each pod:

```bash
cd /workspace/DriftDetect
SEED=43
test -s results/v3_multiseed/cheetah_baseline_seed${SEED}/latest.pt
test -s results/v3_multiseed/cheetah_sharp_seed${SEED}/latest.pt
find results/v3_multiseed/cheetah_baseline_seed${SEED}/rollouts -name 'cheetah_v3_seed*_v2.npz' | wc -l
find results/v3_multiseed/cheetah_sharp_seed${SEED}/rollouts -name 'cheetah_v3_seed*_v2.npz' | wc -l
```

After all three pods complete:

```bash
cd /workspace/DriftDetect
python scripts/evaluate_v3_multiseed.py --task dmc_cheetah_run
python -m json.tool results/tables/v3_cheetah_multiseed_results.json | head -120
```

Success criteria:

- Three checkpoint pairs exist for seeds `43`, `44`, and `45`.
- Each checkpoint has `20` rollout files.
- Mean raw deterministic latent `J_total` reduction is greater than `80%`.
- Reduction std is less than `10%` of the mean.
- SHARP eval return is within `5%` of baseline mean.

## Phase 3: V3 Cartpole Multi-Seed SHARP

Priority: HIGH. Estimated runtime: about `60` GPU-hours.

Pod 5:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 43 cartpole
```

Pod 6:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 44 cartpole
```

Pod 7:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_v3_multiseed.sh 45 cartpole
```

Verify after each pod:

```bash
cd /workspace/DriftDetect
SEED=43
test -s results/v3_multiseed/cartpole_baseline_seed${SEED}/latest.pt
test -s results/v3_multiseed/cartpole_sharp_seed${SEED}/latest.pt
find results/v3_multiseed/cartpole_baseline_seed${SEED}/rollouts -name 'cartpole_v3_seed*_v2.npz' | wc -l
find results/v3_multiseed/cartpole_sharp_seed${SEED}/rollouts -name 'cartpole_v3_seed*_v2.npz' | wc -l
```

After all three pods complete:

```bash
cd /workspace/DriftDetect
python scripts/evaluate_v3_multiseed.py --task dmc_cartpole_swingup
python -m json.tool results/tables/v3_cartpole_multiseed_results.json | head -120
```

Success criteria:

- Three checkpoint pairs exist for seeds `43`, `44`, and `45`.
- Each checkpoint has `20` rollout files.
- Mean raw deterministic latent `J_total` reduction is greater than `90%`.
- Eval return remains healthy relative to same-pod baselines.

## Phase 4: Jacobian Regularization Baseline

Priority: HIGH. Estimated runtime: about `30` GPU-hours.

No separate baseline is needed; reuse V3 multiseed baselines.

Pod 8:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_jacobian_reg.sh 42 cheetah
```

Pod 9:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_jacobian_reg.sh 43 cheetah
```

Pod 10:

```bash
cd /workspace/DriftDetect
bash scripts/runpod_jacobian_reg.sh 44 cheetah
```

Verify after each pod:

```bash
cd /workspace/DriftDetect
SEED=42
test -s results/jacobian_reg/cheetah_jacobian_reg_seed${SEED}/latest.pt
find results/jacobian_reg/cheetah_jacobian_reg_seed${SEED}/rollouts -name 'cheetah_v3_seed*_v2.npz' | wc -l
ls -lh results/jacobian_reg/cheetah_jacobian_reg_seed${SEED}/latest.pt
```

Success criteria:

- Three Jacobian-reg checkpoints exist for seeds `42`, `43`, and `44`.
- Each checkpoint has `20` rollout files.
- Training logs contain active `jacobian_reg_*` metrics after warmup.
- Drift is compared against matched V3 baseline seeds.

## Phase 5: Optional Extensions

Launch only if budget allows:

- Walker SHARP 3-seed: about `40` GPU-hours.
- V4 from-scratch training: about `150` GPU-hours.
- TD-MPC2 reconstruction plus SHARP pilot.

## Parallelism Plan

Total required for Phases 1-4 is about `162` GPU-hours.

Maximum parallel plan:

- Start Pods `1-4` immediately: R2 plus Cheetah seeds.
- Start Pods `5-7` when budget allows: Cartpole seeds.
- Start Pods `8-10` after at least Cheetah seed `43` baseline exists, or launch
  immediately if enough GPUs are available.

Conservative plan:

- Day 1: Pods `1-4`.
- Day 2: Pods `5-7`.
- Day 3: Pods `8-10`.

## Monitoring Commands

Use these on each pod:

```bash
cd /workspace/DriftDetect
nvidia-smi
find results -path '*logs*' -name '*.log' | sort
tail -f results/v3_multiseed/logs/cheetah_seed43/*.log
tail -f results/jacobian_reg/logs/cheetah_seed42/*.log
tail -f results/r2_final/logs/*.log
```

If a process exits early:

```bash
cd /workspace/DriftDetect
find results -path '*logs*' -name '*.log' -print -exec tail -80 {} \;
find results -name latest.pt -print -exec ls -lh {} \;
```

## Data Download Instructions

Set local variables for each RunPod instance:

```bash
export RUNPOD_HOST=<runpod-host>
export RUNPOD_PORT=<ssh-port>
export RUNPOD_USER=root
export REMOTE_REPO=/workspace/DriftDetect
export LOCAL_REPO="/Users/fengye/Desktop/Project/DriftDetect 2/DriftDetect/DriftDetect"
```

Download R2 outputs:

```bash
rsync -avP -e "ssh -p ${RUNPOD_PORT}" \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_REPO}/results/r2_final/ \
  "${LOCAL_REPO}/results/r2_final/"

rsync -avP -e "ssh -p ${RUNPOD_PORT}" \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_REPO}/results/tables/r2_comparison.json \
  "${LOCAL_REPO}/results/tables/"
```

Download V3 multiseed outputs from each V3 pod:

```bash
rsync -avP -e "ssh -p ${RUNPOD_PORT}" \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_REPO}/results/v3_multiseed/ \
  "${LOCAL_REPO}/results/v3_multiseed/"

rsync -avP -e "ssh -p ${RUNPOD_PORT}" \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_REPO}/results/tables/ \
  "${LOCAL_REPO}/results/tables/"
```

Download Jacobian-reg outputs:

```bash
rsync -avP -e "ssh -p ${RUNPOD_PORT}" \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_REPO}/results/jacobian_reg/ \
  "${LOCAL_REPO}/results/jacobian_reg/"
```

After downloads, run local aggregation:

```bash
cd "${LOCAL_REPO}"
python scripts/evaluate_v3_multiseed.py --task dmc_cheetah_run
python scripts/evaluate_v3_multiseed.py --task dmc_cartpole_swingup
python -m json.tool results/tables/v3_cheetah_multiseed_results.json | head -120
python -m json.tool results/tables/v3_cartpole_multiseed_results.json | head -120
```

## Post-Training Checklist

- Confirm every expected checkpoint is non-empty and loadable.
- Confirm every completed checkpoint has `20` rollout `.npz` files and a manifest.
- Archive all timestamped logs.
- Save evaluator JSON outputs under `results/tables/`.
- Record any pod failure, restart, or eval instability in the Month 9 log.
- Update paper figures with multi-seed error bars before writing final claims.
