# R2-Dreamer Restart Plan

## Why This Restart Is Needed

The Month 8 R2-Dreamer baseline and SHARP v2 runs must be restarted from
scratch because the completed runs did not produce usable final checkpoints.
The immediate bug was in `scripts/train_r2_sharp.py`: `save_checkpoint()` used
`torch.save()` without importing `torch` at function scope. The final restart
attempts also failed when the RunPod was closed, so the baseline plus SHARP v2
comparison remains pending.

The restart keeps the R2-Dreamer training setup unchanged and adds stricter
checkpoint verification so a failed or malformed save is detected immediately.

## What Was Fixed

- `save_checkpoint()` now imports `torch` inside the function body.
- Saved checkpoints now include the model state dict, optimizer state dict,
  SHARP config, and a positive integer `step` count.
- `verify_checkpoint(path)` loads the saved checkpoint on CPU, checks required
  keys, verifies that `agent_state_dict` is a non-empty dict, verifies that
  `step` is a positive integer, and prints checkpoint size plus the number of
  model state keys.
- `verify_checkpoint()` is called immediately after every checkpoint save.
- `scripts/runpod_r2_restart.sh` supports independent restart modes through
  `MODE=baseline`, `MODE=sharp`, or `MODE=both`.

## Month 8 R2 Observations

- R2-Dreamer training is highly unstable, with eval swings from `170` to `748`
  across runs.
- R2 baseline drift (`J_total ~= 0.036`) is approximately `5x` lower than the
  V3 baseline (`0.173`).
- SHARP v1 (`beta_mean=1.0`, `beta_var=0.1`) was too aggressive for R2: both
  eval and drift worsened.
- SHARP v2 (`beta_mean=0.1`, `beta_var=0.01`) maintained eval; drift results
  are pending checkpoint recovery and rollout extraction.

## Run Plan

Use `scripts/runpod_r2_restart.sh` on RunPod. The default `MODE=both` runs the
baseline first and SHARP v2 second, but each run can be launched independently:

```bash
MODE=baseline bash scripts/runpod_r2_restart.sh
MODE=sharp bash scripts/runpod_r2_restart.sh
MODE=both bash scripts/runpod_r2_restart.sh
```

The baseline run uses `dmc_cheetah_run`, `500k` steps, `seed=42`,
`batch_size=16`, `envs=16`, and `--disable_sharp`. The checkpoint is written to
`results/r2_final/baseline/latest.pt`.

The SHARP v2 run uses the same task and training settings with
`beta_mean=0.1` and `beta_var=0.01`. The checkpoint is written to
`results/r2_final/sharp_v2/latest.pt`.

After both checkpoints exist and verify, the script runs
`scripts/extract_r2_rollouts.py` with `20` seeds and horizon `200`, then copies
the drift comparison to `results/tables/r2_comparison.json`.

Expected runtime on an RTX 4090 RunPod is approximately `6` hours per run, or
approximately `12` hours total for baseline plus SHARP v2. Rollout extraction
is short relative to training.

## Success Criteria

- `results/r2_final/baseline/latest.pt` exists and passes checkpoint
  verification.
- `results/r2_final/sharp_v2/latest.pt` exists and passes checkpoint
  verification.
- R2 baseline and SHARP v2 rollout files are extracted for `20` seeds.
- `results/tables/r2_comparison.json` exists and contains the `J_total`
  comparison.
- Training and extraction logs are present under `results/r2_final/logs/`.
