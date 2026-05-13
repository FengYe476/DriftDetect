# V3 Walker Multi-Seed SHARP Plan

This follows the shared same-pod methodology in `docs/v3_multiseed_plan.md`.
Each training seed should run a baseline and SHARP pair sequentially on one
RunPod pod, then extract `20` rollout seeds at horizon `200`.

## Why Walker

Walker (`dmc_walker_walk`) adds a third standard DMC proprioceptive task to the
V3 SHARP validation suite. It is larger than Cheetah and Cartpole in both
observation and action dimensionality (`obs_dim=24`, `act_dim=6`), so it tests
whether SHARP's mean/variance hardening remains effective beyond the two Month
8 tasks.

## Configuration

`configs/month8_moment_match_walker.yaml` is copied from the Cheetah Month 8
reference config with only `task` changed from `dmc_cheetah_run` to
`dmc_walker_walk`. The intended settings remain:

- `total_steps=500000`
- `batch_size=32`
- `envs=8`
- `beta_mean=1.0`
- `beta_var=0.1`
- `n_starts=8`
- `warmup_env_steps=50000`

Planned training seeds are `42`, `43`, and `44`.

The unified launcher does not yet include a Walker case. Once Walker support is
added there, the intended launch pattern is:

```bash
bash scripts/runpod_v3_multiseed.sh 42 walker
bash scripts/runpod_v3_multiseed.sh 43 walker
bash scripts/runpod_v3_multiseed.sh 44 walker
```

## Success Criteria

- Both baseline and SHARP checkpoints are saved and loadable for each completed
  Walker training seed.
- Rollouts are extracted for `20` rollout seeds per checkpoint.
- Raw deterministic latent `J_total` is computed over the half-open trim window
  `[25, 175)`.
- SHARP produces a clear mean `J_total` reduction across completed Walker seeds
  without a large degradation in eval return.
