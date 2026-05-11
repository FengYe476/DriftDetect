# V3 Cartpole Multi-Seed SHARP Plan

This follows the shared same-pod methodology in `docs/v3_multiseed_plan.md`.
Each training seed runs a baseline and SHARP pair sequentially on one RunPod
pod, then extracts `20` rollout seeds at horizon `200`.

## Why Cartpole

Cartpole is the strongest cross-task check for V3 SHARP. Month 8 reduced
Cartpole latent `J_total` by `-97.2%`, stronger than the Cheetah result
(`-89.6%`), while maintaining healthy eval performance around `828`.

Cartpole also has a smaller observation space (`5` dims versus Cheetah's `17`)
and a more concentrated drift spectrum: `dc_trend` accounts for about `97%` of
Cartpole V3 drift versus `46.7%` for Cheetah. Multi-seed replication should
test whether this near-total slow-mode suppression is stable across training
seeds.

## Run Plan

Planned training seeds are `43`, `44`, and `45`.

```bash
bash scripts/runpod_v3_multiseed.sh 43 cartpole
bash scripts/runpod_v3_multiseed.sh 44 cartpole
bash scripts/runpod_v3_multiseed.sh 45 cartpole
```

The Cartpole wrapper uses `configs/month8_moment_match_cartpole.yaml` as the
base config. A direct comparison against the Cheetah Month 8 config shows that
the only reference-config difference is `task`; `batch_size=32`, `envs=8`,
`beta_mean=1.0`, `beta_var=0.1`, `n_starts=8`, and warmup `50000` are
identical.

Evaluate completed seeds with:

```bash
python scripts/evaluate_v3_multiseed.py --task dmc_cartpole_swingup
```

If using the Month 8 seed `42` Cartpole reference, ensure the SHARP rollout
files under `results/month8_controlled/rollouts_moment_match_cartpole/` use the
clean `cartpole_v3_seed*_v2.npz` naming before evaluation.

## Success Criteria

- Both baseline and SHARP checkpoints are saved and loadable for each completed
  Cartpole training seed.
- Rollouts are extracted for `20` rollout seeds per checkpoint.
- Mean raw deterministic latent `J_total` reduction is greater than `90%`
  across completed seeds.
- SHARP eval return remains healthy and does not show a systematic degradation
  relative to the same-pod baseline.
