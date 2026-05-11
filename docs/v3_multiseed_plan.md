# V3 Cheetah Multi-Seed SHARP Plan

## Why This Is Needed

The current DreamerV3 Cheetah SHARP result is strong but single-seed: seed `42`
reduced latent `J_total` from `0.173` to `0.018`, a `-89.6%` change, while
peak `eval_return` changed by only `-2.1%`. Multi-seed replication is the
highest-priority reviewer risk because NeurIPS reviewers will reasonably ask
whether this effect is stable across training seeds.

The Month 9 target is to replicate the controlled Cheetah comparison on seeds
`43`, `44`, and `45`, while keeping seed `42` as the Month 8 reference point
when the final evaluation table is generated.

## Controlled Methodology

Each training seed gets one RunPod pod. On that pod, run the baseline first and
SHARP second with the same hardware, software environment, dependency versions,
DreamerV3 commit, and generated config template. The generated configs differ
only in `moment_match.beta_mean` and `moment_match.beta_var`.

This same-pod pairing avoids the Month 7 cross-pod confound. A baseline/SHARP
pair should be interpreted as a controlled comparison for that seed, not as two
independent cloud runs.

The unified wrapper script is:

```bash
bash scripts/runpod_v3_multiseed.sh 43 cheetah
bash scripts/runpod_v3_multiseed.sh 44 cheetah
bash scripts/runpod_v3_multiseed.sh 45 cheetah
```

Launch the three commands on three separate pods in parallel. The script accepts
any positive integer seed, so additional seeds such as `46` or `47` can be run
without modifying the script.

For backward compatibility, `scripts/runpod_v3_cheetah_multiseed.sh` delegates
to the unified launcher with `TASK=cheetah`.

## Runtime And Outputs

Expected runtime is approximately `10` GPU-hours per training run:

- Baseline: `10` GPU-hours per seed.
- SHARP: `10` GPU-hours per seed.
- Three seed pairs: approximately `60` GPU-hours total.

Primary outputs:

- `results/v3_multiseed/cheetah_baseline_seed{SEED}/latest.pt`
- `results/v3_multiseed/cheetah_sharp_seed{SEED}/latest.pt`
- `results/v3_multiseed/cheetah_baseline_seed{SEED}/rollouts/`
- `results/v3_multiseed/cheetah_sharp_seed{SEED}/rollouts/`
- `results/v3_multiseed/logs/seed{SEED}/`
- `results/tables/v3_cheetah_multiseed_results.json`

Evaluate completed seeds with:

```bash
python scripts/evaluate_v3_multiseed.py
```

By default, evaluation also tries to include the Month 8 seed `42` reference
from `results/month8_controlled/rollouts_baseline/` and
`results/month8_controlled/rollouts_moment_match/`. Missing reference files are
reported explicitly rather than treated as fatal.

## Success Criteria

- Both baseline and SHARP checkpoints are saved and loadable for each completed
  training seed.
- Rollouts are extracted for `20` rollout seeds per checkpoint with horizon
  `200`.
- Mean raw deterministic latent `J_total` reduction is greater than `80%`
  across completed seeds.
- Standard deviation of `J_total` reduction is less than `10%` of the mean
  reduction.
- SHARP peak `eval_return` is within `5%` of the baseline mean across completed
  seeds.

The primary reviewer metric is raw deterministic latent `J_total` over the
half-open trim window `[25, 175)`, matching the Month 8 `25:175` convention.
Frequency-band metrics are reported as secondary diagnostics.
