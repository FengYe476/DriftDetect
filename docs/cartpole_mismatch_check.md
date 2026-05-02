# Cartpole Mismatch Check

## Setup

Run the Cartpole posterior/drift mismatch check from the repository root:

```bash
python scripts/run_cartpole_mismatch_check.py
```

The script prefers Cartpole v2 rollout files in `results/rollouts_cartpole/` and falls back to the local rollout convention `results/rollouts/cartpole_v3_seed*_v2.npz` if that directory is absent. It writes `results/tables/cartpole_mismatch_check.json`.

## Results (to be filled)

| r | Posterior/Drift | SFA/Posterior | SFA/Drift |
|---|---:|---:|---:|
| 3 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD |

## Interpretation

Use the posterior/drift overlap to classify the Cartpole result:

- `< 0.4`: supports a cross-task posterior-drift mismatch.
- `> 0.6`: suggests the mismatch may be task-specific to Cheetah.
- `0.4-0.6`: mixed or ambiguous; inspect ranks and SFA comparison.

## Comparison with Cheetah

Reference Cheetah posterior/drift overlaps:

| r | Cheetah Posterior/Drift |
|---|---:|
| 3 | 0.13 |
| 5 | 0.17 |
| 10 | 0.28 |
