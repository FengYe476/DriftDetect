# V4 Mismatch Check

## Setup

Run the V4 Cheetah posterior/drift mismatch check from the repository root:

```bash
python scripts/run_v4_mismatch_check.py
```

The script glob-loads all `.npz` files in `results/rollouts_v4/`, requires at least 20 rollout files, and writes `results/tables/v4_mismatch_check.json`.

## V4 Latent Notes

V4 latents are 512-dimensional packed tokenizer outputs flattened from `(8, 64)`. They are not V3 RSSM deter states, so the script applies no `latent[:, 1024:]` slicing.

V4 uses a multi-task checkpoint, so `U_posterior_v4` reflects multi-task learned visual features rather than a task-specific recurrent GRU state.

## Results (to be filled)

| r | Posterior/Drift | SFA/Posterior | SFA/Drift |
|---|---:|---:|---:|
| 3 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD |

## Cross-Architecture Comparison

| Setting | r=3 | r=5 | r=10 |
|---|---:|---:|---:|
| V3 Cheetah | TBD | TBD | TBD |
| V3 Cartpole | TBD | TBD | TBD |
| V4 Cheetah | TBD | TBD | TBD |

## Interpretation

Use the V4 posterior/drift overlap to classify the result:

- `< 0.4`: supports a cross-architecture posterior-drift mismatch.
- `> 0.6`: suggests the mismatch may be V3/RSSM-specific.
- `0.4-0.6`: mixed or ambiguous; inspect ranks, SFA overlaps, and PCA diagnostics.
