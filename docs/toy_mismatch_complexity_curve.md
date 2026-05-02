# Toy Mismatch Complexity Curve

## Setup

Run the multi-seed toy mismatch sweep from the repository root:

```bash
python scripts/run_toy_mismatch_sweep.py
```

The default configuration trains the fixed-capacity minimal RSSM for 2000 steps
on each complexity level and seed:

| Complexity | Oscillators | obs_dim | Seeds |
|---|---:|---:|---|
| simple | 2 | 4 | 0, 1, 2, 3, 4 |
| medium | 5 | 10 | 0, 1, 2, 3, 4 |
| hard | 10 | 20 | 0, 1, 2, 3, 4 |

The script saves plot-ready results to
`results/tables/toy_mismatch_complexity.json`, including full per-seed overlap
values for Figure 5 scatter plots.

## Results

Full sweep completed with 5 seeds per complexity level. Values are mean +/- std.

| Complexity | obs_dim | r=3 post/drift | r=5 post/drift | r=10 post/drift | dc_trend% |
|---|---:|---:|---:|---:|---:|
| simple | 4 | 0.613 +/- 0.160 | 0.697 +/- 0.113 | 0.797 +/- 0.012 | 19.1% +/- 4.0% |
| medium | 10 | 0.412 +/- 0.143 | 0.672 +/- 0.138 | 0.846 +/- 0.045 | 43.1% +/- 3.4% |
| hard | 20 | 0.378 +/- 0.042 | 0.496 +/- 0.103 | 0.743 +/- 0.183 | 38.3% +/- 14.0% |

The JSON contains 15 per-seed results and 45 flat `plot_points` rows
(3 ranks x 5 seeds x 3 complexity levels).

## Hypothesis Test

The W2 hypothesis is that posterior/drift overlap decreases as environment
complexity increases. The script reports descriptive Pearson correlations
between `obs_dim` and posterior/drift overlap for ranks 3, 5, and 10.

The alternate inverted-U mismatch pattern is:

- simple: higher posterior/drift overlap
- medium: lowest posterior/drift overlap
- hard: overlap recovers because model error becomes more broadband

These tests are descriptive only because there are three complexity levels.

Observed correlations:

| Rank | Pearson(obs_dim, post/drift) | Monotonic decrease? | Medium lowest? |
|---:|---:|---|---|
| 3 | -0.861 | yes | no |
| 5 | -0.965 | yes | no |
| 10 | -0.638 | no | no |

Verdict: mixed partial support. Ranks 3 and 5 support the hypothesis that
posterior/drift overlap falls as complexity increases, while rank 10 remains
high for the medium setting and does not decrease monotonically.

## Connection to Real-Model Mismatch

The real-model mismatch results motivate the toy sweep:

| Setting | posterior/drift overlap |
|---|---|
| V3 Cheetah | 0.06-0.28 |
| V3 Cartpole | 0.06-0.12 |
| V4 Cheetah | 0.41-0.64 |
| simple toy, Month 5 seed 0 | 0.77-0.88 |
| simple toy, 5-seed sweep | 0.61-0.80 |
| medium toy, 5-seed sweep | 0.41-0.85 |
| hard toy, 5-seed sweep | 0.38-0.74 |

If the toy overlap falls as complexity rises, this links the mismatch finding to
the capacity-complexity story: more complex environments produce less alignment
between posterior slow features and the empirical drift subspace.

The current sweep gives partial support: low-rank overlap decreases strongly
from simple to hard, but rank-10 overlap stays comparatively high and variable.

## Figure 5 Implications

Use `plot_points` in `results/tables/toy_mismatch_complexity.json` to create a
scatter plot with:

- x-axis: `obs_dim` or complexity level
- y-axis: posterior/drift overlap
- color or facet: rank
- point identity: seed

A useful companion panel is `dc_trend_pct` versus posterior/drift overlap. This
would show whether the same complexity regime that creates trend-dominated drift
also creates posterior/drift mismatch.
