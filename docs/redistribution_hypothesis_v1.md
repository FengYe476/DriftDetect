# Redistribution Hypothesis v1

## Setup

Run the toy redistribution sweep from the repository root:

```bash
conda run -n env-dreamerv3 python scripts/run_toy_redistribution_sweep.py --parallel
```

The default sweep trains 15 baseline models:

- complexity: simple, medium, hard
- seeds: 0, 1, 2, 3, 4
- train steps: 2000
- RSSM: `deter_dim=64`, `stoch_dim=8`

For each trained model, the script collects one shared baseline rollout set and
then evaluates 12 SMAD configurations:

- rank: 3, 5, 10
- eta: 0.1, 0.2, 0.3, 0.5

The signed band change is:

```text
(damped_band_mse - baseline_band_mse) / baseline_band_mse * 100
```

Negative values mean SMAD reduced that band. Positive values mean SMAD increased
that band.

## Results Summary

To be filled after the full sweep.

Expected output:

- `results/tables/toy_redistribution_sweep.parquet` when parquet support is available
- `results/tables/toy_redistribution_sweep.json` as the fallback output

## Conservation Test

The main conservation diagnostic is `total_mse_change_pct`. If the total MSE is
approximately conserved while band-level changes have opposite signs, the result
supports a redistribution interpretation.

The script also reports `sum_band_change_pct`, the unweighted sum of signed
per-band percentage changes. This quantity is useful as a rough directional
check, but `total_mse_change_pct` is the physically meaningful conservation
measurement because it respects the baseline magnitude of each band.

Default conservation criterion:

```text
abs(total_mse_change_pct) <= 10%
```

## Eta Scaling

Redistribution strength is defined as:

```text
mean(mid_change_pct, high_change_pct) - mean(very_low_change_pct, low_change_pct)
```

Positive values indicate the Phase 2-like pattern where low-frequency bands
decrease and faster bands increase. The script fits descriptive linear and
quadratic curves versus eta and classifies the relationship as approximately
linear, superlinear, or noisy.

## Complexity Dependence

The sweep reports redistribution summaries separately for simple, medium, and
hard toy settings. This tests whether the redistribution effect depends on the
same capacity-complexity regime that produced dc-trend dominance and partial
posterior/drift mismatch.

## Candidate Mechanisms

### Candidate 1: RSSM Prediction Error Capacity Constraint

Total error is approximately conserved, and SMAD shifts error between frequency
bands rather than creating or destroying it. This would appear as small
`total_mse_change_pct` alongside large signed per-band changes.

### Candidate 2: U_drift Damping Forces Compensatory Fast Directions

Damping along empirical drift directions suppresses slow components, but the
patched transition compensates in orthogonal directions that express as mid/high
frequency error. This would appear as consistent low-band decreases and mid/high
increases, especially at larger eta.

### Candidate 3: Numerical Artifact With No Systematic Mechanism

Band changes are inconsistent across seeds, ranks, etas, or complexity levels.
Total error and redistribution strength do not show stable patterns.

## Verdict

To be filled after the full sweep.
