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

The completed sweep produced:

- `180` row-level SMAD evaluations.
- `15` trained baseline toy RSSMs.
- `60` rows per complexity level.
- `12` SMAD evaluations per trained model: ranks `3`, `5`, `10` crossed
  with eta values `0.1`, `0.2`, `0.3`, `0.5`.

The output used for this summary is:

- `results/tables/toy_redistribution_sweep.json`

Aggregate signed band changes by complexity:

| Complexity | dc_trend | very_low | low | mid | high | total | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| simple | +60.1 +/- 65.2% | +32.6 +/- 22.8% | +313.8 +/- 161.8% | +139.8 +/- 40.9% | -5.7 +/- 20.4% | +137.6 +/- 40.9% | SMAD is harmful |
| medium | -1.9 +/- 1.3% | -8.8 +/- 6.5% | -5.7 +/- 4.1% | -1.0 +/- 0.9% | -0.1 +/- 0.1% | -2.4 +/- 1.7% | all bands reduced |
| hard | -11.5 +/- 12.8% | -20.3 +/- 14.0% | -16.4 +/- 11.9% | -9.6 +/- 9.7% | -3.9 +/- 5.9% | -12.5 +/- 12.2% | all bands reduced |

The Phase 2 Cheetah observation was:

| Band | Phase 2 Cheetah change |
|---|---:|
| dc_trend | -4.8% |
| very_low | -20.2% |
| low | -16.2% |
| mid | +39.7% |
| high | +12.9% |

The toy sweep does not reproduce this low-frequency-down plus mid/high-up
redistribution pattern. The `phase2_pattern` rate is `0.000`; none of the
`180` rows match the Phase 2 Cheetah redistribution signature.

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

Result: NOT conserved.

Across all rows, the total MSE change is:

```text
mean total_mse_change_pct = +40.9 +/- 72.9%
```

Only `94/180` rows, or `52.2%`, satisfy the default conservation criterion.
This is not consistent with an approximately conserved error budget.

The non-conservation is driven by the simple toy setting, where SMAD increases
total MSE by `+137.6 +/- 40.9%`. Medium and hard settings reduce total MSE
instead:

- medium: `-2.4 +/- 1.7%`
- hard: `-12.5 +/- 12.2%`

This means the toy result is better described as complexity-dependent
benefit/harm, not as frequency redistribution under a conserved total error
budget.

## Eta Scaling

Redistribution strength is defined as:

```text
mean(mid_change_pct, high_change_pct) - mean(very_low_change_pct, low_change_pct)
```

Positive values indicate the Phase 2-like pattern where low-frequency bands
decrease and faster bands increase. The script fits descriptive linear and
quadratic curves versus eta and classifies the relationship as approximately
linear, superlinear, or noisy.

Result: `superlinear_descriptive`.

The eta-scaling diagnostic fits descriptive linear and quadratic curves to the
redistribution-strength metric:

- linear fit: `R^2 = 0.675`
- quadratic fit: `R^2 = 0.788`

The quadratic fit is stronger, so the script classifies the trend as
`superlinear_descriptive`. However, this should not be interpreted as evidence
for the Phase 2 redistribution mechanism. The redistribution-strength summary
is dominated by the harmful simple setting and does not correspond to a
consistent low-band decrease plus high-band increase.

By eta, mean redistribution strength is:

| eta | redistribution strength | total MSE change |
|---:|---:|---:|
| 0.1 | -35.9 | +48.5% |
| 0.2 | -56.1 | +55.0% |
| 0.3 | -24.7 | +36.0% |
| 0.5 | -0.5 | +24.0% |

The high-band change is not systematically positive, and the
`low_down_high_up_rate` is `0.000`.

## Complexity Dependence

The sweep reports redistribution summaries separately for simple, medium, and
hard toy settings. This tests whether the redistribution effect depends on the
same capacity-complexity regime that produced dc-trend dominance and partial
posterior/drift mismatch.

Result: inverted complexity dependence.

The simple task is harmed strongly by SMAD, while the medium and hard tasks are
benefited:

- simple: total MSE `+137.6 +/- 40.9%`
- medium: total MSE `-2.4 +/- 1.7%`
- hard: total MSE `-12.5 +/- 12.2%`

This is not the Phase 2 Cheetah redistribution pattern. In the toy setting,
the dominant effect is that SMAD can be damaging when the task is too simple
and beneficial once the toy dynamics are more complex.

## Candidate Mechanisms

### Candidate 1: RSSM Prediction Error Capacity Constraint

Total error is approximately conserved, and SMAD shifts error between frequency
bands rather than creating or destroying it. This would appear as small
`total_mse_change_pct` alongside large signed per-band changes.

Verdict: NOT SUPPORTED.

Total error is not conserved. Mean total change is `+40.9 +/- 72.9%`, and the
simple setting shows a large total-error increase. This rules out a clean
capacity-constraint story for the toy sweep.

### Candidate 2: U_drift Damping Forces Compensatory Fast Directions

Damping along empirical drift directions suppresses slow components, but the
patched transition compensates in orthogonal directions that express as mid/high
frequency error. This would appear as consistent low-band decreases and mid/high
increases, especially at larger eta.

Verdict: NOT SUPPORTED in toy.

The Phase 2-like pattern does not appear in any row. The `phase2_pattern_rate`
is `0.000`, and the `low_down_high_up_rate` is also `0.000`. Medium and hard
tasks reduce all bands on average, including mid and high. The toy sweep
therefore does not support systematic compensatory fast-mode activation.

### Candidate 3: Numerical Artifact With No Systematic Mechanism

Band changes are inconsistent across seeds, ranks, etas, or complexity levels.
Total error and redistribution strength do not show stable patterns.

Verdict: MOST LIKELY for the Phase 2 Cheetah mid/high increase.

The toy sweep spans `5` seeds, `3` complexity levels, `3` ranks, and `4` eta
values. It finds no systematic redistribution from low-frequency bands to
mid/high-frequency bands. The simplest interpretation is that the Phase 2
Cheetah mid/high increase was either a single-seed artifact or a
Cheetah-specific phenomenon rather than a general SMAD mechanism.

## Verdict

The Phase 2 Cheetah redistribution observation is likely a single-seed artifact
or Cheetah-specific phenomenon. It should NOT be reported as a general finding.

The stronger toy conclusion is narrower:

```text
SMAD's effect is complexity-dependent in toy RSSMs: harmful in the simple
setting, mildly beneficial in the medium setting, and most beneficial in the
hard setting.
```
