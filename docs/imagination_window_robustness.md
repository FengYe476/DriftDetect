# Imagination Window Robustness

## Purpose

This ablation tests whether DriftDetect's frequency-specific drift patterns
depend on when imagination begins within the episode. The core question is
whether the error profile reflects a steady-state property of the model or a
phase-dependent effect tied to rollout context at imagination onset.

## Method

We evaluated `imagination_start in {50, 200, 500}` with 5 seeds per start and a
fixed `horizon=200`.

- Start `200` reused the existing v2 rollouts for seeds `0-4`.
- Starts `50` and `500` were newly extracted with `include_latent=True`.
- Total steps remained `1000`, so all starts had sufficient rollout context.
- For each start, we computed per-band mean squared error curves with
  `error_curves.aggregate_curves()`.
- Latent-space analysis used `true_latent` vs `imagined_latent`.
- Observation-space analysis used `true_obs` vs `imagined_obs`.
- To reduce bandpass boundary effects, comparisons focused on steps `25`, `100`,
  and `175` rather than the first or last few steps.

## Results

### Latent-space per-band MSE

| Band | start=50 step25 | start=50 step100 | start=50 step175 | start=200 step25 | start=200 step100 | start=200 step175 | start=500 step25 | start=500 step100 | start=500 step175 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dc_trend | 0.1193 | 0.2419 | 0.2720 | 0.1544 | 0.2508 | 0.1947 | 0.0930 | 0.3804 | 0.4024 |
| very_low | 0.1228 | 0.0216 | 0.0288 | 0.1569 | 0.0514 | 0.0500 | 0.0545 | 0.0286 | 0.0376 |
| low | 0.0865 | 0.1356 | 0.1089 | 0.1337 | 0.1496 | 0.1020 | 0.0807 | 0.1376 | 0.1136 |
| mid | 0.0365 | 0.0443 | 0.0514 | 0.0723 | 0.0454 | 0.0341 | 0.0209 | 0.0430 | 0.0445 |
| high | 0.0113 | 0.0227 | 0.0143 | 0.0190 | 0.0201 | 0.0223 | 0.0056 | 0.0170 | 0.0236 |

Step-175 / step-25 ratios:

- `dc_trend`: start50=`2.28`, start200=`1.26`, start500=`4.33`
- `very_low`: start50=`0.23`, start200=`0.32`, start500=`0.69`
- `low`: start50=`1.26`, start200=`0.76`, start500=`1.41`
- `mid`: start50=`1.41`, start200=`0.47`, start500=`2.13`
- `high`: start50=`1.27`, start200=`1.18`, start500=`4.24`

### Observation-space per-band MSE

| Band | start=50 step25 | start=50 step100 | start=50 step175 | start=200 step25 | start=200 step100 | start=200 step175 | start=500 step25 | start=500 step100 | start=500 step175 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dc_trend | 0.5246 | 4.4287 | 4.7349 | 0.6692 | 4.2731 | 4.4133 | 0.2809 | 3.8615 | 4.6317 |
| very_low | 0.3471 | 0.1698 | 0.3801 | 0.4158 | 0.2542 | 0.5050 | 0.2901 | 0.1726 | 0.3479 |
| low | 2.4697 | 3.1895 | 2.3529 | 2.9222 | 1.9105 | 1.1397 | 2.4656 | 3.0579 | 3.6724 |
| mid | 1.2772 | 1.4476 | 2.3272 | 4.8285 | 3.1219 | 1.3107 | 1.5202 | 4.3994 | 2.2082 |
| high | 0.9767 | 1.4986 | 0.7173 | 0.9035 | 2.3651 | 1.2809 | 1.4791 | 0.2995 | 1.6026 |

Step-175 / step-25 ratios:

- `dc_trend`: start50=`9.03`, start200=`6.59`, start500=`16.49`
- `very_low`: start50=`1.10`, start200=`1.21`, start500=`1.20`
- `low`: start50=`0.95`, start200=`0.39`, start500=`1.49`
- `mid`: start50=`1.82`, start200=`0.27`, start500=`1.45`
- `high`: start50=`0.73`, start200=`1.42`, start500=`1.08`

## Analysis

The results are mixed, but the dominant signal is not fully steady-state.

- The `dc_trend` band is the most important result because it is the largest
  error contributor in both latent and observation space. Its growth ratio
  changes substantially across starts, especially in latent space
  (`1.26` vs `4.33`) and obs space (`6.59` vs `16.49`).
- Latent-space `low` is the one clearly stable band. Its ratios stay within the
  consistency threshold and its stepwise magnitudes are fairly similar.
- Observation-space `very_low` and `high` are also fairly consistent across
  starts, but these are not the dominant drift channels.
- `mid` and `high` latent bands vary noticeably, and `low` / `mid` obs-space
  bands also change shape depending on start.

Taken together, the ablation does not support a clean "same curve everywhere in
the episode" story. Some bands are robust, but the largest and most important
trend-like drift channel changes with imagination onset.

## Conclusion

The paper should use a **phase-dependent drift** framing rather than a pure
steady-state drift framing. A more precise statement is that DriftDetect finds a
stable qualitative bias toward low-frequency / trend error, but the magnitude
and growth shape of that drift depend on when imagination begins in the episode.
That is a stronger and more honest claim than saying the same error process
holds uniformly across all phases.

## Limitations

- Five seeds per start is a small sample.
- The ablation covers only one task: `dmc_cheetah_run`.
- Bandpass boundary effects still affect roughly steps `0-10` and `190-199`.
- The latent-space comparison is qualitative across starts because
  `aggregate_curves()` fits PCA pooled across seeds within each start rather
  than one shared PCA basis across all three starts.
