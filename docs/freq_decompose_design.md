# Frequency Decomposition Design - Why Deter-Only Bandpass

## Motivation

DriftDetect's core question is whether world model imagination drift is
frequency-dependent. Aggregate rollout error can tell us that imagination fails,
but not which temporal components fail first. To answer that, we decompose latent
trajectories into temporal frequency bands and measure how per-band error grows
over horizon.

The Month 2 pipeline therefore needs a representation that is both internal to
the model and suitable for spectral filtering.

## Why latent space, not just observation space

Observation-space drift is a downstream symptom. The model does not propagate
flat Cheetah observations during imagination; it propagates RSSM latent state.
If high-frequency error appears in observations, the more mechanistic question is
whether it originates inside the latent trajectory before decoding.

Latent-space diagnostics let us compare the imagined latent path against the
posterior latent path inferred from the true environment rollout under the same
actions. That makes frequency decomposition a diagnostic of the model's internal
rollout dynamics rather than only its decoded output.

Observation-space decomposition is still useful as a sanity check and comparison
path, but the main diagnostic target is latent state.

## Why deter-only

The DreamerV3 RSSM feature returned by `get_feat()` is:

```python
concat([stoch, deter])
```

For the Cheetah checkpoint, this means:

- `stoch`: first 1024 dimensions, from 32 categoricals x 32 classes.
- `deter`: last 512 dimensions, the deterministic GRU hidden state.

The latent probe showed that the stochastic portion is hard one-hot. It has
values in `{0, 1}`, is 96.88% exact zeros, and changes through discrete category
switches. That structure is meaningful, but it is not a smooth signal.

Bandpass filtering assumes a continuous time series with interpretable spectral
content. Applying Butterworth filters to binary switching signals produces
wideband artifacts: sharp transitions spread energy across many frequencies,
which can make the analysis reflect switching discontinuities rather than
meaningful temporal dynamics.

The deterministic portion is a better fit for the Month 2 diagnostic. It is the
GRU hidden state, continuous in approximately `[-1, 1]`, has more than 100k
unique values in the probe, and active dimensions show temporal persistence
such as lag-1 autocorrelation around 0.84. This is the part of the RSSM most
appropriate for spectral analysis.

The stochastic state is not discarded from the project. It should be analyzed
separately in future work with tools designed for discrete state, such as
switching frequency, categorical entropy, dwell time, and transition statistics.
It is excluded only from the bandpass pipeline.

## Why PCA before bandpass

The deter state is 512-dimensional. Filtering every raw dimension is possible,
but difficult to interpret and visualize. PCA gives us a compact, orthogonal
basis for the dominant latent variation before we decompose temporal bands.

The seed-0 probe found:

- 29 PCs capture 90% of deter variance.
- 36 PCs capture 95% of deter variance.

The default is to use the smallest number of PCs whose cumulative explained
variance reaches 95%. This keeps most of the smooth latent signal while avoiding
hundreds of noisy component-wise plots.

PCA should be fit on the true latent deter portion, then applied to both true
and imagined trajectories. For cross-seed analysis, PCA can be fit either per
seed or on deter states pooled across seeds. The module supports both by letting
callers pass in a fitted PCA object; pooled PCA is recommended for cross-seed
figures because it keeps all seeds in a shared basis.

Bandpass filtering is then applied independently to each PC time series.

## Band definitions

The default decomposition uses normalized frequencies in cycles per step with a
sampling rate of `1 step^-1`. For a 200-step horizon, the lowest resolvable
nonzero frequency is `1 / 200 = 0.005` cycles per step, so the following bands
are coarse but resolvable:

| Band | Name | Range |
| --- | --- | --- |
| 0 | `dc_trend` | 0.00-0.02 cycles/step |
| 1 | `very_low` | 0.02-0.05 cycles/step |
| 2 | `low` | 0.05-0.10 cycles/step |
| 3 | `mid` | 0.10-0.20 cycles/step |
| 4 | `high` | 0.20-0.50 cycles/step |

The high band reaches Nyquist at `0.50` cycles per step.

## Filter choice

The default filter is a fourth-order Butterworth filter applied with zero-phase
forward-backward filtering. This keeps phase alignment between true and imagined
signals, which matters for error comparisons.

Filter ablation is planned for Week 4. We should compare Butterworth,
Chebyshev, and FIR filters before treating detailed band boundaries as final.

## Observation-space support

The module also accepts low-dimensional observation-space input. For Cheetah,
`true_obs` and `imagined_obs` are 17-dimensional, so PCA is skipped and bandpass
is applied directly to each observation dimension.

This gives us a sanity-check path: latent-space drift should be compared against
the downstream observation-space drift it eventually causes.

## Interface sketch

```python
decompose(
    signal,
    fs=1.0,
    bands=DEFAULT_BANDS,
    n_pcs=None,
    var_threshold=0.95,
) -> dict[str, np.ndarray]
```

The returned dictionary contains one filtered array per band plus an `_info`
metadata entry with input dimensionality, PCA usage, retained variance, and band
configuration.
