# Filter Ablation Report

## Purpose

Test whether DriftDetect's frequency-band drift results are robust to the choice
of bandpass filter.

## Method

- Filters tested: Butterworth (order 4), Chebyshev Type I (order 4, 1 dB ripple), FIR (`numtaps=51` with automatic shortening for short signals).
- Dataset: 20 Cheetah Run v2 rollouts (seeds `0-19`).
- Metric: per-band mean MSE averaged over steps `[25:175]`.
- Both latent-space and obs-space were analyzed.
- Raw per-step data was written to `results/tables/filter_ablation.csv` because the local environment had pandas but no parquet engine (`pyarrow` / `fastparquet`).

## Results

### Latent-space

| Band | Butterworth | Chebyshev1 | FIR | Max Deviation |
| --- | ---: | ---: | ---: | ---: |
| dc_trend | 0.2340 | 0.1682 | 0.2139 | 32.1% |
| very_low | 0.0521 | 0.0453 | 0.0438 | 17.7% |
| low | 0.1422 | 0.1090 | 0.1129 | 27.4% |
| mid | 0.0518 | 0.0428 | 0.0452 | 19.2% |
| high | 0.0211 | 0.0237 | 0.0208 | 13.2% |

Band rankings:

- Butterworth: `dc_trend > low > very_low > mid > high`
- Chebyshev1: `dc_trend > low > very_low > mid > high`
- FIR: `dc_trend > low > mid > very_low > high`

Rankings identical: **NO**

### Obs-space

| Band | Butterworth | Chebyshev1 | FIR | Max Deviation |
| --- | ---: | ---: | ---: | ---: |
| dc_trend | 3.9388 | 2.6437 | 3.8173 | 37.4% |
| very_low | 0.3243 | 0.2904 | 0.2752 | 16.5% |
| low | 2.5161 | 2.0230 | 1.9270 | 27.3% |
| mid | 2.2550 | 1.9126 | 1.9159 | 16.9% |
| high | 1.3970 | 1.4391 | 1.3949 | 3.1% |

Band rankings:

- Butterworth: `dc_trend > low > mid > high > very_low`
- Chebyshev1: `dc_trend > low > mid > high > very_low`
- FIR: `dc_trend > low > mid > high > very_low`

Rankings identical: **YES**

## Conclusion

The results show some sensitivity to filter choice under the predefined strict
criterion. Latent-space max deviation is `32.1%`, obs-space max deviation is
`37.4%`, and latent-space rankings are not identical because FIR swaps
`very_low` and `mid`.

The core qualitative finding is still stable: `dc_trend` is the dominant band
for all filters in both latent and obs space, and high-frequency latent error
remains the smallest band for all filters. However, absolute magnitudes and the
ordering of small non-dominant latent bands are filter-sensitive. The paper
should report Butterworth as the default and include this ablation as a
supplementary robustness check.

## Recommendation

Keep Butterworth order 4 as the default for the rest of the project. It is the
original analysis choice, has smooth monotonic passbands, is already validated
by the existing tests, and preserves a clear interpretation of Figure 1. When
writing the paper, phrase Finding A as robust in its qualitative direction
(`dc_trend` dominance and high-frequency suppression), but not fully invariant in
exact per-band magnitudes.
