# Month 2 Progress (As of 2026-04-29)

## Overall Status: ~65% Complete (Week 3 of 4)

## Week 1: Latent Collection + Frequency Decomposition — COMPLETE

### Latent Dropping Bug Fix

- Modified `dreamerv3_adapter.py` to collect RSSM latent features alongside decoded observations.
- Added `include_latent=False` flag to `extract_rollout()` in the base adapter.
- New private methods: `_imagine_with_latent()`, `_collect_true_rollout_with_latent()`, `_posterior_latent_from_obs()`.
- All latents use deterministic mode (`sample=False`) for clean comparison.

### Latent Data Probe — Key Discovery

- `get_feat() = concat([stoch, deter])`, not `concat([deter, stoch])`.
- First 1024 dims = stoch: hard one-hot (`32 categoricals x 32 classes`), 96.88% exact zeros, values in `{0,1}`, piecewise-constant.
- Last 512 dims = deter: continuous GRU state, range `[-1,1]`, 100k+ unique values, lag-1 autocorrelation around `0.84`.
- Decision: `freq_decompose` operates on the deter portion only after PCA.
- Rationale: bandpass filtering assumes continuous signals; one-hot stoch produces wideband artifacts.

### v2 Rollout Extraction

- 20 rollouts (seeds `0-19`) with 6 arrays each: `true_obs`, `imagined_obs`, `actions`, `rewards`, `true_latent`, `imagined_latent`.
- All shapes verified: obs `(200, 17)`, latent `(200, 1536)`.
- L2 latent drift confirmed growing in all 20 seeds (`first10 ~4.5-6.8`, `last10 ~10.1-11.1`).
- `manifest_v2.json` added with `latent_spec` metadata.

### Frequency Decomposition Module

- `src/diagnostics/freq_decompose.py`: 5 frequency bands, Butterworth order 4, deter-only PCA (`25-40` PCs for 95% variance).
- Band definitions: `dc_trend (0-0.02)`, `very_low (0.02-0.05)`, `low (0.05-0.10)`, `mid (0.10-0.20)`, `high (0.20-0.50)` cycles/step.
- 7 unit tests all passing.
- Design doc: `docs/freq_decompose_design.md`.

### Integration Check on Real Data (seed 0)

- `dc_trend`: `true_energy=316.7`, `imag_energy=1371.4` (`4.3x` explosion).
- `high`: `true_energy=201.0`, `imag_energy=59.4` (`70%` decay).
- Pattern: DC explodes while high frequency is smoothed out.

## Week 2: Error Curves + Window Ablation — COMPLETE

### error_curves Module

- `src/diagnostics/error_curves.py`: `per_step_band_mse()`, `per_step_band_l2()`, `aggregate_curves()`, `total_error_curve()`.
- `aggregate_curves` uses pooled PCA (Option B): fits PCA on concatenated true signals from all seeds, then applies a shared basis to all seeds.
- 7 unit tests all passing.

### Integration Check (5 seeds, latent-space)

- `dc_trend` dominates at all horizon points.
- Latent-space curves show a plateau / hump shape rather than monotonic growth.
- Obs-space `dc_trend` shows clear monotonic growth (`p ≈ 0.0000`).
- Boundary effects from `sosfiltfilt` affect steps `0-10` and `190-199`.

### Imagination Window Ablation

- Tested `imagination_start in {50, 200, 500} x 5 seeds`.
- 10 new rollouts extracted for starts `50` and `500`.
- Results: qualitative drift pattern (`dc_trend` dominant, high suppressed) is consistent across starts.
- Magnitude and growth ratio vary with start position.
- Conclusion: **phase-dependent drift** framing — consistent qualitative bias toward low-frequency error, but magnitude modulated by episode phase.
- Full report: `docs/imagination_window_robustness.md`.

## Week 3: Figure 1 + Cartpole Training — IN PROGRESS

### Figure 1 (Cheetah)

- `src/analysis/figure1.py` generates 4 PDFs (latent/obs x linear/log).
- 20 seeds, 95% CI, trimmed to steps `[10, 189]`.

### Statistical Summary

- Latent-space band ranking: `dc_trend (46.7%) > low > very_low > mid > high (4.2%)`.
- Frequency asymmetry confirmed: about `11:1` ratio between `dc_trend` and `high`.
- Obs-space: `dc_trend` is the only band with significant positive slope (`p ≈ 0.0000`).
- Latent-space: `0/5` bands have significant positive slope (plateau / hump shape).
- Interpretation: latent drift appears to saturate, but the decoder amplifies small deviations into growing obs-space error.

### Finding A Status: SUPPORTED

- Frequency-asymmetric drift: YES (strong evidence).
- DC/trend dominance: YES (`46.7%` latent, `37.8%` obs).
- High-freq suppression: YES (`4.2%` latent).
- Monotonic growth with horizon: only in obs-space `dc_trend`.
- Paper framing: **frequency-asymmetric drift with trend dominance**.

### Cartpole Training

- Status: IN PROGRESS on RunPod RTX 4090.
- Config: same hyperparameters as Cheetah (`batch_size=32`, `envs=8`, `500k` steps).
- Expected cost: about `$7`.
- Expected completion: pending.

## Week 4: Filter Ablation + Cartpole Diagnostics — IN PROGRESS

### Filter Ablation — COMPLETE

Tested three bandpass filter types on all 20 Cheetah v2 rollouts:

- Butterworth order 4 (default)
- Chebyshev Type I order 4 (1 dB ripple)
- FIR (`numtaps=51`)

Key results:

- `dc_trend` is the dominant band for all three filters in both latent and obs space.
- `high` is the smallest latent band for all three filters.
- Obs-space band rankings are identical across all three filters.
- Latent-space FIR swaps `very_low` and `mid`, but these bands differ by only about 3% in absolute MSE and do not change the main interpretation.
- Max deviation across filters: `32.1%` (latent), `37.4%` (obs). This is driven by Chebyshev's steeper rolloff concentrating energy differently, not by qualitative disagreement.
- Conclusion: Finding A is qualitatively robust to filter choice.
- Default remains Butterworth order 4 for all subsequent analysis.
- Full report: `docs/filter_ablation.md`.
- Raw data: `results/tables/filter_ablation.csv`.

### Cartpole Training — IN PROGRESS

- Running on RunPod RTX 4090, same hyperparameters as Cheetah.
- Expected completion: pending.
- After completion: rollout extraction + Figure 1 comparison.

### Still Planned

- Cartpole rollout extraction and diagnostics.
- Cross-task comparison (Cheetah vs Cartpole drift patterns).
- Month 2 summary + Finding A final judgment.
- Tag: `v0.2-month2-figure1`.

## Key Decisions Made in Month 2

1. Deter-only analysis (`stoch` excluded from bandpass because one-hot signals are incompatible).
2. Deterministic latent mode (`sample=False` for both true and imagined).
3. `include_latent` flag (backward-compatible adapter extension).
4. Pooled PCA across seeds (shared basis for cross-seed comparison).
5. Phase-dependent framing (not pure steady-state).
6. Butterworth as default filter (ablation confirmed qualitative robustness).

## Compute Budget

- Month 1: ~9.5 GPU-hours (`$6.65`) — Cheetah training.
- Month 2: ~9-12 GPU-hours (`~$7`) — Cartpole training (in progress).
- Local MPS: about 1 hour total for rollout extraction and analysis.
- Total spent: about `$13-14`.
- Remaining: about `180` GPU-hours.

## Risk Assessment

- Finding A is supported, but not as strong as "monotonic growth everywhere."
- Single task (Cheetah) until Cartpole completes.
- Boundary effects in bandpass require trimming steps `0-10` and `190-199`.
- Latent vs obs discrepancy (plateau vs growth) needs careful framing.
