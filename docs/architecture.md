# Architecture Notes

Last updated: Month 2, Week 3

## Pipeline Overview

1. **Rollout Extraction**: `DreamerV3Adapter` collects true environment rollouts and runs open-loop imagination via RSSM `img_step()`. v2 rollouts include latent features from `get_feat()`.
2. **Latent Preprocessing**: `get_feat() = concat([stoch(1024), deter(512)])`. Only the deter portion is used for frequency analysis because stoch is hard one-hot and unsuitable for bandpass filtering.
3. **PCA**: Fitted on the deter portion of true latent trajectories, pooled across seeds for cross-seed consistency. In practice, about `25-40` PCs capture 95% variance.
4. **Frequency Decomposition**: Butterworth bandpass (`order=4`, zero-phase via `sosfiltfilt`) into 5 bands: `dc_trend`, `very_low`, `low`, `mid`, `high`. Filter ablation confirmed that Butterworth, Chebyshev Type I, and FIR all produce the same qualitative band ranking. Butterworth is the default.
5. **Error Curves**: Per-step MSE or L2 for each band, aggregated across seeds with mean, std, and 95% CI.
6. **Figures**: Per-band error vs horizon step with confidence-interval shading.

## Key Module Dependencies

- `src/models/adapter.py` -> base interface
- `src/models/dreamerv3_adapter.py` -> DreamerV3 implementation
- `src/diagnostics/freq_decompose.py` -> frequency decomposition
- `src/diagnostics/error_curves.py` -> per-band error curves
- `src/analysis/figure1.py` -> Figure 1 generation

## Data Flow

```text
checkpoint -> DreamerV3Adapter.extract_rollout(include_latent=True)
  -> v2 .npz files (true_obs, imagined_obs, true_latent, imagined_latent, actions, rewards)
    -> freq_decompose.decompose_pair()
      -> per-band filtered signals
        -> error_curves.aggregate_curves()
          -> mean/std/CI per band per step
            -> figure1.py -> PDF figures
```
