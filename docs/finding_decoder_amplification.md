# Finding E Assessment: Decoder Amplification Hypothesis

## Hypothesis

The original Finding E hypothesis was:

> The trained DreamerV3 decoder selectively amplifies `dc_trend` drift, explaining why `dc_trend` dominates in observation space.

This hypothesis predicts not only that the trained decoder amplifies latent drift relative to an untrained decoder, but also that the amplification is directionally biased toward low-frequency or `dc_trend` latent directions.

## Status

**NOT SUPPORTED as originally stated. Partially supported in a modified form.**

The trained decoder does amplify drift overall, but the evidence does not support selective amplification of `dc_trend`. The decoder amplification story is best treated as a supporting mechanistic detail, not as an independent finding explaining trend dominance.

## Evidence Summary

### D1: Random Decoder Control

**Result:** Supports that decoder amplification exists.

D1 compared the trained decoder against a random decoder with the same architecture and freshly initialized weights. Both decoders received the same `true_latent` and `imagined_latent` trajectories from 20 Cheetah Run v2 rollouts.

Key result:

- Trained decoder drift: approximately 12-14 L2.
- Random decoder drift: approximately 8-9 L2.
- Overall amplification ratio: approximately 1.48.
- The ratio grows from approximately 1.38 at step 25 to approximately 1.64 at step 175.

Interpretation:

The trained decoder maps the same latent deviations into larger observation-space deviations than a random decoder. This supports the weak form of decoder amplification: training makes the decoder more sensitive to latent drift.

### D2: Lipschitz Probe

**Result:** Refutes directional amplification toward `dc_trend`.

D2 perturbed true latents along frequency-band directions and measured the trained decoder's observation-space response. If the original Finding E hypothesis were correct, `dc_trend` directions should have shown the largest effective Lipschitz constants.

Instead:

- `dc_trend` had the **smallest** effective Lipschitz constant.
- `mid`, `low`, and `high` directions showed larger decoder sensitivity.
- Effective Lipschitz curves were nearly flat across epsilon values, indicating the decoder is locally approximately linear near the true-latent distribution.

Approximate sensitivity ordering at epsilon = 1.0:

1. `dc_trend`: smallest
2. `very_low`
3. `high`
4. `low`
5. `mid`: largest

Interpretation:

The decoder does not preferentially amplify `dc_trend`. Observation-space trend dominance cannot be explained by a direction-specific decoder gain.

### D3: Linear vs Nonlinear Separation

**Result:** Refutes a horizon-dependent nonlinear amplification mechanism.

D3 compared:

- `decode(imagined_latent) - decode(true_latent)`, the actual decoded observation difference.
- `decode(imagined_latent - true_latent)`, the decoder output on the raw latent difference vector.

For a globally linear decoder, these would match. In practice, the nonlinearity gap was large, with an overall ratio near 1.0.

However, this does **not** mean the decoder is strongly nonlinear in the local sense tested by D2. The key caveat is that `imagined_latent - true_latent` is an out-of-distribution difference vector, not a normal DreamerV3 RSSM feature.

Observed pattern:

- Nonlinearity gap is approximately 10-11 L2 and nearly constant across horizon.
- Actual observation drift grows with horizon.
- The scatter plot shows a horizontal band rather than a proportional relationship between gap and actual drift.

Interpretation:

D3 shows that the decoder is not globally linear and behaves unpredictably on OOD latent-difference inputs. But the effect looks like a constant OOD noise floor, not a growing, horizon-dependent amplification mechanism.

## Revised Interpretation

The combined D1-D3 evidence supports the following revised view:

- Trend dominance is intrinsic to the latent dynamics of the RSSM, likely arising from GRU/tanh saturation, attractor-like behavior, or slow-subspace accumulation during open-loop imagination.
- Decoder amplification exists, but it is broad and non-directional.
- The trained decoder becomes unreliable when imagined latents drift away from the training distribution.
- This helps explain the "latent plateau but obs growth" phenomenon: even if latent error plateaus or changes slowly, the decoder can map OOD latent states into larger observation-space errors.
- This decoder effect does not alter the frequency structure of drift. It amplifies the consequences of latent drift, but it does not create `dc_trend` dominance.

In short:

> The decoder is a magnifier of OOD latent drift, not the source of frequency-band asymmetry.

## Impact on Paper Narrative

Finding E should **not** enter the main paper as an independent positive finding.

Recommended narrative changes:

- Move D1-D3 to the supplementary material as a falsification and mechanism record.
- Keep the main story focused on Finding A: drift is frequency-asymmetric and `dc_trend`-dominated in latent space.
- Use D1-D3 as a discussion clarification: trained decoders can amplify OOD latent drift in observation space, but the amplification is not selectively aligned with `dc_trend`.
- Theorem 1 motivation becomes cleaner: trend dominance is a latent-intrinsic property, not a decoder artifact.
- The decoder amplification section should become a short mechanistic clarification paragraph rather than a headline result.

This strengthens the paper because it removes a weaker explanatory branch and sharpens the central claim: the RSSM latent dynamics themselves produce slow/trend-dominated drift.

## Key Figures

- `results/figures/d1_drift_comparison.pdf`
- `results/figures/d1_amplification_ratio.pdf`
- `results/figures/d2_lipschitz_probe.pdf`
- `results/figures/d2_lipschitz_constant.pdf`
- `results/figures/d2_bar_eps1.pdf`
- `results/figures/d3_linear_nonlinear.pdf`
- `results/figures/d3_scatter.pdf`

## Data Files

- `results/tables/decoder_d1_results.npz`
- `results/tables/decoder_d2_results.npz`
- `results/tables/decoder_d3_results.npz`
