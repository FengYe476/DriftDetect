# Observation Space Alignment for V3 vs V4

## Purpose

This note explains how DriftDetect compares frequency-domain imagination drift
between DreamerV3 and DreamerV4 despite different observation spaces and latent
semantics.

## Cross-Architecture Challenge

DreamerV3 Cheetah is proprioceptive:

- `true_obs` and `imagined_obs`: 17-dimensional proprioceptive vectors.
- `true_latent` and `imagined_latent`: 1536-dimensional RSSM features.
- V3 `deter`: the last 512 dimensions, `latent[:, 1024:1536]`.

DreamerV4 Cheetah is image-native:

- `true_obs` and `imagined_obs`: RGB images, `(3, 128, 128)`.
- `true_latent` and `imagined_latent`: 512-dimensional packed tokenizer
  latents.
- V4 latent: flattened packed dynamics state `(8, 64)`.

The dimensionality match is convenient but not semantic alignment. V3 `deter`
is a recurrent RSSM state trained around proprioceptive prediction and policy
learning. V4 packed tokenizer state is a visual representation used by a
transformer denoising dynamics model.

Raw V3 and V4 MSE magnitudes should not be interpreted as the same physical
quantity.

## Primary Metric: Latent Frequency Structure

The primary comparison is latent-space frequency structure.

For V3:

```text
v3_deter = latent[:, 1024:1536]
```

For V4:

```text
v4_latent = latent[:, :]
```

No V4 slicing is applied. V4 has no stochastic and deterministic split.

The same pipeline is applied to each architecture:

1. Pool true latent trajectories across 20 rollout seeds.
2. Fit PCA retaining 95 percent of true-latent variance.
3. Project true and imagined trajectories into that basis.
4. Decompose projected components with the same Butterworth bands.
5. Compute per-step and trim-window per-band MSE.

Bands:

- `dc_trend`: 0.00 to 0.02 cycles per step.
- `very_low`: 0.02 to 0.05 cycles per step.
- `low`: 0.05 to 0.10 cycles per step.
- `mid`: 0.10 to 0.20 cycles per step.
- `high`: 0.20 to 0.50 cycles per step.

The trim window is steps 25 through 175 inclusive. This avoids over-weighting
the context boundary and the noisy final portion of the 200-step horizon.

This metric asks whether the frequency structure of imagination drift differs
between a GRU-RSSM world model and a transformer denoising world model. The
central question is whether `dc_trend` dominance appears in V4 or is specific
to recurrent RSSM dynamics.

## Secondary Metric: Image-Space Evidence

V4 also provides decoded RGB frames. V3 does not provide matching RGB frames in
this rollout set, so image-space results are secondary evidence.

Two V4 image reductions are used.

Channel-mean RGB:

```text
image_mean[t] = mean over H and W for each RGB channel
```

This gives a 3-dimensional time series. It is architecture-neutral and cheap,
but very lossy: pose, spatial layout, and most motion structure are lost.

Image PCA:

```text
flattened_frame[t] = RGB frame reshaped to 49152 dimensions
```

True V4 frames are pooled across seeds, PCA is fitted with a 95 percent
variance target, and PCs are capped at 50. True and imagined frames are
projected through that basis before the same five-band decomposition.

Image PCA captures more visual structure than channel means, but it introduces
its own V4-specific basis. It is useful supplementary signal, not a replacement
for latent-space comparison.

## What We Can Claim

- Whether `dc_trend` carries a large fraction of V4 latent drift, as it does
  for V3.
- Whether high-frequency latent errors remain relatively suppressed in V4 or
  whether V4 shifts drift energy toward faster bands.
- Fractional band shares within each architecture, since each share is
  normalized by that architecture's total trim-window band MSE.
- V4 image-space results as secondary checks linking latent drift to decoded
  visual behavior.

## What We Cannot Claim

- Raw V3 and V4 MSE magnitudes are not directly comparable.
- The latent coordinates have different meanings, training losses, and decoders.
- The observation spaces differ: V3 uses proprioceptive vectors, while V4 uses
  RGB images.
- The training regimes differ: V3 is task-specific, while V4 is multi-task.
- The rollout sources differ: V3 uses online environment stepping with the V3
  policy, while V4 uses dataset trajectories as the true frame and action source.

## Confounds

- V4 uses dataset trajectories rather than an online V4 policy rollout.
- V4 is multi-task trained; the V3 baseline is task-specific.
- V4 uses a tokenizer and image decoder; V3 uses proprioceptive observations
  and vector decoders.
- V4 actions are padded 16-dimensional vectors with action masks. V3 actions
  use the native Cheetah action space.
- The V4 image-PCA basis is fitted only on V4 true frames. It supports within-V4
  image analysis, but it does not align V3 and V4 observations.
- Channel-mean RGB can miss spatially large errors that preserve global color.

## Paper Framing

The paper should focus on qualitative frequency structure rather than absolute
cross-architecture magnitudes.

The most defensible claim is that we compare the fractional distribution of
imagination error across temporal frequency bands after applying the same PCA
and Butterworth decomposition to each architecture's native latent state.

The key evidence is band ranking and normalized band share, especially the
share assigned to `dc_trend`.

V4 image-space results should be presented as secondary checks, not as the
headline cross-architecture metric.
