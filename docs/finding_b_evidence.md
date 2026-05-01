# Finding B Evidence: Cross-Architecture Trend-Dominated Drift

## Finding B Statement

Trend-dominated imagination drift is a cross-architectural property of
autoregressive world models, not specific to DreamerV3's GRU/tanh RSSM.

## Evidence Summary

Month 4 compared DreamerV3 and DreamerV4 on Cheetah Run using the same
frequency-domain diagnostic protocol:

1. Use each architecture's native 512-dimensional latent state.
2. Pool true latent trajectories across 20 seeds.
3. Fit PCA on pooled true latents with 95 percent variance retained.
4. Project true and imagined latents into the shared PCA basis.
5. Decompose each projected trajectory into the same five temporal bands.
6. Compute band MSE over the trim window `[25, 175]`.
7. Compare fractional band shares within each architecture.

The key result is that `dc_trend` is the dominant band in both architectures.

V3 baseline:

- Architecture: DreamerV3 GRU-RSSM.
- Observation space: 17-dimensional proprioceptive Cheetah state.
- Latent used for diagnostics: 512-dimensional deterministic RSSM state.
- `dc_trend` share: 46.7 percent of latent MSE.
- Band ranking: `dc_trend > low > very_low > mid > high`.
- High-frequency share: 4.2 percent.

V4 comparison:

- Architecture: transformer tokenizer plus diffusion-forcing denoising
  dynamics.
- Observation space: image-based, RGB `(3, 128, 128)`.
- Latent used for diagnostics: 512-dimensional packed tokenizer latent,
  flattened from `(8, 64)`.
- PCA retained 35 components and explained 95.07728441475489 percent variance.
- `dc_trend` share: 63.85715847509169 percent of latent MSE.
- Band ranking: `dc_trend > low > mid > high > very_low`.
- High-frequency share: 4.181199605250795 percent.

Exact V4 latent-space shares from `results/tables/v4_band_errors.json`:

| Band | Mean MSE `[25, 175]` | Share |
|---|---:|---:|
| `dc_trend` | 1.5370213002262525 | 63.85715847509169% |
| `very_low` | 0.04482453766198742 | 1.8622823279240832% |
| `low` | 0.4884244974135168 | 20.292106901746136% |
| `mid` | 0.2360574231797595 | 9.807252689987296% |
| `high` | 0.10064013193250443 | 4.181199605250795% |

Cross-architecture comparison table:

| Band | V3 latent share | V4 latent share |
|---|---:|---:|
| `dc_trend` | 46.7% | 63.9% |
| `very_low` | 10.4% | 1.9% |
| `low` | 28.4% | 20.3% |
| `mid` | 10.3% | 9.8% |
| `high` | 4.2% | 4.2% |

Two details are especially important: the top two bands are identical
(`dc_trend > low`), and the high-frequency error share is nearly identical
(V3 4.2 percent, V4 4.181199605250795 percent).

This is the specific pattern expected if open-loop imagination accumulates
slow, trend-like errors rather than injecting primarily high-frequency noise.

## Strength of the Finding

The V4 result is stronger than a minimal cross-architecture replication.

V4's `dc_trend` share is higher than V3's, not lower:

```text
V3 dc_trend share = 46.7%
V4 dc_trend share = 63.85715847509169%
```

This is the strongest practical version of Finding B. Trend dominance is not
merely detectable in a transformer-denoising world model; it is amplified
relative to the V3 baseline in the primary latent-space diagnostic.

The pattern holds across different dynamics mechanisms:

- V3 uses a single-step GRU RSSM transition.
- V4 uses iterative denoising over packed tokenizer latents.

The pattern holds across different observation spaces:

- V3 is proprioceptive.
- V4 is image-based.

The pattern holds across different latent representations:

- V3 uses an RSSM deterministic hidden state.
- V4 uses a tanh-bounded tokenizer bottleneck state.

The dimensionality match, 512 vs 512, is useful for applying the same PCA and
filtering protocol. It is not what explains the result. The states are learned
under different objectives and decoded through different mechanisms.

## Secondary Image-Space Evidence

Month 4 also ran V4 image-space diagnostics as supplementary evidence.

Channel-mean RGB reduces each frame to `(R_mean, G_mean, B_mean)`. This is very
lossy, but architecture-neutral within V4 image rollouts. It produced:

- `dc_trend` share: 68.06643734275672 percent.
- Ranking: `dc_trend > low > very_low > mid > high`.

Image PCA flattens RGB frames to 49,152 dimensions, fits a randomized PCA basis
on pooled true frames, caps the basis at 50 PCs, and applies the same
five-band decomposition. It produced:

- PCA retained 50 components.
- Variance explained by the capped image-PCA basis: 81.11482382446757 percent.
- `dc_trend` share: 44.13059258253914 percent.
- Ranking: `dc_trend > low > mid > high > very_low`.

Exact V4 image-PCA shares:

| Band | Share |
|---|---:|
| `dc_trend` | 44.13059258253914% |
| `very_low` | 2.7140563751127593% |
| `low` | 25.20473964433875% |
| `mid` | 15.866170757958142% |
| `high` | 12.08444064005121% |

The image-PCA result is weaker than the latent-space result but still has
`dc_trend` as the top band. This supports the qualitative pattern while also
showing that some of the 63.9 percent latent dominance may be
representation-specific.

## Confounds and Limitations

The comparison is not perfectly controlled.

V4 uses dataset trajectories, while V3 uses online policy rollouts. The action
distribution therefore differs.

V4 is multi-task trained, while V3 is single-task. Training data volume,
diversity, and supervision differ.

V4 latent state is a tanh-bounded tokenizer output. V3 `deter` is a GRU hidden
state. These are different latent semantics.

The 512-dimensional latent match is coincidental, not by design.

Image-space PCA comparison shows `dc_trend` at 44.13059258253914 percent, lower
than the V4 latent-space value of 63.85715847509169 percent. This suggests that
some latent-space trend dominance may be amplified by the tokenizer latent
representation.

Only one task, `cheetah-run`, was compared cross-architecturally. The result is
strong for Cheetah but does not yet establish a task-universal property for V4.

Absolute MSE magnitudes should not be compared between V3 and V4. The meaningful
comparison is the normalized band-share pattern and the qualitative ranking.

## Paper Narrative Impact

Finding B upgrades the paper's main empirical claim.

Before Month 4, the strongest claim was:

```text
DreamerV3 has trend-dominated latent drift.
```

After Month 4, the stronger claim is:

```text
Autoregressive imagination generically produces trend-dominated drift.
```

The word "generically" still needs careful framing: the evidence covers two
architectures and one cross-architecture task, not every world model. But the
mechanistic diversity is large enough to make the claim meaningfully broader
than a DreamerV3-specific artifact.

Combined with Finding C, this result points toward a structural explanation.
Finding C showed that `dc_trend` dominance is present from step 5000 in V3
training, before policy convergence. Finding B shows the same qualitative
pattern in a transformer-denoising V4 world model. Together, they suggest that
slow drift is a general failure mode of open-loop autoregressive imagination.

This makes Theorem 1 and the slow-subspace identifiability story more relevant.
The theory should not be framed as explaining only a GRU/tanh RSSM quirk. It
should be framed as a simplified account of why slow components can become
identifiable and consequential in autoregressive world-model rollouts.

Finding B is therefore supported. Since DreamerV3's corresponding value is 46.7 percent,
the V4 result does not weaken the Month 2 story. It strengthens it. Trend-dominated
imagination drift survives a shift from recurrent RSSM dynamics to transformer-denoising
dynamics, from proprioception to images, and from RSSM deter latents to tokenizer packed latents.
