# Toy Experiment Results

## Complexity Sweep

**Date:** 2026-05-02

### Setup

Three environment complexity levels, same RSSM architecture (deter_dim=64, stoch_dim=8), 2000 training steps each:

- Simple: 2 oscillators, obs_dim=4
- Medium: 5 oscillators, obs_dim=10, frequencies [0.3, 0.7, 2.0, 5.0, 12.0] Hz
- Hard: 10 oscillators, obs_dim=20, frequencies [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0] Hz

### Results

| Complexity | obs_dim | dc_trend% | very_low% | low% | mid% | high% | recon_loss |
|---|---|---|---|---|---|---|---|
| simple | 4 | 24.1% | 28.3% | 24.1% | 18.1% | 5.4% | 0.25 |
| medium | 10 | 46.6% | 8.8% | 4.2% | 25.3% | 15.0% | 1.37 |
| hard | 20 | 31.1% | 20.6% | 6.0% | 15.4% | 27.0% | 3.63 |

### Interpretation

dc_trend dominance follows an inverted-U relationship with environment complexity:

1. Simple (capacity >> complexity): Model learns well, residual drift is uniformly distributed. dc_trend=24.1%.
2. Medium (capacity ~ complexity): Model learns fast dynamics but accumulates systematic bias in slow trends. dc_trend peaks at 46.6% -- matching Cheetah Run's 46.7%.
3. Hard (capacity << complexity): Model cannot learn basic structure, drift becomes broadband noise. dc_trend drops to 31.1%, high-frequency error rises to 27.0%.

This suggests that dc_trend dominated drift is a property of the capacity-complexity sweet spot, not an inherent RSSM artifact. DreamerV3's 512-dim RSSM on Cheetah (17-dim obs) sits in this sweet spot. DreamerV4's transformer world model on Cheetah also sits in a similar regime, explaining why both show dc_trend dominance (Finding B).

### SMAD Applicability

SMAD is most effective when the task falls in the medium-complexity regime where dc_trend dominates. For simple tasks (well-learned dynamics) or very hard tasks (broadband errors), directional damping provides less targeted benefit.

### Paper Impact

This result provides a mechanistic explanation for WHY dc_trend dominates in DreamerV3/V4 Cheetah experiments: the model is in a capacity-complexity regime where slow-trend bias accumulation exceeds fast-frequency prediction error. This replaces the vague "architecture-intrinsic" explanation with a testable capacity-complexity hypothesis.

### SMAD Damping Across Complexity Levels

SMAD damping (eta=0.20, U_drift estimated per level, r=10) was applied at each complexity:

| Complexity | dc_trend% | J_slow reduction | J_total reduction |
|---|---|---|---|
| simple | 24.1% | -1.2% (no effect) | -149.8% (harmful) |
| medium | 46.6% | +15.6% | +21.9% |
| hard | 31.1% | +14.1% | +30.5% |

Key findings:

1. On the simple task, SMAD is harmful: the model is already well-fitted, and damping introduces systematic bias that increases total error by 150%.

2. On the medium task (dc_trend=46.6%, matching Cheetah), SMAD provides the best dc_trend-specific improvement (15.6%), confirming the method works best when drift is trend-concentrated.

3. On the hard task, SMAD provides the best total error reduction (30.5%) despite lower dc_trend concentration. This suggests that damping along drift PCA directions acts as a generic drift reducer even when drift is not frequency-concentrated.

4. The inverted pattern (J_total reduction highest on hard, J_slow reduction highest on medium) reveals that SMAD has two operating modes:
   - **Targeted mode** (medium complexity): drift is dc_trend concentrated, damping reduces slow-subspace error specifically.
   - **Generic mode** (hard complexity): drift is broadband, damping along top drift PCs reduces overall error regardless of frequency structure.

5. The simple-task failure is an important safety check: SMAD should NOT be applied to well-fitted models where imagination drift is small and unstructured.
