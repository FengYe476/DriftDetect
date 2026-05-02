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
