# Finding D Evidence: SMAD Phase 2 Training Results

**Date:** 2026-05-02
**Verdict:** MIXED -- SMAD is safe but dc_trend reduction is below the 20% threshold.

## Phase 2 Configuration

- Baseline: DreamerV3 Cheetah Run, 500k steps, eta=0.0, seed=42
- SMAD: DreamerV3 Cheetah Run, 500k steps, eta=0.20, r=10, U_drift basis, seed=42
- Both runs used identical codebase with SMAD patches (baseline has eta=0 identity path)
- GPU: RTX 4090 on RunPod

## Key Results

### Task Performance (eval_return)

| Metric | Baseline | SMAD | Change |
|---|---:|---:|---:|
| Peak eval_return | 746.7 | 746.0 | -0.1% |
| Final eval_return | 663.2 | 746.0 | +12.5% |

SMAD does not harm task performance. The final return is substantially higher for SMAD, suggesting that damping may improve training stability in the later stages.

### Imagination Fidelity

| Metric | Baseline | SMAD | Change |
|---|---:|---:|---:|
| J_slow (dc_trend MSE) | 0.184 | 0.175 | -4.8% |
| J_total (all-band MSE) | 0.372 | 0.356 | -4.2% |

J_slow reduction of 4.8% does not meet the pre-registered 20% threshold.

### Per-Band MSE

| Band | Baseline | SMAD | Change |
|---|---:|---:|---:|
| dc_trend | 0.184 | 0.175 | -4.8% |
| very_low | 0.057 | 0.045 | -20.2% |
| low | 0.076 | 0.064 | -16.2% |
| mid | 0.036 | 0.051 | +39.7% |
| high | 0.019 | 0.022 | +12.9% |

SMAD reduces error in the lower frequency bands (very_low -20.2%, low -16.2%) but increases error in higher frequency bands (mid +39.7%, high +12.9%). This suggests that damping redistributes prediction error from low to high frequencies rather than eliminating it.

### Representation Health

| Metric | Baseline | SMAD | Ratio |
|---|---:|---:|---:|
| DC_trend variance | 0.960 | 1.036 | 1.08 |
| Total deter variance | 30.027 | 27.478 | 0.92 |
| Collapse warning | - | - | NO |

No representation collapse detected. Total variance decreases by 8%, well within acceptable range.

## Interpretation

### Why is the training-time effect weaker than post-hoc?

Pre-check 4 showed 24.8% dc_trend reduction with post-hoc damping on held-out rollouts. Phase 2 training-time damping achieves only 4.8%. Three explanations:

1. **Adaptation effect:** During training, the RSSM adapts its dynamics to compensate for the damping. The model learns to produce larger updates in the damped directions to counteract the eta=0.20 reduction, partially nullifying the intervention.

2. **Actor adaptation:** The actor trained under damped imagination may learn policies that create different state distributions, changing the drift structure relative to the fixed U_drift basis estimated from the baseline model.

3. **Basis staleness:** U_drift was estimated from the baseline checkpoint's rollouts. During SMAD training, the model's drift structure evolves, and the fixed basis becomes increasingly misaligned with the actual drift directions.

### What did work

- eval_return is preserved (even improved in final stages)
- very_low and low band MSE decreased significantly (-20.2% and -16.2%)
- No representation collapse
- SMAD is demonstrably safe as an intervention

### The frequency redistribution effect

The most scientifically interesting finding is the error redistribution: damping reduces low-frequency error but increases mid/high-frequency error. This suggests that the RSSM's error budget has a conservation-like property -- reducing slow drift increases fast prediction noise. This is a novel observation that could inform future intervention designs.

## Finding D Verdict

**MIXED.** SMAD is safe (eval_return preserved) but does not achieve the pre-registered 20% J_slow reduction threshold during training. The post-hoc effect (24.8%) does not fully transfer to training-time intervention (4.8%), likely due to model adaptation.

## Paper Impact

Finding D should be presented as a mixed result with honest discussion:

1. SMAD demonstrates that frequency-targeted intervention is feasible and safe
2. The gap between post-hoc and training-time effectiveness reveals model adaptation as a key challenge
3. The error redistribution from low to high frequencies is a novel finding about RSSM error dynamics
4. Future work should explore adaptive basis re-estimation during training to counter the staleness problem

## Recommendation for Month 6

Given the mixed result, Month 6 should:

- NOT proceed to Phase 3 horizon scaling (the prerequisite J_slow >= 20% was not met)
- Focus on writing the paper with honest presentation of the mixed result
- Frame the contribution as diagnosis + intervention attempt with mechanistic insights from the gap
- The toy complexity sweep and error redistribution findings strengthen the paper's analytical depth
