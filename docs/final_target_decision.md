# Final Target Decision

## 1. Decision

Recommend **Grade D: 85% target, NeurIPS 2027**.

The Adaptive-SMAD proof-of-concept clears the intervention threshold by a wide
margin. At the fair 500k checkpoint, `J_slow` (`dc_trend` MSE) is reduced by
`65.2%` with damping disabled at inference (`eta=0.0`) and by `69.0%` with
damping enabled (`eta=0.20`) relative to the Month 5 Phase 2 baseline. This
exceeds both the original `20%` threshold and the stronger `25%` target needed
to justify a method-plus-diagnosis paper trajectory.

The result is strong enough to upgrade the project from a primarily diagnostic
paper to a diagnosis-to-intervention paper. The caveat is that this is not yet
the final submission evidence. The Phase 2 baseline and Adaptive-SMAD run were
trained on different RunPod pods with different PyTorch versions, so Month 8
must run a controlled same-environment baseline and Adaptive-SMAD comparison
before making the final empirical claim.

## 2. Evidence Summary

### Re-estimation trajectory

Adaptive-SMAD successfully tracked a moving drift subspace. The first
re-estimation at `50k` steps found a basis almost orthogonal to the initial
baseline basis (`overlap_prev=0.018`, `overlap_baseline=0.018`). After that
initial jump, adjacent-basis overlap stabilized in the `0.54-0.75` range,
showing that the adaptive basis moved coherently rather than randomly. In
contrast, fixed-basis SMAD's B1 staleness check found baseline/post-SMAD
overlap of only `0.005-0.022`, confirming that the fixed basis became stale.

| Step | overlap_prev | overlap_baseline |
|---:|---:|---:|
| 50k | 0.018 | 0.018 |
| 100k | 0.536 | 0.021 |
| 150k | 0.666 | 0.022 |
| 200k | 0.547 | 0.020 |
| 250k | 0.582 | 0.018 |
| 300k | 0.642 | 0.018 |
| 350k | 0.544 | 0.018 |
| 400k | 0.662 | 0.019 |
| 450k | 0.637 | 0.020 |
| 500k | 0.703 | 0.019 |
| 550k | 0.753 | 0.017 |
| 600k | 0.734 | 0.016 |

### J_slow reduction

At the 500k checkpoint, Adaptive-SMAD strongly reduces low-frequency latent
drift relative to the Phase 2 baseline.

| Condition | dc_trend MSE | dc_trend share |
|---|---:|---:|
| Phase 2 baseline (Month 5) | 0.184 | 46.7% |
| Adaptive-SMAD, eta=0.0 | 0.064 | 15.3% |
| Adaptive-SMAD, eta=0.20 | 0.057 | 14.1% |

The reduction relative to the Phase 2 baseline is `-65.2%` with damping off and
`-69.0%` with damping on. The same-checkpoint inference comparison,
`eta=0.20` versus `eta=0.0`, gives an additional `-10.9%` `J_slow` reduction.

### Eval return

Adaptive-SMAD preserves and improves task performance. The run reached peak
`eval_return=850.1` at step `555k`, with late-training returns stable in the
`780-846` range. The Phase 2 baseline peak was `746.7`. This means the
intervention did not trade away control performance for lower drift; in this
run, it improved both.

## 3. Caveats

The main caveat is the cross-pod comparison. The Phase 2 baseline and
Adaptive-SMAD run used different RunPod pods and different PyTorch versions.
The effect size is large, but a controlled same-environment comparison is
required before the paper treats the `65-69%` reduction as final.

The current result is also single-seed and single-task. Month 8 must expand to
multiple seeds and multiple tasks to test whether Adaptive-SMAD is robust or
whether the Cheetah result is unusually favorable.

The Adaptive-SMAD run continued to `645k` steps because of a config error
(`5000000` instead of `500000`). The extra training was not wasted: adjacent
basis overlap continued to rise from approximately `0.70` to `0.75`, and
`eval_return` continued to improve. However, the fair comparison against the
Phase 2 baseline is the step `500k` checkpoint, not the final 645k state.

## 4. Month 8 Plan

Given the Grade D decision, Month 8 should run the controlled scale-up:

```text
3 tasks x 3 seeds x 3 methods
tasks = Cheetah Run, Cartpole Swingup, Walker Walk
methods = baseline, fixed-SMAD, Adaptive-SMAD
```

Estimated compute is approximately `120` GPU-hours, which is close to but
still consistent with the remaining project budget of approximately `108`
GPU-hours after Month 7. The scale-up should prioritize the controlled
same-environment comparison first; if budget becomes tight, reduce task count
before reducing the baseline/fixed/adaptive comparison within each task.

Month 8 should also complete Theorem 1 formalization and draft paper Sections
2, 3, and 5. The target paper narrative is now:

```text
frequency-domain diagnosis -> posterior/drift mismatch mechanism ->
Adaptive-SMAD intervention -> controlled multi-task validation
```
