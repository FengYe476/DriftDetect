# Month 7 Summary - DriftDetect

## Executive Summary

Month 7 completed the transition from diagnosing SMAD's fixed-basis failure to
implementing and evaluating Adaptive-SMAD. Week 1 designed and implemented the
adaptive basis machinery. Week 2 numerically supported the posterior/drift
mismatch conjecture in a linear toy system and wired Adaptive-SMAD into the
training workflow. Week 3 produced first drafts of the Introduction and
Findings sections. Week 4 completed the Adaptive-SMAD proof-of-concept training
run and the final target decision.

The central result is that Adaptive-SMAD clears the intervention threshold by a
large margin. At the fair 500k checkpoint, `J_slow` (`dc_trend` MSE) is reduced
by `65.2%` with damping disabled at inference and by `69.0%` with damping
enabled, relative to the Month 5 Phase 2 baseline. The run also improves task
performance, reaching peak `eval_return=850.1` versus the Phase 2 baseline peak
of `746.7`. The result supports a Grade D target: an 85% paper target for
NeurIPS 2027, contingent on controlled same-environment validation in Month 8.

## Deliverables Status

| Deliverable | Status | Notes |
|---|---|---|
| docs/adaptive_smad_design.md | Complete | Adaptive-SMAD design based on B1 staleness evidence |
| src/smad/adaptive_smad/ | Complete | Re-estimator, scheduler, package exports |
| tests/test_adaptive_smad.py | Complete | No-op identity, cadence, state restore, PCA path |
| configs/adaptive_smad_v1.yaml | Complete | Cheetah Run single-seed config |
| docs/mismatch_framing_strategies.md | Complete | Recommended Framing B |
| scripts/run_toy_mismatch_theorem_check.py | Complete | NumPy-only theorem-check script |
| docs/mismatch_theorem_exploration.md | Complete | Results filled after script run |
| scripts/train_adaptive_smad.py | Complete | Adaptive-SMAD training launcher |
| scripts/runpod_adaptive_smad_v1.sh | Complete | RunPod orchestration script |
| scripts/run_b1_post_smad_drift.py update | Complete | Adaptive checkpoint `current_U` inference support |
| paper/sections/01_introduction.md | Complete | First draft committed |
| paper/sections/04_findings.md | Complete | First draft committed |
| Adaptive-SMAD RunPod training | Complete | Cheetah Run proof-of-concept |
| docs/final_target_decision.md | Complete | Grade D recommendation |
| docs/month7_summary.md | This file | Month 7 summary |

## Key Metrics

### Adaptive-SMAD Re-estimation Trajectory

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

`overlap_prev` stabilized in the `0.54-0.75` range after the first update,
showing coherent basis tracking. `overlap_baseline` stayed near `0.016-0.022`,
confirming that the active drift subspace remained nearly orthogonal to the
original baseline basis.

### J_slow and Frequency Metrics

| Condition | dc_trend MSE | dc_trend share | Reduction vs Phase 2 baseline |
|---|---:|---:|---:|
| Phase 2 baseline (Month 5) | 0.184 | 46.7% | - |
| Adaptive-SMAD, eta=0.0 | 0.064 | 15.3% | -65.2% |
| Adaptive-SMAD, eta=0.20 | 0.057 | 14.1% | -69.0% |

The same-checkpoint inference comparison gives an additional `-10.9%`
`dc_trend` reduction for `eta=0.20` relative to `eta=0.0`.

### Eval Return

| Run | Peak eval_return | Notes |
|---|---:|---|
| Phase 2 baseline | 746.7 | Month 5 reference |
| Adaptive-SMAD | 850.1 | Step 555k; late training stable 780-846 |

Adaptive-SMAD preserved and improved task performance in the proof-of-concept
run.

### Mismatch Theorem Check

| Sweep | Result |
|---|---|
| Capacity gap | overlap fell from 0.9993 to 0.6183 |
| Pearson correlation | r = -0.9253 |
| Noise structure | isotropic overlap 0.9792; structured overlap ~0.664-0.666 |

Candidate Theorem B' is numerically supported in the linear setting.

## Finding Status

| Finding | Status | Month 7 Update |
|---|---|---|
| A: Frequency-asymmetric drift | SUPPORTED | Unchanged |
| B: Cross-architecture trend dominance | SUPPORTED | Unchanged |
| C: Training dynamics | SUPPORTED | Unchanged |
| D: SMAD intervention | UPGRADED - Adaptive-SMAD POC SUPPORTED | `J_slow` reduction `65-69%`, eval_return improved |
| E: Decoder amplification | CROSS-ARCH CONFIRMED | Unchanged |
| Mismatch | SUPPORTED with mechanism | Framing B selected; linear toy theorem check supports capacity-gap mechanism |
| Basis staleness | CONFIRMED | Adaptive tracking resolves fixed-basis staleness in Cheetah POC |

## Key Discoveries

1. Adaptive-SMAD directly addresses the B1 staleness failure. The active basis
   remains far from the initial baseline basis while adjacent updates become
   moderately aligned.
2. The intervention effect is much larger than fixed-basis SMAD: `65-69%`
   `J_slow` reduction versus the Phase 2 fixed-basis `4.8%`.
3. Damping-on inference gives an additional same-checkpoint `10.9%` improvement
   over the Adaptive-SMAD-trained model with damping disabled.
4. Adaptive-SMAD improved task performance in the proof-of-concept run,
   reaching peak `eval_return=850.1`.
5. The run's accidental extension to `645k` steps was informative: overlap and
   eval_return continued improving past 500k.
6. Candidate Theorem B' gives a numerical mechanism for posterior/drift
   mismatch: overlap decreases with capacity gap and inference-noise
   anisotropy.
7. The paper trajectory is now diagnosis-to-intervention, not diagnosis-only.

## Compute Budget

| Item | GPU-hours | Cost |
|---|---:|---:|
| Months 1-6 total | ~72 | ~$50 |
| Month 7 local design, implementation, theorem check, writing | ~0 | $0 |
| Month 7 Adaptive-SMAD RunPod training | ~20 | ~$14 |
| Project total through Month 7 | ~92 | ~$64 |
| Remaining budget | ~108 | ~$76 |

Month 8 controlled validation is estimated at approximately `120` GPU-hours.
This is slightly above the rounded remaining budget, so Month 8 should monitor
cost closely and prioritize same-environment controls over optional task
expansion if needed.

## Plan vs Actual

| Month 7 Plan | Actual |
|---|---|
| Adaptive-SMAD design | Complete |
| Adaptive-SMAD implementation | Complete with tests |
| Mismatch framing strategy | Complete; Framing B selected |
| Toy mismatch theorem exploration | Complete; conjecture numerically supported |
| Wire Adaptive-SMAD into training launcher | Complete |
| RunPod launcher for Adaptive-SMAD | Complete |
| Paper Section 1 and 4 drafts | Complete and committed |
| Adaptive-SMAD first training run | Complete; strong POC result |
| Final target decision | Complete; Grade D recommended |

## Month 8 Readiness

- [x] Adaptive-SMAD proof-of-concept complete
- [x] Re-estimation trajectory analyzed
- [x] Frequency metrics at 500k checkpoint analyzed
- [x] Target decision documented
- [x] Paper Introduction and Findings drafts complete
- [x] Mismatch framing and theorem-check documents complete
- [ ] Controlled same-pod baseline + Adaptive-SMAD comparison
- [ ] Multi-seed/multi-task validation
- [ ] Theorem 1 formalization completion
- [ ] Paper Section 2 draft
- [ ] Paper Section 3 draft
- [ ] Paper Section 5 draft

## Git Tag

Month 7 release tag: `v0.7-month7-adaptive-smad-poc-and-decision`
