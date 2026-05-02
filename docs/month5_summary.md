# Month 5 Summary - DriftDetect

## Executive Summary

Month 5 completed SMAD Phase 2 training, Pre-checks 3-4, V4 D2 cross-architecture decoder test, and toy experiments. Finding D (SMAD intervention) is MIXED: the method is safe (eval_return preserved) but dc_trend reduction (4.8%) falls below the pre-registered 20% threshold. The gap between post-hoc effectiveness (24.8%) and training-time effectiveness (4.8%) reveals that the RSSM adapts to damping during training, partially nullifying the intervention. The toy complexity sweep discovered that dc_trend dominance follows an inverted-U pattern with environment complexity. V4 D2 confirmed that decoder non-directional amplification is a cross-architectural property.

## Deliverables Status

| Deliverable | Status | Notes |
|---|---|---|
| Pre-check 3a (SFA basis) | Complete | Posterior slow != drift confirmed as real finding |
| Pre-check 3b (Visual sanity) | Complete | SFA dirs genuinely slow, posterior dirs are not |
| Pre-check 3c (Trim sensitivity) | Complete | Results stable across trim windows |
| Pre-check 3 documentation | Complete | docs/smad_precheck_3_methodology.md |
| Pre-check 4 (Anti-tautology xval) | Complete | Transfer ratio 0.97-0.98, circularity resolved |
| Pre-check 4 documentation | Complete | docs/smad_precheck_4_anti_tautology.md |
| SMAD U_estimation module | Complete | src/smad/U_estimation.py |
| SMAD img_step_patch module | Complete | src/smad/img_step_patch.py, identity test passed |
| SMAD anchor_loss module | Complete | src/smad/anchor_loss.py, 7 tests passed |
| U_drift basis saved | Complete | results/smad/U_drift_cheetah_r10.npy |
| V4 D2 Lipschitz probe | Complete | dc_trend smallest in V4 (3.04), matching V3 (1.22) |
| V4 D2 documentation | Complete | docs/finding_e_v4_extension.md |
| Phase 2 training script | Complete | scripts/train_smad_phase2.py |
| Phase 2 training (RunPod) | Complete | Baseline + SMAD, 500k steps each, RTX 4090 |
| Phase 2 rollout extraction | Complete | 20 seeds x 2 runs |
| Phase 2 evaluation | Complete | results/tables/phase2_metrics.json |
| Finding D evidence | Complete | docs/finding_d_evidence.md, verdict MIXED |
| Toy experiment | Complete | Dual oscillator + minimal RSSM |
| Toy complexity sweep | Complete | Inverted-U dc_trend dominance pattern |
| Toy SMAD damping | Complete | Effective on medium complexity, harmful on simple |
| Month 5 summary | Complete | This file |

## Key Metrics

### Phase 2 Training Results

| Metric | Baseline | SMAD (eta=0.20) | Change |
|---|---:|---:|---:|
| Peak eval_return | 746.7 | 746.0 | -0.1% |
| Final eval_return | 663.2 | 746.0 | +12.5% |
| J_slow (dc_trend MSE) | 0.184 | 0.175 | -4.8% |
| J_total (all-band MSE) | 0.372 | 0.356 | -4.2% |
| very_low MSE | 0.057 | 0.045 | -20.2% |
| low MSE | 0.076 | 0.064 | -16.2% |
| mid MSE | 0.036 | 0.051 | +39.7% |
| high MSE | 0.019 | 0.022 | +12.9% |
| DC_trend variance ratio | - | - | 1.08 (healthy) |
| Collapse warning | - | - | NO |

### Pre-check Results

| Pre-check | Result |
|---|---|
| 3a: SFA vs posterior vs drift overlap | All near-orthogonal (0.003-0.283) |
| 3b: SFA derivative variance | SFA: 0.004, Posterior: 1.6, Drift: 0.18 |
| 3c: Trim sensitivity | Max deviation ~0.03 across windows |
| 4: Cross-validation transfer ratio | 0.97-0.98 |

### V4 D2 Lipschitz Probe

| Band | V3 Lipschitz | V4 Lipschitz |
|---|---:|---:|
| dc_trend | 1.221 (smallest) | 3.035 (smallest) |
| very_low | 1.666 | 4.093 |
| low | 2.140 | 5.206 |
| mid | 2.344 | 5.158 |
| high | 2.168 | 5.587 |

### Toy Complexity Sweep

| Complexity | obs_dim | dc_trend% | SMAD J_slow reduction | SMAD J_total reduction |
|---|---:|---:|---:|---:|
| simple | 4 | 24.1% | -1.2% (harmful) | -149.8% (harmful) |
| medium | 10 | 46.6% | +15.6% | +21.9% |
| hard | 20 | 31.1% | +14.1% | +30.5% |

## Finding Status

| Finding | Status | Evidence |
|---|---|---|
| A: Frequency-asymmetric drift | SUPPORTED | Month 2, unchanged |
| B: Cross-architecture trend dominance | SUPPORTED | Month 4 V4, unchanged |
| C: Training dynamics | SUPPORTED | Month 3, unchanged |
| D: SMAD intervention | MIXED | Phase 2: safe but J_slow -4.8% < 20% threshold |
| E: Decoder amplification | CROSS-ARCH CONFIRMED | V4 D2: dc_trend Lipschitz smallest in both architectures |

## Key Discoveries

1. Posterior slow features != drift directions is a real finding, not a methodological artifact. SFA confirms this independently.
2. Post-hoc damping (24.8% reduction) does not transfer to training-time damping (4.8%). The RSSM adapts to compensate.
3. Error redistribution: SMAD shifts error from low frequencies to high frequencies, suggesting an error budget conservation property.
4. dc_trend dominance follows an inverted-U pattern with environment complexity, peaking at medium complexity where model capacity matches task difficulty.
5. V4 tokenizer decoder also does not directionally amplify dc_trend, upgrading Finding E to cross-architecture status.

## Compute Budget

| Item | GPU-hours | Cost | Notes |
|---|---:|---:|---|
| Months 1-4 total | ~40.2 | ~$28.15 | Previous work |
| V4 D2 Lipschitz probe | ~0.5 | ~$0.35 | RunPod RTX 4090 |
| Phase 2 baseline training | ~10 | ~$7.00 | RunPod RTX 4090 |
| Phase 2 SMAD training | ~10 | ~$7.00 | RunPod RTX 4090 |
| Phase 2 rollout extraction | ~0.5 | ~$0.35 | RunPod RTX 4090 |
| Month 5 total | ~21 | ~$14.70 | |
| Project total | ~61.2 | ~$42.85 | |
| Remaining budget | ~138.8 | ~$97.15 | |

## Month 6 Plan

Given the MIXED Finding D result:

- Phase 3 (horizon scaling) is NOT recommended -- J_slow threshold not met
- Focus on paper writing with honest presentation of mixed results
- The paper's strength is in diagnosis (Findings A, B, C, E) not intervention
- SMAD serves as a proof-of-concept that frequency-targeted intervention is feasible and safe
- The post-hoc vs training-time gap and error redistribution are novel findings
- Theorem 1 formalization remains to be done
