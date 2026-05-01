# Month 3 Summary - DriftDetect

## Executive Summary

Month 3 converted DriftDetect's Month 2 frequency-domain finding into a stronger mechanistic story. Decoder amplification experiments D1-D3 tested whether the observation-space growth seen in Month 2 came from selective decoder amplification of `dc_trend` latent errors; the answer was no. The trained decoder amplifies off-distribution latent drift overall, but not preferentially along the trend direction, so trend dominance is best interpreted as latent-intrinsic. In parallel, the training-dynamics pipeline analyzed 102 high-frequency Cheetah checkpoints and showed that `dc_trend` dominance is present from the first checkpoint at step 5000, when the policy is still weak. This supports Finding C: trend-dominated latent drift is an architecture-intrinsic property of DreamerV3's RSSM dynamics, not a late training artifact. Month 3 also completed DreamerV4 deep reconnaissance and made a conditional GO decision for Month 4 cross-architecture comparison.

## Deliverables Status

| Deliverable | Status | Notes |
|---|---|---|
| D1 random decoder control | Complete | `scripts/run_decoder_d1.py`; trained decoder amplifies drift about 48% more than random decoder |
| D2 Lipschitz probe | Complete | `scripts/run_decoder_d2.py`; `dc_trend` has the smallest effective Lipschitz constant |
| D3 linear/nonlinear separation | Complete | `scripts/run_decoder_d3.py`; decoder nonlinearity appears as constant OOD noise |
| Finding E verdict | Complete | `docs/finding_decoder_amplification.md`; original directional decoder-amplification hypothesis not supported |
| High-frequency Cheetah checkpoint archive | Complete | 102 archived checkpoints from step 5k to 510k |
| Training dynamics pipeline | Complete | `scripts/run_training_dynamics.py`; 102 checkpoints x 5 bands |
| Figure 3 | Complete | Curves, heatmap, and share plots generated in `results/figures/` |
| Finding C evidence | Complete | `dc_trend` dominance present from step 5000 and persists through training |
| Cartpole training dynamics decision | Complete | `docs/cartpole_dynamics_decision.md`; decision to skip redundant Cartpole dynamics run |
| V4 HF deep recon | Complete | `docs/v4_hf_deep_recon.md`; code-level DreamerV4 analysis |
| V4 go/no-go decision | Complete | `docs/v4_decision.md`; conditional GO for Month 4 inference-only V4 comparison |
| `month3_summary.md` | Complete | This file |

## Key Metrics

### D1: Random Decoder Control

- Evaluated 20 Cheetah v2 rollout seeds.
- Compared trained decoder against an architecture-identical random decoder.
- Step 25 obs drift: trained 12.16, random 8.80, ratio 1.38.
- Step 100 obs drift: trained 14.23, random 8.87, ratio 1.60.
- Step 175 obs drift: trained 13.43, random 8.20, ratio 1.64.
- Overall mean ratio over trim window `[10, 189]`: 1.479.
- Interpretation: trained decoder amplifies drift overall, and the ratio grows with horizon.

### D2: Lipschitz Probe

Effective Lipschitz constants at `eps = 1.0`:

| Band | Effective Lipschitz |
|---|---:|
| `dc_trend` | 1.221 |
| `very_low` | 1.666 |
| `low` | 2.140 |
| `mid` | 2.344 |
| `high` | 2.168 |

- `dc_trend` has the smallest decoder response, not the largest.
- Constants are nearly flat across epsilon values `[0.01, 0.05, 0.1, 0.5, 1.0]`.
- Interpretation: the decoder is locally approximately linear and does not preferentially amplify trend direction.

### D3: Linear/Nonlinear Separation

- Compared `decode(imag) - decode(true)` against `decode(imag - true)` across 20 Cheetah rollouts.
- Nonlinearity gap is approximately 10-11 and nearly constant across horizon.
- Actual observation difference grows from about 2 to about 15.
- Overall ratio is approximately 1.008.
- Scatter plot shows a horizontal band: the gap is uncorrelated with actual observation-difference magnitude.
- Interpretation: nonlinear decoder contribution is an OOD noise floor, not a horizon-growing amplification mechanism.

### Training Dynamics

- Checkpoints analyzed: 102 Cheetah checkpoints from step 5k to 510k.
- Rollouts per checkpoint: 1 seed.
- Horizon: 200.
- Trim window: `[25, 175]`.
- Frequency bands: 5.
- Output: `results/tables/v3_cheetah_training_dynamics.npz`.
- Runtime: about 2 hours on local MPS.
- Step 5000 `dc_trend` share: 79%.
- Step 500k `dc_trend` share: about 33%.
- Main result: `dc_trend` is the top band throughout training.

### Compute and Cost

- Month 3 high-frequency training: 2 Cheetah training runs, about 20 GPU-hours total.
- Month 3 GPU cost: about $14.
- Local MPS analysis time in Month 3: about 4 hours.
- Total project spend through Month 3: about $27.65.
- Remaining budget: about 160.5 GPU-hours, about $112.

## Finding C: Training Dynamics

**SUPPORTED - `dc_trend` dominance is architecture-intrinsic to DreamerV3 RSSM imagination.**

The original training-dynamics question was when trend dominance emerges. A late-emergence result would have suggested that saturation, overfitting, or converged policy behavior created the frequency asymmetry. The Month 3 result rules that out for Cheetah Run: `dc_trend` dominance is already present at step 5000, when `eval_return` is only about 28.

Evidence chain:

1. A high-frequency Cheetah checkpoint archive was created with checkpoints every 5000 steps from early training through 510k steps.
2. The retrained high-frequency run reproduced the Month 1 baseline quality, reaching peak `eval_return` 791.3 at step 485k, matching the Month 1 peak of 791.1.
3. The training-dynamics pipeline loaded each checkpoint, extracted a rollout, ran frequency decomposition, and computed per-band MSE over the same 200-step horizon.
4. Figure 3A shows `dc_trend` MSE above the other bands across training on a log scale.
5. Figure 3B shows a consistently bright `dc_trend` heatmap row across checkpoint steps.
6. Figure 3C shows the fractional share starts at 79% at step 5k and declines to about 33% by step 500k, while remaining the largest band.

Conclusion:

Training does not produce trend dominance; it slightly alleviates it. The best current interpretation is that DreamerV3's GRU/tanh RSSM architecture has an inherent slow-mode error bias during open-loop imagination. Optimization improves the model and reduces the extremity of that bias, but does not eliminate the structural tendency for low-frequency drift to dominate.

Impact on the paper:

- Finding C strengthens Finding A by showing the frequency asymmetry is not just a property of a mature checkpoint.
- The result reframes the saturation hypothesis: saturation may affect magnitude, but it is not the origin of `dc_trend` dominance.
- Figure 3 becomes the primary evidence that the phenomenon is architecture-intrinsic within DreamerV3.

## Finding E: Decoder Amplification

**NOT SUPPORTED as originally stated.**

Original hypothesis:

The trained decoder selectively amplifies `dc_trend` latent drift, causing small trend-direction latent errors to become disproportionately large observation-space errors.

D1-D3 evidence chain:

1. D1 confirmed that a trained decoder amplifies drift more than a random decoder. The trained/random ratio averaged 1.479 over the analysis window and increased with horizon.
2. D2 tested directionality directly by perturbing true latents along each frequency-band direction. `dc_trend` had the smallest effective Lipschitz constant at `eps = 1.0`, while `mid` and `high` bands had larger responses.
3. D2 also showed that Lipschitz constants are nearly flat across epsilon, indicating approximately linear local decoder behavior.
4. D3 separated linear and nonlinear effects. The nonlinearity gap stayed near 10-11 while the actual observation difference grew from about 2 to about 15.
5. D3 scatter results showed the gap forms a horizontal band, consistent with constant OOD noise rather than a growing mechanism.

Revised interpretation:

The decoder does amplify off-distribution latent inputs, but it does so broadly rather than selectively along `dc_trend`. Trend dominance is already present in latent space and therefore originates upstream in RSSM open-loop dynamics. The decoder converts latent errors into larger observation-space errors, but it does not explain the frequency structure of those errors.

Paper implication:

Finding E should not appear as an independent main finding. D1-D3 are still valuable as a supplementary falsification record because they rule out an intuitive but incorrect explanation for observation-space drift growth. The main paper can say that decoder amplification exists as a uniform OOD instability, while the trend-dominant frequency structure is latent-intrinsic.

## Plan vs Actual

| Month 3 Plan | Actual | Outcome |
|---|---|---|
| Test decoder amplification hypothesis | Completed D1-D3 with random decoder, Lipschitz probe, and linear/nonlinear separation | Hypothesis falsified in its directional form |
| Establish Finding E if supported | Finding E not supported as an independent main finding | Revised interpretation documented |
| Launch high-frequency checkpoint training | Completed after one failed foreground archive attempt and one successful `nohup` retrain/archive run | 102 checkpoints archived |
| Run training dynamics across intermediate checkpoints | Completed over all 102 Cheetah checkpoints | Finding C supported |
| Generate Figure 3 | Completed curves, heatmap, and share plots | Ready for paper draft |
| Decide whether to run Cartpole dynamics | Decision documented: skip | Avoided redundant compute |
| Conduct V4 deep recon | Completed code-level recon of `external/dreamer4/` | Month 4 adapter risks clarified |
| Make V4 Month 4 decision | Completed conditional GO decision | V4 moves forward as inference-only target |

## Key Discoveries Not in Original Plan

1. `dc_trend` dominance appears at the first Cheetah checkpoint, step 5000, long before the policy becomes strong. This was stronger and earlier than expected.
2. Training reduces `dc_trend` share from 79% to about 33% rather than increasing it, revising the saturation hypothesis.
3. Decoder amplification exists, but it is not directional. This falsifies the most natural decoder-based explanation for Finding A.
4. The decoder nonlinearity gap behaves like a constant OOD floor rather than a horizon-growing mechanism.
5. The checkpoint archival process itself became a reproducibility lesson: foreground archive monitoring is unsafe over SSH, while `nohup` training plus `nohup` archiving worked.
6. Cartpole training dynamics were judged scientifically redundant after Cheetah answered the emergence question and Month 2 already established cross-task `dc_trend` dominance.
7. DreamerV4's imagination API is fundamentally different from V3's `img_step`: it is iterative denoising over token latents, not a single RSSM transition.
8. V4's tokenizer decoder creates a new cross-architectural decoder-amplification question: if V4 shows less observation-growth-despite-latent-plateau, decoder architecture may matter even though V3 directional amplification was not supported.

## Compute Budget

| Item | GPU-hours | Cost | Notes |
|---|---:|---:|---|
| Cheetah training (Month 1) | ~9.5 | ~$6.65 | DreamerV3 Cheetah checkpoint, peak 791.1 |
| Cartpole training (Month 2) | ~10 | ~$7.00 | DreamerV3 Cartpole checkpoint, peak ~854 |
| Month 3 high-frequency Cheetah training, attempt 1 | ~10 | ~$7.00 | Checkpoints lost when SSH/foreground archive process stopped |
| Month 3 high-frequency Cheetah training, attempt 2 | ~10 | ~$7.00 | Successful archive of 102 checkpoints |
| Local MPS analysis | ~4 local hours | $0 | Rollout extraction, D1-D3, training dynamics pipeline |
| Total spent | ~39.5 GPU-hours | ~$27.65 | Cloud GPU cost only |
| Remaining budget | ~160.5 GPU-hours | ~$112 | Sufficient for inference-only V4 Month 4 work |

## Risk Assessment for Month 4

1. V4 adapter complexity is the highest risk. V4 uses tokenizer latents plus iterative denoising, not a single-step RSSM `img_step`, so adapter parity with V3 is nontrivial.
2. Observation-space alignment is a medium risk. V3 diagnostics use proprio observations and 1536-dimensional RSSM features, while V4 is image-based with tokenized 128x128 frame latents.
3. Methodological comparison is a medium risk. A V4 image/token FFT can answer whether trend dominance appears in transformer-token imagination, but it is not a one-to-one proprio comparison.
4. V4 codebase stability is a medium risk. The inspected implementation is unofficial and may have checkpoint or API rough edges.
5. Compute budget is low risk. Inference-only V4 work is expected to need about 5-10 GPU-hours, well within the remaining budget.
6. Schedule risk is medium. If V4 adapter work is not functioning by Month 4 Week 2, the project should switch to a fallback rather than consume the full month.

## Month 4 Readiness

- [x] Finding A supported across Cheetah and Cartpole.
- [x] Finding C supported with 102-checkpoint training dynamics.
- [x] Finding E falsification record complete and documented.
- [x] Figure 1 complete for Month 2.
- [x] Figure 3 complete for Month 3.
- [x] V4 code-level recon complete.
- [x] V4 conditional GO decision documented.
- [x] RunPod CUDA path identified for V4 inference.
- [ ] DreamerV4 adapter: NOT STARTED.
- [ ] V4 checkpoint download and load smoke test: NOT STARTED.
- [ ] V4 rollout collection: NOT STARTED.
- [ ] V4 frequency decomposition and Figure 2: NOT STARTED.
- [ ] Finding B evidence document: NOT STARTED.

Month 4 fallback decision point:

If the V4 adapter is not working by Month 4 Week 2, switch to either PlaNet as a simpler RSSM-family comparison or additional DreamerV3 tasks such as Walker and Hopper to strengthen the single-architecture story.

## Git Tag

Month 3 release tag: `v0.3-month3-dynamics-decoder`
