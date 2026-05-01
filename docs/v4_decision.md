# V4 Cross-Architecture Go/No-Go Decision

**Date:** 2026-05-01

**Decision:** GO (conditional)

## Conditions for GO

- V4 adapter scope is limited to inference-only; no V4 training.
- Use HuggingFace multi-task checkpoints as-is; no additional V4 checkpoint training is needed.
- Observation space: Option A, image-space temporal FFT, is the primary path. Option B, physics state extraction, is a stretch goal.
- Minimum viable experiment: 20 rollouts from the V4 HF checkpoint on `cheetah_run`, run `freq_decompose` on image-space latent trajectories, and compare dc_trend share against V3.
- RunPod CUDA is required for V4 inference, with approximately 5-10 GPU-hours estimated.

## Scientific Questions V4 Answers

- Does dc_trend dominance exist in transformer-based world models?
- If yes, trend dominance is a general property of autoregressive imagination, not RSSM-specific.
- If no, trend dominance is GRU/tanh-specific, and transformer plus diffusion-forcing avoids it.
- Either answer strengthens the paper.

## Risk Assessment

- HIGH: V4 adapter implementation complexity, because iterative denoising is fundamentally different from single-step `img_step`.
- MEDIUM: image-space versus proprio-space comparison methodology.
- LOW: compute budget, because V4 inference is cheap at approximately 5-10 GPU-hours.
- MEDIUM: V4 codebase stability, because the implementation is unofficial and may have bugs.

## Month 4 Timeline

- Week 1: V4 adapter implementation and minimal example on RunPod.
- Week 2: V4 rollout collection and `freq_decompose`.
- Week 3: Cross-architecture comparison and Figure 2.
- Week 4: Finding B evidence document and Month 4 summary.

## Fallback if V4 Fails

- Option 1: PlaNet, a simpler RSSM variant with an easier adapter, to confirm Finding A across the RSSM family.
- Option 2: Skip cross-architecture comparison and strengthen the single-architecture story with additional DreamerV3 tasks such as Walker and Hopper.
- Decision point: if the V4 adapter is not working by Month 4 Week 2, switch to a fallback.

## New Question from Month 3

Can V4's tokenizer decoder help test the decoder amplification hypothesis cross-architecturally?

V4's quantized tokenizer decoder is fundamentally different from V3's MLP decoder. If V4 shows less observation-growth-despite-latent-plateau, it would confirm that decoder architecture matters for the amplification effect.
