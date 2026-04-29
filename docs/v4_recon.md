# DreamerV4 Reconnaissance

Date: 2026-04-29

## Repository Info

- Repository: `nicklashansen/dreamer4`
- Local path: `external/dreamer4/`
- Documented cloned commit: `bdeddfe`
- Local commit inspected: `bdeddfe` (`2026-03-24`, merge pull request #4)
- Maintenance status: recent and plausibly active. The README identifies the implementation as unofficial and incomplete relative to the original Dreamer 4 paper, but open to contributions.
- Scope note: this reconnaissance used the already-cloned local repository only. No training or model execution was performed.

## Architecture Differences

DreamerV3 in DriftDetect currently uses the NM512 RSSM implementation:

- Recurrent state-space model with deterministic and stochastic latent state.
- Open-loop imagination advances latent state via `dynamics.img_step()`.
- Proprio observations are decoded directly from RSSM features.
- The policy lives in the same agent object and can provide action sequences for rollout collection.

The inspected Dreamer4 implementation is structurally different:

- It uses a causal tokenizer plus an interactive dynamics model.
- The tokenizer and dynamics model both use block-causal transformer architecture.
- The tokenizer encodes image patches into low-dimensional latent tokens and decodes patches back to frames.
- The dynamics model operates over interleaved action, shortcut noise/step, and tokenizer representation sequences.
- Sampling is iterative denoising rather than a single RSSM transition.

Key API differences:

- V3 exposes a single `Dreamer` agent with environment policy, encoder, RSSM dynamics, and decoder.
- V4 separates tokenizer and dynamics checkpoints and uses helper functions such as `sample_one_timestep_packed()` for one-step latent generation.
- V4 appears image-first at 128x128 resolution, while our Month 1 V3 pipeline is proprio-only with 17-dimensional observations.
- V4 does not provide a directly equivalent actor/policy path for collecting environment actions in the same way the V3 agent does.

## Dependencies

The repo provides `environment.yaml` rather than `requirements.txt`.

Conda dependencies:

- `python=3.10`
- `glew=2.1.0`
- `glib=2.86.0`
- `pip=25.2`

Pip dependencies:

- `aiohttp==3.13.3`
- `pillow==12.0.0`
- `glfw==2.10.0`
- `gpustat==1.1.1`
- `lpips==0.1.4`
- `numpy==1.24.4`
- `tensordict==0.10.0`
- `torch==2.8.0`
- `torchvision==0.23.0`
- `torchrl==0.10.0`
- `transformers==4.56.2`
- `wandb==0.22.1`

Compatibility with `env-dreamerv3`:

- A separate conda environment is strongly recommended.
- Dreamer4 expects `torch==2.8.0`, while `env-dreamerv3` uses PyTorch 2.1.2 for the local MPS workflow.
- The README states that inference requires a CUDA-enabled GPU; local Apple Silicon MPS is not a supported target for this implementation.
- The repo is image/tokenizer heavy and depends on packages not needed by the V3 proprio pipeline, including `lpips`, `transformers`, `torchrl`, and web serving packages.

## Pretrained Checkpoints

The README states that pretrained tokenizer and dynamics checkpoints are available from Hugging Face:

- Tokenizer checkpoint: trained for 90k steps, 128x128 resolution, 30 tasks.
- Dynamics checkpoint: trained for 40k steps with actions, 128x128 resolution, 30 tasks.

The provided checkpoints target a multitask DMControl/MMBench dataset rather than a single Cheetah Run proprio-only setup. This is useful for cross-architecture analysis, but it is not a drop-in replacement for the current V3 Cheetah proprio checkpoint.

## Training Cost Estimate

The README training estimates are:

- Tokenizer: about 24 hours on 8x RTX 3090 GPUs.
- Dynamics model: about 48 hours on 8x RTX 3090 GPUs.
- Recommended training hardware: more than 256 GB RAM and 8 GPUs with more than 24 GB memory each.
- Full preprocessed dataset storage: about 350 GB.

Approximate GPU-hour and cost estimates at `$0.70/hr` per GPU:

- Reproduce README tokenizer: `24 * 8 = 192` GPU-hours, about `$134`.
- Reproduce README dynamics: `48 * 8 = 384` GPU-hours, about `$269`.
- Combined README-scale training: `576` GPU-hours, about `$403`.

The repo does not define an online "500k environment step" training run comparable to the V3 training we ran. If dynamics training were naively scaled from 40k to 500k optimizer steps, the dynamics phase alone would be about `4800` GPU-hours, about `$3360`, before tokenizer cost. That extrapolation is rough, but it highlights that self-training V4 is much more expensive than the Month 1 V3 Cheetah run.

## API Compatibility With WorldModelAdapter

A `DreamerV4Adapter` is feasible, but it would require a substantially different implementation from `DreamerV3Adapter`.

Likely method mapping:

- `load_checkpoint()`: load both tokenizer and dynamics checkpoints; configure image resolution, tokenizer args, dynamics args, and task/action metadata.
- `reset()`: create or wrap a DMC/MMBench environment and return either image observations or a projected vector representation.
- `encode()`: tokenize image observations into packed latent tokens using the V4 encoder.
- `imagine()`: iteratively sample future packed latents with action conditioning, then decode through the tokenizer decoder.
- `collect_true_rollout()`: needs a policy/action source. The V4 repo is mainly a world model interface, so actions would likely come from dataset trajectories, a separate controller, scripted actions, or our V3 policy.

Complexity estimate:

- Adapter skeleton and checkpoint load: 1-2 days.
- Single-task offline rollout extraction from existing dataset trajectories: 3-5 days.
- Online environment rollout with a policy-compatible action source: 1-2 weeks.
- Full parity with the current V3 proprio diagnostic pipeline: 2+ weeks, mainly due to image-token output alignment and action-source decisions.

## Recommendation

Dreamer4 is feasible for Month 4, but it is not a low-risk Month 2 dependency.

- Risk level: Medium-High.
- Suggested action: defer V4 implementation until Month 3 reconnaissance/prototyping, then decide whether Month 4 uses pretrained V4 checkpoints or a narrower adapter over offline trajectories.
- Do not train Dreamer4 now. The hardware and storage requirements are too large for the current Month 1/2 critical path.
- Keep Month 2 focused on frequency decomposition and drift analysis using the completed V3 checkpoint, adapter, and 20-rollout batch.

The best near-term path is to treat Dreamer4 as an architecture-comparison target, not as a blocker for the core DriftDetect diagnostic pipeline.
