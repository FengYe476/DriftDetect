# DreamerV4 HF Deep Reconnaissance

Date: 2026-05-01

Scope: code-level reconnaissance of `external/dreamer4/` for Month 4 cross-architecture comparison. No local inference was run because the current machine is Apple Silicon/MPS and the DreamerV4 README requires a CUDA-enabled GPU for inference.

Local source inspected:

- Repo: `external/dreamer4/`
- Commit: `bdeddfe2058c1025344424d1dbcdb743d9228fc9`
- Key files: `README.md`, `dreamer4/model.py`, `dreamer4/train_tokenizer.py`, `dreamer4/train_dynamics.py`, `dreamer4/interactive.py`, `dreamer4/wm_dataset.py`, `dreamer4/task_set.py`

## 1. HF Checkpoint Details

`external/dreamer4/README.md` says the released model checkpoints are hosted at `https://huggingface.co/nicklashansen/dreamer4` and the dataset at `https://huggingface.co/datasets/nicklashansen/dreamer4`. It states that inference requires a CUDA-enabled GPU, and that the full preprocessed dataset requires approximately 350 GB of disk storage.

Local README checkpoint table:

- Tokenizer: `https://huggingface.co/nicklashansen/dreamer4/resolve/main/tokenizer_ckpt.pt`
  - Trained for 90k steps
  - Resolution: 128x128
  - Tasks: 30
- Dynamics model: `https://huggingface.co/nicklashansen/dreamer4/resolve/main/dynamics_ckpt.pt`
  - Trained for 40k steps with actions
  - Resolution: 128x128
  - Tasks: 30

Important filename discrepancy: the current HuggingFace file listing, checked on 2026-05-01, shows `tokenizer.pt` and `dynamics.pt`, not `tokenizer_ckpt.pt` and `dynamics_ckpt.pt`.

Current HF file sizes:

- `nicklashansen/dreamer4::tokenizer.pt`: 257 MB
- `nicklashansen/dreamer4::dynamics.pt`: 512 MB
- `nicklashansen/dreamer4::config.json`: 191 bytes
- HF model repository total: 769 MB

Storage requirement for Month 4:

- Checkpoint-only inference: about 0.8 GB for the two model files, plus local cache/temp overhead. Budget 2 GB.
- Full preprocessed dataset: about 350 GB, per README. This is not needed for a minimum inference smoke test, but may be needed if we want to reuse the released offline dataset at scale.
- No checkpoint files are present in the local clone; `external/dreamer4/` is only about 18 MB.

The 30-task checkpoint task set is defined in `dreamer4/task_set.py`:

`walker-stand`, `walker-walk`, `walker-run`, `cheetah-run`, `reacher-easy`, `reacher-hard`, `acrobot-swingup`, `pendulum-swingup`, `cartpole-balance`, `cartpole-balance-sparse`, `cartpole-swingup`, `cartpole-swingup-sparse`, `cup-catch`, `finger-spin`, `finger-turn-easy`, `finger-turn-hard`, `hopper-stand`, `hopper-hop`, `walker-walk-backward`, `walker-run-backward`, `cheetah-run-backward`, `cheetah-run-front`, `cheetah-run-back`, `cheetah-jump`, `hopper-hop-backward`, `cup-spin`, `pendulum-spin`, `jumper-jump`, `reacher-three-easy`, `reacher-three-hard`.

Note: `tasks.json` contains metadata for many more tasks, but training uses `TASK_SET` from `dreamer4/task_set.py`.

## 2. Architecture Deep Dive

Main model file: `external/dreamer4/dreamer4/model.py`.

Training and inference entry points:

- Tokenizer training: `dreamer4/train_tokenizer.py`
- Dynamics training/evaluation helpers: `dreamer4/train_dynamics.py`
- Interactive inference server: `dreamer4/interactive.py`
- Dataset with actions: `dreamer4/wm_dataset.py`

### Tokenizer

The tokenizer is implemented by `Encoder`, `Decoder`, and `Tokenizer` in `model.py`.

Default tokenizer hyperparameters from `train_tokenizer.py`:

- Input frames: `(B, T, 3, 128, 128)`
- Patch size: `4`
- Number of image patches: `(128 / 4) * (128 / 4) = 1024`
- Patch dimension: `4 * 4 * 3 = 48`
- `d_model = 256`
- `depth = 8`
- `n_heads = 4`
- `n_latents = 16`
- `d_bottleneck = 32`
- Tokenizer latent shape: `(B, T, 16, 32)`
- Flattened tokenizer latent size per frame: `16 * 32 = 512`

Encoding path:

- `temporal_patchify(videos_btchw, patch)` maps frames to `(B, T, Np, Dp)`.
- `Encoder.forward(patch_tokens_btnd)` projects each patch with `patch_proj`.
- During training, `MAEReplacer` masks image patch tokens.
- Learned latent tokens of shape `(16, d_model)` are prepended to image patch tokens.
- `BlockCausalTransformer` processes the combined latent/image token sequence.
- The first `n_latents` outputs are projected through `bottleneck_proj` and `tanh`.
- Output is continuous bottleneck latent `z` with shape `(B, T, 16, 32)`.

Decoding path:

- `Decoder.forward(z_btLd)` takes `(B, T, 16, 32)`.
- Latents are projected up to `d_model`.
- Learned patch queries are appended.
- The decoder transformer predicts patch tokens.
- `patch_head` plus `sigmoid` returns `(B, T, 1024, 48)`.
- `temporal_unpatchify(...)` reconstructs `(B, T, 3, 128, 128)` frames.

Transformer structure:

- `BlockCausalTransformer` alternates space attention and causal time attention.
- `SpaceSelfAttentionModality` builds modality-specific attention masks.
- `TimeSelfAttention` uses causal attention along the time dimension.
- For tokenizer defaults, `latents_only_time=True`, so temporal attention is applied to latent tokens only.

Comparison to V3 RSSM:

- V3 DriftDetect latent features are 1536-dimensional per timestep: 1024 stochastic plus 512 deterministic.
- V4 tokenizer latents are 512-dimensional per frame: 16 continuous tokens x 32 bottleneck dims.
- There is no clear deterministic/stochastic split analogous to V3 RSSM.
- V4 tokenizer latents are continuous tanh-bounded bottleneck tokens, not an RSSM state with separate posterior stochastic and deterministic recurrent components.

### Dynamics Model

The dynamics model is `Dynamics` in `model.py`.

Default dynamics hyperparameters from `train_dynamics.py`:

- `d_model_dyn = 512`
- `dyn_depth = 8`
- `n_heads = 4`
- `packing_factor = 2`
- `n_register = 4`
- `n_agent = 1`
- `k_max = 8`
- `time_every = 1`
- `space_mode = "wm_agent_isolated"`
- Action conditioning is enabled by running `train_dynamics.py --use_actions`.

Packing converts tokenizer latents before dynamics:

- Tokenizer latent: `(B, T, 16, 32)`
- `pack_bottleneck_to_spatial(..., n_spatial=8, k=2)` maps this to `(B, T, 8, 64)`
- Total latent scalar count remains `8 * 64 = 512`

Dynamics token layout per timestep with defaults:

- 1 action token
- 1 shortcut signal token
- 1 shortcut step token
- 8 spatial latent tokens
- 4 register tokens
- 1 agent token
- Total: 16 tokens per timestep

`Dynamics.forward(...)` signature:

```python
forward(
    actions: Optional[torch.Tensor],      # (B,T,16) or None
    step_idxs: torch.Tensor,              # (B,T)
    signal_idxs: torch.Tensor,            # (B,T)
    packed_enc_tokens: torch.Tensor,      # (B,T,n_spatial,d_spatial)
    *,
    act_mask: Optional[torch.Tensor] = None,
    agent_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]
```

It projects packed spatial tokens, action tokens, shortcut-step embeddings, shortcut-signal embeddings, register tokens, and optional agent tokens into a block-causal transformer. The output head `flow_x_head` predicts `x1_hat` with shape `(B, T, n_spatial, d_spatial)`, i.e. the clean packed latent target for shortcut denoising.

Closest equivalent to V3 `img_step()`:

- There is no direct single-call RSSM-style `img_step()` in this codebase.
- The closest one-step imagination primitive is `sample_one_timestep_packed(...)` in `train_dynamics.py` and a near-duplicate in `interactive.py`.
- It samples a future packed latent by initializing the next latent from Gaussian noise and repeatedly calling `Dynamics.forward(...)` over a shortcut/Euler schedule.

## 3. Imagination API

The main code-level imagination helpers are in `dreamer4/train_dynamics.py`:

```python
sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,            # (B,t,n_spatial,d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor                         # (B,n_spatial,d_spatial)
```

and:

```python
sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,            # (B,T,n_spatial,d_spatial)
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor                         # (B,T',n_spatial,d_spatial)
```

One-step generation details:

- `past_packed` provides the context latents.
- The next latent is initialized as random Gaussian noise.
- `sched["K"]` controls the number of denoising/Euler integration steps.
- For each Euler step, the code concatenates `past_packed` with the current noisy candidate latent and calls `dyn(...)`.
- Only the last timestep prediction is used as `x1_hat`.
- The update is `(x1_hat - z) / (1 - tau)` times `dt`.

Schedule details:

- `make_tau_schedule(k_max=8, schedule="finest")` gives `K = 8`.
- `schedule="shortcut"` uses `K = 1 / eval_d`.
- `train_dynamics.py` evaluation default is `eval_d = 0.25`, so `K = 4`.
- `interactive.py` default is `eval_d = 0.5`, so `K = 2`.

Open-loop feasibility:

- Yes, the code supports open-loop autoregressive imagination.
- `sample_autoregressive_packed_sequence(...)` uses ground-truth packed latents for the context, then repeatedly appends generated latents for the requested horizon.
- It accepts an action sequence, so V4 can be conditioned on dataset or scripted action sequences.
- Passing `actions=None` is supported by `ActionEncoder`, which emits a learned base action token, but the released dynamics checkpoint was described as trained with actions, so action-conditioned evaluation is the safer path.

How many steps can it imagine in one call?

- There is no explicit hard horizon constant in the helper.
- The helper clips `T'` to `min(T, ctx_length + horizon)` because it uses `z_gt_packed` as the container for total available length.
- Training default sequence length is 32. Interactive default context window is 24.
- Practically, Month 4 should start with context 8 and horizon 16, matching `train_dynamics.py` eval defaults, then test horizon 32 only after the smoke test works.

Decoder helper:

```python
decode_packed_to_frames(
    decoder,
    *,
    z_packed: torch.Tensor,               # (B,T',n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor                         # (B,T',C,H,W)
```

`interactive.py` provides checkpoint loaders that are useful for an adapter:

- `load_tokenizer_from_ckpt(tokenizer_ckpt, device)`
- `load_dynamics_from_ckpt(dynamics_ckpt, device, d_bottleneck, n_latents, packing_factor)`
- `decode_single_packed_frame(...)`

## 4. Observation Space Alignment

Current DriftDetect V3 pipeline:

- Proprioceptive observations for Cheetah Run and Cartpole Swingup.
- V3 latent feature size: 1536.
- Drift metrics compare imagined latent trajectories against posterior/encoded true trajectories.

DreamerV4 inspected repo:

- RGB-native world model.
- Default image size: 128x128.
- `WMDataset.__getitem__()` returns `obs`, `act`, `act_mask`, `rew`, `lang_emb`, and `emb_id`.
- No physics/proprio state tensor is returned by the V4 dataset path.
- The model code does not expose a built-in proprio prediction head.

Option A: V4 image/token temporal FFT.

- Run frequency decomposition over tokenizer latents `(B,T,16,32)` or packed dynamics latents `(B,T,8,64)` along the time axis.
- Optionally run the same decomposition over decoded RGB error `(B,T,3,128,128)` for a pixel-space sanity check.
- This is the most native comparison because it uses the V4 tokenizer and dynamics directly.
- It avoids needing a DMC physics-state extraction path.
- The caveat is comparability: V4 512-dim token latents are not semantically identical to V3 1536-dim RSSM features.

Option B: Extract DMC physics state alongside V4 images.

- This would require collecting fresh environment rollouts that save both rendered 128x128 frames and physics/proprio state.
- It also requires an action source. The V4 repo has a world model and interactive action interface, but no complete trained policy/actor path equivalent to our V3 agent.
- The released offline dataset path appears to provide frames/actions/rewards, not physics state.
- This option would enable direct proprio comparison but adds environment instrumentation and action-source risk.

Recommendation for Month 4: choose Option A first.

Minimum defensible comparison:

- Use overlapping tasks from `TASK_SET`: `cheetah-run` and `cartpole-swingup`.
- Encode true image rollouts with the V4 tokenizer.
- Imagine future packed latents with V4 dynamics under the recorded or scripted action sequence.
- Compare imagined packed latents to encoded true packed latents over time.
- Apply the existing frequency decomposition to the resulting latent error trajectories.
- Decode frames only for visual QA and optional pixel-space FFT.

## 5. Adapter Design Sketch

A `DreamerV4Adapter` should be separate from the V3 adapter because the model contract is fundamentally different.

Suggested methods:

### `load_checkpoint(tokenizer_path, dynamics_path, device="cuda:0")`

Implementation:

- Use `interactive.load_tokenizer_from_ckpt(...)`.
- Use `interactive.load_dynamics_from_ckpt(...)`.
- Read tokenizer metadata: `H`, `W`, `C`, `patch`, `n_latents`, `d_bottleneck`.
- Set `packing_factor=2` unless checkpoint args say otherwise.
- Build schedule with `make_tau_schedule(k_max=dyn_info["k_max"], schedule="shortcut", d=0.25)` for initial experiments.

Complexity: Low to medium.

Risks:

- HF filenames differ from local README.
- Checkpoint internal keys must match `_get_state_dict(...)`; the helper is robust to several common key layouts, but we have not loaded the actual HF files yet.
- Requires CUDA runtime and the DreamerV4 conda environment.

### `reset(task, source="dataset", start_idx=None)`

Implementation:

- For minimum viable offline mode, load a frame sequence from preprocessed shards and raw action `.pt` files, mirroring `WMDataset`.
- Encode the first `ctx_length` frames into tokenizer latents.
- Keep action masks from `tasks.json`.

Complexity: Medium.

Risks:

- Full dataset preprocessing is large if we rely on the official dataset.
- A lighter smoke test can use a locally rendered DMC frame sequence instead, but then action/state alignment becomes our responsibility.

### `encode(frames)`

Implementation:

- Accept `(B,T,3,128,128)` float frames in `[0,1]` or uint8 frames.
- Call `temporal_patchify(frames, patch)`.
- Call `tokenizer.encoder(patches)`.
- Return both raw tokenizer latents `(B,T,16,32)` and packed latents `(B,T,8,64)`.

Complexity: Low.

Risk: Mostly preprocessing and device/dtype hygiene.

### `imagine(context_packed, actions, horizon, schedule)`

Implementation:

- Construct a container `z_gt_packed` with context plus placeholder length, or write a small adapter loop around `sample_one_timestep_packed(...)`.
- Use action tensor `(B,T,16)` and `act_mask`.
- Generate autoregressively for `horizon`.
- Support repeated samples with different seeds because the shortcut sampler starts each next latent from random noise.

Complexity: Medium.

Risks:

- Stochastic denoising variance can contaminate frequency diagnostics if not controlled.
- Long horizons may be out of distribution relative to the default 32-step training windows.
- Compute cost scales with `horizon * K` dynamics calls.

### `decode(z_packed)`

Implementation:

- Call `decode_packed_to_frames(...)`.
- Return `(B,T,3,128,128)` frames.

Complexity: Low.

Risk: None beyond memory for decoded video batches.

Biggest technical risks overall:

- CUDA environment setup on RunPod with `torch==2.8.0` and matching dependencies.
- HF checkpoint filename/key mismatch.
- No native policy/actor path in the inspected V4 repo.
- Dataset storage if we need official offline trajectories at scale.
- Scientific comparability between V3 RSSM latent drift and V4 tokenizer/dynamics token drift.
- V4 implementation is explicitly unofficial and not a complete reproduction of the original paper.

## 6. Go/No-Go Assessment for Month 4

Can we run V4 inference on RunPod within budget?

Yes, for inference. The README says the interactive web server requires a CUDA GPU with at least 2 GB memory. A RunPod RTX 4090 is more than sufficient for checkpoint loading and small-batch rollouts. With about 168 GPU-hours remaining, a bounded V4 inference/adaptation experiment is affordable. Full V4 training is not affordable or necessary for Month 4.

Is the codebase stable enough to build an adapter?

Yes, with caution. The model code is compact and the key pieces are exposed as ordinary PyTorch modules/functions. The most useful functions already exist:

- `load_tokenizer_from_ckpt(...)`
- `load_dynamics_from_ckpt(...)`
- `sample_one_timestep_packed(...)`
- `sample_autoregressive_packed_sequence(...)`
- `decode_packed_to_frames(...)`

The main caution is that this is an unofficial implementation and the HF file names currently differ from the local README.

Minimum viable V4 experiment for the paper:

1. Download `tokenizer.pt` and `dynamics.pt` from `nicklashansen/dreamer4` on RunPod.
2. Load both checkpoints with the existing interactive loaders, adjusting filenames if necessary.
3. Run a smoke test: encode a short 128x128 frame context, imagine 16 steps with actions, decode frames.
4. Run one overlapping task, preferably `cheetah-run`, using V4 packed latents `(B,T,8,64)`.
5. Compare imagined packed latents against encoded true packed latents and apply the frequency decomposition along time.
6. Repeat on `cartpole-swingup` only if the first task works cleanly and budget remains.

Recommendation: GO, with conditions.

Conditions:

- Use RunPod CUDA; do not attempt MPS inference.
- Do not train V4.
- Start with checkpoint load plus one 16-step rollout smoke test.
- Use Option A, V4 latent-token FFT, as the Month 4 primary path.
- Cap the first implementation pass to 1-2 days and about 8-16 GPU-hours.
- Treat V4 as a supplementary cross-architecture comparison, not a dependency for the main Finding A/C/E story.
- If checkpoint loading fails because of filename/key/API drift, downgrade to NO-GO for the current submission and mention V4 as future work.

Bottom line: V4 is feasible for a narrow Month 4 inference-only comparison. The strongest path is not full parity with V3 proprio rollouts; it is a native V4 token-latent drift experiment on overlapping tasks, with decoded frames used for QA.

## Explicit Non-Findings

- No V3-style `img_step()` function was found.
- No deterministic/stochastic RSSM state split was found.
- No trained policy/actor API equivalent to the V3 Dreamer agent was found.
- No built-in physics-state/proprio output path was found in the V4 dataset/model code.
- No local HF checkpoint files were found under `external/dreamer4/`.

## Sources

- Local source: `external/dreamer4/README.md`
- Local source: `external/dreamer4/dreamer4/model.py`
- Local source: `external/dreamer4/dreamer4/train_tokenizer.py`
- Local source: `external/dreamer4/dreamer4/train_dynamics.py`
- Local source: `external/dreamer4/dreamer4/interactive.py`
- Local source: `external/dreamer4/dreamer4/wm_dataset.py`
- Local source: `external/dreamer4/dreamer4/task_set.py`
- HF model repository listing checked 2026-05-01: `https://huggingface.co/nicklashansen/dreamer4/tree/main`
