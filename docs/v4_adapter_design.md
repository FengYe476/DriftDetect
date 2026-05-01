# DreamerV4 Adapter Design

Date: 2026-05-01

Scope: design only. Do not add V4 adapter code in this step. This document
specifies the implementation target for `src/models/dreamerv4_adapter.py` and
associated rollout scripts.

Inspection basis:

- DriftDetect adapter contract: `src/models/adapter.py`
- V3 reference adapter: `src/models/dreamerv3_adapter.py`
- V4 source: `external/dreamer4/` at
  `bdeddfe2058c1025344424d1dbcdb743d9228fc9`
- V4 prior notes: `docs/v4_recon.md`, `docs/v4_hf_deep_recon.md`,
  `docs/v4_decision.md`
- No V4 checkpoint was loaded locally. CUDA-only behavior remains a RunPod
  smoke-test item.

## 1. Checkpoint Loading

### HF repositories

Model checkpoints are in the HF model repo:

- Repo ID: `nicklashansen/dreamer4`
- Pin: `10e04fce900745e266ab68925b6779831551f157`
- Current files on HF:
  - `tokenizer.pt`
    - Size: 256,658,341 bytes
    - LFS sha256: `8a2cb0e3f381a5c5375a666e3f3012793482c3033975a4d40337269009f100d1`
  - `dynamics.pt`
    - Size: 511,868,041 bytes
    - LFS sha256: `7c0e0d3d7bf366947509fb6a0a90953785ab6f41e617b4edf7f491f87db3d773`
  - `config.json`

Important discrepancy: `external/dreamer4/README.md` links to
`tokenizer_ckpt.pt` and `dynamics_ckpt.pt`, but HF currently lists
`tokenizer.pt` and `dynamics.pt`. The adapter should prefer the current HF file
names and allow overrides.

Dataset trajectories are in the HF dataset repo:

- Repo ID: `nicklashansen/dreamer4`
- Repo type: `dataset`
- HF page short revision observed: `7dceb05`
- Contains directories `expert`, `mixed-small`, and `mixed-large`.
- Dataset card says 7,200 trajectories, 3.6M frames, 30 tasks, total HF raw
  file size about 30.9 GB. The V4 README estimates about 350 GB after full
  preprocessing.

### Download policy

Implementation should use `huggingface_hub.hf_hub_download()` only on RunPod,
with `repo_id="nicklashansen/dreamer4"`, filenames `tokenizer.pt` and
`dynamics.pt`, and revision `10e04fce900745e266ab68925b6779831551f157`. Do not
download the full dataset for the first smoke test; for rollout collection,
selectively download one task/source first, e.g. `expert` or `mixed-small` for
`cheetah-run`.

### V4 loader entry points

No `from_pretrained()` API exists in `external/dreamer4/`. Use the custom
helpers in `external/dreamer4/dreamer4/interactive.py`:

- `load_tokenizer_from_ckpt(tokenizer_ckpt: str, device: torch.device)`
- `load_dynamics_from_ckpt(dynamics_ckpt: str, *, device: torch.device,
  d_bottleneck: int, n_latents: int, packing_factor: int)`
- `make_tau_schedule(*, k_max: int, schedule: str = "finest",
  d: Optional[float] = None)`

Prefer the `interactive.py` helpers because they accept several checkpoint key
layouts through `_get_state_dict()`. The tokenizer-only alternative is
`train_dynamics.py:load_frozen_tokenizer_from_pt_ckpt`.

### Required config fields

Tokenizer construction in `interactive.load_tokenizer_from_ckpt()` reads
`ckpt["args"]` and falls back to defaults:

```text
H=128, W=128, C=3, patch=4
d_model=256, n_heads=4, depth=8
n_latents=16, d_bottleneck=32
dropout=0.0, mlp_ratio=4.0, time_every=1
scale_pos_embeds=True
```

Derived tokenizer fields:

```text
n_patches = (H // patch) * (W // patch) = 1024
d_patch = patch * patch * C = 48
latent shape = (B, T, n_latents, d_bottleneck) = (B, T, 16, 32)
```

Dynamics construction in `interactive.load_dynamics_from_ckpt()` reads
`ckpt["args"]` and falls back to defaults:

```text
d_model = args["d_model_dyn"] or args["dyn_d_model"] or args["d_model"] or 256
n_heads=4
depth = args["dyn_depth"] or args["depth"] or 8
dropout=0.0, mlp_ratio=4.0, time_every=4
k_max=8, n_register=4, n_agent=0
space_mode="wm_agent_isolated"
scale_pos_embeds=True
```

Training defaults in `train_dynamics.py` differ in important places:

```text
d_model_dyn=512
dyn_depth=8
n_heads=4
time_every=1
packing_factor=2
n_register=4
n_agent=1
k_max=8
space_mode="wm_agent_isolated"
```

Implementation requirement: after `torch.load()`, log `ckpt.keys()` and
`ckpt.get("args", {})` for both HF files. If `args` is missing or inconsistent,
use the training defaults above rather than the weaker `interactive.py` fallback.
This is UNKNOWN until the RunPod checkpoint smoke test.

Derived dynamics fields with expected Month 4 defaults:

```text
packing_factor = 2
n_spatial = n_latents // packing_factor = 8
d_spatial = d_bottleneck * packing_factor = 64
packed latent shape = (B, T, 8, 64)
dynamics action shape = (B, T, 16)
```

### Memory footprint

Checkpoint disk size is about 0.77 GB. In fp32, parameter memory should be about
the same order, plus PyTorch/CUDA overhead and activations.

Expected inference VRAM for batch `B=1`, context `ctx<=24`, horizon streaming
one step at a time:

- Minimum stated by V4 README/interactive server: at least 2 GB CUDA VRAM.
- Practical DriftDetect budget: 3-4 GB with fp32, 2-3 GB with autocast.
- RunPod target should reserve at least 6 GB free VRAM to avoid allocator and
  decode spikes.

## 2. V4 Imagination API

### One-step primitive

Primary code reference:
`external/dreamer4/dreamer4/train_dynamics.py:sample_one_timestep_packed`

Current signature:

```python
@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,          # (B,t,n_spatial,d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,     # (B,T,A) aligned to frames or None
    act_mask: Optional[torch.Tensor] = None,    # (B,T,A) or (A,) or None
) -> torch.Tensor:
    ...
    return z[:, 0]  # (B,n_spatial,d_spatial)
```

The near-duplicate in `interactive.py` adds `use_amp: bool = True`,
`torch.inference_mode()`, and CUDA autocast. Use that implementation logic in
the adapter.

### Autoregressive helper

Code reference:
`external/dreamer4/dreamer4/train_dynamics.py:sample_autoregressive_packed_sequence`.
It takes `z_gt_packed: (B,T,n_spatial,d_spatial)`, `ctx_length`, `horizon`,
`k_max`, `sched`, optional `actions: (B,T,A)`, optional `act_mask`, and returns
`torch.stack(outs, dim=1)`.

Output shape is `(B, ctx_length + horizon, n_spatial, d_spatial)`, capped by
the supplied `z_gt_packed.shape[1]`. Compare generated latents with:

```python
z_pred = sample_autoregressive_packed_sequence(...)
true_eval = z_gt_packed[:, ctx_length : ctx_length + horizon]
imag_eval = z_pred[:, ctx_length : ctx_length + horizon]
```

### Iterative denoising loop

Schedule builder:
`external/dreamer4/dreamer4/train_dynamics.py:make_tau_schedule`

Schedule fields:

```text
K = number of Euler/denoising steps
e = log2(K), used as shortcut step index
scale = k_max // K
tau = [i / K for i in range(K)] + [1.0]
tau_idx = [i * scale for i in range(K)] + [k_max]
dt = 1.0 / K
```

Defaults and recommended initial settings:

- `k_max=8`.
- `schedule="finest"` gives `K=8`.
- `schedule="shortcut", d=0.25` gives `K=4`; this is the
  `train_dynamics.py` eval default and should be the first Month 4 rollout
  setting.
- `schedule="shortcut", d=0.5` gives `K=2`; this is the `interactive.py`
  default.
- Smoke test may use `schedule="shortcut", d=1.0` for `K=1`.

For each generated timestep:

1. Initialize the candidate next packed latent with Gaussian noise:
   `z = torch.randn((B, 1, n_spatial, d_spatial), dtype=past_packed.dtype)`.
2. Build `step_idxs_full` with shape `(B, t+1)`.
   Context timesteps use `emax=log2(k_max)`; the generated timestep uses `e`.
3. Build `signal_idxs_full` with shape `(B, t+1)`.
   Context timesteps are initialized to `k_max - 1`; the generated timestep is
   updated to `tau_idx[i]` each denoising step.
4. Concatenate context and candidate:
   `packed_seq = torch.cat([past_packed, z], dim=1)`.
5. Call `dyn(actions_in, step_idxs_full, signal_idxs_full, packed_seq,
   act_mask=actmask_in, agent_tokens=None)`.
6. Take `x1_hat_full[:, -1:, :, :]` as the clean packed latent prediction.
7. Euler update:
   `b = (x1_hat.float() - z.float()) / max(1e-4, 1.0 - tau_i)`;
   `z = (z.float() + b * dt).to(dtype)`.

### Action conditioning

Code reference: `external/dreamer4/dreamer4/model.py:ActionEncoder` and
`external/dreamer4/dreamer4/model.py:Dynamics.forward`.

`Dynamics.forward()` signature:

```python
def forward(
    self,
    actions: Optional[torch.Tensor],          # (B,T,16) or None
    step_idxs: torch.Tensor,                  # (B,T)
    signal_idxs: torch.Tensor,                # (B,T)
    packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
    *,
    act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
    agent_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

Action injection:

- Actions are continuous and expected in `[-1, 1]`.
- The V4 model hardcodes `ActionEncoder(..., action_dim=16)`.
- Dataset actions are padded to 16 dimensions and masked by task.
- `ActionEncoder.forward()` applies `actions * act_mask`, clamps to `[-1, 1]`,
  then maps through `fc1 -> silu -> fc2` and adds a learned base token.
- The action token is concatenated before shortcut signal, shortcut step,
  spatial latent, register, and optional agent tokens.

Training alignment from `train_dynamics.py`:

```python
frames = obs_u8[:, :-1].float() / 255.0
actions = torch.zeros_like(act)
actions[:, 1:] = act[:, :-1]
act_mask = torch.zeros_like(mask)
act_mask[:, 1:] = mask[:, :-1]
```

Adapter must preserve this alignment. For a model frame sequence
`frames[:, 0:T]`, `actions[:, 0]` is zero and `actions[:, t]` for `t>=1` is
the transition action that produced `frames[:, t]`.

### V4 latent definition

There are two equivalent latent views:

- Tokenizer bottleneck latent:
  - Shape: `(B, T, 16, 32)`
  - Scalar size per frame: 512
  - Continuous, tanh-bounded, no V3-style stochastic/deterministic split.
- Packed dynamics latent:
  - Shape: `(B, T, 8, 64)` with `packing_factor=2`
  - Scalar size per frame: 512
  - This is the native dynamics/imagination state.

Primary DriftDetect V4 latent should be the packed dynamics latent flattened to
`(T, 512)` as `float32`:

```python
latent_2d = z_packed.reshape(T, 8 * 64).detach().cpu().numpy().astype(np.float32)
```

### V4 image output

Decoder reference:
`external/dreamer4/dreamer4/train_dynamics.py:decode_packed_to_frames`

```python
def decode_packed_to_frames(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,     # (B,T',n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    ...
    return frames.clamp(0, 1)
```

Expected output shape is `(B, T, 3, 128, 128)` with float values in `[0, 1]`.
True dataset frames are loaded as uint8 `(T+1, 3, 128, 128)` by
`WMDataset` and converted to float by training/eval code.

## 3. Action Source Decision

Recommended option: (c) use V4 HF dataset trajectories.

Justification:

- The V4 codebase has no inspected V4 policy/actor equivalent to V3's policy.
- The dynamics checkpoint was trained with actions; dataset trajectories are the
  only native source that supplies images, 16-D padded actions, masks, and
  rewards in the exact V4 format.
- `WMDataset.__getitem__()` returns:
  - `obs`: `(T+1, 3, H, W)` uint8
  - `act`: `(T, 16)` float32 padded
  - `act_mask`: `(T, 16)` float32
  - `rew`: `(T,)` float32
  - `lang_emb`, `emb_id`
- This avoids out-of-distribution random controls and avoids uncertain V3-to-V4
  action replay.

Rejection of option (a), replay V3 policy actions on V4:

- `cheetah-run` action dimension is compatible at the dimension level:
  `external/dreamer4/tasks.json` says action_dim 6, and V3 DMC Cheetah also
  uses 6 continuous controls.
- However, tasks.json does not document the joint ordering or normalization
  beyond action_dim and a text description.
- V3 policy actions would require separately collecting V4-compatible rendered
  image rollouts from DMC under the same actions. That adds environment/render
  alignment risk and no longer uses the V4 training distribution.

Rejection of option (b), scripted/random actions:

- Easy to implement, but likely out of distribution for an action-conditioned
  dynamics checkpoint trained on expert/mixed TD-MPC2 trajectories.
- It would make any drift growth ambiguous: model architecture versus OOD action
  sequence.

Failure modes of recommended option:

- Uses released training trajectories, not a held-out online environment. This
  is acceptable for an inference/architecture comparison but should be named in
  reports.
- Dataset preprocessing can be storage-heavy. Do selective task/source download
  first; do not preprocess all 30 tasks for Month 4 W1.
- HF raw dataset format/path selection needs RunPod verification.
- `sample_autoregressive_packed_sequence()` examples use short horizons
  (`eval_horizon=16`); 200-step open-loop stability is UNKNOWN until W1.

## 4. Observation Alignment and FFT Input

Decision: primary V4 frequency decomposition runs on image-native tokenizer
latents, not proprio and not raw pixels.

Exact tensors to store and analyze per rollout:

```text
true_obs:        (H, 3, 128, 128), float32 or uint8 for QA
imagined_obs:    (H, 3, 128, 128), float32 or uint8 for QA
actions:         (H + ctx_length, 16) or (H, 16), float32, document alignment
action_mask:     (H + ctx_length, 16) or (H, 16), float32
rewards:         (H,), float32
true_latent:     (H, 512), float32, z_gt_packed[:, ctx:ctx+H].reshape(H, 512)
imagined_latent: (H, 512), float32, z_pred[:, ctx:ctx+H].reshape(H, 512)
```

The exact input to `src.diagnostics.freq_decompose.decompose_pair()` is:

```python
decompose_pair(
    rollout["true_latent"],      # (200, 512)
    rollout["imagined_latent"],  # (200, 512)
    var_threshold=0.95,
)
```

Why not per-pixel raw RGB as the primary metric:

- Raw frame flattening gives `(T, 49152)`, which the current
  `freq_decompose.py` intentionally rejects. It supports V3 full latent 1536,
  512-D latent/deter, or <=50-D observation/PCA inputs.
- Per-pixel FFT would be expensive and would require a new shared PCA or
  projection path before filtering.
- Mean-pooled RGB `(T, 3)` is too lossy for the primary claim.

Optional pixel-space sanity check: decode frames and compute channel mean
`(T, 3)` or a future shared PCA projection from flattened RGB to <=50 PCs. Do
not use this as the headline V4 frequency result without extending the
diagnostic code.

### Storage estimate

For one 200-step rollout:

- One float32 image array `(200, 3, 128, 128)` is about 39.3 MB.
- True plus imagined float32 images are about 78.6 MB.
- True plus imagined float32 latents `(200, 512)` are about 0.82 MB total.
- Actions, masks, rewards, metadata are negligible (<0.1 MB).

Recommended storage mode:

- Store `true_latent` and `imagined_latent` as float32.
- Store `true_obs` as uint8 if sourced from dataset.
- Store `imagined_obs` as uint8 after clipping/scaling to `[0, 255]` for QA.
- With uint8 images, one rollout is about 20-25 MB uncompressed plus NPZ
  overhead; 20 seeds are about 0.4-0.6 GB.
- If images are stored as float32, one rollout is about 80 MB and 20 seeds are
  about 1.6 GB.

Storage location:

```text
results/rollouts_v4/
```

Suggested filenames: `results/rollouts_v4/cheetah_v4_seed{seed}_v1.npz`.

## 5. Adapter Interface Mapping

`WorldModelAdapter.load_checkpoint(path: str) -> None`

- Does not map cleanly because V4 has two checkpoints. Implementation should
  accept `path` as a local directory containing `tokenizer.pt` and `dynamics.pt`
  or a small manifest path.
- Proposed future interface extension:
  `load_checkpoint(self, path: str, *, tokenizer_path: str | None = None,
  dynamics_path: str | None = None)`.

`WorldModelAdapter.reset(seed: int = 0) -> np.ndarray`

- In dataset mode, select a deterministic dataset window for the given seed and
  return the first frame `(3, 128, 128)` as `np.float32` in `[0, 1]`.
- This is mainly for contract compatibility; V4 rollout collection should use
  `collect_true_rollout()` or `extract_rollout()` because context frames are
  needed.

`WorldModelAdapter.encode(obs: np.ndarray) -> dict`

- Accept either a single frame `(3, 128, 128)` or a sequence
  `(T, 3, 128, 128)`, normalize to torch `(1, T, 3, 128, 128)`, call
  `temporal_patchify()`, then `tokenizer.encoder`.
- Return `{"z_btLd": (T,16,32), "z_packed": (T,8,64), "flat": (T,512)}` as
  CPU numpy arrays or detached tensors.

`WorldModelAdapter.imagine(init_obs, actions, horizon) -> np.ndarray`

- `init_obs` must be V4 context frames `(ctx_length, 3, 128, 128)`, not a single
  proprio vector.
- Encode context, build aligned `(1, ctx_length+horizon, 16)` actions, run
  autoregressive packed sampling, and decode only generated steps.
- Return decoded generated images `(horizon, 3, 128, 128)` for the base method.
  The latent path should be exposed through an overridden `extract_rollout()`.

`WorldModelAdapter.collect_true_rollout(seed, total_steps, imagination_start)`

- In V4, this selects a dataset window rather than stepping an online env.
- Return true frames, aligned actions, rewards, and cache/record the packed true
  latents needed for imagination.
- Needs an override with `include_latent=True` and `horizon`, mirroring the V3
  adapter pattern.

`WorldModelAdapter.extract_rollout(...) -> dict`

- Must be overridden for V4.
- Return the standard keys plus `action_mask`, `true_latent`, and
  `imagined_latent`.
- The v1 base implementation is not sufficient because V4 needs context images,
  action masks, and latent outputs.

Proposed non-breaking extension for V4 and future image models: add optional
keyword-only `context_length: int | None = None` to `extract_rollout()`.

## 6. Environment Setup

Use a new conda environment on RunPod only:

```text
env name: env-dreamerv4
YAML source: external/dreamer4/environment.yaml
```

Create with:

```bash
conda env create -n env-dreamerv4 -f external/dreamer4/environment.yaml
conda activate env-dreamerv4
```

Key dependencies from the upstream YAML: `python=3.10`, `torch==2.8.0`,
`torchvision==0.23.0`, `torchrl==0.10.0`, `tensordict==0.10.0`,
`transformers==4.56.2`, `lpips==0.1.4`, `aiohttp==3.13.3`,
`pillow==12.0.0`, `numpy==1.24.4`, and `wandb==0.22.1`.

This must not share `env-dreamerv3`, which uses PyTorch 2.1.2. V4 inference
requires CUDA. Do not implement or document a local Apple MPS path for V4.

## 7. Validation Plan Before Rollout Collection

Minimal RunPod smoke test: create `env-dreamerv4`, download pinned
`tokenizer.pt` and `dynamics.pt`, add `external/dreamer4/dreamer4` to
`sys.path`, load tokenizer and dynamics through `interactive.py`, make one
`(1,1,3,128,128)` frame, run `temporal_patchify()`, `tok.encoder()`,
`pack_bottleneck_to_spatial(..., n_spatial=8, k=2)`, build
`make_tau_schedule(k_max=dyn_info["k_max"], schedule="shortcut", d=1.0)`,
run one `sample_one_timestep_packed()` call with zero actions/mask, then decode
with `decode_single_packed_frame()` or `decode_packed_to_frames()`.

Success criteria:

- Checkpoint loads with strict state dict.
- Shapes match:
  - `z_btLd`: `(1, 1, 16, 32)`
  - `z_packed`: `(1, 1, 8, 64)`
  - `z_next`: `(1, 8, 64)`
  - decoded frame: `(3, 128, 128)` or `(1, 1, 3, 128, 128)`
- No NaNs or infs in latents or decoded image.
- Decoded image values are within `[0, 1]`.
- CUDA memory stays under 4 GB for B=1.

Smoke-test GPU-hour estimate:

- Checkpoint download time is network-bound and should not be counted as GPU.
- Model load plus one encode, one denoising step, and decode should be under
  5 minutes of GPU time.
- Budget 0.1 GPU-hour including iteration and logging.

## 8. Risks and Unknowns for W1 Implementation

These require actual RunPod execution or HF download inspection:

- HF checkpoint internal keys and `ckpt["args"]` contents. If `args` is missing,
  loader defaults may build the wrong dynamics width or agent-token count.
- Whether `scale_pos_embeds` in checkpoint args matches loader defaults.
- Whether `interactive.load_tokenizer_from_ckpt()` and
  `interactive.load_dynamics_from_ckpt()` load the current HF files strictly
  without code changes.
- Selective HF dataset download paths for a single task/source. The repo code
  expects raw `.pt` demos plus preprocessed frame shards, while the raw dataset
  stores PNG strips that must be preprocessed.
- Exact full dataset revision SHA for `nicklashansen/dreamer4` dataset if we
  need a full 40-char pin. The HF UI exposed short `7dceb05`.
- Online V4 environment path is intentionally not designed here.
- V3 action replay joint ordering for `cheetah-run` remains undocumented in V4
  `tasks.json`; this is only relevant if we later revisit option (a).
- 200-step V4 open-loop stability. V4 evaluation examples use short horizons
  such as 16, while interactive generation rolls one step at a time.
- Actual runtime and VRAM for `K=4`, `ctx_length=8 or 24`, `horizon=200`.
- Whether using released training trajectories is acceptable for final paper
  framing or whether we need a held-out V4 dataset split later.
