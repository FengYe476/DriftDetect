# DriftDetect
### Frequency-Domain Diagnostic for World Model Latent Drift

![Status](https://img.shields.io/badge/Status-Month%204%20Complete-238636)
![Compute Budget](https://img.shields.io/badge/Compute-~200%20GPU--hours-2ea44f)
![Timeline](https://img.shields.io/badge/Timeline-6%20months-f59e0b)

**TL;DR:** DriftDetect is a compute-efficient research project that systematically diagnoses how existing world models drift over long horizons by analyzing rollout failure in the frequency domain.

## What is this?

DriftDetect is a research project on latent drift in world models for reinforcement learning, especially in long-horizon imagination and planning. While recent world models have improved substantially, we still lack a mechanistic understanding of which temporal components fail first and how those failures accumulate. This project studies that question through frequency-domain diagnostics, using pre-trained checkpoints where available and targeted training when necessary. In other words, the goal is to **diagnose existing models**, not to propose a new world model method.

## Key Contributions

- 🎯 **Diagnostic Protocol**: A frequency-domain evaluation pipeline for measuring how rollout error grows across temporal bands.
- 📐 **Theory**: A lightweight theoretical component on slow-subspace identifiability in a simplified setting.
- 🔍 **Implications**: Empirical insights that can inform future world model architectures and training objectives.

See [PROPOSAL.md](./PROPOSAL.md) for the full proposal.

## Current Status

Month 4 is complete. DriftDetect now has cross-architecture evidence for
trend-dominated imagination drift and a scoped Month 5 SMAD Phase 2 path.

- [x] Month 1: Infrastructure, checkpoint, adapter, 20 v1 rollouts
- [x] Month 2 Week 1: Latent collection fix, 20 v2 rollouts, `freq_decompose` module
- [x] Month 2 Week 2: `error_curves` module, imagination window ablation
- [x] Month 2 Week 3: Figure 1 generated, Cartpole training complete
- [x] Month 2 Week 4: Filter ablation, Cartpole diagnostics, cross-task comparison
- [x] Month 3 Week 1: D1 decoder amplification + high-frequency checkpoint training launched
- [x] Month 3 Week 2: D2 Lipschitz probe + D3 linear/nonlinear separation
- [x] Month 3 Week 3: Training dynamics pipeline + Figure 3
- [x] Month 3 Week 4: V4 deep recon + Finding E verdict + Month 3 summary
- [x] Month 4 Week 1: DreamerV4 adapter design + implementation
- [x] Month 4 Week 2: V4 rollout collection + frequency decomposition
- [x] Month 4 Week 3: Figure 2 + Finding B evidence
- [x] Month 4 Week 4: SMAD pre-checks + Phase 1 verdict + Month 4 summary

### Recent Achievement

- Finding A SUPPORTED: imagination drift is frequency-asymmetric and trend-dominated across both Cheetah Run and Cartpole Swingup.
- `dc_trend` carries 46.7% (Cheetah) to 97.0% (Cartpole) of latent MSE.
- Cross-task comparison confirms the drift pattern is intrinsic to RSSM, not task-specific.
- Full diagnostic pipeline operational: `freq_decompose` + `error_curves` + Figure 1 generation.
- Two strong checkpoints: Cheetah (peak 791.1) and Cartpole (peak ~854).
- Decoder amplification experiments D1-D3 complete: trained decoder amplifies drift overall, but not preferentially along `dc_trend`.
- Finding E not supported as an independent finding; trend dominance appears latent-intrinsic.
- Finding C SUPPORTED: Cheetah training dynamics across 102 checkpoints show `dc_trend` dominance from step 5000.
- Figure 3 complete: curves, heatmap, and `dc_trend` share over training.
- Finding B SUPPORTED: DreamerV4 latent drift is also trend-dominated, with `dc_trend` carrying 63.9% of V4 latent MSE.
- Figure 2 complete: V3 solid curves vs V4 dashed curves for cross-architecture frequency-band error.
- SMAD Phase 1 complete: `U_posterior` is NO GO, Path B with `U_drift` is WEAK GO for Month 5.
- V4 adapter infrastructure complete: adapter, smoke tests, batch extractor, and RunPod CUDA environment.

### Finding Status

| Finding | Status | Evidence |
|---|---|---|
| Finding A: frequency-asymmetric drift with trend dominance | SUPPORTED | Month 2 Cheetah + Cartpole diagnostics |
| Finding B: cross-architecture trend-dominated drift | SUPPORTED | Month 4 V4 latent `dc_trend` share 63.9%; Figure 2 |
| Finding C: training dynamics | SUPPORTED | `dc_trend` dominant from step 5000 across 102 Cheetah checkpoints |
| Finding D: SMAD intervention | PENDING Month 5 | Phase 2 Path B with `U_drift` basis |
| Finding E: decoder amplification | NOT SUPPORTED as independent finding | D1-D3 show uniform OOD amplification, not directional `dc_trend` amplification |

### Progress Table

| Month | Week | Focus | Status | Date |
|---|---|---|---|---|
| Month 1 | Week 1 | Infrastructure setup | ✅ Complete | 2026-04 |
| Month 1 | Week 2 | Baseline extraction pipeline | ✅ Complete | 2026-04 |
| Month 1 | Week 3 | DreamerV3 Cheetah checkpoint training | ✅ Complete | 2026-04-29 |
| Month 1 | Week 4 | Adapter integration + 20 v1 rollouts | ✅ Complete | 2026-04 |
| Month 2 | Week 1 | Latent fix + v2 rollouts + `freq_decompose` | ✅ Complete | 2026-04 |
| Month 2 | Week 2 | `error_curves` + window ablation | ✅ Complete | 2026-04 |
| Month 2 | Week 3 | Figure 1 + Cartpole training | ✅ Complete | 2026-04-30 |
| Month 2 | Week 4 | Filter ablation + Cartpole diagnostics | ✅ Complete | 2026-04-30 |
| Month 2 | Summary | Final judgment + release tag | ✅ Complete | 2026-04-30 |
| Month 3 | Week 1 | D1 decoder amplification + high-freq checkpoint training launched | ✅ Complete | 2026-04-30 |
| Month 3 | Week 2 | D2 Lipschitz probe + D3 linear/nonlinear separation | ✅ Complete | 2026-04-30 |
| Month 3 | Week 3 | Training dynamics pipeline + Figure 3 | ✅ Complete | 2026-05-01 |
| Month 3 | Week 4 | V4 deep recon + Finding E verdict + Month 3 summary | ✅ Complete | 2026-05-01 |
| Month 3 | Summary | Training dynamics + decoder verdict + V4 decision | ✅ Complete | 2026-05-01 |
| Month 4 | Week 1 | V4 adapter design + implementation | ✅ Complete | 2026-05-01 |
| Month 4 | Week 2 | V4 rollout collection + frequency decomposition | ✅ Complete | 2026-05-01 |
| Month 4 | Week 3 | Figure 2 + cross-architecture evidence | ✅ Complete | 2026-05-01 |
| Month 4 | Week 4 | SMAD pre-checks + Phase 1 verdict | ✅ Complete | 2026-05-01 |
| Month 4 | Summary | Finding B supported + SMAD Path B selected | ✅ Complete | 2026-05-01 |

**Last updated:** May 1, 2026 (Month 4 complete)

## Timeline

| Phase | Status | ETA | Notes |
|---|---|---|---|
| Month 1: Setup & Baseline | ✅ Complete | 2026-04 | Infrastructure, checkpoint, adapter, and 20 v1 rollouts completed |
| Month 2: Core Diagnostics | ✅ Complete | 2026-04 | Finding A supported across two tasks with filter and window ablations complete |
| Month 3: Decoder & Training Dynamics | ✅ Complete | 2026-05 | D1-D3 complete, Finding E revised, Finding C supported with Figure 3 |
| Month 4: Cross-Architecture + SMAD Phase 1 | ✅ Complete | 2026-05 | Finding B supported with V4; SMAD Path B selected |
| Month 5: SMAD Phase 2 + Theory | ⚪ Pending | 2026-05 | Real intervention test with `U_drift` basis |
| Month 6: Revision & Submission | ⚪ Pending | 2026-06 | Consolidate findings, theorem, and paper draft |

---

## Week 1 Summary (April 28, 2026)

**Environment Setup - COMPLETE ✅**

### What was done:
- Created project skeleton (src/, configs/, results/, docs/, tests/, etc.)
- Configured `.gitignore` with comprehensive rules for Python + ML artifacts
- Set up `pyproject.toml` and `setup.cfg` for package management
- Created two isolated conda environments:
  - **env-analysis**: Python 3.11 + numpy/scipy/matplotlib/jupyter (for diagnostics)
  - **env-dreamerv3**: Python 3.10 + PyTorch 2.1.2 MPS + gymnasium + dm-control (for model inference)
- Cloned external dependencies:
  - NM512/dreamerv3-torch (commit `6ef8646`)
  - nicklashansen/dreamer4 (commit `bdeddfe`)
- Documented setup process in `docs/setup.md` and `docs/external_repos.md`

### Platform:
- macOS Apple Silicon M5 Pro
- PyTorch 2.1.2 with **MPS GPU acceleration verified** (`torch.backends.mps.is_available() = True`)
- Conda 25.11.0

### Key achievements:
- ✅ Both environments verified working (all imports successful)
- ✅ `src/` package installed as editable in both environments
- ✅ Full reproducibility documented (167-line setup guide)
- ✅ Git history clean (7 semantic commits)

### Next steps (Week 2):
1. Check NM512 repo for pretrained DreamerV3 checkpoints
2. Implement `src/diagnostics/extract_rollout.py`
3. Run first rollout extraction on Cheetah Run
4. Create sanity check notebook

**Compute used:** ~0 GPU-hours (setup only, no training)  
**Remaining budget:** ~200 GPU-hours

---

## Week 2 Summary (April 28, 2026)

**DreamerV3 Baseline & Pipeline Validation - COMPLETE ✅**

### What was done:
1. **Checkpoint exploration (Task 1)**:
   - Investigated NM512/dreamerv3-torch for pretrained weights
   - **Conclusion**: No pretrained checkpoints available
   - Evidence: Repo marked "Outdated", maintainer confirmed no plans to share (Issue #70)
   - Alternative: Use random-init agent for pipeline validation

2. **Pipeline implementation (Task 2)**:
   - Created `src/diagnostics/extract_rollout.py` (304 lines)
   - Open-loop RSSM imagination via `dynamics.img_step()`
   - Verified on dmc_cheetah_run with MPS device
   - Data format validated: true_obs vs imagined_obs aligned correctly

3. **External dependencies**:
   - Added: gym, tensorboard, ruamel.yaml to env-dreamerv3
   - Applied temporary patches to external/dreamerv3-torch for MPS support:
     - `envs/dmc.py`: Fix list+tuple concatenation
     - `networks.py`, `tools.py`: Change default device from "cuda" to "cpu"
     - `models.py`: Pass device parameter to ImagBehavior.actor

### Key findings:
- ✅ Pipeline works end-to-end: env reset/step, imagination loop, decoder, NPZ storage
- ✅ Data shapes correct: (200, 17) obs, (200, 6) actions for Cheetah Run
- ✅ MPS device functional for inference (M5 Pro GPU acceleration working)
- ⚠️ No scientifically meaningful rollouts yet (random weights as expected)

### Technical validation:
```python
# Verified data format
true_obs:      (200, 17)  # position(8) + velocity(9)
imagined_obs:  (200, 17)  # same shape, aligned to actions
actions:       (200, 6)   # Cheetah joint actions
rewards:       (200,)
metadata:      task, seed, alignment info
```

### Next steps (Week 2 remainder):
1. Create sanity check notebook (`notebooks/01_sanity_rollouts.ipynb`)
2. Visualize true vs imagined trajectories
3. Document external patches for reproducibility
4. Week 3-4: Explore checkpoint sources or test M5 Pro training feasibility

**Compute used:** ~0 GPU-hours (pipeline validation only, ~5 min MPS inference)  
**Remaining budget:** ~200 GPU-hours

---

## Week 3 Summary (April 29, 2026)

**DreamerV3 Checkpoint Training - COMPLETE ✅**

### What changed from the original plan:
- Week 3 was originally scoped for adapter design.
- It was reallocated to checkpoint training after confirming that no public DreamerV3 checkpoints were available for DMC.
- Scope was narrowed to a single high-value task, `dmc_cheetah_run`, to secure a strong baseline within Month 1.

### Training run:
- Platform: RunPod cloud GPU
- Hardware: NVIDIA RTX 4090 (24GB)
- Steps: 505k (target 500k)
- Batch size: 32
- Parallel environments: 8
- Wall-clock time: ~9.5 hours
- Cost: ~$6.65

### Key results:
- ✅ Peak `eval_return`: 791.1 at Step 485k
- ✅ Stable `eval_return`: 730-750
- ✅ Average `train_return`: ~715-720
- ✅ Performance exceeded the DreamerV3 paper upper range for this setting (650-750)

### Output artifacts:
- `results/checkpoints/cheetah_run_500k.pt`
- `results/checkpoints/manifest.json`
- `docs/training_record_cheetah_500k.md`

### Next steps (Week 4):
1. Verify checkpoint integrity and adapter assumptions
2. Run `extract_rollout.py` against the trained checkpoint
3. Batch extract rollouts for multiple seeds
4. Begin DreamerV4 reconstruction path for comparison

**Compute used:** ~9.5 RTX 4090 GPU-hours  
**Checkpoint status:** Ready for Week 4 rollout extraction

---

## Month 3 Summary (May 1, 2026)

**Decoder Amplification & Training Dynamics - COMPLETE ✅**

See [`docs/month3_summary.md`](./docs/month3_summary.md) for the full Month 3 record.

Release tag: `v0.3-month3-dynamics-decoder`

### Decoder amplification:
- ✅ D1 random decoder control complete: trained decoder amplifies drift by ~48% relative to a random decoder.
- ✅ D2 Lipschitz probe complete: amplification is **not** directional toward `dc_trend`; `dc_trend` has the smallest effective Lipschitz constant.
- ✅ D3 linear/nonlinear separation complete: decoder nonlinearity appears as constant OOD noise, not a horizon-growing amplification mechanism.
- ✅ Finding E verdict: **not supported** as an independent finding. Decoder amplification is uniform OOD instability, while trend dominance is latent-intrinsic.

### Training dynamics:
- ✅ High-frequency Cheetah checkpoint archive complete: 102 checkpoints from step 5k to 510k.
- ✅ Peak `eval_return`: 791.3 at step 485k, matching the Month 1 peak of 791.1.
- ✅ Training dynamics pipeline complete: 102 checkpoints x 5 bands, horizon 200, trim `[25, 175]`.
- ✅ Figure 3 complete: curves, heatmap, and fractional share plots.
- ✅ Finding C supported: `dc_trend` dominance is present from step 5000 and remains the top band throughout training.
- Key interpretation: training does not create trend dominance; it slightly alleviates it from 79% share at step 5k to about 33% by step 500k.

### V4 Month 4 readiness:
- ✅ `docs/v4_hf_deep_recon.md` complete.
- ✅ `docs/v4_decision.md` complete.
- ✅ Month 4 V4 comparison is GO (conditional), limited to inference-only RunPod CUDA work using HF checkpoints.
- Fallback if V4 adapter fails by Month 4 Week 2: PlaNet or additional DreamerV3 tasks.

## Month 4 Summary (May 1, 2026)

**DreamerV4 Cross-Architecture + SMAD Phase 1 - COMPLETE ✅**

See [`docs/month4_summary.md`](./docs/month4_summary.md) for the full Month 4 record.

Release tag: `v0.4-month4-v4-smad-precheck`

### DreamerV4 cross-architecture comparison:
- ✅ V4 adapter infrastructure complete: CUDA-only adapter, smoke tests, batch extractor, and RunPod environment.
- ✅ V4 rollout collection complete: 20 Cheetah seeds, 200 imagination steps, image observations and 512-dimensional packed tokenizer latents.
- ✅ V4 frequency decomposition complete: latent, channel-mean image, and image-PCA diagnostics.
- ✅ Figure 2 complete: V3 solid lines vs V4 dashed lines across five frequency bands.
- ✅ Finding B supported: V4 latent `dc_trend` share is 63.9%, higher than the V3 Cheetah baseline of 46.7%.
- Key interpretation: trend-dominated imagination drift survives the shift from GRU-RSSM proprioception to transformer-denoising image-token dynamics.

### SMAD Phase 1:
- ✅ Pre-check 1 complete: `U_posterior` is NO GO, with overlap only 0.132 to 0.283 for ranks 3 to 10.
- ✅ Pre-check 2 complete: `U_drift` post-hoc damping reaches 47.75% `dc_trend` reduction at `r=10, eta=0.50`.
- ✅ Phase 1 verdict complete: Path B is WEAK GO with `U_drift`, conservative `eta=0.20`, and a narrow Month 5 scope.
- Key interpretation: posterior slow modes are not aligned enough for intervention, but empirical drift directions are a plausible Phase 2 basis.

## Key References

- Hafner et al. (2023), **DreamerV3**
- Hafner et al. (2025), **DreamerV4**
- **TritonCast** (arXiv:2505.19432)
- **THICK Dreamer** (ICLR 2024)

## Compute Budget

- 💻 **Hardware**: Single RTX 4090 (24GB)
- 📊 **Total Budget**: ~200 GPU-hours
- ✅ **Cheetah training**: ~9.5 GPU-hours, ~$6.65
- ✅ **Cartpole training**: ~10 GPU-hours, ~$7.00
- ✅ **Month 3 high-frequency Cheetah training**: ~20 GPU-hours, ~$14.00 (two runs)
- ✅ **Month 4 V4 work**: ~0.7 GPU-hours, ~$0.50
- 💸 **Total spent**: ~40.2 GPU-hours, ~$28.15
- ⏳ **Remaining budget**: ~159.8 GPU-hours, ~$112
- ✅ **Strategy**: Use pre-trained checkpoints when available, and run targeted training only when necessary to unblock the diagnostic pipeline

## Repository Structure

- `docs/` — research notes, design documents, and project architecture.
- `src/diagnostics/` — core diagnostic code for frequency decomposition and rollout analysis.
- `src/models/` — model-loading utilities and interfaces for pretrained world models.
- `src/analysis/` — statistical analysis, summaries, and plotting utilities.
- `src/utils/` — shared helpers and general-purpose utilities.
- `experiments/` — experiment configs, scripts, and run-specific outputs.
- `results/rollouts/` — rollout datasets, manifests, and window ablation outputs.
- `results/rollouts/window_ablation/` — imagination-start ablation rollouts.
- `results/figures/` — generated plots and visual results.
- `results/tables/` — tabular summaries and evaluation outputs.
- `notebooks/` — exploratory notebooks and prototype analysis.
- `configs/` — configuration files for experiments and evaluation.

## Getting Started

Detailed instructions will be added in Phase 1.

## License & Acknowledgments

- **License**: TBD
- **Acknowledgments**: Special thanks to [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) and [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) for open implementations that make checkpoint-based analysis possible.
