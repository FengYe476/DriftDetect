# DriftDetect
### Frequency-Domain Diagnostic for World Model Latent Drift

![Status](https://img.shields.io/badge/Status-Month%201%20Week%203%20Complete-1f6feb)
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

Week 3 completed (Month 1, Week 3 of 4).

- [x] Proposal
- [x] Environment setup (macOS M5 Pro, PyTorch 2.1.2 MPS, conda 25.11.0)
- [x] DreamerV3 baseline and rollout extraction pipeline
- [x] DreamerV3 checkpoint training on `dmc_cheetah_run`
- [🔄] Week 4: Adapter + batch rollouts + DreamerV4 reconstruction
- [ ] Cross-architecture analysis
- [ ] Toy validation
- [ ] Paper draft

### Recent Achievement

- Successfully trained DreamerV3 on the `dmc_cheetah_run` task
- Peak performance: 791.1 (exceeds DreamerV3 paper SOTA range of 650-750)
- Stable performance: 730-750
- Training run: 505k steps in 9.5 hours on RunPod RTX 4090
- Cost: $6.65

### Progress Table

| Phase | Task | Status | Date |
|---|---|---|---|
| Week 1 | Infrastructure setup | ✅ Complete | 2026-04 |
| Week 2 | `extract_rollout.py` | ✅ Complete | 2026-04 |
| Week 3 | DreamerV3 checkpoint training | ✅ Complete | 2026-04-29 |
| Week 4 | Adapter + batch rollouts + V4 recon | 🔄 Starting | 2026-05 (target) |

**Last updated:** April 29, 2026 (Week 3 completed)

## Timeline

| Phase | Status | ETA | Notes |
|---|---|---|---|
| Month 1: Setup & Baseline | 🟢 Week 3 Complete | 2026-05 | Checkpoint trained, Week 4 adapter/rollouts starting |
| Month 2: Core Diagnostics | ⚪ Not started | 2026-06 | Next: batch rollout extraction and frequency analysis |
| Month 3: Cross-Architecture Analysis | ⚪ Not started | 2026-07 | |
| Month 4: Theory & Toy Validation | ⚪ Not started | 2026-08 | |
| Month 5: Analysis & Writing | ⚪ Not started | 2026-09 | |
| Month 6: Revision & Submission | ⚪ Not started | 2026-10 | |

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

## Key References

- Hafner et al. (2023), **DreamerV3**
- Hafner et al. (2025), **DreamerV4**
- **TritonCast** (arXiv:2505.19432)
- **THICK Dreamer** (ICLR 2024)

## Compute Budget

- 💻 **Hardware**: Single RTX 4090 (24GB)
- 📊 **Total Budget**: ~200 GPU-hours
- ✅ **Strategy**: Use pre-trained checkpoints when available, and run targeted training only when necessary to unblock the diagnostic pipeline

## Repository Structure

- `docs/` — research notes, design documents, and project architecture.
- `src/diagnostics/` — core diagnostic code for frequency decomposition and rollout analysis.
- `src/models/` — model-loading utilities and interfaces for pretrained world models.
- `src/analysis/` — statistical analysis, summaries, and plotting utilities.
- `src/utils/` — shared helpers and general-purpose utilities.
- `experiments/` — experiment configs, scripts, and run-specific outputs.
- `results/figures/` — generated plots and visual results.
- `results/tables/` — tabular summaries and evaluation outputs.
- `notebooks/` — exploratory notebooks and prototype analysis.
- `configs/` — configuration files for experiments and evaluation.

## Getting Started

Detailed instructions will be added in Phase 1.

## License & Acknowledgments

- **License**: TBD
- **Acknowledgments**: Special thanks to [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) and [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4) for open implementations that make checkpoint-based analysis possible.
