# DriftDetect
### Frequency-Domain Diagnostic for World Model Latent Drift

![Status](https://img.shields.io/badge/Status-Proposal%20Phase-1f6feb)
![Compute Budget](https://img.shields.io/badge/Compute-~200%20GPU--hours-2ea44f)
![Timeline](https://img.shields.io/badge/Timeline-6%20months-f59e0b)

**TL;DR:** DriftDetect is a compute-efficient research project that systematically diagnoses how existing world models drift over long horizons by analyzing rollout failure in the frequency domain.

## What is this?

DriftDetect is a research project on latent drift in world models for reinforcement learning, especially in long-horizon imagination and planning. While recent world models have improved substantially, we still lack a mechanistic understanding of which temporal components fail first and how those failures accumulate. This project studies that question through frequency-domain diagnostics, using mostly pre-trained checkpoints rather than large-scale retraining. In other words, the goal is to **diagnose existing models**, not to propose a new world model method.

## Key Contributions

- 🎯 **Diagnostic Protocol**: A frequency-domain evaluation pipeline for measuring how rollout error grows across temporal bands.
- 📐 **Theory**: A lightweight theoretical component on slow-subspace identifiability in a simplified setting.
- 🔍 **Implications**: Empirical insights that can inform future world model architectures and training objectives.

See [PROPOSAL.md](./PROPOSAL.md) for the full proposal.

## Current Status

- [x] Proposal
- [x] Environment setup (macOS M5 Pro, PyTorch 2.1.2 MPS, conda 25.11.0)
- [🔄] DreamerV3 baseline (external repos cloned, checkpoint exploration pending)
- [ ] Diagnostic pipeline
- [ ] Cross-architecture analysis
- [ ] Toy validation
- [ ] Paper draft

**Last updated:** April 28, 2026 (Week 1 complete)

## Timeline

| Phase | Status | ETA | Notes |
|---|---|---|---|
| Month 1: Setup & Baseline | 🟢 Week 1 Complete | 2026-05 | macOS M5 Pro, PyTorch MPS, envs verified |
| Month 2: Core Diagnostics | ⚪ Not started | 2026-06 | Next: extract_rollout.py |
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

## Key References

- Hafner et al. (2023), **DreamerV3**
- Hafner et al. (2025), **DreamerV4**
- **TritonCast** (arXiv:2505.19432)
- **THICK Dreamer** (ICLR 2024)

## Compute Budget

- 💻 **Hardware**: Single RTX 4090 (24GB)
- 📊 **Total Budget**: ~200 GPU-hours
- ✅ **Strategy**: Rely on pre-trained checkpoints and inference-only analysis whenever possible

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
