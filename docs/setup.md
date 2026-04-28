# DriftDetect Environment Setup Guide

**Last updated:** April 28, 2026  
**Platform:** macOS Apple Silicon (M5 Pro)

This guide covers setting up the DriftDetect development environment from scratch on macOS with Apple Silicon.

---

## Prerequisites

### Hardware
- macOS machine with Apple Silicon (M1/M2/M3/M4/M5)
- Recommended: 16GB+ RAM (unified memory shared between CPU/GPU)
- ~10GB free disk space for conda environments

### Software
- macOS 11.0 or later
- Command Line Tools for Xcode (install via `xcode-select --install`)
- Git (comes with Command Line Tools)

---

## Step 1: Install Miniconda

If you don't have conda installed:

1. Download Miniconda for macOS Apple Silicon (arm64):
```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

2. Install:
```bash
   bash Miniconda3-latest-MacOSX-arm64.sh
```
   - Accept the license
   - Install to default location (`~/miniconda3`)
   - Say "yes" to `conda init`

3. Close and reopen your terminal (or run `source ~/.zshrc`)

4. Verify:
```bash
   conda --version  # Should show 25.11.0 or newer
```

---

## Step 2: Clone the Repository

```bash
cd ~/Desktop  # or wherever you want the project
git clone https://github.com/FengYe476/DriftDetect.git
cd DriftDetect
```

---

## Step 3: Create Conda Environments

### Environment 1: env-analysis (for diagnostics and plotting)

```bash
conda env create -f configs/env-analysis.yaml
```

This creates a Python 3.11 environment with numpy, scipy, matplotlib, jupyter, and dev tools.

**Verify:**
```bash
conda activate env-analysis
python -c "import numpy, scipy, matplotlib; print('Core packages OK')"
python -c "from src.diagnostics import *; print('src importable')"
conda deactivate
```

Expected output: Both commands should print "OK" / "importable" with no errors.

---

### Environment 2: env-dreamerv3 (for running world models)

```bash
conda env create -f configs/env-dreamerv3.yaml
```

This creates a Python 3.10 environment with PyTorch 2.1.2 (MPS support), gymnasium, dm-control, and mujoco.

**Installation time:** 10-20 minutes (downloads ~70MB of packages).

**Verify:**
```bash
conda activate env-dreamerv3
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}')"
python -c "import gymnasium, dm_control, mujoco; print('All OK')"
conda deactivate
```

Expected output:
````
PyTorch 2.1.2, MPS available: True
All OK
````

**Important:** `MPS available: True` confirms GPU acceleration is working on your Apple Silicon chip.

---

## Step 4: Clone External Repositories

```bash
cd external
git clone https://github.com/NM512/dreamerv3-torch.git
git clone https://github.com/nicklashansen/dreamer4.git
cd ..
```

These are third-party implementations of DreamerV3 and DreamerV4 used for checkpoint loading and inference.

---

## Troubleshooting

### Issue: `MPS available: False`

If PyTorch can't detect the GPU:
- Verify you're on Apple Silicon (run `uname -m`, should show `arm64`)
- Update macOS to 12.3+ (required for MPS)
- Reinstall pytorch: `conda install pytorch=2.1.* torchvision -c pytorch`

### Issue: `dm-control` or `mujoco` installation fails

These packages can be finicky on macOS. If pip fails:
```bash
conda activate env-dreamerv3
pip install --no-build-isolation dm-control mujoco
```

### Issue: Path with spaces causes pip errors

If your project path has spaces (e.g., "DriftDetect 2"), the `-e` flag in the yaml should have quotes:
```yaml
- -e "/path/to/DriftDetect 2/..."
```

This is already handled in the repo's yaml files.

---

## Verification Checklist

After setup, verify everything works:

- [ ] `conda env list` shows both `env-analysis` and `env-dreamerv3`
- [ ] Both environments can import their key packages (see verification commands above)
- [ ] `external/dreamerv3-torch/` and `external/dreamer4/` directories exist
- [ ] `src/` is importable in both environments (editable install worked)

---

## Next Steps

- **Week 2:** Start working on `src/diagnostics/extract_rollout.py` (use env-dreamerv3)
- **Analysis work:** Use env-analysis for notebooks and frequency-domain diagnostics

See [README.md](../README.md) for the full project timeline.
