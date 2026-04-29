# DriftDetect Quickstart

## Setup

1. Clone the repo.
2. Create environments:

```bash
conda env create -f configs/env-dreamerv3.yaml
conda env create -f configs/env-analysis.yaml
```

3. Download or verify the checkpoint if it is not already present:

See `results/checkpoints/manifest.json`

## Reproduce 20 rollouts

```bash
conda activate env-dreamerv3
python scripts/batch_extract_v3.py
```

## Run sanity check

```bash
conda activate env-analysis
jupyter notebook notebooks/01_sanity_rollouts.ipynb
```

## Run tests

```bash
conda activate env-dreamerv3
pytest tests/test_v3_adapter.py
```

## Expected outputs

- 20 `.npz` files in `results/rollouts/`
- L2 distance grows from about `3-5` at step 0 to about `10-20` by step 100 and beyond
- All 4 adapter tests pass
