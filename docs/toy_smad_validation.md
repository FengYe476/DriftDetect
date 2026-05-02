# Toy SMAD Validation

**Date:** 2026-05-02
**Status:** Completed CPU smoke/full run.

## Purpose

The toy experiment validates the SMAD intervention on a small controllable RSSM before Phase 2 DreamerV3 training. It tests whether frequency-structured imagination drift and SMAD damping effects can be reproduced in a minimal oscillator environment.

## Setup

- Environment: dual oscillator with slow and fast components.
- Observation dimension: 4 (`x_slow`, `x_fast`, `dx_slow`, `dx_fast`).
- RSSM: Gaussian stochastic state, `deter_dim=64`, `stoch_dim=8`.
- Training: 2000 steps on CPU.
- Horizon: 50.
- Evaluation rollouts: 10.
- Trim window: `[5, 45]`.

## Full Run Results

Subspace overlaps:

| Rank | SFA/posterior | SFA/drift | posterior/drift |
|---:|---:|---:|---:|
| 3 | 0.1569 | 0.1397 | 0.8806 |
| 5 | 0.3128 | 0.2100 | 0.7664 |
| 10 | 0.6700 | 0.4918 | 0.7888 |

Baseline frequency-band MSE:

| Band | MSE | Share |
|---|---:|---:|
| dc_trend | 0.345864 | 19.3% |
| very_low | 0.295017 | 16.4% |
| low | 0.455098 | 25.4% |
| mid | 0.515815 | 28.7% |
| high | 0.182704 | 10.2% |

SMAD damping:

| Eta | J_slow reduction | J_total reduction |
|---:|---:|---:|
| 0.10 | -49.5% | 16.0% |
| 0.20 | -39.2% | 32.0% |
| 0.30 | -37.6% | 39.3% |

## Interpretation

The simple toy setting does not reproduce Cheetah's strong dc_trend dominance. Instead, error mass is spread across low and mid bands, with `dc_trend` only 19.3% of total band MSE. This supports the idea that strong trend dominance is not automatic in every RSSM, even when the environment contains slow latent structure.

Unlike Cheetah Pre-check 3, the toy posterior and drift subspaces align strongly (`posterior/drift` overlap 0.77-0.88). This suggests the Cheetah mismatch may arise from task/model complexity rather than a universal property of RSSMs.

Damping reduces total band MSE substantially, but it increases the dc_trend component in this simple setting. That is scientifically useful: SMAD's effectiveness appears to depend on the drift subspace and dominant frequency structure being aligned with the target low-frequency error mode.

## Follow-up

Run a complexity-scaling sweep with fixed RSSM capacity and increasing environment dimensionality to test whether `dc_trend` dominance grows as the capacity/complexity gap widens.

## References

- Script: `scripts/run_toy_smad.py`
- Results: `results/tables/toy_smad_results.json`
- Toy modules: `src/toy/dual_oscillator_env.py`, `src/toy/minimal_rssm.py`, `src/toy/smad_intervention.py`
