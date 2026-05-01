# Month 4 Summary - DriftDetect

## Executive Summary

Month 4 completed the cross-architecture DreamerV4 comparison and the SMAD
Phase 1 pre-checks.

Finding B is SUPPORTED: V4's image-token world model shows stronger
trend-dominated latent drift than V3, with `dc_trend` carrying 63.9 percent of
V4 latent MSE versus 46.7 percent for V3.

Figure 2 is complete and compares V3 solid curves against V4 dashed curves.

The observation-space alignment methodology is documented.

The DreamerV4 adapter, smoke tests, batch extractor, and CUDA environment file
are implemented.

Twenty V4 Cheetah rollouts were collected on RTX 4090 using HF checkpoints and
dataset trajectories.

SMAD Pre-check 1 was a NO GO for `U_posterior`: overlap with the empirical
drift subspace was only 0.132 to 0.283 for ranks 3 to 10.

SMAD Pre-check 2 was a conditional PASS for `U_drift`: post-hoc damping reached
47.75 percent `dc_trend` reduction at `r=10, eta=0.50`.

The combined SMAD verdict is Path B, WEAK GO with `U_drift` basis and a
conservative Phase 2 scope.

The three major Month 4 outputs are V4 frequency decomposition with Figure 2,
SMAD Phase 1 verdict, and V4 adapter infrastructure.

## Deliverables Status

| Deliverable | Status | Notes |
|---|---|---|
| V4 adapter design | Complete | `docs/v4_adapter_design.md` |
| V4 adapter implementation | Complete | `src/models/dreamerv4_adapter.py` |
| V4 smoke tests | Complete | `tests/test_v4_adapter.py`; RunPod CUDA-only |
| V4 batch extractor | Complete | `scripts/batch_extract_v4.py` |
| V4 RunPod environment | Complete | `configs/env-dreamerv4.yaml` |
| V4 rollouts | Complete | 20 seeds in `results/rollouts_v4/`; input artifacts not committed |
| V4 frequency decomposition | Complete | `scripts/run_v4_freq_decompose.py` and `results/tables/v4_band_errors.json` |
| Figure 2 | Complete | `src/analysis/figure2.py`; `results/figures/figure2_v3_vs_v4_v1.pdf` |
| Observation alignment document | Complete | `docs/observation_space_alignment.md` |
| Finding B evidence | Complete | `docs/finding_b_evidence.md` |
| SMAD Pre-check 1 | Complete | `scripts/run_smad_alignment_check.py`, `docs/smad_precheck_alignment.md`, `results/tables/smad_alignment_check.json` |
| SMAD Pre-check 1 figures | Complete | `results/figures/smad_alignment_*.pdf` |
| SMAD Pre-check 2 | Complete | `scripts/run_smad_eta_sweep.py`, `docs/smad_precheck_eta_sweep.md`, `results/tables/smad_eta_sweep.json` |
| SMAD Pre-check 2 figures | Complete | `results/figures/smad_eta_sweep_*.pdf` |
| SMAD Phase 1 verdict | Complete | `docs/smad_phase1_verdict.md` |
| Month 4 summary | Complete | This file |

## Key Metrics

### V4 Rollout Collection

- Task: `cheetah-run`.
- Seeds: 20, seeds 0 through 19.
- Context length: 8 frames.
- Horizon: 200 imagination steps.
- True and imagined latents: `(200, 512)`.
- True and imagined images: `(200, 3, 128, 128)`.
- Mean extraction time from manifest: 3.418918360339012 seconds per seed.
- Maximum extraction time: 4.095765591016971 seconds.
- Approximate Month 4 V4 GPU budget: 0.7 GPU-hours on RTX 4090.
- Approximate Month 4 V4 cost: about $0.50.

### V4 Frequency Decomposition

Latent-space, primary diagnostic:

- Input: 512-dimensional packed tokenizer latents.
- PCA retained 35 components.
- PCA variance explained: 0.9507728441475489.
- Trim window: `[25, 175]`.
- `dc_trend` mean MSE: 1.5370213002262525.
- `dc_trend` share: 63.85715847509169 percent.
- Band ranking: `dc_trend > low > mid > high > very_low`.

Image-PCA, secondary diagnostic:

- PCA retained 50 components.
- Capped image-PCA variance explained: 0.8111482382446757.
- `dc_trend` mean MSE: 2.2491048731353898.
- `dc_trend` share: 44.13059258253914 percent.
- Band ranking: `dc_trend > low > mid > high > very_low`.

Cross-architecture comparison:

| Band | V3 latent share | V4 latent share | V4 image-PCA share |
|---|---:|---:|---:|
| `dc_trend` | 46.7% | 63.9% | 44.1% |
| `very_low` | 10.4% | 1.9% | 2.7% |
| `low` | 28.4% | 20.3% | 25.2% |
| `mid` | 10.3% | 9.8% | 15.9% |
| `high` | 4.2% | 4.2% | 12.1% |

### SMAD Phase 1

Pre-check 1, `U_posterior` alignment:

| r | overlap | drift energy in `U_posterior` | verdict |
|---:|---:|---:|---|
| 3 | 0.13226411396636534 | 0.061430762849638415 | NO GO |
| 5 | 0.1674721367245057 | 0.12859728054913727 | NO GO |
| 10 | 0.2827199884234227 | 0.19044422968948987 | NO GO |

Pre-check 2, post-hoc `U_drift` damping:

- Frequency PCA retained 38 components and explained 0.9504360611578495
  variance.
- `r=10, eta=0.20`: `dc_trend` reduction 23.19819459702299 percent.
- `r=10, eta=0.20`: total MSE change -24.472830609668662 percent.
- `r=10, eta=0.50`: `dc_trend` reduction 47.7488897838189 percent.
- `r=10, eta=0.50`: total MSE change -50.985063770143014 percent.
- Phase 1 verdict: Path B, WEAK GO with `U_drift` basis.

## Finding Status

| Finding | Status | Evidence |
|---|---|---|
| Finding A: frequency-asymmetric drift with trend dominance | SUPPORTED | Month 2 Cheetah and Cartpole diagnostics; V3 Cheetah `dc_trend` share 46.7% |
| Finding B: cross-architecture trend-dominated drift | SUPPORTED | Month 4 V4 latent `dc_trend` share 63.9%; Figure 2 |
| Finding C: training dynamics | SUPPORTED | Month 3 showed `dc_trend` dominance from step 5000 across 102 Cheetah checkpoints |
| Finding D: SMAD intervention | PENDING Month 5 | Phase 2 Path B with `U_drift` basis remains to be tested |
| Finding E: decoder amplification | NOT SUPPORTED as independent finding | Month 3 D1-D3 show uniform OOD amplification, not directional `dc_trend` amplification |

## Finding B Interpretation

The V4 result upgrades the central empirical story.

Month 2 showed that DreamerV3 Cheetah and Cartpole have trend-dominated
latent drift. Month 4 shows that the same qualitative pattern appears in a
fundamentally different world model: image-based DreamerV4 with transformer
tokenizer latents and iterative denoising dynamics.

The strongest evidence is that V4's `dc_trend` share is higher than V3's:

```text
V3 Cheetah latent dc_trend = 46.7%
V4 Cheetah latent dc_trend = 63.85715847509169%
```

The top-two ranking is identical:

```text
dc_trend > low
```

The high-frequency share is nearly identical:

```text
V3 high = 4.2%
V4 high = 4.181199605250795%
```

This supports the claim that trend-dominated drift is not merely a GRU/tanh RSSM
artifact. It appears to be a broader property of autoregressive world-model
imagination.

## SMAD Phase 1 Interpretation

SMAD Phase 1 produced a mixed but actionable result.

The original posterior slow-mode idea did not pass. `U_posterior` has too little
overlap with empirical drift directions, so it should not be the Month 5 basis.

The alternate `U_drift` basis did pass a post-hoc screen. Damping along empirical
drift PCs reduced both `dc_trend` and total MSE in the existing V3 rollouts.

This is not a full intervention result. It is a linear post-hoc approximation
that does not re-run the RSSM. The next question is whether the effect transfers
to real recurrent dynamics.

Month 5 should therefore run one careful Path B configuration:

```text
basis = U_drift
r = 10
eta = 0.20 conservative default
scope = combined damping + anchor
```

## Plan vs Actual

| Month 4 Plan | Actual | Outcome |
|---|---|---|
| Write V4 adapter design | Completed `docs/v4_adapter_design.md` | V4 implementation risks clarified |
| Implement V4 adapter and RunPod scripts | Completed adapter, smoke tests, batch extractor, and env YAML | V4 infrastructure ready |
| Collect V4 rollouts | Completed 20 seeds, 200-step horizon | Inputs ready for Finding B |
| Run V4 frequency decomposition | Completed latent, channel-mean, and image-PCA analyses | Finding B supported |
| Generate Figure 2 | Completed `figure2_v3_vs_v4_v1.pdf` | Cross-architecture plot ready |
| Document observation alignment | Completed `docs/observation_space_alignment.md` | Methodology risk addressed |
| Run SMAD pre-checks | Completed alignment check and eta sweep | Path B selected |
| Decide SMAD Phase 2 path | Completed `docs/smad_phase1_verdict.md` | Weak GO with `U_drift` |
| Draft Finding B evidence | Completed `docs/finding_b_evidence.md` | Paper claim upgraded |

## Key Discoveries Not in Original Plan

1. V4 did not merely replicate V3 trend dominance; it amplified it in latent
   space.
2. V4 image-PCA also ranked `dc_trend` first, but with a lower 44.1 percent
   share than the latent metric.
3. Posterior slow modes and empirical drift directions are not the same object.
4. The original SMAD `U_posterior` basis failed the alignment threshold at every
   requested rank.
5. Post-hoc damping along `U_drift` was much stronger at `r=10` than at `r=3`
   or `r=5`.
6. The conservative SMAD setting `r=10, eta=0.20` already clears the 20 percent
   `dc_trend` reduction threshold in the post-hoc screen.
7. V4 inference was much cheaper than expected once the adapter was working.

## Compute Budget

| Item | GPU-hours | Cost | Notes |
|---|---:|---:|---|
| Cheetah training, Month 1 | ~9.5 | ~$6.65 | DreamerV3 Cheetah checkpoint |
| Cartpole training, Month 2 | ~10 | ~$7.00 | DreamerV3 Cartpole checkpoint |
| Month 3 high-frequency Cheetah training | ~20 | ~$14.00 | Two runs, one successful archive |
| Month 4 V4 work | ~0.7 | ~$0.50 | Adapter smoke tests and 20 V4 rollouts |
| Total spent | ~40.2 | ~$28.15 | Cloud GPU cost only |
| Remaining budget | ~159.8 | ~$112 | Enough for Month 5 SMAD Phase 2 |

Local CPU/MPS analysis time is not counted in cloud GPU spend.

## Risk Assessment for Month 5

1. Post-hoc damping may not transfer to real RSSM recurrence. This is the main
   risk for Finding D.
2. `U_drift` is estimated from baseline drift vectors, so there is circularity
   in the pre-check. Phase 2 must be evaluated on real re-run rollouts.
3. The fixed `U_drift` basis may become stale during SMAD training.
4. Damping may reduce error by collapsing useful slow variance. Representation
   health checks are mandatory.
5. `eta=0.20` may be too weak in the real intervention, even though it passed
   the post-hoc screen.
6. `eta=0.50` may be too aggressive despite the large post-hoc gain.
7. The Month 5 scope must stay narrow to preserve compute and schedule.

## Month 5 Readiness Checklist

- [x] Finding A supported.
- [x] Finding B supported with DreamerV4.
- [x] Finding C supported.
- [x] Finding E revised and documented.
- [x] V4 adapter infrastructure complete.
- [x] Figure 2 complete.
- [x] Observation-space alignment documented.
- [x] SMAD Pre-check 1 complete.
- [x] SMAD Pre-check 2 complete.
- [x] SMAD Phase 1 verdict complete.
- [ ] Implement real SMAD Path B patch in DreamerV3.
- [ ] Verify identity run with `eta=0, beta=0`.
- [ ] Estimate fixed `U_drift` from baseline calibration rollouts.
- [ ] Run one combined damping-plus-anchor configuration.
- [ ] Evaluate `J_slow`, band MSE, eval return, and representation health.
- [ ] Decide whether Finding D is supported.

## Month 5 Default Plan

Run one SMAD Phase 2 configuration:

```text
basis = U_drift
r = 10
eta = 0.20
anchor = enabled
beta search = {0.01, 0.1, 1.0} only if needed
```

Proceed to broader ablations only if the single configuration succeeds and
compute remains.

## Git Tag

Month 4 release tag: `v0.4-month4-v4-smad-precheck`
