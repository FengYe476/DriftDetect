# SMAD Phase 1 Verdict: Path B Weak GO with U_drift Basis

Date: 2026-05-01

Status: MONTH 4 DECISION COMPLETE

## 1. Executive Summary

SMAD Phase 1 produced a mixed but actionable result. Pre-check 1 is a clear
NO GO for using `U_posterior` as the SMAD basis: the posterior slow subspace
estimated from PCA plus lag-1 autocorrelation does not align with the empirical
drift subspace. Pre-check 2 is a conditional PASS for damping along `U_drift`:
post-hoc projection of cumulative drift along empirical drift PCs reduces
`dc_trend` MSE and total MSE on existing Cheetah rollouts. The combined verdict
is Path B: WEAK GO with caveats. SMAD Phase 2 may proceed, but only with
`U_drift` as the basis, a conservative `r=10, eta=0.20` setting, and a reduced
single-configuration scope. This is not Path A because the original slow-mode
hypothesis failed. It is not Path C because drift-direction damping is
empirically promising. The decisive Phase 2 question is whether the post-hoc
damping result transfers to a real RSSM re-run with recurrent dynamics.

## 2. Pre-check 1 Recap

Pre-check 1 tested the central Month 5 assumption that posterior slow modes
would be a usable proxy for imagination drift directions.

Data:

```text
results/rollouts/cheetah_v3_seed{0..19}_v2.npz
true_latent, imagined_latent: (200, 1536)
true_deter = true_latent[:, 1024:]
imag_deter = imagined_latent[:, 1024:]
```

Method:

1. Pool true posterior deter trajectories across 20 seeds: `(4000, 512)`.
2. Fit PCA and retain 38 PCs, explaining 0.9504360611578495 variance.
3. Score retained PCs by lag-1 autocorrelation, averaged across seeds.
4. Select the top `r` autocorrelated PCs as `U_posterior`.
5. Compute drift vectors `Delta_t = imag_deter_t - true_deter_t`.
6. Trim first and last 25 steps, producing a drift PCA matrix `(3000, 512)`.
7. Select the top `r` drift PCs as `U_drift`.

Metrics:

```text
overlap = ||U_posterior.T @ U_drift||_F^2 / r
variance ratio = sum ||U_posterior.T @ Delta_t||^2 / sum ||Delta_t||^2
```

Exact results from `results/tables/smad_alignment_check.json`:

| r | overlap | drift energy in U_posterior | verdict |
|---:|---:|---:|---|
| 3 | 0.13226411396636534 | 0.061430762849638415 | NO GO with U_posterior |
| 5 | 0.1674721367245057 | 0.12859728054913727 | NO GO with U_posterior |
| 10 | 0.2827199884234227 | 0.19044422968948987 | NO GO with U_posterior |

Interpretation: all requested ranks fall below the 0.4 weak-go threshold. At
`r=3`, posterior slow modes capture only 0.061430762849638415 of raw drift
energy. At `r=10`, they still capture only 0.19044422968948987. Posterior slow
modes and drift directions are therefore nearly orthogonal for DreamerV3
Cheetah Run. The intuition that "drift equals slow subspace wandering" is
empirically wrong for this checkpoint and rollout distribution. Drift
accumulates along directions that are not the highest-autocorrelation posterior
PCs.

Paper value: this is itself a reportable finding. "Drift subspace is not
posterior slow subspace" challenges a natural assumption about RSSM latent
drift. It suggests that temporal smoothness in posterior latents and open-loop
imagination error accumulation are different geometric objects.

## 3. Pre-check 2 Recap

Pre-check 2 asked whether SMAD could still work if the basis is changed from
`U_posterior` to `U_drift`.

Method:

1. Recompute `U_drift` for `r in {3, 5, 10}` from the trimmed drift matrix.
2. Sweep `eta in {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}`.
3. Apply post-hoc damping to the existing imagined deter trajectories:

```text
damped_imag_deter_t = true_deter_t + (I - eta * U_drift U_drift.T) Delta_t
```

4. Run the five-band frequency decomposition on true vs damped deter rollouts.
5. Compute band MSE and raw total deter MSE on the trim window `[25, 175)`.

Frequency evaluation used the existing DriftDetect bands:

```text
dc_trend: 0.00-0.02
very_low: 0.02-0.05
low:      0.05-0.10
mid:      0.10-0.20
high:     0.20-0.50
```

The frequency PCA retained 38 PCs and explained 0.9504360611578495 variance.
Filtering used Butterworth order 4 with `sosfiltfilt`.

Exact key results from `results/tables/smad_eta_sweep.json`:

| r | eta | dc_trend MSE | total MSE | dc_trend change % | total change % | guardrail |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.20 | 0.1797786857328603 | 0.08565529541970997 | -23.19819459702299 | -24.472830609668662 | PASS |
| 10 | 0.50 | 0.12231009249665047 | 0.05558779544681446 | -47.7488897838189 | -50.985063770143014 | PASS |

At `r=10, eta=0.20`, `dc_trend` MSE drops by 23.19819459702299 percent and
total MSE drops by 24.472830609668662 percent. This clears the a priori 20
percent `dc_trend` reduction threshold without triggering the total-MSE
guardrail. At `r=10, eta=0.50`, `dc_trend` MSE drops by 47.7488897838189
percent and total MSE drops by 50.985063770143014 percent, but this is the
boundary of the tested eta range.

Circularity caveat: `U_drift` is derived from PCA on the same drift vectors
that are then projected out. Some MSE reduction is therefore expected by
construction. The formula above damps cumulative drift after it has already
been observed; it does not modify `img_step()`, re-run the RSSM, or capture
nonlinear trajectory feedback. These results motivate Phase 2, but do not
confirm that real SMAD damping will work.

## 4. Formal Verdict

Verdict: Path B, WEAK GO with caveats.

Phase 2 basis:

```text
U_drift = top-r PCs of baseline imagination drift vectors
```

Rejected basis:

```text
U_posterior = posterior PCA directions ranked by lag-1 autocorrelation
```

Recommended Phase 2 setting:

```text
r = 10
eta = 0.20
```

The conservative `eta=0.20` setting is preferred over `eta=0.50`. It already
exceeds the 20 percent `dc_trend` threshold, while avoiding the extreme sweep
boundary where the circularity concern is strongest.

Phase 2 scope is one configuration only:

```text
combined damping + anchor with U_drift basis
```

Do not run the original three-configuration ablation by default. There should
be no separate damping-only or anchor-only training runs unless the single
combined run succeeds and spare compute remains. The roughly 20 GPU-hours saved
by this reduction should be reallocated to strengthening the DreamerV4
cross-architecture comparison or to additional toy environment work.

## 5. Implications for Month 5

What stays from the original Month 4-5 structure:

1. Implement the SMAD patch in DreamerV3.
2. Include both inference-time damping and training-time anchor loss.
3. Run the identity test: `eta=0, beta=0` must match the baseline checkpoint.
4. Estimate the intervention basis from a fixed baseline checkpoint.
5. Keep Theorem 1 LaTeX work in scope.
6. Keep the toy simple version in scope.
7. Evaluate `J_slow`, frequency-band drift, eval return, and representation health.

What changes:

1. The basis is `U_drift`, estimated from baseline open-loop drift vectors.
2. The basis is not `U_posterior` from PCA plus lag-1 autocorrelation.
3. The primary damping setting is `r=10, eta=0.20`.
4. Only one training configuration is authorized by default.
5. The single configuration combines damping with anchoring.
6. The anchor coefficient `beta` may be selected from `{0.01, 0.1, 1.0}`.
7. Separate damping-only and anchor-only ablations are removed from the plan.
8. About 20 GPU-hours are saved relative to the original ablation plan.
9. Representation health checks remain mandatory.
10. `dc_trend` variance collapse remains a critical abort condition.

Representation health is still critical. If `dc_trend` variance drops by more
than 30 percent, the run should be treated as representation collapse even if
MSE improves. Phase 3 horizon scaling proceeds only if the single Phase 2
configuration satisfies:

```text
J_slow drop >= 20 percent
eval_return drop <= 5 percent
```

New Finding D framing:

1. Posterior slow modes do not equal drift directions.
2. Drift-direction damping is promising in post-hoc analysis.
3. Real RSSM damping remains to be tested in Phase 2.

The Theorem 1 damping corollary needs to be reframed. The theorem's "slow
subspace" should not be casually identified with `U_posterior`. The slow-feature
objective identifies high-autocorrelation posterior directions, but Pre-check 1
shows these are not the directions along which DreamerV3 Cheetah drift
accumulates. For this project, the empirically relevant damping subspace is
`U_drift`. Slow temporal variation and imagination instability are related, but
not equivalent.

## 6. Risk Table

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Post-hoc results do not transfer to real RSSM damping | Medium-High | Phase 2 fails | Single-config design minimizes wasted compute |
| `eta=0.20` is too conservative for real damping | Medium | Weak Phase 2 result | Re-run at `eta=0.30` only if first run shows less than 10 percent `dc_trend` drop and eval return is healthy |
| `eta=0.20` is too aggressive for real damping | Medium | Eval return drops or latent dynamics degrade | Monitor eval return during training and abort early if return drops by more than 10 percent |
| `U_drift` changes during SMAD training | Medium | Fixed basis becomes stale | Use fixed `U_drift` from baseline; re-estimation is a later ablation option |
| Representation collapse | Low-Medium | SMAD kills useful slow dynamics | Check `dc_trend` variance on the SMAD checkpoint and abort if it drops by more than 30 percent |

## 7. Limitations

These pre-checks used only Cheetah Run. Cartpole Swingup was not tested.

Pre-check 2 used a post-hoc linear approximation, not real nonlinear RSSM
dynamics. The actual SMAD patch will alter recurrent trajectories step by step.

There is a circularity concern in `U_drift`: the basis is estimated from drift
PCA and then used to remove drift components. That is useful as a screen, but
not a causal validation.

The selected rank `r=10` captures only part of the drift subspace. Higher ranks
might be needed for other tasks, longer horizons, or later checkpoints.

The 20 percent `dc_trend` threshold was set before the eta sweep, but it is
still an arbitrary operational threshold rather than a theorem.

Bottom line: Path B authorizes one careful Phase 2 test of `U_drift`-based SMAD.
It does not establish SMAD as solved.
