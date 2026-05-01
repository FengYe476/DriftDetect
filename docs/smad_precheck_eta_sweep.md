# SMAD Pre-check 2: Eta Sweep with U_drift Basis

Date: 2026-05-01

Status: COMPLETE

## Method

Pre-check 1 found that `U_posterior` is not aligned with the empirical drift
subspace, so this eta sweep uses `U_drift` only.

The basis was estimated from the same 20 Cheetah Run v2 rollout files:

```text
results/rollouts/cheetah_v3_seed{0..19}_v2.npz
```

Each rollout contains `true_latent` and `imagined_latent` with shape
`(200, 1536)`. The deterministic RSSM state is the last 512 dimensions:

```text
true_deter = true_latent[:, 1024:]
imag_deter = imagined_latent[:, 1024:]
```

`U_drift` was computed by pooling trimmed drift vectors:

```text
Delta_t = imag_deter_t - true_deter_t
trim window = [25, 175)
pooled drift shape = (3000, 512)
```

PCA was fit to this pooled drift matrix. The top `r` PCs define `U_drift` for
`r in {3, 5, 10}`.

For each `eta`, the script applied post-hoc linear damping to the cumulative
drift vector:

```text
damped_imag_deter_t = true_deter_t + (I - eta * U_drift U_drift.T) Delta_t
```

The stochastic part of the imagined latent was left unchanged, but all reported
metrics are computed on the deterministic state.

This is an approximation. It does not modify `img_step()`, does not re-run the
RSSM, and does not capture recurrent feedback from damping one step into later
steps. It is only an inference-time screen for whether damping along `U_drift`
is directionally promising.

Frequency evaluation used the existing DriftDetect band definitions:

```text
dc_trend: 0.00-0.02
very_low: 0.02-0.05
low:      0.05-0.10
mid:      0.10-0.20
high:     0.20-0.50
```

The script fit a shared 95% PCA basis on pooled true deter trajectories before
bandpass filtering. This retained 38 PCs and explained 95.04% of posterior deter
variance. Filtering used Butterworth order 4 with `sosfiltfilt`.

`total_mse` is raw deterministic-state MSE over the same trim window.

## Results

| r | eta | dc_trend | very_low | low | mid | high | total | dc change % | total change % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.00 | 0.2341 | 0.0522 | 0.1423 | 0.0518 | 0.0210 | 0.1134 | 0.00 | 0.00 |
| 3 | 0.05 | 0.2275 | 0.0512 | 0.1320 | 0.0514 | 0.0209 | 0.1090 | -2.82 | -3.88 |
| 3 | 0.10 | 0.2214 | 0.0502 | 0.1223 | 0.0510 | 0.0209 | 0.1048 | -5.43 | -7.57 |
| 3 | 0.15 | 0.2157 | 0.0493 | 0.1131 | 0.0506 | 0.0208 | 0.1009 | -7.86 | -11.05 |
| 3 | 0.20 | 0.2105 | 0.0484 | 0.1045 | 0.0502 | 0.0207 | 0.0971 | -10.09 | -14.34 |
| 3 | 0.30 | 0.2014 | 0.0467 | 0.0889 | 0.0495 | 0.0206 | 0.0904 | -13.96 | -20.31 |
| 3 | 0.40 | 0.1942 | 0.0451 | 0.0755 | 0.0488 | 0.0204 | 0.0845 | -17.05 | -25.49 |
| 3 | 0.50 | 0.1888 | 0.0437 | 0.0644 | 0.0482 | 0.0203 | 0.0795 | -19.35 | -29.87 |
| 5 | 0.00 | 0.2341 | 0.0522 | 0.1423 | 0.0518 | 0.0210 | 0.1134 | 0.00 | 0.00 |
| 5 | 0.05 | 0.2278 | 0.0499 | 0.1315 | 0.0508 | 0.0208 | 0.1077 | -2.68 | -5.02 |
| 5 | 0.10 | 0.2220 | 0.0478 | 0.1212 | 0.0498 | 0.0206 | 0.1023 | -5.16 | -9.78 |
| 5 | 0.15 | 0.2167 | 0.0457 | 0.1115 | 0.0488 | 0.0205 | 0.0972 | -7.44 | -14.28 |
| 5 | 0.20 | 0.2118 | 0.0438 | 0.1024 | 0.0479 | 0.0203 | 0.0924 | -9.51 | -18.53 |
| 5 | 0.30 | 0.2036 | 0.0403 | 0.0858 | 0.0462 | 0.0199 | 0.0836 | -13.04 | -26.25 |
| 5 | 0.40 | 0.1972 | 0.0373 | 0.0715 | 0.0447 | 0.0196 | 0.0761 | -15.76 | -32.94 |
| 5 | 0.50 | 0.1927 | 0.0347 | 0.0594 | 0.0435 | 0.0194 | 0.0696 | -17.66 | -38.60 |
| 10 | 0.00 | 0.2341 | 0.0522 | 0.1423 | 0.0518 | 0.0210 | 0.1134 | 0.00 | 0.00 |
| 10 | 0.05 | 0.2193 | 0.0488 | 0.1304 | 0.0482 | 0.0205 | 0.1059 | -6.31 | -6.63 |
| 10 | 0.10 | 0.2053 | 0.0455 | 0.1191 | 0.0448 | 0.0200 | 0.0988 | -12.28 | -12.92 |
| 10 | 0.15 | 0.1922 | 0.0425 | 0.1084 | 0.0415 | 0.0196 | 0.0920 | -17.91 | -18.86 |
| 10 | 0.20 | 0.1798 | 0.0396 | 0.0984 | 0.0385 | 0.0191 | 0.0857 | -23.20 | -24.47 |
| 10 | 0.30 | 0.1574 | 0.0344 | 0.0800 | 0.0330 | 0.0183 | 0.0741 | -32.75 | -34.67 |
| 10 | 0.40 | 0.1383 | 0.0298 | 0.0642 | 0.0283 | 0.0176 | 0.0641 | -40.93 | -43.51 |
| 10 | 0.50 | 0.1223 | 0.0259 | 0.0508 | 0.0243 | 0.0169 | 0.0556 | -47.75 | -50.99 |

## Eta Selection

The selection rule was: maximize dc_trend reduction relative to `eta=0`,
subject to total MSE increasing by no more than 10%.

All tested eta values passed the total-MSE guardrail because post-hoc damping
reduced raw drift energy.

The best candidate is:

```text
r = 10
eta = 0.50
dc_trend reduction = 47.75%
total MSE change = -50.99%
```

`r=3, eta=0.50` reached only 19.35% dc_trend reduction, just below the 20%
GO threshold. `r=5, eta=0.50` reached 17.66%. The actionable setting is
therefore `r=10, eta=0.50`.

## Combined Pre-check Verdict

Pre-check 1 verdict: `U_posterior` is not a valid SMAD basis.

Pre-check 2 verdict: SMAD Phase 2 GO with `U_drift` basis at `r=10`,
`eta=0.50`.

Scope should be narrow: run one Month 5 Phase 2 configuration using
`U_drift` with damping plus anchoring combined. Do not spend budget on the
original three-configuration ablation unless this first configuration works.

The result supports the damping idea directionally, but only when the basis is
estimated from empirical drift vectors.

## Limitations

This is a post-hoc linear approximation, not real RSSM damping.

Real SMAD will alter latent transitions recursively, so later-step dynamics may
differ from this per-timestep cumulative-drift projection.

The result is single-task: Cheetah Run only.

The sweep uses the existing 20 rollouts from one checkpoint and one rollout
distribution.

The basis is linear and global. It may miss state-dependent or curved drift
directions.

The largest tested eta was 0.50. Since the best candidate is at this boundary,
Phase 2 should treat `eta=0.50` as a starting point, not proof that larger eta
would be safe in a real recurrent rollout.
