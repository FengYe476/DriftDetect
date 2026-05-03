# 4. Empirical Findings

## 4.1 Finding A: Frequency-Asymmetric Drift

The first finding is that imagination drift in DreamerV3 is strongly frequency-asymmetric. Rather than spreading uniformly across temporal scales, latent error concentrates in the slowest components of the deterministic RSSM state. On Cheetah Run, the DC/trend band accounts for 46.7% of latent mean-squared error over the evaluation window. On Cartpole Swingup, the same band accounts for 97.0%. By contrast, the high-frequency band contributes only 4.2% of latent error on Cheetah and 0.3% on Cartpole. The dominant failure mode is therefore not high-frequency instability, but slow displacement of imagined latent trajectories away from posterior trajectories.

This conclusion is robust across Butterworth, Chebyshev, and FIR filters, and across three episode phases defined by different imagination start positions. Exact band shares vary with filter design and phase, but DC/trend remains dominant while high-frequency error remains suppressed. The result also separates latent-space and observation-space behavior. In latent space, drift tends to saturate over the finite 200-step horizon. In observation space, DC/trend error grows monotonically, indicating that the decoder expresses bounded latent displacement as growing observation discrepancy. The frequency structure, however, is already present in latent space.

## 4.2 Finding B: Cross-Architecture Trend Dominance

The second finding is that trend-dominated drift is not specific to DreamerV3's GRU/tanh RSSM. Applying the same diagnostic to DreamerV4 shows that the pattern survives a substantial architectural change. DreamerV4 uses image-token latents and transformer-style iterative denoising rather than the single-step recurrent transition used in DreamerV3 [CITATION], yet its latent imagination errors remain dominated by slow components.

In the Cheetah comparison, DreamerV4 has a DC/trend share of 63.9%, higher than DreamerV3's 46.7% on the same task. The high-frequency share is nearly identical: both architectures allocate approximately 4.2% of latent error to the high band. The top-two ranking is also identical, with DC/trend first and low-frequency error second. These similarities across representation type and imagination mechanism weaken the hypothesis that trend dominance is merely a GRU/tanh artifact. The more conservative interpretation is that slow drift is a broader property of autoregressive world-model imagination, with architecture modulating magnitude rather than eliminating the phenomenon.

## 4.3 Finding C: Architecture-Intrinsic Training Dynamics

The third finding is that trend dominance is present from the beginning of training and persists throughout optimization. A high-frequency DreamerV3 Cheetah archive was constructed with checkpoints every 5000 steps from early training through approximately 510k steps. At step 5000, when the policy is still weak, the DC/trend share is 79%. By step 500k, the share falls to approximately 33%, but it remains the largest band throughout the run.

This temporal profile rules out a simple late-training explanation. If trend dominance were caused by converged policy behavior, representation saturation, or end-of-training overfitting, it should emerge gradually or intensify late. Instead, it is strongest early and partially alleviated by optimization. Training improves the RSSM and reduces the extremity of the slow-error bias, but does not remove it. Finding C therefore supports an architecture-intrinsic interpretation for DreamerV3: the RSSM transition induces a slow-mode error preference under open-loop rollout even before the agent has learned a strong task policy.

## 4.4 Finding E: Decoder Amplification is Uniform, Not Directional

The fourth finding concerns the origin of observation-space drift. A natural hypothesis is that the decoder selectively amplifies latent error along the DC/trend direction, turning small slow latent deviations into large observation-level deviations. Three decoder analyses reject this directional explanation.

D1 shows that the trained decoder amplifies off-distribution latent drift approximately 48% more than an architecture-matched random decoder, with the ratio growing over horizon. D2 tests directionality directly by perturbing true latent states along frequency-band directions. In DreamerV3, the DC/trend direction has the smallest effective Lipschitz constant, 1.221, while other bands have larger constants. DreamerV4 shows the same qualitative pattern: the tokenizer decoder also has its smallest measured Lipschitz response along DC/trend, with value 3.035. D3 further shows that decoder nonlinearity behaves like a constant off-distribution noise floor rather than a horizon-growing mechanism. Thus, the decoder amplifies off-distribution latent inputs broadly, but it does not explain why slow latent error dominates. Trend dominance is latent-intrinsic and originates upstream in the learned dynamics.

## 4.5 Mismatch Finding: Posterior Slow Features != Drift Directions

The fifth finding is that posterior slow features and imagination drift directions are not the same subspace. The strength of this mismatch is architecture-and-complexity dependent: strongest in DreamerV3, weaker in DreamerV4, and graded across toy environments. This is the paper's recommended Framing B. Mismatch is not claimed to have universal magnitude; instead, it varies systematically with model architecture and task complexity.

Posterior/drift subspace overlap is computed as the squared Frobenius norm of the basis cross-product divided by rank. Lower values indicate stronger mismatch.

| Setting | r=3 overlap | r=5 overlap | r=10 overlap | Interpretation |
|---|---:|---:|---:|---|
| V3 Cheetah | 0.13 | 0.17 | 0.28 | strong mismatch |
| V3 Cartpole | 0.06 | 0.08 | 0.12 | strongest mismatch |
| V4 Cheetah | 0.64 | 0.41 | 0.43 | moderate mismatch |
| Toy simple | 0.61 | 0.70 | 0.80 | weak mismatch |
| Toy medium | 0.41 | 0.67 | 0.85 | moderate mismatch |
| Toy hard | 0.38 | 0.50 | 0.74 | moderate mismatch |

The DreamerV3 results are the clearest cases. In Cheetah, overlap stays below 0.30 across ranks 3, 5, and 10. In Cartpole, overlap is even lower, ranging from 0.06 to 0.12. DreamerV4 provides a contrast case rather than a contradiction. Its rank-3 overlap is 0.64, indicating partial alignment, while rank-5 and rank-10 overlaps are 0.41 and 0.43. The toy results extend the same graded story: simple settings are closer to aligned, while medium and hard settings show stronger low-rank mismatch. SFA controls confirm that the result is not a PCA artifact, since independently estimated slow features are also misaligned with empirical drift directions in the real-model settings.

Linear toy experiments provide a mechanism. In a 10-dimensional contractive linear system, posterior/drift overlap decreases monotonically as the model-environment capacity gap increases. The capacity-gap sweep yields Pearson r = -0.925 between perturbation size and overlap, with overlap falling from near identity to 0.618. A separate noise-structure sweep shows that isotropic inference noise produces the highest overlap, while anisotropic structured noise lowers alignment. These results support the conjecture that posterior slow features and drift directions are determined by different operators, and that alignment depends on model capacity and inference-noise geometry.

The operational implication is direct. SMAD Pre-check 1 failed because it used `U_posterior`, the posterior slow-feature basis, as a candidate intervention subspace. In V3 Cheetah, that basis overlaps the drift basis by only 0.13-0.28, so damping it cannot reliably target the actual drift direction. The empirical `U_drift` basis was therefore necessary. The same distinction explains the later staleness result: fixed-basis SMAD reduces DC/trend by only 4.8% during training, but after the drift basis is re-estimated, post-hoc damping recovers 22.6% reduction. The old and new drift bases have overlap only 0.005-0.022, showing that even the drift subspace itself evolves during training. Adaptive-SMAD addresses this moving-target problem, with final results [TO BE UPDATED].
