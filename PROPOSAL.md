# Decoding Latent Drift: A Frequency-Domain Diagnostic Framework for World Models with Theoretical Foundations

## 1. Introduction & Motivation

Latent drift in recurrent state-space world models is a central bottleneck for long-horizon planning. Even when one-step reconstructions appear accurate, small autoregressive errors can accumulate over imagined rollouts and gradually move the latent state away from the true trajectory. This mismatch is especially damaging in model-based reinforcement learning, where downstream planning and policy improvement depend on reliable imagined futures rather than short-horizon prediction alone.

Recent world models have made substantial engineering progress on this problem. In particular, the Dreamer family has steadily improved latent dynamics modeling, and DreamerV4's diffusion-forcing style training is explicitly motivated by better rollout stability. Yet these advances remain primarily algorithmic and empirical: they improve behavior, but they do not tell us which parts of the signal drift first, whether drift is concentrated in particular temporal scales, or how such failures evolve across architectures and during training. As a result, the field still lacks a mechanistic account of latent drift.

This proposal reframes the problem accordingly. Rather than introducing a new full-scale world model and competing primarily through end-to-end benchmark scores, we propose the first frequency-domain diagnostic framework tailored to latent drift in world models. Our core hypothesis is that drift is frequency-structured: prediction error does not grow uniformly across temporal components, but instead concentrates in specific bands, with high-frequency dynamics exhibiting disproportionately steep error accumulation over rollout horizon. If validated, this view would turn latent drift from an opaque failure mode into a measurable pathology.

The project is deliberately compute-friendly. Its main empirical component operates on publicly released pre-trained checkpoints, allowing us to isolate diagnosis from large-scale retraining. Our own computation is limited primarily to inference, frequency decomposition, and analysis, with only one lightweight training component used to study checkpoint dynamics over time. This shift makes the proposal substantially more feasible while also better aligned with its scientific objective: understanding why world models drift, not merely building another architecture that appears to drift less.

The proposal makes three contributions:

- **Diagnostic framework:** a frequency-domain protocol for measuring how rollout error grows across temporal bands in pre-trained world models.
- **Theoretical foundation:** a simplified identifiability result showing why slow-feature recovery is well motivated under linear-Gaussian latent dynamics.
- **Design implications:** empirically grounded guidance for future world model architectures, regularizers, and training objectives.

## 2. Diagnostic Framework

### 2.1 Frequency Decomposition Protocol

The central empirical contribution is a diagnostic pipeline that takes as input a pre-trained world model checkpoint and a sequence of ground-truth observations collected from the environment. The protocol then decomposes the observed signal by temporal frequency and measures how prediction error accumulates within each band.

The workflow is as follows:

1. Given an evaluation trajectory, project the observation sequence into disjoint temporal bands using band-pass filters. We will use a practical partition such as DC, 0-2 Hz, 2-5 Hz, 5-10 Hz, and above 10 Hz, with task-specific adjustment when environment frame rates require it.
2. For each band, run the world model in rollout mode and compare model predictions against the corresponding band-limited target signal over increasing horizon lengths.
3. Compute band-specific error curves as a function of horizon, and summarize each curve by its error growth rate, e.g. the slope of MSE with respect to horizon, $dMSE/dH$, together with confidence intervals across trajectories and seeds.
4. Compare these growth rates across bands to determine whether some temporal components are systematically more unstable than others.

The primary output is a family of **frequency-vs-error-growth curves**, which form the conceptual center of Figure 1. These curves are designed to answer a mechanistic question that aggregate rollout metrics cannot: is long-horizon degradation mainly a generic accumulation phenomenon, or does it disproportionately target fast-varying components of the environment state?

This section substantially expands the pilot diagnostic idea in the original proposal. There, frequency analysis served mainly as motivation for a subsequent model intervention. Here, the diagnosis itself becomes the paper's main object of study.

### 2.2 Cross-Architecture Extension

To ensure that any observed pathology is not an artifact of a single codebase or checkpoint, we extend the same diagnostic protocol across multiple publicly available RSSM-style world models:

- **DreamerV3**, using released checkpoints from `NM512/dreamerv3-torch`.
- **DreamerV4**, using the open-source implementation from `nicklashansen/dreamer4`, with inference-only evaluation and no additional large-scale training.
- **Optional extensions**, such as PlaNet or DreamerV2, if stable public checkpoints are available.

The goal is comparative diagnosis rather than leaderboard-style competition. If similar frequency-specific error growth appears across multiple architectures, then the resulting evidence would support the stronger claim that frequency pathology is a pervasive property of the RSSM family. If, instead, the pathology meaningfully differs across architectures, those differences become a source of mechanistic insight: they may reveal which design choices selectively stabilize slow dynamics, which help with high-frequency reconstruction, and which merely improve aggregate metrics without addressing long-horizon drift.

Because this stage relies primarily on released checkpoints, the compute cost is modest. The study requires evaluation rollouts, filtering, and metric computation rather than full training of several world models from scratch.

### 2.3 Training Dynamics Diagnosis

In addition to cross-sectional diagnosis on fixed checkpoints, we will analyze how frequency-specific errors evolve during training. For a single DreamerV3 training run, we will save intermediate checkpoints at regular intervals, for example every 10k environment steps or gradient updates depending on the implementation.

At each checkpoint, we will apply the same frequency decomposition protocol and track:

- how each band's rollout error curve changes over training,
- which frequency bands are learned first,
- which bands remain persistently fragile,
- whether late-stage performance gains correspond to genuine improvements in high-frequency stability or only to better short-horizon reconstruction.

This analysis turns training into a developmental trajectory rather than a black box. It allows us to test whether world models exhibit a coarse-to-fine learning pattern, whether high-frequency instability is merely delayed or fundamentally persistent, and whether architectural modifications should target early representation learning or late-stage rollout robustness.

### 2.4 Evaluation Tasks

All diagnostic experiments will be conducted on three DeepMind Control Suite tasks:

- **Cheetah Run** — strongly periodic locomotion with rich high-frequency dynamics
- **Cartpole Swingup** — non-periodic task requiring long-horizon credit assignment
- **Reacher Hard** — sparse-reward task with mixed-frequency targets

This selection spans distinct dynamics regimes to test whether frequency pathology is task-invariant.

## 3. Theoretical Foundation: Local Identifiability in a Simplified Setting

### 3.1 Motivation

Our main diagnosis is empirical and focuses on modern nonlinear RSSMs. Still, a lightweight theoretical result is valuable because it clarifies why slow-fast decomposition is a scientifically meaningful lens rather than just an intuitive engineering trick. We therefore include a simplified identifiability result that formalizes when slow latent structure can, in principle, be recovered from temporal smoothness.

### 3.2 Simplified Setting

Assume the true latent dynamics are linear Gaussian and admit a slow-fast decomposition:

$$
z_t = \begin{bmatrix} z_t^{slow} \\ z_t^{fast} \end{bmatrix},
$$

$$
z_{t+1}^{slow} = A_s z_t^{slow} + \epsilon_t^s, \qquad \|A_s\| < 1,\qquad \epsilon_t^s \sim \mathcal{N}(0, \sigma_s^2 I),
$$

$$
z_{t+1}^{fast} = A_f z_t^{fast} + \epsilon_t^f, \qquad \epsilon_t^f \sim \mathcal{N}(0, \sigma_f^2 I), \qquad \sigma_f^2 \gg \sigma_s^2,
$$

$$
o_t = W \begin{bmatrix} z_t^{slow} \\ z_t^{fast} \end{bmatrix} + \eta_t.
$$

Intuitively, the slow component varies smoothly and has lower process noise, whereas the fast component changes more rapidly and carries substantially larger temporal variation.

### 3.3 Theorem 1: Subspace Identifiability

We study a recovered slow representation $\hat{z}^{slow}$ trained under a slow-feature objective

$$
\mathcal{L}_{slow} = \mathbb{E}\left[\left\|\hat{z}_{t+1}^{slow} - \hat{z}_t^{slow}\right\|^2\right]
$$

subject to the variance constraint

$$
\mathbb{E}\left[\|\hat{z}^{slow}\|^2\right] = c.
$$

**Theorem 1 (informal).** Under the linear-Gaussian assumptions above, and assuming a sufficient spectral gap between slow and fast temporal variation induced by $\sigma_s^2 \ll \sigma_f^2$, any global minimizer of the constrained slow-feature objective spans the same subspace as the true slow latent component $z^{slow}$, up to an orthogonal rotation.

The proof sketch proceeds in four steps:

1. Rewrite the constrained optimization problem using a Lagrangian that combines temporal smoothness with the variance-preserving constraint.
2. Derive the KKT conditions for stationary points of the constrained problem.
3. Show that the optimal directions solve a generalized eigenvalue problem associated with the slow dynamics operator and the covariance structure of the observed process.
4. Use the slow-fast variance separation assumption to prove subspace separation, yielding recovery of the slow latent subspace up to rotation.

The full proof belongs in the appendix, but the main role of the theorem in the proposal is conceptual. It provides a principled reason to expect temporally smooth latent directions to correspond to meaningful slow structure, even though the real empirical setting is far more nonlinear and partially observed.

### 3.4 Empirical Validation on a Toy Environment

To bridge the theorem and the real-world diagnostics, we will construct a minimal dual-frequency oscillator environment. The true state contains:

- a 2D slow oscillator at 0.5 Hz,
- a 2D fast oscillator at 5 Hz.

A minimal RSSM with a small MLP transition and observation model will be trained on this environment. The aim is not benchmark performance but controlled validation: we will test whether the learned slow representation aligns with the true slow subspace, whether the frequency-diagnostic curves cleanly separate the two components, and whether the theory's assumptions predict qualitative behavior in practice.

### 3.5 Connection to Anti-Collapse Mechanisms

The theorem also helps reinterpret practical anti-collapse mechanisms. The variance constraint in the simplified setting corresponds naturally to mechanisms such as LayerNorm or other variance-preserving normalization on a designated slow channel. Likewise, the empirical need for an additional predictive target suggests a smoothed predictive constraint: if a latent channel is meant to encode long-range structure, then its supervision should emphasize temporally aggregated or low-frequency task information rather than instantaneous noisy targets.

In the revised proposal, these ideas are no longer presented as the centerpiece of a new model called DT-RSSM. Instead, they serve as theoretically motivated design implications that emerge from diagnosis and toy analysis.

## 4. Empirical Findings on Real World Models

This section is the core of the paper. Rather than centering the narrative on a new training recipe, we organize the empirical study around a set of diagnosis-driven findings that the proposed framework is designed to establish.

### 4.1 Finding A: Latent Drift is Frequency-Structured

Our primary claim is that rollout error is not temporally uniform. We expect frequency-vs-error-growth curves to show that high-frequency components suffer disproportionately steep degradation as horizon increases, even when low-frequency structure remains relatively stable. This would operationalize latent drift as a measurable frequency pathology rather than a vague notion of compounding error.

### 4.2 Finding B: The Pathology Generalizes Across RSSM Architectures

By applying the same diagnostic protocol to DreamerV3, DreamerV4, and optional earlier RSSM variants, we aim to determine whether frequency-structured drift is architecture-specific or family-wide. A cross-architecture pattern would support the stronger claim that the pathology is intrinsic to recurrent latent imagination in this model class. Divergences, if they appear, would be equally informative because they would identify which architectural choices alter the shape of the pathology.

### 4.3 Finding C: Training Learns Temporal Bands Unevenly

Checkpoint-based diagnosis during a DreamerV3 training run will reveal whether temporal scales are acquired in stages. We hypothesize that low-frequency components are learned earlier and more robustly, while high-frequency bands remain noisy for longer and may never fully stabilize in long rollouts. This would suggest that final performance metrics obscure an important developmental asymmetry in representation learning.

### 4.4 Finding D: Diagnostics Yield Actionable Design Implications

The end goal of the paper is not diagnosis for its own sake. If drift consistently concentrates in identifiable frequency bands, then future architectures can be designed with targeted stabilizers rather than generic regularization. Examples include explicitly protected slow channels, multi-timescale latent transitions, band-aware training objectives, or selective long-horizon supervision concentrated on fragile temporal components. The value of the proposal lies in making such interventions principled rather than ad hoc.

### 4.5 Related Work

- **World Models**: DreamerV3 (Hafner et al. 2023), DreamerV4 (Hafner et al. 2025), PlaNet (Hafner et al. 2019)
- **Slow-Fast Hierarchies**: THICK Dreamer (ICLR 2024), Hieros (2023), Clockwork VAE (Saxena et al. 2021)
- **Frequency Analysis in Other Domains**: TritonCast (arxiv 2505.19432) — spectral error analysis in earth system models validates that frequency pathology is domain-general
- **Identifiability Theory**: Hyvärinen & Morioka (2016), Klindt et al. (2021) — temporal contrastive learning and nonlinear ICA

## 6. Feasibility, Evaluation, and Scientific Honesty

The revised proposal is intentionally lightweight in computation. Most experiments use public checkpoints and require only inference-time rollouts plus offline analysis. The only training-heavy component is a single checkpointed DreamerV3 run for the training-dynamics study, along with a small toy-environment experiment for validating the simplified theorem. This is a substantial reduction from the original plan, which required training and comparing multiple full models and ablations.

Evaluation will combine three complementary levels:

- **Mechanistic metrics:** band-specific rollout error curves, error growth rates, and cross-checks based on spectral analysis of prediction residuals.
- **Representation probes:** in the toy setting, alignment between recovered slow subspaces and ground-truth slow states.
- **Comparative evidence:** consistency or divergence of frequency pathology across released world-model architectures.

The project also preserves the original proposal's emphasis on scientific honesty. We explicitly acknowledge three failure modes:

- **No meaningful band separation:** frequency-specific growth curves may turn out to be flat or weakly differentiated, in which case the core hypothesis would be falsified.
- **Architecture-specific effects:** the pathology may appear only in some world models, weakening the claim of generality but still yielding useful comparative insight.
- **Theory-practice gap:** the identifiability result may illuminate the toy environment without directly explaining behavior in large nonlinear RSSMs; in that case, it remains a bounded theoretical supplement rather than a grand explanatory claim.

Even under these outcomes, the proposal remains scientifically valuable. A negative or qualified result would still sharpen how the community studies latent drift, establish a reusable diagnostic benchmark, and clarify which future interventions are worth pursuing.

## 7. Expected Contribution

The reoriented proposal replaces a method-first narrative with a diagnosis-first research agenda. Its central contribution is a frequency-domain framework for decoding how and where world models drift, backed by a modest identifiability theorem and a compute-efficient empirical plan built around public checkpoints. If successful, the project will contribute a new measurement language for world-model failure, a minimal theoretical justification for slow-feature structure, and a clearer foundation for future architecture design.

## 8. Timeline & Milestones (6 months)

| Phase | Duration | Tasks | GPU-hours |
|-------|----------|-------|-----------|
| **Month 1** | Setup & Baseline | Environment setup, reproduce DreamerV3 baseline on 3 DMC tasks | ~80h |
| **Month 2** | Core Diagnostics | Implement frequency decomposition pipeline, generate Figure 1 | ~30h |
| **Month 3** | Cross-Architecture | DreamerV4 inference analysis, training dynamics tracking | ~40h |
| **Month 4** | Theory & Toy | Prove Theorem 1, validate on dual-oscillator toy environment | ~20h |
| **Month 5** | Analysis & Writing | Deep analysis, visualization, draft paper | ~20h buffer |
| **Month 6** | Revision & Submission | Internal review, supplementary experiments, submission | ~10h buffer |

**Total compute**: ~200 GPU-hours (feasible on single RTX 4090 over 6 months)
