# Proposal Addendum: From Diagnosis to Intervention

## Status

This addendum extends the original PROPOSAL.md with a concrete intervention component motivated by Months 1–3 empirical results. It does not replace the original proposal. It adds a targeted method (Slow-Mode Anchored Dynamics) that operationalizes Finding D and strengthens the paper from a pure diagnostic study into a diagnosis-to-intervention contribution.

## A1. Empirical Basis for the Extension

The original proposal hypothesized that drift would be frequency-structured (Finding A) and that diagnostics would yield actionable design implications (Finding D). Months 1–3 have now established the empirical foundation:

**Finding A — Confirmed.** Imagination drift in DreamerV3 is frequency-asymmetric and trend-dominated. The DC/trend band carries 46.7% (Cheetah Run) to 97.0% (Cartpole Swingup) of latent MSE, while high-frequency error accounts for only 4.2% to 0.3%. This pattern is robust across two tasks, three filter types (Butterworth, Chebyshev, FIR), and multiple imagination start positions.

**Decoder amplification — Falsified as independent finding.** Experiments D1–D3 confirmed that the trained decoder amplifies out-of-distribution latent drift overall (~48%), but does not preferentially amplify DC/trend directions. The decoder's effective Lipschitz constant is smallest along `dc_trend`. Trend dominance is therefore latent-intrinsic, arising from RSSM open-loop dynamics rather than decoder-specific amplification.

**Implication.** Because drift concentrates in an identifiable low-dimensional subspace of the latent dynamics — not in the decoder, not uniformly across frequencies — a targeted intervention on that subspace is both motivated and feasible. This is the basis for the method described below.

## A2. Slow-Mode Anchored Dynamics

### A2.1 Overview

The method has two components and introduces zero new model parameters:

1. **Inference-time trend damping:** selectively reduce the per-step update magnitude along the slow/trend subspace of the RSSM deterministic state.
2. **Training-time anchor loss:** penalize slow-subspace divergence between imagined and posterior latents over the imagination horizon.

Both components depend on a fixed slow-mode basis $U$ estimated from training data. The full technical specification is provided in the companion document (`slow_mode_anchored_dynamics.md`). This section summarizes the method and its relationship to the original proposal.

### A2.2 Slow-Mode Basis

Let $U \in \mathbb{R}^{d \times r}$ be a matrix whose columns span the slow/trend subspace of the RSSM deterministic state, with $U^\top U = I_r$.

$U$ is constructed in three steps from posterior deterministic latents collected during training:

1. PCA on deter states, retaining directions that explain 95% of variance.
2. For each PC, compute lag-1 autocorrelation of its temporal projection.
3. Select the $r$ directions with highest autocorrelation. Default $r = 3$.

Because $U$ is drawn from a subset of PCA orthonormal columns, the orthogonality constraint is satisfied automatically. The construction is compatible with the existing DriftDetect pipeline, which already performs deter-only PCA as a preprocessing step.

### A2.3 Core Formulas

**Inference-time dynamics:**

$$z_{t+1}^{imag} = z_t^{imag} + \Delta z_t - \eta \, UU^\top \Delta z_t$$

where $\Delta z_t$ is the original RSSM `img_step` update, $\eta \in (0, \eta_{max}]$ with $\eta_{max} < 1$ controls damping strength, and $UU^\top \Delta z_t$ is the projection of the update onto the slow subspace. Non-slow components propagate unchanged; slow components are scaled by $(1 - \eta)$.

**Training-time loss:**

$$\mathcal{L} = \mathcal{L}_{Dreamer} + \beta \, \gamma(s) \cdot \frac{1}{|K|} \sum_{k \in K} \left\| U^\top z_{t+k}^{imag} - \mathrm{sg}(U^\top z_{t+k}^{post}) \right\|_2^2$$

where $K \subseteq \{1, \ldots, H_{imag}\}$ are anchor horizon points within the training imagination window, $\mathrm{sg}(\cdot)$ denotes stop-gradient, $\beta$ controls anchor strength, and $\gamma(s)$ is a step-based activation schedule.

### A2.4 Key Design Choices

**Why not correct toward ground truth at inference time?** Open-loop imagination has no access to true observations. Any inference-time correction that requires $z^{true}$ is structurally invalid. The damping formula uses only $\Delta z_t$ and the fixed projection $UU^\top$, both of which are available during imagination.

**Why stop-gradient on the posterior?** The anchor loss should train the dynamics model to stay aligned with the posterior in the slow subspace. Without stop-gradient, gradients would also flow through the encoder, potentially distorting the posterior representation to match imagination rather than observations.

**Why a step-based schedule rather than online diagnostics?** Running frequency decomposition inside the training loop is expensive and fragile. A predetermined schedule $\gamma(s)$ activated after the step at which `dc_trend` dominance stabilizes (determined by the Month 3 training-dynamics analysis) achieves the same effect at zero runtime cost.

**Why not simply increase the existing KL dynamics loss?** The DreamerV3 dynamics loss is one-step: it penalizes the divergence between prior and posterior at each transition. Drift is a multi-step accumulation phenomenon. The anchor loss operates over $K = \{5, 10, 15\}$ steps of imagination, directly targeting the accumulated slow-subspace error that one-step losses cannot see.

## A3. Relationship to Original Proposal Structure

### A3.1 Finding D: From Aspiration to Implementation

The original proposal describes Finding D as: *"If drift consistently concentrates in identifiable frequency bands, then future architectures can be designed with targeted stabilizers rather than generic regularization."* It lists examples — "explicitly protected slow channels, band-aware training objectives, selective long-horizon supervision" — but does not commit to implementing any of them.

Slow-Mode Anchored Dynamics is a concrete realization of Finding D. The damping component is a form of "explicitly protected slow channels." The anchor loss is a form of "selective long-horizon supervision concentrated on fragile temporal components." Both are derived directly from the diagnostic output of Finding A rather than designed ad hoc.

### A3.2 Theorem 1: From Motivation to Mechanism

Theorem 1 in the original proposal proves that, under linear-Gaussian dynamics with a slow-fast spectral gap, any minimizer of a slow-feature objective recovers the true slow subspace up to rotation.

The anchor loss is the practical, finite-horizon counterpart of this objective. It constrains the RSSM to preserve slow-subspace alignment over multi-step imagination, penalizing exactly the divergence that Theorem 1's slow-feature objective would minimize in the idealized setting. The $U$ basis estimated via PCA + autocorrelation is the empirical analogue of the theorem's slow subspace.

A supplementary theoretical result can formalize this connection: under the same linear-Gaussian assumptions, applying damping with rate $\eta$ along the true slow subspace provably reduces imagination MSE when the slow eigenvalues of the transition operator dominate drift accumulation. The proof reduces to showing that damping decreases the spectral radius of the transition operator restricted to the slow subspace without affecting the fast subspace. This result provides a sufficient condition for the method to work and a clear failure mode (misalignment between $U$ and the true drift subspace).

### A3.3 Finding C: Training Dynamics Determine Schedule

The Month 3 training-dynamics experiment (100 archived checkpoints) was originally designed to answer Finding C: which frequency bands are learned first? This analysis now has an additional operational purpose. The step at which `dc_trend` dominance stabilizes determines $s_0$ and $s_1$ for the activation schedule $\gamma(s)$. Finding C is no longer purely descriptive — it directly parameterizes the intervention.

## A4. Experimental Plan

The experiments are organized in three phases of increasing cost and commitment. Each phase has a clear go/no-go criterion.

### Phase 1: Inference-Time Validation (~2 GPU-hours)

Using existing Cheetah and Cartpole checkpoints without retraining:

1. Estimate $U$ from 20 v2 rollout posterior deter latents (PCA + autocorrelation).
2. Verify subspace alignment: compute overlap between posterior slow subspace and drift direction subspace. If overlap is low, the method's premise is undermined and Phase 2 should not proceed without switching to an alternative basis estimation method (e.g., SFA or direct drift-direction PCA).
3. Apply inference-time damping at $\eta \in \{0.1, 0.2, 0.3, 0.5\}$.
4. Run the existing `freq_decompose` + `error_curves` pipeline on damped rollouts.
5. Report per-band MSE changes.

**Go criterion:** `dc_trend` MSE decreases by $\geq$20% at $\eta = 0.3$ without increasing other bands' MSE by more than 10%.

### Phase 2: Full Training Experiment (~30 GPU-hours)

If Phase 1 succeeds:

1. Train DreamerV3 on Cheetah Run in three configurations:
   - Baseline (no modification).
   - Damping only ($\eta = 0.3$, $\beta = 0$).
   - Damping + anchor loss ($\eta = 0.3$, $\beta$ sweep over $\{0.01, 0.1, 1.0\}$).
2. For each configuration, evaluate:
   - $J_{slow}$: slow-subspace imagination fidelity. Expected to decrease.
   - $J_{total}$: total imagination fidelity. Expected to not increase.
   - `eval_return`: task performance. Expected to not decrease.
3. Generate per-band error curves for each configuration to show whether the frequency structure of drift changes.

**Go criterion:** $J_{slow}$ decreases and `eval_return` does not decrease by more than 5% relative to baseline.

### Phase 3: Horizon Scaling Experiment (~20 GPU-hours)

If Phase 2 succeeds, this is the strongest possible demonstration:

1. Train baseline and damping+anchor DreamerV3 at $H_{imag} \in \{15, 30, 50\}$.
2. Compare imagination fidelity and `eval_return` as a function of horizon.
3. The hypothesis is that baseline DreamerV3 degrades at longer horizons due to trend-dominated drift, while the modified version sustains imagination fidelity, enabling longer-horizon planning.

This reframes the contribution from "marginal improvement at default settings" to "enabling longer imagination horizons," which is a more impactful claim.

## A5. Revised Contribution Statement

The original proposal lists three contributions: diagnostic framework, theoretical foundation, and design implications. With this addendum, the contributions become:

1. **Diagnostic framework:** a frequency-domain protocol for measuring how rollout error grows across temporal bands in world models. *(Unchanged.)*
2. **Theoretical foundation:** a subspace identifiability result under linear-Gaussian dynamics, extended with a damping optimality result that provides sufficient conditions for the proposed intervention. *(Extended.)*
3. **Diagnostic-driven intervention:** Slow-Mode Anchored Dynamics, a zero-parameter method that translates the diagnostic finding (trend-dominated drift) into a targeted correction. This replaces the original proposal's aspirational "design implications" with a concrete, testable mechanism. *(New.)*

## A6. Revised Timeline

The original 6-month timeline is adjusted to accommodate the intervention experiments in Months 5–6. Months 1–4 are unchanged.

| Phase | Duration | Tasks | GPU-hours |
|-------|----------|-------|-----------|
| **Month 1** | Setup & Baseline | Infrastructure, checkpoint training, adapter, rollouts | ~10h |
| **Month 2** | Core Diagnostics | `freq_decompose`, `error_curves`, Figure 1, filter/window ablation | ~10h |
| **Month 3** | Mechanistic Analysis | Decoder amplification (D1–D3), training dynamics checkpoints | ~12h |
| **Month 4** | Theory & Toy | Prove Theorem 1, damping optimality result, toy environment | ~20h |
| **Month 5** | Intervention Experiments | Phase 1 (inference validation) + Phase 2 (training experiment) | ~32h |
| **Month 6** | Writing & Horizon Scaling | Phase 3 (if warranted), paper draft, revision | ~20h |

**Total compute:** ~104 GPU-hours used or planned, well within the 200 GPU-hour budget.

## A7. Failure Modes Specific to the Extension

The original proposal's failure modes (§6) remain valid. The following are additional risks specific to the intervention component:

- **Subspace misalignment:** The posterior slow subspace (estimated via PCA + autocorrelation) may not align with the actual drift direction subspace. Phase 1 includes an explicit overlap check to detect this before committing to training experiments.
- **Anchor loss too weak at short horizons:** DreamerV3's default $H_{imag} = 15$ may be too short for meaningful slow-subspace drift to accumulate, resulting in negligible anchor gradients. Mitigation: check `dc_trend` MSE at steps 5, 10, 15 in existing data before designing the anchor experiment.
- **Damping suppresses legitimate slow dynamics:** If $\eta$ is too large, the model loses the ability to track real slow changes in the environment. Mitigation: keep $\eta \leq 0.5$; validate `eval_return` does not drop; ablate $\eta$.
- **Interaction between components:** Damping and anchor loss together may over-correct the slow subspace. Mitigation: test each component independently (Phase 2 ablation groups) before combining.

Even if the intervention produces negative or null results, it remains scientifically informative: it would indicate that trend-dominated drift, while diagnosable, cannot be mitigated by simple subspace-level corrections, suggesting the problem is more deeply embedded in the RSSM transition function than a linear subspace view can capture.

## A8. Summary

This addendum proposes a natural extension of the DriftDetect project from diagnosis to intervention. The key insight is that Finding A's trend-dominated drift concentrates in a low-dimensional, identifiable subspace of the RSSM latent dynamics. This structural property enables a minimal, targeted correction — Slow-Mode Anchored Dynamics — that requires no new model parameters, integrates with the existing DriftDetect pipeline, and is testable within the remaining compute budget. The extension strengthens the paper by closing the loop from observation to action, transforming Finding D from an aspiration into a demonstrated capability.
