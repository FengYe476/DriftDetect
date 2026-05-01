# Slow-Mode Anchored Dynamics: A Diagnostic-Driven Extension to DriftDetect

## 1. Context and Motivation

DriftDetect (Months 1–3) established that open-loop imagination drift in RSSM world models is **frequency-asymmetric and trend-dominated**:

- `dc_trend` carries 46.7% (Cheetah Run) to 97.0% (Cartpole Swingup) of latent MSE.
- High-frequency error is strongly suppressed (4.2% to 0.3%).
- The pattern holds across two tasks and three filter types (Finding A).
- Decoder amplification experiments (D1–D3) confirmed that trend dominance is **latent-intrinsic**, not a decoder artifact.

This raises a natural follow-up question: if drift concentrates in an identifiable slow subspace, can a minimal, targeted intervention reduce it without disrupting the rest of the dynamics?

We propose **Slow-Mode Anchored Dynamics** — a two-component extension that uses DriftDetect's diagnostic output to construct a principled drift-mitigation mechanism. The method adds no new model capacity. It applies a subspace-selective damping at inference time and a slow-subspace anchor loss at training time.

## 2. Method

### 2.1 Slow-Mode Basis Estimation

Let $z_1^{post}, z_2^{post}, \ldots, z_T^{post}$ be posterior deterministic latent states collected from training rollouts. Following DriftDetect's existing pipeline (deter-only, PCA to 95% variance), we:

1. Compute PCA basis $V = [v_1, \ldots, v_k]$ on the deter portion of posterior latents.
2. For each PC direction $v_j$, compute the lag-1 autocorrelation of its projection:

$$\rho_j = \frac{\sum_t (c_j(t) - \bar{c}_j)(c_j(t+1) - \bar{c}_j)}{\sum_t (c_j(t) - \bar{c}_j)^2}, \quad c_j(t) = v_j^\top z_t^{post}$$

3. Select the top $r$ directions with largest $\rho_j$ to form $U = [u_1, \ldots, u_r]$.

Since $U$ is a subset of PCA orthonormal columns, $U^\top U = I_r$ holds automatically, and $P = UU^\top$ is a valid orthogonal projection.

**Default:** $r = 3$, with ablation over $r \in \{1, 3, 5\}$.

**Note:** Slow Feature Analysis (SFA) provides a theoretically cleaner alternative that directly minimizes temporal derivative variance. We use PCA + autocorrelation ranking for simplicity and compatibility with the existing DriftDetect pipeline; SFA is a natural future refinement.

### 2.2 Inference-Time Dynamics (Trend Damping)

The standard RSSM open-loop imagination step can be written as:

$$z_{t+1}^{imag} = z_t^{imag} + \Delta z_t$$

where $\Delta z_t$ is the update produced by `img_step`. We modify this to:

$$\boxed{z_{t+1}^{imag} = z_t^{imag} + \Delta z_t - \eta \, UU^\top \Delta z_t}$$

Equivalently:

$$z_{t+1}^{imag} = z_t^{imag} + (1 - \eta) \, UU^\top \Delta z_t + (I - UU^\top) \Delta z_t$$

- The non-slow component $(I - UU^\top)\Delta z_t$ propagates unchanged.
- The slow component $UU^\top \Delta z_t$ is scaled by $(1 - \eta)$, reducing the per-step accumulation rate in the drift-dominant subspace.

**Constraint:** $0 < \eta \leq \eta_{max} < 1$. We recommend $\eta_{max} = 0.5$; setting $\eta = 1$ would freeze the slow subspace entirely, preventing the model from capturing legitimate slow dynamics.

At inference time, damping is always active. During training, it is gated by a schedule $\gamma(s)$ (see §2.4).

### 2.3 Training-Time Anchor Loss

During training, the RSSM has access to both imagined and posterior latents. We add an auxiliary loss that penalizes slow-subspace divergence over the imagination horizon:

$$\boxed{\mathcal{L} = \mathcal{L}_{Dreamer} + \beta \, \gamma(s) \cdot \frac{1}{|K|} \sum_{k \in K} \left\| U^\top z_{t+k}^{imag} - \text{sg}(U^\top z_{t+k}^{post}) \right\|_2^2}$$

- $z_{t+k}^{imag}$: latent from open-loop imagination at step $k$.
- $z_{t+k}^{post}$: posterior latent from the true trajectory.
- $\text{sg}(\cdot)$: stop-gradient. Gradients flow only through $z^{imag}$, updating the dynamics model. The posterior/encoder is not pulled by this loss.
- $K \subseteq \{1, \ldots, H_{imag}\}$: anchor horizon points within the training imagination window. For DreamerV3's default $H_{imag} = 15$, we use $K = \{5, 10, 15\}$.
- $\beta$: anchor weight hyperparameter.

**Relationship to existing DreamerV3 losses:** The DreamerV3 dynamics loss (KL between prior and posterior) operates one-step. The anchor loss differs in two ways: (a) it targets only the slow subspace, and (b) it operates over multi-step imagination horizons, directly penalizing the accumulated drift that one-step losses cannot see.

### 2.4 Training Schedule

Both damping and anchor loss are gated by a shared schedule:

$$\gamma(s) = \text{clip}\left(\frac{s - s_0}{s_1 - s_0}, \; 0, \; 1\right)$$

- $s < s_0$: no intervention (model learns freely).
- $s_0 < s < s_1$: gradual activation.
- $s > s_1$: fully active.

The values of $s_0$ and $s_1$ are determined empirically from DriftDetect's Month 3 training-dynamics analysis. For example, if `dc_trend` dominance stabilizes after 150k steps, set $s_0 = 150k$, $s_1 = 250k$.

At inference time: $\gamma = 1$ (always fully active).

### 2.5 Summary of Variables

| Symbol | Source | Fixed/Learned |
|--------|--------|---------------|
| $U$ | PCA + autocorrelation on posterior deter latents | Fixed after warmup |
| $\Delta z_t$ | Original RSSM `img_step` output | From model |
| $\eta$ | Damping strength | Hyperparameter ($\leq 0.5$) |
| $\beta$ | Anchor loss weight | Hyperparameter |
| $\gamma(s)$ | Step-based schedule | Predetermined |
| $K$ | Anchor horizon points | $\subseteq \{1, \ldots, H_{imag}\}$ |

All variables are available from existing DriftDetect/DreamerV3 infrastructure. No new model parameters are introduced.

## 3. Connection to the Existing Proposal

The original PROPOSAL.md defines four findings (A–D) and a theoretical component (Theorem 1). Slow-Mode Anchored Dynamics integrates as follows:

### 3.1 Completing Finding D

Finding D in the proposal states: *"Diagnostics Yield Actionable Design Implications."* The proposal describes this as "examples include explicitly protected slow channels, multi-timescale latent transitions, band-aware training objectives." Slow-Mode Anchored Dynamics is a concrete realization of Finding D — it translates the diagnostic output (Finding A) into a specific, minimal architectural intervention. Without it, Finding D remains aspirational.

### 3.2 Connecting Theorem 1 to Practice

Theorem 1 proves subspace identifiability for a slow-feature objective under linear-Gaussian dynamics. The anchor loss $\mathcal{L}_{anchor}$ is the practical analogue: it constrains the RSSM to preserve slow-subspace alignment over multi-step imagination, which is the nonlinear, finite-horizon counterpart of the slow-feature objective in the theorem. The $U$ basis estimated via PCA + autocorrelation is the empirical analogue of the theorem's slow subspace $z^{slow}$.

This connection strengthens both components: the theorem motivates why targeting the slow subspace is principled, and the method demonstrates that the theoretical insight translates into a concrete intervention.

### 3.3 Extending Cross-Architecture Analysis (Finding B)

If the damping mechanism reduces drift in DreamerV3, the same $U$-estimation and damping procedure can be applied to other RSSM variants. Whether the method transfers across architectures becomes an additional cross-architecture question complementing Finding B.

### 3.4 Leveraging Training Dynamics (Finding C)

The training-dynamics analysis (Month 3, 100 archived checkpoints) directly determines $s_0$ and $s_1$ for the activation schedule. This makes Finding C not just descriptive but operationally useful — it tells us *when* during training the intervention should activate.

## 4. Experimental Plan

### 4.1 Validation Without Retraining (Immediate, ~2 GPU-hours)

Using the existing Cheetah and Cartpole checkpoints:

1. Estimate $U$ from 20 v2 rollout posterior deter latents.
2. In `img_step`, apply $\Delta z_t - \eta UU^\top \Delta z_t$ for $\eta \in \{0.1, 0.2, 0.3, 0.5\}$.
3. Run the existing `freq_decompose` + `error_curves` pipeline.
4. Report per-band MSE changes, especially `dc_trend` reduction and whether other bands are affected.

This validates the damping component at zero training cost using the existing DriftDetect pipeline.

### 4.2 Full Training Experiment (~20–30 GPU-hours per configuration)

If §4.1 shows promise:

1. Train DreamerV3 on Cheetah Run with anchor loss ($\beta$ sweep) and damping ($\eta$ sweep).
2. Three ablation groups: damping only ($\beta = 0$), anchor only ($\eta = 0$), both.
3. Evaluate:
   - $J_{slow} = \sum_t \| U^\top z_t^{imag} - U^\top z_t^{post} \|^2$ (slow-subspace fidelity) — should decrease.
   - $J_{total} = \sum_t \| z_t^{imag} - z_t^{post} \|^2$ (total fidelity) — should not increase.
   - `eval_return` — should not decrease; ideally improves.

### 4.3 Long-Horizon Imagination Experiment (Stretch Goal)

The strongest demonstration: compare original DreamerV3 vs. the modified version at $H_{imag} = 50$ (instead of the default 15).

- **Hypothesis:** Original DreamerV3 degrades at $H_{imag} = 50$ because of trend-dominated drift. The modified version maintains imagination fidelity, enabling longer-horizon planning.
- This reframes the contribution as *enabling longer imagination horizons* rather than *marginal eval_return improvement at the default horizon*.

## 5. Budget Impact

| Experiment | GPU-hours | Cost |
|------------|-----------|------|
| §4.1 Inference-time validation | ~2 | ~$1.40 |
| §4.2 Full training (3 configs × 1 task) | ~30 | ~$21 |
| §4.3 Long-horizon (2 configs × 1 task) | ~20 | ~$14 |
| **Total** | **~52** | **~$36** |

Within the remaining ~168.5 GPU-hour budget.

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| $U$ estimated at warmup drifts as representation changes during training | Medium | Re-estimate $U$ every 50k steps; verify overlap metric |
| Anchor loss conflicts with DreamerV3's existing KL dynamics loss | Medium | Start with small $\beta$; monitor KL loss stability |
| Damping suppresses legitimate slow dynamics, harming policy learning | High | Keep $\eta \leq 0.5$; validate eval_return does not decrease |
| 15-step $H_{imag}$ is too short for anchor loss to have meaningful gradient signal | Medium | Check dc_trend MSE magnitude at step 5/10/15 in existing data |
| Both components together over-correct, double-suppressing slow updates | Medium | Test each component independently before combining |

## 7. Paper Positioning

**Without this extension:** DriftDetect is a diagnostic paper. Contribution is Finding A + characterization + Theorem 1. Reviewer question: "So what?"

**With this extension:** DriftDetect becomes a diagnosis-to-intervention paper. The narrative arc is:

1. We diagnose drift structure (Finding A).
2. We understand why it happens (Theorem 1 + D1–D3 falsification).
3. We show the diagnosis is actionable (Slow-Mode Anchored Dynamics).

This positions the paper as a complete research program rather than a standalone observation.

**Recommended venue framing:** "Frequency-domain diagnosis reveals that RSSM imagination drift concentrates in a low-dimensional slow subspace. This structural insight enables a targeted, zero-parameter intervention that reduces trend-dominated drift without disrupting fast dynamics."
