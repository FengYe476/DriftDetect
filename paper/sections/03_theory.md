# 3. Theoretical Analysis

Long-horizon drift has two ingredients: source error $\varepsilon_t$, and
amplification geometry $\Phi_{t,T}$. Prior interventions mostly target
particular drift directions after accumulation; SHARP targets the source error
before accumulation. This section formalizes that distinction. It first writes
autoregressive drift as the accumulation of one-step prediction errors through
Jacobian chains. It then shows that partial methods lack a guarantee, and in
our experiments each exploited an unregularized escape route. Finally, it
proves that the SHARP population loss is a sufficient finite-horizon condition
for eliminating latent-coordinate drift in $L^2$.

## 3.1 Autoregressive Drift as Amplified One-Step Error

Let $z^*_0,\ldots,z^*_T$ denote a posterior latent trajectory used as the
reference coordinate system, and let $a_0,\ldots,a_{T-1}$ be the same action
sequence used for both the reference and imagined trajectories. The imagined
trajectory is teacher-forced in actions but recurrent in latent state:
$$
\hat z_{t+1}=f(\hat z_t,a_t), \qquad \hat z_0=z^*_0.
$$
This same-action construction isolates model dynamics from policy drift. The
policy is not asked to respond to the imagined state; the question is whether a
fixed action-conditioned transition can keep its latent coordinates aligned
with the posterior reference. All expectations below are over the evaluation
distribution of reference trajectories, action sequences, and any stochasticity
used to define the latent transition.

Define the drift, one-step prediction error, and single-step Jacobian as
$$
\Delta_t=\hat z_t-z^*_t,\qquad
\varepsilon_t=f(z^*_t,a_t)-z^*_{t+1},\qquad
A_t=\partial_z f(z^*_t,a_t).
$$
If $f$ is $C^2$ near the reference trajectory, Taylor expansion gives
$$
\Delta_{t+1}
=A_t\Delta_t+\varepsilon_t+R_t,
\qquad |R_t|\le C|\Delta_t|^2
$$
for a local constant $C$. Thus drift has a source term,
$\varepsilon_t$, and an amplification term, $A_t\Delta_t$.

For $t<T$, define the accumulated Jacobian chain
$$
\Phi_{t,T}=A_{T-1}A_{T-2}\cdots A_{t+1},
$$
with the empty product $\Phi_{T-1,T}=I$. Iterating the linearized recursion
from $\Delta_0=0$ yields
$$
\Delta_T
=\sum_{t=0}^{T-1}\Phi_{t,T}\varepsilon_t+\rho_T,
$$
where $\rho_T$ collects the second-order remainders. This decomposition is the
basic object of the theory. A method can reduce drift either by changing the
amplification geometry $\Phi_{t,T}$ or by reducing the source errors
$\varepsilon_t$. SHARP is analyzed as a source-level method. The distinction is
important because a small accumulated error in one direction can be produced
by many different source-error patterns, while controlling the source term
directly constrains what can enter every later Jacobian chain.

## 3.2 Why Local Interventions Lack Guarantees

The following propositions are no-guarantee statements. They do not say that a
partial intervention cannot help. They say that the intervention objective
alone does not force total latent drift to vanish. Each proposition identifies
one unregularized degree of freedom in the source error or in the recurrent
rollout distribution, then pairs that gap with the corresponding empirical
witness from Month 8.

### Proposition 1: Subspace No-Guarantee

**Statement.** Let $P_U$ be an orthogonal projection onto a rank-$r$ subspace
$U\subset\mathbb{R}^D$ with $r<D$. Any loss that constrains only
$P_U\varepsilon_t$ does not bound $P_{U^\perp}\varepsilon_t$. Consequently,
for
$$
L_U=\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}|P_U\varepsilon_t|^2,
$$
$L_U\to 0$ does not imply
$\frac{1}{T}\sum_t\mathbb{E}|\varepsilon_t|^2\to 0$, and therefore does not
imply total drift elimination.

**Proof sketch.** Choose any nonzero vector $v\in U^\perp$ and set
$\varepsilon_t=v$ for every $t$. Then $P_U\varepsilon_t=0$, so $L_U=0$, while
$\mathbb{E}|\varepsilon_t|^2=|v|^2>0$. Through the accumulation formula in
Section 3.1, this residual source error can still be propagated by
$\Phi_{t,T}$ into nonzero terminal drift. The projection loss has no term that
controls the complement. More generally, a sequence of models can make
$\mathbb{E}|P_U\varepsilon_t|^2\to 0$ while keeping
$\mathbb{E}|P_{U^\perp}\varepsilon_t|^2$ bounded away from zero. The projected
objective therefore certifies alignment on $U$, not total latent-coordinate
accuracy.

**Empirical witness.** Anchor loss gives the corresponding experimental case.
It reduced `J_slow` by `-79.5%` on the target subspace, but drift on other
directions increased by `+486%`, and `J_total` increased by `+4.1%`. The
targeted component improved, but the unconstrained complement absorbed the
error.

### Proposition 2: Off-Support Non-Identifiability

**Statement.** Let $S$ be the support of training latents. For an inference-time
correction $z'=z+\delta$ with $z'\notin S$, the training objective imposes no
constraint on $f(z',a)$. There exist two transition functions that agree on
$S$ for all actions observed during training but produce arbitrarily different
rollouts from $S+\delta$.

**Proof sketch.** Let $f$ be any transition function matching the training
objective on $S$. Construct $g(z,a)=f(z,a)+b\,\psi(z)$, where $b$ is an
arbitrary vector and $\psi$ is a smooth bump function that is zero on $S$ and
nonzero near a corrected point $z'\notin S$. Then $f$ and $g$ have identical
training loss because they agree on the support where the objective is
evaluated. However, once a corrected latent is fed back into recurrence, their
next states differ by $b\,\psi(z')$, and the resulting rollouts can separate
arbitrarily as $|b|$ grows. The bump construction can be chosen smooth, so
local differentiability of the transition does not by itself identify behavior
away from support. A training objective must either include the corrected
states or avoid feeding them back into the recurrent state.

**Empirical witness.** HIAD recurrent correction increased `J_total` by about
`+45%`, and recurrent DriftHead correction increased it by roughly `+74000%`.
The same DriftHead correction reduced `J_total` by `-42.9%` when applied
non-recurrently as a post-hoc adjustment. This comparison isolates the failure
point: correcting a latent can be useful, but feeding the corrected latent back
into recurrence moves the model off the support constrained by training.

### Proposition 3: Partial-Moment Escape

**Statement.** Any objective constraining only a strict subset of the first two
moments of $\varepsilon_t$ does not imply
$\mathbb{E}|\varepsilon_t|^2\to 0$. In particular, constraining only the mean
leaves covariance unconstrained, and constraining only a subspace leaves the
complement unconstrained.

**Proof sketch.** Let
$$
\mu_t=\mathbb{E}[\varepsilon_t],\qquad
\sigma_t^2=\mathbb{E}|\varepsilon_t-\mu_t|^2
          =\operatorname{tr}(\operatorname{Cov}(\varepsilon_t)).
$$
The second moment decomposes as
$$
\mathbb{E}|\varepsilon_t|^2
=|\mathbb{E}[\varepsilon_t]|^2+\operatorname{tr}(\operatorname{Cov}(\varepsilon_t))
=|\mu_t|^2+\sigma_t^2.
$$
A mean-only loss can drive $|\mu_t|^2$ to zero while leaving
$\sigma_t^2=c>0$; for example, $\varepsilon_t$ may be zero-mean noise with
fixed covariance. A subspace loss can drive the projected moment to zero while
leaving the complementary moment positive. In either case, the full second
moment remains nonzero. Since Section 3.1 shows that terminal drift is driven
by the accumulated source energy, leaving either component unbounded leaves a
route by which finite-horizon drift can persist.

**Empirical witness.** The true bias loss constrained the mean term
$|\mu_t|^2$ and was integrated into the model loss correctly, but the RSSM
escaped through variance. The noise variance diagnostic increased from `0` to
`27.5` within `30k` post-activation training steps, after which the run was
stopped.

Together, the three propositions separate the intervention problem into three
failure surfaces: unpenalized directions, unsupported recurrent states, and
uncontrolled moments. They do not rank methods by usefulness, since each method
can still be valuable in a regime where its assumptions hold. They clarify why
a guarantee for total latent drift requires controlling the full source error
on the support used by the recurrent rollout.

## 3.3 SHARP: Source-Level Statistical Hardening

SHARP (Statistical Hardening of Autoregressive Rollout Predictions) directly
regularizes the population moments that determine the one-step second moment.
For fixed horizon $T$, define
$$
L_{\mathrm{SHARP}}
=\frac{1}{T}\sum_{t=0}^{T-1}
\left[
\beta_\mu|\mu_t|^2+\beta_\sigma\sigma_t^2
\right],
$$
where
$$
\mu_t=\mathbb{E}[\varepsilon_t],
\qquad
\sigma_t^2=\mathbb{E}|\varepsilon_t-\mu_t|^2.
$$
Here $\beta_\mu>0$ and $\beta_\sigma>0$. By the decomposition above,
controlling both terms controls the full one-step source energy:
$$
\mathbb{E}|\varepsilon_t|^2=|\mu_t|^2+\sigma_t^2.
$$

This is the key distinction from the partial objectives in Section 3.2.
Subspace penalties leave directions unconstrained. Mean-only bias penalties
leave variance unconstrained. In contrast, SHARP is the minimal moment-based
loss among tested methods that directly controls both components of
$\mathbb{E}|\varepsilon_t|^2$.

This statement is deliberately limited to the class of moment-based losses
tested here. It does not rule out other objectives that might control source
error through likelihood, contrastive prediction, or uncertainty calibration.
It says that within the tested family, the mean term and the variance term are
the two terms needed to upper-bound the source second moment used in the
finite-horizon theorem below.

The theory uses the population loss rather than a single minibatch estimate.
The DreamerV3 implementation uses a squared per-dimension batch variance term
for numerical convenience. Appendix X translates between the implementation and
the theorem: if $\Sigma=\operatorname{Cov}(\varepsilon_t)$ and
$\operatorname{Var}[\varepsilon_t]$ denotes the vector of per-dimension
variances, then
$$
\operatorname{tr}(\Sigma)
\le \sqrt{D}\|\operatorname{Var}[\varepsilon_t]\|_2
$$
by Cauchy-Schwarz. Thus convergence of the implemented variance norm still
implies convergence of the trace term, while finite-loss constants become
square-root rather than linear.

## 3.4 SHARP Finite-Horizon Sufficiency

The theorem below formalizes the sufficient condition supplied by SHARP. The
assumptions are stated explicitly because each one marks a boundary of the
claim. Same-action rollout avoids confounding transition error with policy
feedback. Population loss separates the mathematical guarantee from minibatch
estimation noise. The reference-coordinate assumption says that posterior
latents are the coordinates in which drift is measured. The bounded-chain and
remainder assumptions keep the nonlinear recurrence in the finite-horizon
regime described by the Taylor expansion in Section 3.1.

**Theorem 1 (finite-horizon SHARP sufficiency).** Fix a finite horizon $T$.
Assume:

1. The rollout is same-action, or teacher-forced in actions: imagined and
   reference trajectories use the same action sequence $a_t$.
2. $f$ is $C^2$ in a neighborhood of the reference trajectory
   $z^*_0,\ldots,z^*_T$.
3. The Jacobian chains are bounded:
   $|\Phi_{t,T}|_{\mathrm{op}}\le M_T$ for all $t<T$.
4. $L_{\mathrm{SHARP}}$ is the population-level loss.
5. The posterior latent $z^*$ is a valid reference coordinate.
6. The second-order Taylor remainder is controlled so that the accumulated
   remainder contributes
   $O(\mathbb{E}\max_{0\le t\le T}|\Delta_t|^4)$ to the terminal squared
   drift.

Let $\beta_{\min}=\min(\beta_\mu,\beta_\sigma)$. Then
$$
\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}|\varepsilon_t|^2
\le
\frac{L_{\mathrm{SHARP}}}{\beta_{\min}},
$$
and
$$
\mathbb{E}|\Delta_T|^2
\le
T^2M_T^2
\left(\frac{L_{\mathrm{SHARP}}}{\beta_{\min}}\right)
+O\!\left(\mathbb{E}\max_{0\le t\le T}|\Delta_t|^4\right).
$$
Consequently, for fixed finite $T$, if $L_{\mathrm{SHARP}}\to 0$ and the
controlled remainder vanishes, then $\mathbb{E}|\Delta_T|^2\to 0$. The
convergence claim is in $L^2$. Appendix X gives the full proof and notes that
a limiting random variable with zero second moment is zero almost surely.

**Proof sketch.** The second-moment identity gives
$$
\mathbb{E}|\varepsilon_t|^2=|\mu_t|^2+\sigma_t^2.
$$
Since $\beta_{\min}\le\beta_\mu,\beta_\sigma$,
$$
\beta_{\min}\mathbb{E}|\varepsilon_t|^2
\le
\beta_\mu|\mu_t|^2+\beta_\sigma\sigma_t^2.
$$
Averaging over $t$ proves the first inequality.

For the drift bound, unroll the Taylor recursion:
$$
\Delta_T
=\sum_{t=0}^{T-1}\Phi_{t,T}\varepsilon_t+\rho_T.
$$
Using $|\Phi_{t,T}|_{\mathrm{op}}\le M_T$ and
$|\sum_t x_t|^2\le T\sum_t |x_t|^2$,
$$
\mathbb{E}\left|\sum_{t=0}^{T-1}\Phi_{t,T}\varepsilon_t\right|^2
\le
T\sum_{t=0}^{T-1}M_T^2\mathbb{E}|\varepsilon_t|^2
=
T^2M_T^2
\left(\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}|\varepsilon_t|^2\right).
$$
Substituting the first inequality gives the leading term. The controlled
second-order Taylor remainder contributes the stated
$O(\mathbb{E}\max_t|\Delta_t|^4)$ term.

The proof is intentionally finite-horizon. No uniform-in-time stability claim
is made from the loss alone. The constant $M_T$ summarizes the amplification
geometry over the chosen horizon, and the bound degrades if the Jacobian chains
grow. This matches the empirical use of fixed diagnostic horizons such as
`25:175`, where the goal is to reduce measured latent drift over the
imagination window used by the agent.

**Corollary 1 (temporal uncorrelation).** If the centered errors
$\varepsilon_t-\mu_t$ are uncorrelated across time, define
$$
\bar\mu^2=\frac{1}{T}\sum_{t=0}^{T-1}|\mu_t|^2,
\qquad
\bar\sigma^2=\frac{1}{T}\sum_{t=0}^{T-1}\sigma_t^2.
$$
Then
$$
\mathbb{E}|\Delta_T|^2
\lesssim
M_T^2\left(T^2\bar\mu^2+T\bar\sigma^2\right)
+\text{higher-order terms}.
$$
The bias term scales as $T^2$: $T$ steps of same-direction bias produce
$|\Delta_T|^2$ on the order of $T^2|\mu|^2$. The variance term scales as $T$
under temporal uncorrelation, and as $T^2$ in the worst case covered by the
theorem.

This corollary explains why the mean term is especially important for
long-horizon drift. A small persistent bias can add coherently across time
before squaring, while zero-mean fluctuations add linearly when they are
temporally uncorrelated. SHARP includes both terms because the worst-case bound
does not assume uncorrelated errors, and because the true bias-loss experiment
showed that variance can become the active escape route when it is left
unpenalized.

## 3.5 Empirical Validation of Theoretical Assumptions

The theorem is a finite-horizon latent-coordinate statement. Section 3.5 does
not re-prove its assumptions in trained world models; it checks whether the
measured behavior follows the source-error and accumulation pattern predicted
by the theory.

First, SHARP reduces the source error. On V3 Cheetah, the per-step drift
increment decreases from `0.0062` to `0.0009`, a `-84.8%` change. This is the
quantity closest to the theoretical $\mathbb{E}|\varepsilon_t|^2$ source term.
It is also the measurement that most directly distinguishes source reduction
from post-accumulation cancellation.

Second, cumulative drift decreases at the same order of magnitude. Latent
`J_total` decreases by `-89.6%` on V3 Cheetah, `-97.2%` on V3 Cartpole, and
`-64.1%` on the V4 Cheetah Transformer fine-tune. These results are consistent
with the theorem's prediction that reducing one-step source energy reduces
finite-horizon latent drift after amplification by bounded Jacobian chains.
The V4 result is especially relevant to the role of $\Phi_{t,T}$, because the
Transformer dynamics have a different amplification geometry from the V3 RSSM
but still respond to the same source-level constraint.

Third, the reductions do not appear to arise from horizon redistribution. On
V3 Cheetah, early, mid, and late band drift decrease by `-88.9%`, `-89.7%`,
and `-90.2%`, respectively. On V4 Cheetah, the corresponding decreases are
`-61.6%`, `-65.7%`, and `-64.8%`. The uniformity across bands is important:
it matches the source-level account, where SHARP reduces the error before it
is accumulated rather than moving accumulated drift to a different region of
the horizon.

Together, these checks do not prove the assumptions from data. They show that
the measured behavior is aligned with the theorem's mechanism: one-step source
energy decreases first, accumulated latent drift decreases next, and the
reduction is not explained by moving error to a later horizon band.
