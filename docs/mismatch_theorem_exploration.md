# Mismatch Theorem Exploration

## 1. Candidate Theorem B'

Candidate Theorem B' is currently a conjecture supported by numerical evidence,
not a proved result.

Conjecture: consider a finite-horizon linear-Gaussian world model with
contractive dynamics:

```text
x_{t+1} = A_true x_t + w_t
z_t = x_t + e_t
z^imag_{t+h+1} = A_model z^imag_{t+h}
```

where `rho(A_true) < 1`, `rho(A_model) < 1`, process noise `w_t` is Gaussian,
inference noise `e_t` has bounded covariance, and imagination is evaluated over
a finite horizon. The posterior slow subspace is defined by the top
eigenvectors of the posterior latent covariance, while the drift subspace is
defined by the top eigenvectors of the accumulated open-loop drift covariance:

```text
drift_{t,h} = z^imag_{t+h} - x_{t+h}
```

The conjecture is that these two subspaces are determined by different
operators:

- posterior slow features depend on the covariance of observed posterior
  latents under inference updates;
- drift directions depend on the covariance induced by open-loop model error,
  propagated inference noise, and finite-horizon accumulation.

Their alignment angle is zero only under special symmetry conditions, such as
isotropic inference noise and no effective capacity gap between `A_model` and
`A_true`. In the capacity-bounded regime, where `A_model` differs from
`A_true` through a non-commuting model error, the posterior/drift angle is
strictly positive and tends to grow as the capacity gap increases.

The theorem-check script uses a 10-dimensional contractive system with three
slow modes:

```text
eig(A_true) = [0.99, 0.97, 0.95, 0.90, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
```

The dimensionality is intentionally larger than the rank-3 subspace being
tested, so random subspace overlap has room to fall well below perfect
alignment. This experiment provides numerical support for the conjecture, not
a proof.

## 2. Numerical Verification Plan

Run:

```bash
python scripts/run_toy_mismatch_theorem_check.py
```

The script writes:

```text
results/tables/toy_mismatch_theorem_check.json
```

### Capacity Gap Sweep

The capacity sweep fixes isotropic inference noise and varies:

```text
A_model = A_true + epsilon * perturbation
```

where the perturbation is non-commuting with `A_true` and maps slow coordinates
into faster coordinates in the `A_true` eigenbasis. The sweep uses
`epsilon in np.linspace(0.0, 0.3, 10)`.

Supportive outcome:

- posterior/drift overlap decreases as `epsilon` increases;
- the endpoint overlap at `epsilon=0.3` is lower than at `epsilon=0.0`;
- Pearson correlation between `epsilon` and overlap is negative.

Refuting or weakening outcome:

- overlap is flat or increases with `epsilon`;
- the endpoint drop is near zero;
- the trend depends strongly on random seed.

### Noise Structure Sweep

The noise sweep fixes `epsilon=0.1` and compares isotropic inference noise
against structured covariance conditions. Structured noise uses rotated
diagonal covariances with 10 eigenvalues:

- `isotropic`: all eigenvalues equal;
- `mild_structured`: eigenvalues from `1.0` down to `0.1`;
- `structured`: log-spaced eigenvalues from `1.0` down to `0.001`;
- `strong_structured`: log-spaced eigenvalues from `1.0` down to `0.0003`.

Supportive outcome:

- isotropic noise produces the highest posterior/drift overlap;
- structured noise lowers overlap, showing that posterior and drift operators
  separate when inference covariance is anisotropic.

Refuting or weakening outcome:

- structured noise produces overlap equal to or higher than isotropic noise;
- the ordering is unstable across seeds.

## 3. Results

### Capacity Gap Sweep

| epsilon | spectral_radius | overlap |
|---:|---:|---:|
| 0.000 | 0.9900 | 0.9993 |
| 0.033 | 0.9900 | 0.9982 |
| 0.067 | 0.9900 | 0.9928 |
| 0.100 | 0.9900 | 0.9789 |
| 0.133 | 0.9900 | 0.7712 |
| 0.167 | 0.9900 | 0.6605 |
| 0.200 | 0.9900 | 0.6550 |
| 0.233 | 0.9900 | 0.6470 |
| 0.267 | 0.9900 | 0.6353 |
| 0.300 | 0.9900 | 0.6183 |

The capacity-gap sweep supports the conjecture. Posterior/drift overlap falls
from `0.9993` at `epsilon=0.0` to `0.6183` at `epsilon=0.3`, an endpoint drop
of `0.3810`. The Pearson correlation between `epsilon` and overlap is
`-0.9253`, showing a strong negative relationship between model-environment
capacity gap and posterior/drift alignment.

### Noise Structure Sweep

| condition | overlap |
|---|---:|
| isotropic | 0.9792 |
| mild_structured | 0.6640 |
| structured | 0.6663 |
| strong_structured | 0.6663 |

The noise-structure sweep also supports the conjecture. Isotropic inference
noise yields the highest overlap (`0.9792`), while structured noise conditions
drop to approximately `0.664-0.666`. This shows that anisotropic inference
noise separates the posterior covariance operator from the accumulated drift
covariance operator.

### Interpretation

Verdict: **Candidate Theorem B' is numerically supported in the linear
setting. The mismatch is a structural consequence of the capacity gap between
model and environment, modulated by inference noise structure.**

The endpoint drop from `0.9993` to `0.6183` demonstrates that a
model-environment capacity gap is sufficient to create posterior/drift
mismatch in the linear setting. The spectral radius stays at approximately
`0.99` across the sweep, so the effect is not caused by loss of contractivity.
Instead, it comes from the non-commuting model error changing the operator that
governs accumulated open-loop drift.

The `dim=10`, `r=3` setup gives the trend meaningful geometric room. The random
rank-3 overlap baseline is approximately `r / dim = 0.3`, so the observed drop
from near-perfect alignment to `0.6183` is well above random but clearly away
from identity. This matches the intended interpretation: the theorem check
does not claim arbitrary orthogonality, but it does show that posterior slow
features and drift directions separate systematically as the capacity gap
increases.

## 4. Implications

If the numerical results support the conjecture, the mismatch finding is not a
pathology of a single trained RSSM. It is a structural consequence of how
posterior inference and open-loop imagination use a learned model differently.
Posterior slow features describe high-variance, slowly changing directions
under observed posterior updates; imagination drift directions describe where
model error and propagated inference uncertainty accumulate without future
observations.

This would support the capacity-complexity interaction story from Framing B.
DreamerV3's GRU/tanh RSSM may have a larger effective capacity gap on complex
tasks, producing stronger posterior/drift mismatch. DreamerV4's
transformer-tokenizer model may have a smaller effective capacity gap, producing
weaker but still present mismatch. Toy simple settings can remain near-aligned,
while harder toy settings show stronger low-rank mismatch.

This would provide theoretical grounding for the Section 4 narrative:

```text
Mismatch exists, varies systematically with architecture and complexity, and
suggests a capacity-task-interaction mechanism that determines how much
posterior structure aligns with imagination stability.
```

If the numerical results are mixed or negative, the mismatch finding remains
empirically supported but lacks a clean linear-theory explanation. In that
case, the paper should present mismatch as an empirical Section 4 result rather
than a Section 3 theorem, and should state that explaining the phenomenon may
require nonlinear RSSM analysis, state-dependent drift bases, or
architecture-specific inference dynamics beyond this linear toy setting.
