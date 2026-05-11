# Jacobian Regularization Baseline

## Motivation

SHARP penalizes the mean and variance structure of one-step RSSM prediction
errors. A natural reviewer question is whether the same drift reduction could
come from a simpler local sensitivity penalty: regularizing the Jacobian of the
learned dynamics model, as in Jacobian-control style dynamics regularizers.

This baseline answers that question directly for DreamerV3. It keeps the
training loop, model, task, rollout extraction, and checkpoint format aligned
with the Month 8 SHARP experiments, but replaces SHARP with a Jacobian norm
penalty.

## Objective

For deterministic RSSM state `d_t`, stochastic posterior state `s_t`, and
action `a_{t+1}`, define the deterministic transition:

```text
d_{t+1} = f_theta(s_t, d_t, a_{t+1})
```

The Jacobian baseline adds:

```text
L_total = L_DreamerV3 + lambda_jac * || d f_theta / d d_t ||_F^2
```

Only the deterministic transition Jacobian is regularized. The stochastic state
is treated as conditioning context, not as a differentiated input.

## Hutchinson Estimator

The full Jacobian is expensive for a 512-dimensional deterministic state. The
implementation uses a Hutchinson-style random projection. For a standard normal
output-space vector `v`:

```text
E_v || J^T v ||^2 = ||J||_F^2
```

The code samples one or more random projections per sampled start timestep and
computes `J^T v` with `torch.autograd.grad(..., create_graph=True)`, so the
result remains differentiable with respect to RSSM transition parameters.

## Expected Outcome

Jacobian regularization should reduce some local error amplification. It is not
expected to match SHARP when drift is dominated by systematic low-frequency
bias, because a local sensitivity penalty does not directly constrain the mean
or variance of one-step prediction errors. A low-Jacobian transition can still
produce biased predictions that accumulate during open-loop imagination.

This makes the baseline useful as a reviewer-facing comparison: if SHARP
outperforms Jacobian regularization at similar eval return, the result supports
the claim that statistical error shaping addresses failure modes that local
sensitivity control misses.

## References

- Fang et al. 2024: Jacobian regularization for learned dynamics stability.
- Hoffman et al. 2019: Hutchinson-style random projection estimators for
  Jacobian norm regularization.
