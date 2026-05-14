# SHARP-v2 Design

## Motivation

SHARP v1 reduced raw latent drift very strongly, but the follow-up diagnostics
showed that part of the gain came from latent compression rather than better
open-loop dynamics.

The strongest signals were:

- Cheetah: raw `J_total` fell by about `85%`, but normalized `J_total` fell by
  only about `31%`, implying that much of the raw improvement was a scale
  artifact.
- Cartpole: latent trace fell by about `96%`, but Step 0/1 observation error
  was already `8-10x` worse before meaningful dynamics accumulation.
- Cartpole low-band one-step error increased, and multi-step amplification
  grew sharply despite the raw drift reduction.

The likely mechanism is a gradient shortcut. SHARP v1 computes one-step RSSM
prediction error against posterior states produced by the encoder/inference
path. Its target `post["deter"][t + 1]` is detached, but its input state
`post["stoch"][t]`, `post["deter"][t]`, and any other posterior state keys are
fed live into `dynamics.img_step()`. SHARP gradients can therefore flow back
through the posterior representation and pressure the encoder/RSSM posterior to
shrink the latent manifold.

## Run A: Transition-Only SHARP

Run A keeps the SHARP v1 loss and hyperparameters unchanged:

```text
L_SHARP = beta_mean * ||E_batch[epsilon_t]||^2
        + beta_var  * ||Var_batch[epsilon_t]||^2

epsilon_t = f(post_t, a_t) - post_{t+1}
```

The only functional change is the gradient boundary in
`scripts/train_moment_match_v2.py`:

- Already detached in v1: deterministic target `post["deter"][t + 1]`.
- Newly detached in v2: `post["stoch"][t]`, `post["deter"][t]`, and every
  other posterior state key passed through `post.items()`.
- Newly detached as a no-op guard: replay `action`, which normally does not
  require gradients.

With this boundary, SHARP gradients update the transition path through
`dynamics.img_step()` but do not update encoder/posterior inputs through the
SHARP auxiliary objective. The ordinary Dreamer model loss still trains the
encoder, posterior, decoder, reward head, KL terms, and actor-related world
model surfaces exactly as before.

## Run A Outcome

Run A traded some of SHARP v1's raw collapse for healthier decoded behavior,
but still left a scale artifact:

- Raw `J_total` reduction was `69.6%`.
- Normalized `J_total` reduction was only about `21.5%`, below the `>40%`
  target.
- Imag trace reduction was `65.9%`, still above the `<40%` target.
- Step 0/1 observation error was healthy at `0.107`.
- Eval return reached a `702` peak, then dropped to about `350`.

Run A supports transition-only SHARP as the right gradient direction, but shows
that the raw loss can still reward latent-scale contraction.

## Run B: Normalized Transition-Only SHARP

Run B makes the SHARP loss scale-invariant by normalizing one-step transition
errors by detached posterior scale:

```text
sigma_t = stopgrad(std_batch(post_deter[t + 1]) + eps)
epsilon_norm_t = (img_step(post_t, action_t) - post_deter[t + 1]) / sigma_t

L_SHARP = beta_mean * ||E_batch[epsilon_norm_t]||^2
        + beta_var  * ||Var_batch[epsilon_norm_t] - tau||^2
```

The normalization uses per-dimension posterior deterministic-state standard
deviation across the batch, clamped with `eps=1e-6`. `sigma_t` is detached, so
SHARP cannot reduce the loss by changing the normalization denominator through
the posterior path.

Run B also changes the variance term from a zero target to a calibrated target:
`tau=1.0`. This prevents the auxiliary loss from directly rewarding collapse
of normalized residual variance. The training script logs `mm_norm_scale`, the
mean sampled `sigma_t`, so contraction can be monitored during training. If
`mm_norm_scale` trends toward zero, posterior scale is still shrinking despite
normalization.

The screening configs use `300k` environment steps. Expected Run B outcomes:

- Normalized `J_total` reduction above `35-40%`.
- Imag trace reduction below `60%`.
- Step 0/1 observation error remains healthy.

## Run C: Trace Floor + Optional Jacobian Reg

Run C returns to Run A's raw transition-only SHARP objective because it gave the
strongest drift pressure, then adds an explicit posterior trace floor to block
the latent-collapse route. SHARP still detaches every posterior-derived input
to `dynamics.img_step()`, so SHARP gradients remain transition-only.

The trace floor is intentionally different: it uses live `post["deter"]`
without detach, so its gradients can reach the representation path and resist
posterior compression:

```text
current_trace = mean_t sum_dim Var_batch(post_deter[t])
L_trace = beta_trace * relu(trace_floor - current_trace)^2
```

Trace floor starts from step 0. Optional Jacobian regularization is warmup-gated
with SHARP and penalizes local transition amplification through a Hutchinson
estimate of `||d deter_{t+1} / d deter_t||_F^2`.

Run C screening variants:

| Run | SHARP mode | Trace floor | beta_trace | Jacobian lambda | Purpose |
|---|---:|---:|---:|---:|---|
| C1 | raw | 20.0 | 0.1 | 0.0 | Run A drift pressure with 0.5x baseline trace floor |
| C2 | raw | 24.0 | 0.1 | 0.0 | Slightly stronger 0.6x baseline trace floor |
| C3 | raw | 20.0 | 0.1 | 0.001 | C1 plus small local Jacobian control |

## Future Runs

- Run D: overshooting, extending the transition-only objective beyond one step
  while preserving the posterior detach boundary.
- Run E: prior fidelity, adding checks or losses that keep transition priors
  useful for decoder and reward prediction, not only latent drift metrics.
