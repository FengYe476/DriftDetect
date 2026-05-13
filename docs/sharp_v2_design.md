# SHARP-v2 Run A Design

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

## Expected Outcomes

Run A should trade some raw scale collapse for healthier normalized and decoded
behavior:

- Imag trace reduction below `40%`, instead of the v1 `82-96%` collapse.
- Normalized `J_total` reduction above `50%`, improving on the weak Cheetah
  normalized result.
- Step 0/1 observation error ratio below `2x`, fixing the Cartpole early
  fidelity failure.
- Eval return within `5%` of the baseline.

If these outcomes hold, the result supports the interpretation that SHARP's
useful part is transition hardening, while the v1 failure mode was posterior
representation compression.

## Future Runs

- Run B: normalized SHARP, measuring prediction errors in a scale-normalized
  latent space to remove latent magnitude incentives.
- Run C: variance floor, preventing the auxiliary loss from rewarding
  near-zero posterior spread.
- Run D: overshooting, extending the transition-only objective beyond one step
  while preserving the posterior detach boundary.
- Run E: prior fidelity, adding checks or losses that keep transition priors
  useful for decoder and reward prediction, not only latent drift metrics.
