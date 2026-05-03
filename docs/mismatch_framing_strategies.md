# Mismatch Framing Strategies

## 1. The Data

The mismatch finding compares posterior slow-feature directions against the
empirical imagination drift directions. Overlap is computed as:

```text
overlap = ||U_posterior.T @ U_drift||_F^2 / r
```

Lower overlap means stronger mismatch. The current evidence spans DreamerV3,
DreamerV4, and toy RSSM settings:

| Setting | r=3 overlap | r=5 overlap | r=10 overlap | dc_trend% |
|---|---:|---:|---:|---:|
| V3 Cheetah | 0.13 | 0.17 | 0.28 | 46.7% |
| V3 Cartpole | 0.06 | 0.08 | 0.12 | 97.0% |
| V4 Cheetah | 0.64 | 0.41 | 0.43 | 63.9% |
| Toy simple | 0.61 | 0.70 | 0.80 | 19.1% |
| Toy medium | 0.41 | 0.67 | 0.85 | 43.1% |
| Toy hard | 0.38 | 0.50 | 0.74 | 38.3% |

The strength of mismatch varies systematically across settings. DreamerV3 RSSM
models show strong mismatch, especially Cartpole. DreamerV4 Cheetah shows
weaker but still incomplete posterior/drift alignment. The toy models show a
complexity-dependent pattern: simple settings are closer to aligned, while more
complex settings show stronger low-rank mismatch. Proper SFA slow features are
also misaligned with drift directions, confirming that the result is not merely
a PCA plus autocorrelation artifact.

## 2. Three Candidate Framings

### Framing A: Strong

**Claim:** Mismatch is universal in real-task autoregressive world models.
Posterior slow features and drift directions are fundamentally different
objects in any trained autoregressive world model.

This framing interprets the V4 Cheetah overlap values, `0.41-0.64`, as weaker
but still present mismatch.

Pros:

- Gives the paper the strongest possible headline claim.
- Produces a clean narrative: posterior slow features are not drift directions.
- Directly explains why `U_posterior` failed and why `U_drift` was necessary.

Cons:

- V4 Cheetah rank-3 overlap is `0.64`, which is arguably partial alignment
  rather than strong mismatch.
- A reviewer could challenge universality with one counterexample.
- Toy simple reaches rank-10 overlap of `0.80`, which weakens the claim that
  mismatch is universal across learned dynamical systems.

Assessment: This framing is punchy but brittle. It risks turning a nuanced and
interesting graded result into an overclaim.

### Framing B: Moderate

**Claim:** Mismatch strength is architecture-and-complexity dependent.
Posterior/drift mismatch exists in all tested settings, but its magnitude
varies systematically. It is strongest in capacity-constrained RSSM models
like DreamerV3, weaker in the higher-capacity DreamerV4 transformer-tokenizer
setting, and complexity-dependent in toy models.

This framing creates a mechanism story: mismatch emerges when model capacity
cannot simultaneously represent posterior structure and open-loop imagination
stability.

Pros:

- Honestly reflects all six data points without cherry-picking.
- Turns V4's partial alignment into evidence for a graded phenomenon rather
  than evidence against the finding.
- Connects naturally to the Month 5 toy inverted-U `dc_trend` complexity
  pattern and the broader capacity-complexity interaction story.
- Is hard for reviewers to attack because the nuance is acknowledged upfront.
- Still preserves the operational conclusion that `U_posterior` is unsafe as a
  default SMAD basis and that `U_drift` must be estimated directly.

Cons:

- Has a weaker headline than Framing A.
- Requires more exposition because the paper must explain why partial
  alignment in V4 is scientifically meaningful.

Assessment: This is the best match to the evidence. It is strong enough to be a
paper finding, but careful enough to survive cross-architecture scrutiny.

### Framing C: Conservative

**Claim:** Strong mismatch is specific to the GRU/tanh RSSM architecture in
DreamerV3. The V4 evidence is inconclusive.

Pros:

- Maximally safe.
- Very difficult for reviewers to challenge.
- Cleanly fits the strongest empirical cases: V3 Cheetah and V3 Cartpole.

Cons:

- Abandons the cross-architectural contribution.
- Makes the mismatch finding less impactful.
- Fails to use the V4 and toy evidence constructively.
- Leaves the capacity-complexity mechanism underdeveloped, even though the toy
  sweep was designed to test exactly that pattern.

Assessment: This framing is too cautious. It protects against overclaiming but
unnecessarily discards the strongest story that the full evidence supports.

## 3. Recommendation

Recommend Framing B: **Mismatch strength is architecture-and-complexity
dependent.**

Rationale:

- It honestly reflects all six data points without cherry-picking.
- It connects mismatch to the capacity-complexity interaction mechanism, which
  links the real-model evidence to the toy inverted-U result and creates a
  coherent Section 4 storyline.
- It is robust to reviewer pushback because it does not require claiming that
  mismatch is universal or equally strong in every architecture.
- It preserves the key intervention lesson: posterior slow features and drift
  directions are distinct enough in practice that `U_posterior` failed the SMAD
  pre-check, while `U_drift` became the necessary operational basis.

The recommended Section 4 narrative becomes:

```text
Mismatch exists, varies systematically with architecture and complexity, and
suggests a capacity-task-interaction mechanism that determines how much
posterior structure aligns with imagination stability.
```

Under this framing, V4's partial alignment is evidence for the graded-mismatch
story, not evidence against the finding. DreamerV4 does not invalidate the
mismatch result; it shows that a higher-capacity transformer-tokenizer world
model can align posterior structure and drift directions more than a
capacity-constrained RSSM while still failing to make them identical.

## 4. Implications for Paper Section 4

The Mismatch subsection should be structured as follows:

1. Lead with the strong DreamerV3 evidence. V3 Cheetah and V3 Cartpole both
   show low posterior/drift overlap, with Cartpole reaching only `0.06-0.12`.
   This establishes the core phenomenon in RSSM world models.
2. Introduce DreamerV4 as the contrast case. V4 Cheetah shows weaker mismatch,
   with rank-3 overlap of `0.64` but rank-5 and rank-10 overlap around `0.41`
   to `0.43`. This supports partial alignment rather than either complete
   alignment or complete mismatch.
3. Use the toy complexity sweep to propose the mechanism. Toy simple is
   near-aligned, while medium and hard settings show stronger low-rank
   mismatch. This suggests that mismatch grows when the model must trade off
   posterior representation quality against open-loop stability.
4. Connect back to SMAD. The mismatch explains why `U_posterior` failed in
   SMAD Pre-check 1 and why `U_drift` was necessary. Posterior slow features
   identify directions of slow variation under observed posterior inference;
   drift directions identify where imagined trajectories actually diverge.
   SMAD needs the second object.

This gives Section 4 a clear arc: strong RSSM mismatch, partial V4 contrast,
toy mechanism, and intervention consequence.
