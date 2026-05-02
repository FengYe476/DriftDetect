# SMAD Basis Design Decision

**Date:** 2026-05-01  
**Chosen basis:** U_drift (PCA on empirical drift vectors)  
**Rank:** r = 10  
**Source data:** 20 Cheetah v2 rollouts, trim window [25, 175]  
**Saved path:** results/smad/U_drift_cheetah_r10.npy

## Rationale

Pre-check 3 established that neither SFA slow features nor PCA+autocorrelation posterior features align with the empirical drift subspace. Pre-check 4 confirmed that U_drift generalizes across rollout splits with transfer ratio 0.97-0.98.

U_drift is the only basis that:

1. Aligns with the actual drift phenomenon by construction
2. Generalizes to unseen rollouts (Pre-check 4)
3. Produces meaningful damping effect (24.8% dc_trend reduction at eta=0.20 on held-out data)

## Circularity Acknowledgment

U_drift is estimated from the same checkpoint whose training we aim to improve. This means:

- The basis captures the drift structure of the BASELINE model
- After SMAD training, the model's drift structure will change
- The fixed basis may become partially stale

This is acceptable for Phase 2 because:

1. Pre-check 4 showed the basis is stable across rollout noise (transfer ratio ~0.98)
2. The basis captures checkpoint-level structural properties, not per-rollout noise
3. Iterative basis re-estimation (train -> re-estimate U -> retrain) is a natural extension but out of scope for Phase 2

## Connection to Pre-check 3 Finding

The discovery that posterior slow features != drift directions is itself a paper contribution. It means:

- Drift is temporally slow (dc_trend frequency dominance, Finding A)
- But drift accumulates in spatial directions that are NOT the environment's slow-feature directions
- The drift subspace is a model-specific error attractor, distinct from both SFA slow features and high-variance posterior modes
- This has implications for Theorem 1's framing: the theorem's slow subspace identifiability result applies to the idealized setting, but in practice the empirical drift subspace serves as the operational target
