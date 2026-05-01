# Cartpole Training Dynamics Decision

**Date:** 2026-05-01

**Decision:** SKIP Cartpole training dynamics

## Rationale

- The core question, when dc_trend dominance emerges, has been clearly answered by Cheetah: it is architecture-intrinsic and present from step 5000.
- Month 2 already confirmed cross-task generality of Finding A on Cartpole, where dc_trend accounts for 97.0% of latent MSE, so Cartpole's drift pattern at the final checkpoint is known.
- Repeating 102-checkpoint training dynamics on Cartpole would cost approximately 10 GPU-hours, about $7, plus approximately 3 hours of local pipeline time for a result that is very likely to confirm the same pattern.
- The marginal scientific value does not justify the compute and time cost.
- If reviewers request cross-task training dynamics, it can be added during the revision period.

## What We Already Have for Cartpole

- Trained checkpoint with peak return of approximately 854.
- 20 v2 rollouts with latent features.
- Finding A confirmed, with dc_trend accounting for 97.0% of latent MSE.
- Figure 1 for Cartpole generated.

## Risk Assessment

Low. The Cheetah training dynamics result is strong and clear. Cartpole's even more extreme dc_trend dominance, 97% versus 47%, suggests the same architecture-intrinsic pattern would hold.
