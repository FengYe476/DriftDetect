# Month 9 Plan

Month 9 turns the Month 8 single-seed SHARP result into reviewer-ready
multi-seed evidence and adds comparison baselines.

## Weeks 1-2: GPU Training

Primary goal: launch and monitor all high-priority GPU runs.

Launch in parallel where budget allows:

```bash
# R2-Dreamer
MODE=both bash scripts/runpod_r2_restart.sh

# V3 Cheetah multi-seed
bash scripts/runpod_v3_multiseed.sh 43 cheetah
bash scripts/runpod_v3_multiseed.sh 44 cheetah
bash scripts/runpod_v3_multiseed.sh 45 cheetah

# V3 Cartpole multi-seed
bash scripts/runpod_v3_multiseed.sh 43 cartpole
bash scripts/runpod_v3_multiseed.sh 44 cartpole
bash scripts/runpod_v3_multiseed.sh 45 cartpole

# Jacobian regularization baseline
bash scripts/runpod_jacobian_reg.sh 42 cheetah
bash scripts/runpod_jacobian_reg.sh 43 cheetah
bash scripts/runpod_jacobian_reg.sh 44 cheetah
```

Daily monitoring:

- Check `nvidia-smi`, training logs, checkpoint timestamps, and eval metrics.
- Verify each finished run immediately with checkpoint existence, rollout count,
  and manifest presence.
- Download completed results from RunPod before terminating pods.
- Record failures, restarts, and anomalous eval swings.

## Week 3: Analysis And Paper Writing

Analysis:

```bash
python scripts/evaluate_v3_multiseed.py --task dmc_cheetah_run
python scripts/evaluate_v3_multiseed.py --task dmc_cartpole_swingup
python -m json.tool results/tables/r2_comparison.json | head -120
```

Core analysis tasks:

- Summarize V3 Cheetah SHARP multi-seed `J_total` reduction, std, and eval
  retention.
- Summarize V3 Cartpole SHARP multi-seed `J_total` reduction, std, and eval
  retention.
- Compare SHARP against Jacobian regularization on Cheetah drift and eval.
- Analyze R2 baseline versus SHARP v2, with explicit caveats about training
  instability.
- Update frequency-band decomposition tables for all complete seeds.

Paper tasks:

- Rewrite Section 1, Introduction v2, around the multi-seed story.
- Draft Section 2, Related Work, including Jacobian regularization and learned
  dynamics stability.
- Rewrite Section 4, Findings v2, with multi-seed Cheetah and Cartpole results.
- Finalize Theorem 1 LaTeX and align notation with Sections 3-4.

## Week 4: Paper Polish And Submission Prep

Paper polish:

- Draft Section 7, Discussion and Conclusion.
- Run AI Reviewer Round 4.
- Update all figures with multi-seed error bars.
- Update abstract with final multi-seed and baseline comparison claims.
- Compile full paper and complete a full read-through for claim consistency.

Release prep:

```bash
git status --short
git tag v0.9-month9-multiseed-validation
git push origin main
git push origin v0.9-month9-multiseed-validation
```

Final checks:

- All key result JSON files are under `results/tables/`.
- All plots used in the paper are regenerated from committed scripts.
- Every numerical paper claim has a source table or log.
- R2 limitations are stated clearly and not overclaimed.
- Jacobian-reg comparison is framed as a reviewer-facing baseline, not a
  replacement for SHARP.
