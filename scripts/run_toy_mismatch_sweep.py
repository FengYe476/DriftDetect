#!/usr/bin/env python3
"""Multi-seed toy mismatch sweep across oscillator complexity levels.

This script reruns the fixed-capacity toy RSSM at simple, medium, and hard
environment complexity, then measures whether posterior/drift subspace overlap
falls as complexity increases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_toy_complexity_sweep import (  # noqa: E402
    MultiOscillatorEnv,
    complexity_specs,
    relative_path,
    resolve_path,
    sample_batch,
    set_seed,
    to_jsonable,
)
from src.toy.minimal_rssm import MinimalRSSM  # noqa: E402
from src.toy.smad_intervention import analyze_drift, collect_rollouts  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "toy_mismatch_complexity.json"
RANKS = (3, 5, 10)
OVERLAP_PAIRS = ("posterior_drift", "sfa_posterior", "sfa_drift")
MONTH5_REFERENCE_DC_TREND_PCT = {
    "simple": 24.1,
    "medium": 46.6,
    "hard": 31.1,
}
REAL_MODEL_REFERENCE_PATHS = {
    "V3 Cheetah": REPO_ROOT / "results" / "tables" / "smad_alignment_check.json",
    "V3 Cartpole": REPO_ROOT / "results" / "tables" / "cartpole_mismatch_check.json",
    "V4 Cheetah": REPO_ROOT / "results" / "tables" / "v4_mismatch_check.json",
}


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    specs = complexity_specs()

    print("Toy mismatch complexity sweep", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Seeds per level: {seeds}", flush=True)
    print(f"Train steps per seed: {args.train_steps}", flush=True)
    print(f"Horizon: {args.horizon}", flush=True)
    print(f"Eval episodes: {args.eval_episodes}", flush=True)
    print(f"Trim window: [{args.trim[0]}, {args.trim[1]}]", flush=True)
    print("", flush=True)

    per_seed_results = []
    for spec in specs:
        for seed in seeds:
            per_seed_results.append(run_seed(spec, seed, args))

    summaries = summarize_by_complexity(per_seed_results, specs, seeds)
    hypothesis = hypothesis_test(summaries)
    real_model_references = load_real_model_references()
    plot_points = build_plot_points(per_seed_results)

    output = {
        "analysis": "toy_mismatch_complexity_sweep",
        "hypothesis": (
            "posterior/drift overlap may decrease as toy environment "
            "complexity increases, linking mismatch to the inverted-U "
            "complexity pattern"
        ),
        "notes": {
            "correlation_tests": (
                "Descriptive only: there are three complexity levels, so "
                "correlations are not inferential statistics."
            ),
            "seed_policy": "Seeds 0..n-1 are reused for every complexity level.",
        },
        "config": {
            "seeds": seeds,
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "horizon": int(args.horizon),
            "eval_episodes": int(args.eval_episodes),
            "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
            "lr": float(args.lr),
            "free_bits": float(args.free_bits),
            "kl_scale": float(args.kl_scale),
            "device": str(args.device),
            "rssm": {
                "deter_dim": int(args.deter_dim),
                "stoch_dim": int(args.stoch_dim),
                "hidden_dim": int(args.hidden_dim),
            },
        },
        "complexity_levels": [complexity_metadata(spec) for spec in specs],
        "per_seed_results": per_seed_results,
        "plot_points": plot_points,
        "summary_by_complexity": summaries,
        "hypothesis_test": hypothesis,
        "real_model_references": real_model_references,
        "month5_reference_dc_trend_pct": MONTH5_REFERENCE_DC_TREND_PCT,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(output), indent=2) + "\n")

    print_master_table(summaries)
    print_hypothesis_summary(hypothesis)
    print(f"\nSaved: {relative_path(args.output)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed toy posterior/drift mismatch sweep."
    )
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--eval_episodes",
        "--n_episodes",
        dest="eval_episodes",
        type=int,
        default=10,
    )
    parser.add_argument("--trim", type=int, nargs=2, default=(5, 45))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--free_bits", type=float, default=0.0)
    parser.add_argument("--kl_scale", type=float, default=0.1)
    parser.add_argument("--deter_dim", type=int, default=64)
    parser.add_argument("--stoch_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_seeds", "--n-seeds", dest="n_seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output = resolve_path(args.output)
    args.trim = (int(args.trim[0]), int(args.trim[1]))
    if args.train_steps <= 0:
        raise ValueError("--train_steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")
    if not 0 <= args.trim[0] <= args.trim[1] < args.horizon:
        raise ValueError("--trim must be inside horizon.")
    if args.n_seeds <= 0:
        raise ValueError("--n_seeds must be positive.")
    if args.seeds is not None and len(args.seeds) == 0:
        raise ValueError("--seeds must contain at least one seed.")
    if args.deter_dim < max(RANKS):
        raise ValueError(f"--deter_dim must be at least {max(RANKS)}.")
    if args.stoch_dim <= 0:
        raise ValueError("--stoch_dim must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be positive.")
    return args


def run_seed(spec: Any, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    env = MultiOscillatorEnv(
        frequencies=spec.frequencies,
        amplitudes=spec.amplitudes,
        episode_length=args.horizon,
        coupling_strength=spec.coupling_strength,
        seed=seed,
    )
    rssm = MinimalRSSM(
        obs_dim=env.obs_dim,
        action_dim=1,
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=args.device,
    )

    print(
        f"[{spec.name} seed={seed}] obs_dim={env.obs_dim} "
        f"n_osc={env.n_osc} frequencies={list(spec.frequencies)}",
        flush=True,
    )
    last_metrics: dict[str, float] = {}
    for step in range(1, args.train_steps + 1):
        obs, actions, rewards = sample_batch(env, args.batch_size, args.horizon)
        last_metrics = rssm.train_step(
            obs,
            actions,
            reward_seq=rewards,
            free_bits=args.free_bits,
            kl_scale=args.kl_scale,
            sample=True,
        )
        if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
            print(
                f"  [{spec.name} seed={seed}] step {step}/{args.train_steps} "
                f"loss={last_metrics['loss']:.4f} "
                f"recon={last_metrics['recon_loss']:.4f} "
                f"kl={last_metrics['kl_loss']:.4f}",
                flush=True,
            )

    rollouts = collect_rollouts(
        rssm,
        env,
        n_episodes=args.eval_episodes,
        horizon=args.horizon,
    )
    analysis = analyze_drift(
        rollouts["true_deter"],
        rollouts["imagined_deter"],
        trim=args.trim,
    )
    frequency = analysis["frequency"]
    dc_trend_pct = 100.0 * frequency["band_share"]["dc_trend"]
    print(
        f"  [{spec.name} seed={seed}] dc_trend={dc_trend_pct:.1f}% "
        f"post/drift r10={analysis['overlaps']['10']['posterior_drift']:.3f}",
        flush=True,
    )
    print("", flush=True)

    return {
        "complexity": spec.name,
        "seed": int(seed),
        "obs_dim": int(env.obs_dim),
        "n_osc": int(env.n_osc),
        "frequencies": list(spec.frequencies),
        "amplitudes": list(spec.amplitudes),
        "coupling_strength": float(spec.coupling_strength),
        "sampled_coupling": env.coupling.tolist(),
        "final_train_metrics": last_metrics,
        "rollout_shapes": {
            "true_deter": list(rollouts["true_deter"].shape),
            "imagined_deter": list(rollouts["imagined_deter"].shape),
        },
        "trim_window_inclusive": analysis["trim_window_inclusive"],
        "ranks": analysis["ranks"],
        "overlaps": analysis["overlaps"],
        "dc_trend_pct": float(dc_trend_pct),
        "frequency": {
            "band_mse": frequency["band_mse"],
            "band_share": frequency["band_share"],
            "J_slow": frequency["J_slow"],
            "J_total": frequency["J_total"],
            "pca_components": frequency["pca_components"],
            "pca_var_explained": frequency["pca_var_explained"],
        },
        "drift_pca": analysis["drift_pca"],
        "posterior": analysis["posterior"],
        "sfa": analysis["sfa"],
    }


def summarize_by_complexity(
    per_seed_results: list[dict[str, Any]],
    specs: list[Any],
    seeds: list[int],
) -> list[dict[str, Any]]:
    summaries = []
    for spec in specs:
        rows = [row for row in per_seed_results if row["complexity"] == spec.name]
        if not rows:
            continue
        overlap_summary = {
            pair: {
                str(rank): summarize_values(
                    [row["overlaps"][str(rank)][pair] for row in rows]
                )
                for rank in RANKS
            }
            for pair in OVERLAP_PAIRS
        }
        metric_summary = {
            "dc_trend_pct": summarize_values([row["dc_trend_pct"] for row in rows]),
            "J_slow": summarize_values([row["frequency"]["J_slow"] for row in rows]),
            "J_total": summarize_values([row["frequency"]["J_total"] for row in rows]),
            "recon_loss": summarize_values(
                [row["final_train_metrics"]["recon_loss"] for row in rows]
            ),
        }
        first = rows[0]
        summaries.append(
            {
                "complexity": spec.name,
                "obs_dim": int(first["obs_dim"]),
                "n_osc": int(first["n_osc"]),
                "seeds": [int(seed) for seed in seeds],
                "frequencies": list(spec.frequencies),
                "amplitudes": list(spec.amplitudes),
                "overlaps": overlap_summary,
                "metrics": metric_summary,
                "month5_reference_dc_trend_pct": MONTH5_REFERENCE_DC_TREND_PCT.get(
                    spec.name
                ),
            }
        )
    return summaries


def summarize_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)) if array.size else float("nan"),
        "std": std,
        "min": float(np.min(array)) if array.size else float("nan"),
        "max": float(np.max(array)) if array.size else float("nan"),
        "values": [float(value) for value in array.tolist()],
    }


def hypothesis_test(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    obs_dim = np.asarray([row["obs_dim"] for row in summaries], dtype=np.float64)
    complexity_names = [row["complexity"] for row in summaries]
    by_rank = {}
    monotonic_ranks = []
    medium_lowest_ranks = []

    for rank in RANKS:
        values = np.asarray(
            [
                row["overlaps"]["posterior_drift"][str(rank)]["mean"]
                for row in summaries
            ],
            dtype=np.float64,
        )
        pearson = safe_corr(obs_dim, values)
        monotonic_decrease = bool(np.all(np.diff(values) < 0))
        medium_lowest = bool(
            len(values) == 3 and values[1] < values[0] and values[1] < values[2]
        )
        if monotonic_decrease:
            monotonic_ranks.append(rank)
        if medium_lowest:
            medium_lowest_ranks.append(rank)
        by_rank[str(rank)] = {
            "complexities": complexity_names,
            "obs_dim": obs_dim.tolist(),
            "posterior_drift_mean": values.tolist(),
            "pearson_obs_dim_vs_overlap": pearson,
            "negative_correlation": bool(pearson < 0.0),
            "monotonic_decrease": monotonic_decrease,
            "medium_lowest_overlap": medium_lowest,
        }

    if len(monotonic_ranks) == len(RANKS):
        verdict = "negative_correlation_complexity_overlap"
        interpretation = (
            "Posterior/drift overlap decreases monotonically with complexity "
            "at all reported ranks."
        )
    elif len(medium_lowest_ranks) == len(RANKS):
        verdict = "inverted_u_mismatch_medium_lowest_overlap"
        interpretation = (
            "The medium-complexity task has the lowest posterior/drift overlap "
            "at all reported ranks, matching an inverted-U mismatch pattern."
        )
    elif monotonic_ranks or medium_lowest_ranks:
        verdict = "mixed_partial_support"
        interpretation = (
            "Some ranks support a complexity-linked mismatch pattern, but the "
            "rank-wise evidence is not uniform."
        )
    else:
        verdict = "ambiguous_or_no_support"
        interpretation = (
            "The three-level toy sweep does not show a clear negative "
            "correlation or medium-lowest overlap pattern."
        )

    return {
        "rank_results": by_rank,
        "monotonic_decrease_ranks": monotonic_ranks,
        "medium_lowest_overlap_ranks": medium_lowest_ranks,
        "verdict": verdict,
        "interpretation": interpretation,
        "descriptive_only": True,
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_plot_points(per_seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in per_seed_results:
        for rank in RANKS:
            rank_key = str(rank)
            overlaps = row["overlaps"][rank_key]
            points.append(
                {
                    "complexity": row["complexity"],
                    "obs_dim": row["obs_dim"],
                    "n_osc": row["n_osc"],
                    "seed": row["seed"],
                    "rank": rank,
                    "posterior_drift": overlaps["posterior_drift"],
                    "sfa_posterior": overlaps["sfa_posterior"],
                    "sfa_drift": overlaps["sfa_drift"],
                    "dc_trend_pct": row["dc_trend_pct"],
                }
            )
    return points


def load_real_model_references() -> dict[str, Any]:
    references = {}
    for label, path in REAL_MODEL_REFERENCE_PATHS.items():
        if not path.exists():
            references[label] = {
                "available": False,
                "path": relative_path(path),
                "reason": "file not found",
            }
            continue
        try:
            data = json.loads(path.read_text())
            references[label] = {
                "available": True,
                "path": relative_path(path),
                "posterior_drift": extract_posterior_drift_reference(data),
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            references[label] = {
                "available": False,
                "path": relative_path(path),
                "reason": str(exc),
            }
    return references


def extract_posterior_drift_reference(data: dict[str, Any]) -> dict[str, float]:
    if "overlaps" in data:
        overlaps = data["overlaps"]
        return {
            str(rank): float(overlaps[str(rank)]["posterior_vs_drift"])
            for rank in RANKS
            if str(rank) in overlaps
        }
    if "rank_results" in data:
        rank_results = data["rank_results"]
        return {
            str(rank): float(rank_results[str(rank)]["overlap"])
            for rank in RANKS
            if str(rank) in rank_results
        }
    raise KeyError("Could not find overlaps or rank_results.")


def complexity_metadata(spec: Any) -> dict[str, Any]:
    return {
        "complexity": spec.name,
        "n_osc": int(len(spec.frequencies)),
        "obs_dim": int(2 * len(spec.frequencies)),
        "frequencies": list(spec.frequencies),
        "amplitudes": list(spec.amplitudes),
        "coupling_strength": float(spec.coupling_strength),
    }


def print_master_table(summaries: list[dict[str, Any]]) -> None:
    print("Toy mismatch complexity master table:", flush=True)
    print(
        "| Complexity | obs_dim | r=3 post/drift | r=5 post/drift | "
        "r=10 post/drift | dc_trend% |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|", flush=True)
    for row in summaries:
        overlaps = row["overlaps"]["posterior_drift"]
        dc_trend = row["metrics"]["dc_trend_pct"]
        print(
            f"| {row['complexity']} | {row['obs_dim']} | "
            f"{format_mean_std(overlaps['3'])} | "
            f"{format_mean_std(overlaps['5'])} | "
            f"{format_mean_std(overlaps['10'])} | "
            f"{format_mean_std(dc_trend, precision=1, suffix='%')} |",
            flush=True,
        )


def print_hypothesis_summary(hypothesis: dict[str, Any]) -> None:
    print("\nHypothesis test (descriptive, n=3 complexity levels):", flush=True)
    for rank in RANKS:
        result = hypothesis["rank_results"][str(rank)]
        print(
            f"  r={rank}: pearson(obs_dim, post/drift)="
            f"{result['pearson_obs_dim_vs_overlap']:.3f}; "
            f"monotonic_decrease={result['monotonic_decrease']}; "
            f"medium_lowest={result['medium_lowest_overlap']}",
            flush=True,
        )
    print(f"Verdict: {hypothesis['verdict']}", flush=True)
    print(hypothesis["interpretation"], flush=True)


def format_mean_std(
    summary: dict[str, Any],
    *,
    precision: int = 3,
    suffix: str = "",
) -> str:
    mean = summary["mean"]
    std = summary["std"]
    return f"{mean:.{precision}f}{suffix} +/- {std:.{precision}f}{suffix}"


if __name__ == "__main__":
    main()
