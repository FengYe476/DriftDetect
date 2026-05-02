#!/usr/bin/env python3
"""Toy SMAD frequency-redistribution sweep.

Train one fixed-capacity toy RSSM per (complexity, seed), then evaluate a grid
of SMAD ranks and damping strengths on the shared baseline rollout set.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
from src.toy.smad_intervention import (  # noqa: E402
    analyze_drift,
    collect_rollouts,
    frequency_metrics,
    patch_toy_img_step,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "toy_redistribution_sweep.parquet"
BANDS = ("dc_trend", "very_low", "low", "mid", "high")
DEFAULT_COMPLEXITIES = ("simple", "medium", "hard")
DEFAULT_RANKS = (3, 5, 10)
DEFAULT_ETAS = (0.1, 0.2, 0.3, 0.5)
CONSERVATION_TOLERANCE_PCT = 10.0


def main() -> None:
    args = parse_args()
    specs = select_specs(args.complexities)
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    tasks = [make_task(spec, seed, args) for spec in specs for seed in seeds]

    print_header(args, specs, seeds, tasks)
    if args.parallel:
        rows = run_parallel(tasks, args.workers)
    else:
        rows = []
        for task in tasks:
            rows.extend(run_model_task(task))

    rows = sorted(
        rows,
        key=lambda row: (
            row["complexity_order"],
            row["seed"],
            row["rank"],
            row["eta"],
        ),
    )
    summary = compute_summary(rows, args.conservation_tolerance_pct)
    output_path = write_results(rows, summary, args)
    print_summaries(summary)
    print(f"\nSaved: {relative_path(output_path)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run toy SMAD frequency-redistribution sweep."
    )
    parser.add_argument("--complexities", nargs="+", default=list(DEFAULT_COMPLEXITIES))
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--etas", type=float, nargs="+", default=list(DEFAULT_ETAS))
    parser.add_argument("--n_seeds", "--n-seeds", dest="n_seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
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
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--conservation_tolerance_pct",
        type=float,
        default=CONSERVATION_TOLERANCE_PCT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.quick:
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_OUTPUT.with_name("toy_redistribution_sweep_quick.json")
        args.ranks = [3]
        args.etas = [0.1]
        args.n_seeds = 1
        args.seeds = [0]
        args.train_steps = 20
        args.eval_episodes = 2
        args.log_interval = 10

    args.output = resolve_path(args.output)
    args.trim = (int(args.trim[0]), int(args.trim[1]))
    args.ranks = sorted(set(int(rank) for rank in args.ranks))
    args.etas = sorted(set(float(eta) for eta in args.etas))
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
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
    if not args.ranks:
        raise ValueError("--ranks must contain at least one rank.")
    if not args.etas:
        raise ValueError("--etas must contain at least one eta.")
    if max(args.ranks) > args.deter_dim:
        raise ValueError("--deter_dim must be at least max(--ranks).")
    if args.stoch_dim <= 0:
        raise ValueError("--stoch_dim must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive.")
    if any(rank <= 0 for rank in args.ranks):
        raise ValueError("--ranks must be positive.")
    if any(eta < 0 for eta in args.etas):
        raise ValueError("--etas must be non-negative.")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.conservation_tolerance_pct < 0:
        raise ValueError("--conservation_tolerance_pct must be non-negative.")


def select_specs(names: list[str]) -> list[Any]:
    available = {spec.name: spec for spec in complexity_specs()}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            "Unknown complexities "
            f"{unknown}; expected one or more of {sorted(available)}."
        )
    return [available[name] for name in names]


def make_task(spec: Any, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "complexity": spec.name,
        "complexity_order": list(DEFAULT_COMPLEXITIES).index(spec.name),
        "frequencies": list(spec.frequencies),
        "amplitudes": list(spec.amplitudes),
        "coupling_strength": float(spec.coupling_strength),
        "seed": int(seed),
        "ranks": list(args.ranks),
        "etas": list(args.etas),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "horizon": int(args.horizon),
        "eval_episodes": int(args.eval_episodes),
        "trim": [int(args.trim[0]), int(args.trim[1])],
        "lr": float(args.lr),
        "free_bits": float(args.free_bits),
        "kl_scale": float(args.kl_scale),
        "deter_dim": int(args.deter_dim),
        "stoch_dim": int(args.stoch_dim),
        "hidden_dim": int(args.hidden_dim),
        "device": str(args.device),
        "log_interval": int(args.log_interval),
        "parallel_worker": bool(args.parallel),
    }


def run_parallel(tasks: list[dict[str, Any]], workers: int | None) -> list[dict[str, Any]]:
    if workers is None:
        workers = max(1, min(len(tasks), max(1, (os.cpu_count() or 2) // 2)))
    print(f"Running {len(tasks)} model tasks with {workers} workers", flush=True)
    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        future_to_task = {executor.submit(run_model_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            model_id = model_id_for(task["complexity"], task["seed"])
            try:
                model_rows = future.result()
            except Exception as exc:
                raise RuntimeError(f"Model task failed for {model_id}: {exc}") from exc
            rows.extend(model_rows)
            print(
                f"[{model_id}] completed {len(model_rows)} redistribution rows",
                flush=True,
            )
    return rows


def run_model_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    if task["parallel_worker"]:
        torch.set_num_threads(1)

    complexity = task["complexity"]
    seed = int(task["seed"])
    model_id = model_id_for(complexity, seed)
    trim = (int(task["trim"][0]), int(task["trim"][1]))
    set_seed(seed)
    env = MultiOscillatorEnv(
        frequencies=task["frequencies"],
        amplitudes=task["amplitudes"],
        episode_length=task["horizon"],
        coupling_strength=task["coupling_strength"],
        seed=seed,
    )
    rssm = MinimalRSSM(
        obs_dim=env.obs_dim,
        action_dim=1,
        deter_dim=task["deter_dim"],
        stoch_dim=task["stoch_dim"],
        hidden_dim=task["hidden_dim"],
        lr=task["lr"],
        device=task["device"],
    )

    print(
        f"[{model_id}] train obs_dim={env.obs_dim} n_osc={env.n_osc}",
        flush=True,
    )
    last_metrics: dict[str, float] = {}
    for step in range(1, task["train_steps"] + 1):
        obs, actions, rewards = sample_batch(env, task["batch_size"], task["horizon"])
        last_metrics = rssm.train_step(
            obs,
            actions,
            reward_seq=rewards,
            free_bits=task["free_bits"],
            kl_scale=task["kl_scale"],
            sample=True,
        )
        if step == 1 or step % task["log_interval"] == 0 or step == task["train_steps"]:
            print(
                f"  [{model_id}] step {step}/{task['train_steps']} "
                f"loss={last_metrics['loss']:.4f} "
                f"recon={last_metrics['recon_loss']:.4f} "
                f"kl={last_metrics['kl_loss']:.4f}",
                flush=True,
            )

    baseline_rollouts = collect_rollouts(
        rssm,
        env,
        n_episodes=task["eval_episodes"],
        horizon=task["horizon"],
    )
    analysis = analyze_drift(
        baseline_rollouts["true_deter"],
        baseline_rollouts["imagined_deter"],
        trim=trim,
    )
    baseline_frequency = analysis["frequency"]
    max_rank = max(task["ranks"])
    U_drift = analysis["bases"]["U_drift"][:, :max_rank]

    rows: list[dict[str, Any]] = []
    for rank in task["ranks"]:
        U_rank = U_drift[:, :rank]
        for eta in task["etas"]:
            damped_imagined = collect_damped_imagined_on_baseline(
                rssm,
                baseline_rollouts,
                U_rank,
                eta,
            )
            damped_frequency = frequency_metrics(
                baseline_rollouts["true_deter"],
                damped_imagined,
                trim=trim,
            )
            row = build_result_row(
                task=task,
                env=env,
                model_id=model_id,
                rank=rank,
                eta=eta,
                train_metrics=last_metrics,
                baseline_frequency=baseline_frequency,
                damped_frequency=damped_frequency,
                analysis=analysis,
            )
            rows.append(row)
            print(
                f"  [{model_id}] r={rank} eta={eta:.2f} "
                f"low={row['low_change_pct']:+.1f}% "
                f"mid={row['mid_change_pct']:+.1f}% "
                f"high={row['high_change_pct']:+.1f}% "
                f"total={row['total_mse_change_pct']:+.1f}%",
                flush=True,
            )
    return rows


def collect_damped_imagined_on_baseline(
    rssm: MinimalRSSM,
    baseline_rollouts: dict[str, np.ndarray],
    U_drift: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Run damped imagination on the exact baseline observation/action episodes."""

    device = next(rssm.parameters()).device
    rssm.eval()
    with torch.no_grad():
        obs_tensor = torch.as_tensor(
            baseline_rollouts["true_obs"],
            device=device,
            dtype=torch.float32,
        )
        action_tensor = torch.as_tensor(
            baseline_rollouts["actions"],
            device=device,
            dtype=torch.float32,
        )
        post, _ = rssm.observe(obs_tensor, action_tensor, sample=False)
        start = {key: value[:, 0] for key, value in post.items()}

        restore = patch_toy_img_step(rssm, U_drift, eta)
        try:
            imag = rssm.imagine(start, action_tensor, sample=False)
        finally:
            restore()
    return imag["deter"].detach().cpu().numpy().astype(np.float32)


def build_result_row(
    *,
    task: dict[str, Any],
    env: MultiOscillatorEnv,
    model_id: str,
    rank: int,
    eta: float,
    train_metrics: dict[str, float],
    baseline_frequency: dict[str, Any],
    damped_frequency: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": model_id,
        "complexity": task["complexity"],
        "complexity_order": int(task["complexity_order"]),
        "obs_dim": int(env.obs_dim),
        "n_osc": int(env.n_osc),
        "seed": int(task["seed"]),
        "rank": int(rank),
        "eta": float(eta),
        "train_loss": float(train_metrics["loss"]),
        "train_recon_loss": float(train_metrics["recon_loss"]),
        "train_kl_loss": float(train_metrics["kl_loss"]),
        "train_reward_loss": float(train_metrics["reward_loss"]),
        "train_grad_norm": float(train_metrics["grad_norm"]),
        "baseline_total_mse": float(baseline_frequency["J_total"]),
        "damped_total_mse": float(damped_frequency["J_total"]),
        "total_mse_change_pct": percent_change(
            baseline_frequency["J_total"],
            damped_frequency["J_total"],
        ),
        "baseline_pca_components": int(baseline_frequency["pca_components"]),
        "damped_pca_components": int(damped_frequency["pca_components"]),
        "trim_start": int(task["trim"][0]),
        "trim_end": int(task["trim"][1]),
        "eval_episodes": int(task["eval_episodes"]),
        "horizon": int(task["horizon"]),
    }

    band_changes = {}
    for band in BANDS:
        baseline_mse = float(baseline_frequency["band_mse"][band])
        damped_mse = float(damped_frequency["band_mse"][band])
        change = percent_change(baseline_mse, damped_mse)
        band_changes[band] = change
        row[f"baseline_{band}_mse"] = baseline_mse
        row[f"damped_{band}_mse"] = damped_mse
        row[f"{band}_change_pct"] = change
        row[f"baseline_{band}_share"] = float(baseline_frequency["band_share"][band])
        row[f"damped_{band}_share"] = float(damped_frequency["band_share"][band])

    row["low_band_change_pct"] = row["low_change_pct"]
    row["mid_band_change_pct"] = row["mid_change_pct"]
    row["high_band_change_pct"] = row["high_change_pct"]
    row["sum_band_change_pct"] = float(sum(band_changes.values()))
    row["redistribution_strength"] = float(
        np.mean([band_changes["mid"], band_changes["high"]])
        - np.mean([band_changes["very_low"], band_changes["low"]])
    )
    row["very_low_down"] = bool(band_changes["very_low"] < 0.0)
    row["low_down"] = bool(band_changes["low"] < 0.0)
    row["mid_up"] = bool(band_changes["mid"] > 0.0)
    row["high_up"] = bool(band_changes["high"] > 0.0)
    row["dc_trend_down"] = bool(band_changes["dc_trend"] < 0.0)
    row["phase2_pattern"] = bool(
        row["very_low_down"]
        and row["low_down"]
        and row["mid_up"]
        and row["high_up"]
    )
    row["drift_pca_cumulative_top10"] = float(analysis["drift_pca"]["cumulative_top10"])
    for index, value in enumerate(
        analysis["drift_pca"]["explained_variance_ratio_top10"],
        start=1,
    ):
        row[f"drift_pca_var_ratio_{index}"] = float(value)
    return row


def compute_summary(rows: list[dict[str, Any]], tolerance_pct: float) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_models": len({row["model_id"] for row in rows}),
        "n_rows_by_model": summarize_rows_by_model(rows),
        "by_complexity": summarize_groups(rows, ["complexity"]),
        "by_condition": summarize_groups(rows, ["complexity", "rank", "eta"]),
        "conservation": conservation_summary(rows, tolerance_pct),
        "consistency": consistency_summary(rows),
        "eta_scaling": eta_scaling_summary(rows),
        "complexity_dependence": complexity_dependence_summary(rows),
    }


def summarize_rows_by_model(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["model_id"]] += 1
    return dict(sorted(counts.items()))


def summarize_groups(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    summaries = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {key: value for key, value in zip(keys, group_key, strict=True)}
        summary["n"] = len(group_rows)
        for metric in band_change_columns() + [
            "total_mse_change_pct",
            "sum_band_change_pct",
            "redistribution_strength",
        ]:
            summary[metric] = summarize_values([row[metric] for row in group_rows])
        summary["phase2_pattern_rate"] = float(
            np.mean([row["phase2_pattern"] for row in group_rows])
        )
        summary["low_down_high_up_rate"] = float(
            np.mean([row["low_down"] and row["high_up"] for row in group_rows])
        )
        summaries.append(summary)
    return summaries


def conservation_summary(rows: list[dict[str, Any]], tolerance_pct: float) -> dict[str, Any]:
    conserved = [abs(row["total_mse_change_pct"]) <= tolerance_pct for row in rows]
    return {
        "tolerance_pct": float(tolerance_pct),
        "total_mse_change_pct": summarize_values(
            [row["total_mse_change_pct"] for row in rows]
        ),
        "sum_band_change_pct": summarize_values(
            [row["sum_band_change_pct"] for row in rows]
        ),
        "fraction_total_mse_conserved": float(np.mean(conserved)) if rows else 0.0,
        "n_total_mse_conserved": int(sum(conserved)),
        "n_rows": len(rows),
    }


def consistency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    flags = ("very_low_down", "low_down", "mid_up", "high_up", "dc_trend_down")
    return {
        **{f"{flag}_rate": float(np.mean([row[flag] for row in rows])) for flag in flags},
        "phase2_pattern_rate": float(np.mean([row["phase2_pattern"] for row in rows])),
        "low_down_high_up_rate": float(
            np.mean([row["low_down"] and row["high_up"] for row in rows])
        ),
    }


def eta_scaling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_eta = summarize_groups(rows, ["eta"])
    eta_values = np.asarray([row["eta"] for row in by_eta], dtype=np.float64)
    strength = np.asarray(
        [row["redistribution_strength"]["mean"] for row in by_eta],
        dtype=np.float64,
    )
    linear = fit_polynomial(eta_values, strength, degree=1)
    quadratic = fit_polynomial(eta_values, strength, degree=2)
    classification = classify_eta_scaling(linear, quadratic)
    return {
        "metric": (
            "mean(mid_change_pct, high_change_pct) - "
            "mean(very_low_change_pct, low_change_pct)"
        ),
        "by_eta": by_eta,
        "linear_fit": linear,
        "quadratic_fit": quadratic,
        "classification": classification,
    }


def complexity_dependence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = summarize_groups(rows, ["complexity"])
    return [
        {
            "complexity": row["complexity"],
            "n": row["n"],
            "redistribution_strength": row["redistribution_strength"],
            "phase2_pattern_rate": row["phase2_pattern_rate"],
            "low_down_high_up_rate": row["low_down_high_up_rate"],
            "total_mse_change_pct": row["total_mse_change_pct"],
        }
        for row in summaries
    ]


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, Any]:
    if x.size <= degree or y.size <= degree:
        return {"available": False, "reason": "not enough eta points"}
    coeff = np.polyfit(x, y, degree)
    pred = np.polyval(coeff, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "available": True,
        "degree": int(degree),
        "coefficients_high_to_low": [float(value) for value in coeff.tolist()],
        "r2": float(r2),
    }


def classify_eta_scaling(linear: dict[str, Any], quadratic: dict[str, Any]) -> str:
    if not linear.get("available") or not quadratic.get("available"):
        return "insufficient_eta_points"
    quad_coeff = quadratic["coefficients_high_to_low"][0]
    r2_gain = quadratic["r2"] - linear["r2"]
    if quad_coeff > 0.0 and r2_gain >= 0.05:
        return "superlinear_descriptive"
    if linear["r2"] >= 0.80:
        return "approximately_linear_descriptive"
    return "nonlinear_or_noisy_descriptive"


def write_results(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    metadata = {
        "analysis": "toy_redistribution_sweep",
        "signed_change_definition": "(damped_mse - baseline_mse) / baseline_mse * 100",
        "baseline_policy": (
            "One trained model and one shared baseline rollout set per "
            "(complexity, seed); ranks and etas are evaluation-only parameters."
        ),
        "config": {
            "complexities": list(args.complexities),
            "ranks": list(args.ranks),
            "etas": list(args.etas),
            "n_seeds": int(args.n_seeds),
            "seeds": args.seeds if args.seeds is not None else list(range(args.n_seeds)),
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "horizon": int(args.horizon),
            "eval_episodes": int(args.eval_episodes),
            "trim_window_inclusive": [int(args.trim[0]), int(args.trim[1])],
            "quick": bool(args.quick),
            "parallel": bool(args.parallel),
            "workers": args.workers,
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        write_json_output(output, metadata, rows, summary)
        return output

    try:
        import pandas as pd  # type: ignore

        frame = pd.DataFrame(rows)
        frame.to_parquet(output, index=False)
        summary_path = output.with_name(f"{output.stem}_summary.json")
        summary_payload = {**metadata, "summary": summary, "row_output": relative_path(output)}
        summary_path.write_text(json.dumps(to_jsonable(summary_payload), indent=2) + "\n")
        return output
    except Exception as exc:
        fallback = output.with_suffix(".json")
        metadata["parquet_fallback_reason"] = str(exc)
        print(
            "Parquet output unavailable; writing JSON fallback instead: "
            f"{relative_path(fallback)} ({exc})",
            flush=True,
        )
        write_json_output(fallback, metadata, rows, summary)
        return fallback


def write_json_output(
    output: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "rows": rows, "summary": summary}
    output.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def print_header(
    args: argparse.Namespace,
    specs: list[Any],
    seeds: list[int],
    tasks: list[dict[str, Any]],
) -> None:
    n_rows = len(tasks) * len(args.ranks) * len(args.etas)
    print("Toy SMAD frequency redistribution sweep", flush=True)
    print(f"Complexities: {[spec.name for spec in specs]}", flush=True)
    print(f"Seeds per complexity: {seeds}", flush=True)
    print(f"Ranks: {args.ranks}", flush=True)
    print(f"Etas: {args.etas}", flush=True)
    print(f"Train steps per model: {args.train_steps}", flush=True)
    print(f"Eval episodes per baseline/config: {args.eval_episodes}", flush=True)
    print(f"Model trainings: {len(tasks)}", flush=True)
    print(f"Expected result rows: {n_rows}", flush=True)
    print(f"Parallel: {args.parallel}", flush=True)
    print("", flush=True)


def print_summaries(summary: dict[str, Any]) -> None:
    print("\nSummary by complexity: signed band MSE change %", flush=True)
    print(
        "Complexity   n   dc_trend       very_low       low            "
        "mid            high           total",
        flush=True,
    )
    for row in summary["by_complexity"]:
        print(
            f"{row['complexity']:<12}"
            f"{row['n']:>3}  "
            f"{format_mean_std(row['dc_trend_change_pct']):>12}  "
            f"{format_mean_std(row['very_low_change_pct']):>12}  "
            f"{format_mean_std(row['low_change_pct']):>12}  "
            f"{format_mean_std(row['mid_change_pct']):>12}  "
            f"{format_mean_std(row['high_change_pct']):>12}  "
            f"{format_mean_std(row['total_mse_change_pct']):>12}",
            flush=True,
        )

    conservation = summary["conservation"]
    print("\nConservation test:", flush=True)
    print(
        f"  total_mse_change_pct mean="
        f"{format_mean_std(conservation['total_mse_change_pct'])}; "
        f"fraction within +/-{conservation['tolerance_pct']:.1f}% = "
        f"{conservation['fraction_total_mse_conserved']:.3f} "
        f"({conservation['n_total_mse_conserved']}/{conservation['n_rows']})",
        flush=True,
    )
    print(
        "  unweighted sum_band_change_pct mean="
        f"{format_mean_std(conservation['sum_band_change_pct'])}",
        flush=True,
    )

    consistency = summary["consistency"]
    print("\nConsistency test:", flush=True)
    print(
        f"  very_low_down={consistency['very_low_down_rate']:.3f} "
        f"low_down={consistency['low_down_rate']:.3f} "
        f"mid_up={consistency['mid_up_rate']:.3f} "
        f"high_up={consistency['high_up_rate']:.3f} "
        f"phase2_pattern={consistency['phase2_pattern_rate']:.3f}",
        flush=True,
    )

    eta = summary["eta_scaling"]
    print("\nEta scaling:", flush=True)
    for row in eta["by_eta"]:
        print(
            f"  eta={row['eta']:.2f} "
            f"redistribution_strength={format_mean_std(row['redistribution_strength'])}",
            flush=True,
        )
    print(
        f"  classification={eta['classification']} "
        f"linear_r2={eta['linear_fit'].get('r2', float('nan')):.3f} "
        f"quadratic_r2={eta['quadratic_fit'].get('r2', float('nan')):.3f}",
        flush=True,
    )


def band_change_columns() -> list[str]:
    return [f"{band}_change_pct" for band in BANDS]


def summarize_values(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def format_mean_std(summary: dict[str, Any], precision: int = 1) -> str:
    return f"{summary['mean']:+.{precision}f} +/- {summary['std']:.{precision}f}"


def percent_change(original: float, new: float) -> float:
    if original == 0.0:
        return 0.0 if new == 0.0 else float("inf")
    return float((new - original) / original * 100.0)


def model_id_for(complexity: str, seed: int) -> str:
    return f"{complexity}_seed{seed}"


if __name__ == "__main__":
    main()
