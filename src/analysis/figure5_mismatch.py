"""Generate Figure 5: posterior-drift mismatch master figure."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "driftdetect-matplotlib-cache"),
)

import matplotlib.pyplot as plt  # noqa: E402


sys.path.insert(0, str(REPO_ROOT))

TABLE_DIR = REPO_ROOT / "results" / "tables"
FIGURE_DIR = REPO_ROOT / "results" / "figures"
OUTPUT_FIGURE = FIGURE_DIR / "figure5_mismatch_master_v1.pdf"
OUTPUT_TABLE = TABLE_DIR / "figure5_mismatch_summary.json"

RANKS = (3, 5, 10)
MISMATCH_THRESHOLD = 0.4
PANEL_A_ORDER = (
    "V3 Cartpole",
    "V3 Cheetah",
    "Toy hard",
    "V4 Cheetah",
    "Toy medium",
    "Toy simple",
)

GROUP_COLORS = {
    "V3": "#2563eb",
    "V4": "#f97316",
    "Toy": "#16a34a",
}
SETTING_COLORS = {
    "V3 Cartpole": "#60a5fa",
    "V3 Cheetah": "#2563eb",
    "V4 Cheetah": "#f97316",
    "Toy hard": "#166534",
    "Toy medium": "#22c55e",
    "Toy simple": "#86efac",
}
MARKERS = {
    "V3 Cartpole": "o",
    "V3 Cheetah": "s",
    "V4 Cheetah": "D",
    "Toy hard": "^",
    "Toy medium": "v",
    "Toy simple": "P",
}


@dataclass(frozen=True)
class SettingSpec:
    label: str
    group: str
    source_path: Path
    fallback_overlaps: dict[int, float]
    fallback_std: dict[int, float | None]
    fallback_dc_trend_pct: float | None


SETTING_SPECS = {
    "V3 Cheetah": SettingSpec(
        label="V3 Cheetah",
        group="V3",
        source_path=TABLE_DIR / "smad_alignment_check.json",
        fallback_overlaps={3: 0.13, 5: 0.17, 10: 0.28},
        fallback_std={3: None, 5: None, 10: None},
        fallback_dc_trend_pct=46.7,
    ),
    "V3 Cartpole": SettingSpec(
        label="V3 Cartpole",
        group="V3",
        source_path=TABLE_DIR / "cartpole_mismatch_check.json",
        fallback_overlaps={3: 0.06, 5: 0.08, 10: 0.12},
        fallback_std={3: None, 5: None, 10: None},
        fallback_dc_trend_pct=97.0,
    ),
    "V4 Cheetah": SettingSpec(
        label="V4 Cheetah",
        group="V4",
        source_path=TABLE_DIR / "v4_mismatch_check.json",
        fallback_overlaps={3: 0.64, 5: 0.41, 10: 0.43},
        fallback_std={3: None, 5: None, 10: None},
        fallback_dc_trend_pct=63.9,
    ),
    "Toy simple": SettingSpec(
        label="Toy simple",
        group="Toy",
        source_path=TABLE_DIR / "toy_mismatch_complexity.json",
        fallback_overlaps={3: 0.613, 5: 0.697, 10: 0.797},
        fallback_std={3: 0.160, 5: 0.113, 10: 0.012},
        fallback_dc_trend_pct=19.1,
    ),
    "Toy medium": SettingSpec(
        label="Toy medium",
        group="Toy",
        source_path=TABLE_DIR / "toy_mismatch_complexity.json",
        fallback_overlaps={3: 0.412, 5: 0.672, 10: 0.846},
        fallback_std={3: 0.143, 5: 0.138, 10: 0.045},
        fallback_dc_trend_pct=43.1,
    ),
    "Toy hard": SettingSpec(
        label="Toy hard",
        group="Toy",
        source_path=TABLE_DIR / "toy_mismatch_complexity.json",
        fallback_overlaps={3: 0.378, 5: 0.496, 10: 0.743},
        fallback_std={3: 0.042, 5: 0.103, 10: 0.183},
        fallback_dc_trend_pct=38.3,
    ),
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    settings = load_all_settings()
    plot_figure(settings)
    save_summary(settings)
    print_summary(settings)


def load_all_settings() -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    cached_json: dict[Path, tuple[dict[str, Any] | None, str | None]] = {}

    for label, spec in SETTING_SPECS.items():
        if spec.source_path not in cached_json:
            cached_json[spec.source_path] = load_json(spec.source_path)
        data, load_error = cached_json[spec.source_path]

        overlaps, overlap_std, source_detail = load_setting_values(spec, data, load_error)
        dc_trend_pct, dc_source = load_dc_trend_pct(label, spec, data, load_error)
        settings[label] = {
            "setting": label,
            "group": spec.group,
            "color": SETTING_COLORS[label],
            "marker": MARKERS[label],
            "line_style": "--" if spec.group == "Toy" else "-",
            "source_path": display_path(spec.source_path),
            "source_detail": source_detail,
            "overlap": {str(rank): float(overlaps[rank]) for rank in RANKS},
            "overlap_std": {
                str(rank): none_or_float(overlap_std.get(rank)) for rank in RANKS
            },
            "r10_overlap": float(overlaps[10]),
            "r10_overlap_std": none_or_float(overlap_std.get(10)),
            "dc_trend_pct": none_or_float(dc_trend_pct),
            "dc_trend_source": dc_source,
            "uses_fallback": source_detail["used_fallback"],
        }

    return settings


def load_setting_values(
    spec: SettingSpec,
    data: dict[str, Any] | None,
    load_error: str | None,
) -> tuple[dict[int, float], dict[int, float | None], dict[str, Any]]:
    if data is None:
        return (
            dict(spec.fallback_overlaps),
            dict(spec.fallback_std),
            {
                "used_fallback": True,
                "reason": load_error or "source JSON unavailable",
            },
        )

    try:
        if spec.label == "V3 Cheetah":
            overlaps = {
                rank: float(data["rank_results"][str(rank)]["overlap"])
                for rank in RANKS
            }
            return overlaps, dict(spec.fallback_std), {
                "used_fallback": False,
                "json_key": "rank_results[*].overlap",
            }

        if spec.label in {"V3 Cartpole", "V4 Cheetah"}:
            overlaps = {
                rank: float(data["overlaps"][str(rank)]["posterior_vs_drift"])
                for rank in RANKS
            }
            return overlaps, dict(spec.fallback_std), {
                "used_fallback": False,
                "json_key": "overlaps[*].posterior_vs_drift",
            }

        toy_level = spec.label.replace("Toy ", "")
        row = find_toy_summary(data, toy_level)
        overlaps = {
            rank: float(row["overlaps"]["posterior_drift"][str(rank)]["mean"])
            for rank in RANKS
        }
        overlap_std = {
            rank: float(row["overlaps"]["posterior_drift"][str(rank)]["std"])
            for rank in RANKS
        }
        return overlaps, overlap_std, {
            "used_fallback": False,
            "json_key": f"summary_by_complexity[{toy_level}].overlaps.posterior_drift",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return (
            dict(spec.fallback_overlaps),
            dict(spec.fallback_std),
            {
                "used_fallback": True,
                "reason": f"could not parse source JSON: {exc}",
            },
        )


def load_dc_trend_pct(
    label: str,
    spec: SettingSpec,
    data: dict[str, Any] | None,
    load_error: str | None,
) -> tuple[float | None, dict[str, Any]]:
    if label.startswith("Toy ") and data is not None:
        try:
            toy_level = label.replace("Toy ", "")
            row = find_toy_summary(data, toy_level)
            return float(row["metrics"]["dc_trend_pct"]["mean"]), {
                "used_fallback": False,
                "path": display_path(spec.source_path),
                "json_key": f"summary_by_complexity[{toy_level}].metrics.dc_trend_pct.mean",
            }
        except (KeyError, TypeError, ValueError) as exc:
            return spec.fallback_dc_trend_pct, {
                "used_fallback": True,
                "reason": f"could not parse toy dc_trend_pct: {exc}",
            }

    if label == "V4 Cheetah":
        v4_path = TABLE_DIR / "v4_band_errors.json"
        v4_data, v4_error = load_json(v4_path)
        if v4_data is not None:
            try:
                share = float(v4_data["latent-space"]["bands"]["dc_trend"]["share"])
                return 100.0 * share, {
                    "used_fallback": False,
                    "path": display_path(v4_path),
                    "json_key": "latent-space.bands.dc_trend.share",
                }
            except (KeyError, TypeError, ValueError) as exc:
                return spec.fallback_dc_trend_pct, {
                    "used_fallback": True,
                    "reason": f"could not parse V4 dc_trend share: {exc}",
                }
        return spec.fallback_dc_trend_pct, {
            "used_fallback": True,
            "reason": v4_error or "v4_band_errors.json unavailable",
        }

    reason = load_error or "dc_trend share is documented but not stored in overlap JSON"
    return spec.fallback_dc_trend_pct, {"used_fallback": True, "reason": reason}


def find_toy_summary(data: dict[str, Any], complexity: str) -> dict[str, Any]:
    for row in data["summary_by_complexity"]:
        if row["complexity"] == complexity:
            return row
    raise KeyError(f"missing toy complexity {complexity}")


def plot_figure(settings: dict[str, dict[str, Any]]) -> None:
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    plot_panel_a(ax_bar, settings)
    plot_panel_b(ax_line, settings)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)


def plot_panel_a(ax: plt.Axes, settings: dict[str, dict[str, Any]]) -> None:
    labels = list(PANEL_A_ORDER)
    x = np.arange(len(labels))
    values = np.asarray([settings[label]["r10_overlap"] for label in labels], dtype=np.float64)
    errors = [
        0.0 if settings[label]["r10_overlap_std"] is None else settings[label]["r10_overlap_std"]
        for label in labels
    ]
    colors = [settings[label]["color"] for label in labels]

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        color=colors,
        edgecolor="#111827",
        linewidth=0.7,
        capsize=3,
        width=0.72,
    )
    ax.axhline(
        MISMATCH_THRESHOLD,
        color="#6b7280",
        linestyle="--",
        linewidth=1.1,
    )
    ax.text(
        len(labels) - 0.1,
        MISMATCH_THRESHOLD + 0.015,
        "mismatch threshold",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )

    for bar, label, error in zip(bars, labels, errors, strict=True):
        dc = settings[label]["dc_trend_pct"]
        y = bar.get_height() + float(error) + 0.035
        annotation = "dc n/a" if dc is None else f"dc {dc:.0f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#111827",
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Subspace overlap")
    ax.set_title("A. Posterior-Drift Overlap at r=10", fontsize=12)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_b(ax: plt.Axes, settings: dict[str, dict[str, Any]]) -> None:
    for label in PANEL_A_ORDER:
        row = settings[label]
        y = [row["overlap"][str(rank)] for rank in RANKS]
        ax.plot(
            RANKS,
            y,
            color=row["color"],
            marker=row["marker"],
            linestyle=row["line_style"],
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )

    ax.axhline(
        MISMATCH_THRESHOLD,
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.text(
        10.0,
        MISMATCH_THRESHOLD + 0.02,
        "mismatch threshold",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#4b5563",
    )
    ax.set_xlim(2.6, 10.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(RANKS)
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Subspace overlap")
    ax.set_title("B. Overlap vs Rank", fontsize=12)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=8.5,
    )


def save_summary(settings: dict[str, dict[str, Any]]) -> None:
    output = {
        "analysis": "figure5_mismatch_master",
        "figure_path": display_path(OUTPUT_FIGURE),
        "panel_a_order": list(PANEL_A_ORDER),
        "ranks": list(RANKS),
        "mismatch_threshold": MISMATCH_THRESHOLD,
        "color_groups": GROUP_COLORS,
        "settings": [settings[label] for label in PANEL_A_ORDER],
    }
    OUTPUT_TABLE.write_text(json.dumps(output, indent=2) + "\n")


def print_summary(settings: dict[str, dict[str, Any]]) -> None:
    print("Figure 5 mismatch master summary")
    print("| Setting | r=3 | r=5 | r=10 | dc_trend% |")
    print("|---|---:|---:|---:|---:|")
    for label in PANEL_A_ORDER:
        row = settings[label]
        dc = row["dc_trend_pct"]
        dc_text = "n/a" if dc is None else f"{dc:.1f}"
        print(
            f"| {label} | "
            f"{row['overlap']['3']:.3f} | "
            f"{row['overlap']['5']:.3f} | "
            f"{row['overlap']['10']:.3f} | "
            f"{dc_text} |"
        )
    print(f"\nSaved figure: {display_path(OUTPUT_FIGURE)}")
    print(f"Saved table:  {display_path(OUTPUT_TABLE)}")


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, f"missing {display_path(path)}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {display_path(path)}: {exc}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def none_or_float(value: float | None) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
