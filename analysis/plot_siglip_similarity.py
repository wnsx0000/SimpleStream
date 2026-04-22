from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import build_experiment_summary
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

EXCLUDED_PLOT_TASKS = frozenset({"HLD"})
RECENT_FRAME_LABELS = ("oldest", "older", "newer", "current")


def load_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "summary.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_excluded_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("task", "")) not in EXCLUDED_PLOT_TASKS]


def detect_similarity_backend(summary: dict[str, Any]) -> str:
    backends = summary.get("config", {}).get("similarity_backends", [])
    if backends:
        return backends[0] + "_similarity" if not backends[0].endswith("_similarity") else backends[0]
    metrics = summary.get("metrics", {})
    for key in metrics:
        if "similarity" in key:
            return key
    raise ValueError("Cannot detect similarity backend from summary.json")


def task_average_metric(task_summaries: dict[str, Any], backend: str) -> dict[str, Any]:
    metrics = [
        task_summary.get("metrics", {}).get(backend, {})
        for task, task_summary in task_summaries.items()
        if task not in EXCLUDED_PLOT_TASKS
    ]
    metrics = [metric for metric in metrics if metric]
    if not metrics:
        return {}
    return {
        "count": int(sum(metric.get("count", 0) for metric in metrics)),
        "recent4_mean_percentile_mean": float(
            np.mean([metric.get("recent4_mean_percentile_mean", 0.0) for metric in metrics])
        ),
        "recent4_mean_percentile_std": float(
            np.mean([metric.get("recent4_mean_percentile_std", 0.0) for metric in metrics])
        ),
    }


def filtered_summary_fallback(summary: dict[str, Any], backend: str) -> dict[str, Any]:
    filtered = dict(summary)
    filtered["tasks"] = {
        task: task_summary
        for task, task_summary in summary.get("tasks", {}).items()
        if task not in EXCLUDED_PLOT_TASKS
    }
    filtered["metrics"] = {}
    total_metric = task_average_metric(filtered["tasks"], backend)
    if total_metric:
        filtered["metrics"][backend] = total_metric

    splits: dict[str, Any] = {}
    for split_name, split_summary in summary.get("splits", {}).items():
        split_copy = dict(split_summary)
        split_copy["tasks"] = {
            task: task_summary
            for task, task_summary in split_summary.get("tasks", {}).items()
            if task not in EXCLUDED_PLOT_TASKS
        }
        split_copy["metrics"] = {}
        split_metric = task_average_metric(split_copy["tasks"], backend)
        if split_metric:
            split_copy["metrics"][backend] = split_metric
        splits[split_name] = split_copy
    filtered["splits"] = splits
    return filtered


def extract_bar_data(
    summary: dict[str, Any],
    backend: str,
) -> tuple[list[str], list[float], list[float], list[str]]:
    """Returns (labels, means, stds, group_types) in the order:
    realtime tasks -> Real-time avg -> backward tasks -> Backward avg -> Total avg
    group_types: "realtime", "realtime_avg", "backward", "backward_avg", "total_avg"
    """
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    group_types: list[str] = []

    task_data = summary.get("tasks", {})

    # Realtime tasks
    for task in REAL_TIME_TASKS:
        if task not in task_data:
            continue
        m = task_data[task].get("metrics", {}).get(backend, {})
        if not m:
            continue
        labels.append(task)
        means.append(float(m.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(m.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("realtime")

    # Real-time average
    rt_metrics = summary.get("splits", {}).get("realtime", {}).get("metrics", {}).get(backend, {})
    if rt_metrics:
        labels.append("Real-time\nAvg")
        means.append(float(rt_metrics.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(rt_metrics.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("realtime_avg")

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        if task not in task_data:
            continue
        m = task_data[task].get("metrics", {}).get(backend, {})
        if not m:
            continue
        labels.append(task)
        means.append(float(m.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(m.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("backward")

    # Backward average
    bw_metrics = summary.get("splits", {}).get("backward", {}).get("metrics", {}).get(backend, {})
    if bw_metrics:
        labels.append("Backward\nAvg")
        means.append(float(bw_metrics.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(bw_metrics.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("backward_avg")

    # Total average
    total_metrics = summary.get("metrics", {}).get(backend, {})
    if total_metrics:
        labels.append("Total\nAvg")
        means.append(float(total_metrics.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(total_metrics.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("total_avg")

    return labels, means, stds, group_types


COLOR_MAP = {
    "backward": "#6baed6",
    "backward_avg": "#2171b5",
    "realtime": "#fc9272",
    "realtime_avg": "#cb181d",
    "total_avg": "#525252",
}


def recent4_percentile_vector(
    metric: dict[str, Any],
    *,
    percentile_field: str,
    index_field: str,
) -> np.ndarray | None:
    if percentile_field not in metric or index_field not in metric:
        return None

    percentiles = np.asarray(metric.get(percentile_field), dtype=np.float64)
    if percentiles.ndim != 1 or percentiles.size < 1:
        return None

    indices = [int(index) for index in metric.get(index_field, [])]
    if len(indices) < 4:
        return None
    indices = indices[-4:]
    if any(index < 0 or index >= percentiles.size for index in indices):
        return None

    values = percentiles[indices].astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        return None
    return values


def record_recent4_percentile_vector(record: dict[str, Any], backend: str) -> np.ndarray | None:
    metric = record.get("metrics", {}).get(backend, {})
    if not isinstance(metric, dict):
        return None
    return recent4_percentile_vector(
        metric,
        percentile_field="frame_percentiles",
        index_field="recent_frame_indices_within_analysis",
    )


def collect_recent4_percentile_stats(
    records: list[dict[str, Any]],
    backend: str,
    *,
    task_name: str | None = None,
    split_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows: list[np.ndarray] = []
    for record in records:
        if record.get("error"):
            continue
        if task_name is not None and str(record.get("task", "")) != task_name:
            continue
        if split_name is not None and str(record.get("split", "")) != split_name:
            continue
        values = record_recent4_percentile_vector(record, backend)
        if values is not None:
            rows.append(values)

    if not rows:
        return None

    stacked = np.vstack(rows).astype(np.float64, copy=False)
    return stacked.mean(axis=0), stacked.std(axis=0)


def extract_recent4_per_frame_bar_data(
    records: list[dict[str, Any]],
    backend: str,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    labels: list[str] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    group_types: list[str] = []

    for task in REAL_TIME_TASKS:
        stats = collect_recent4_percentile_stats(records, backend, task_name=task)
        if stats is None:
            continue
        mean_row, std_row = stats
        labels.append(task)
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("realtime")

    rt_stats = collect_recent4_percentile_stats(records, backend, split_name="realtime")
    if rt_stats is not None:
        mean_row, std_row = rt_stats
        labels.append("Real-time\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("realtime_avg")

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        stats = collect_recent4_percentile_stats(records, backend, task_name=task)
        if stats is None:
            continue
        mean_row, std_row = stats
        labels.append(task)
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("backward")

    bw_stats = collect_recent4_percentile_stats(records, backend, split_name="backward")
    if bw_stats is not None:
        mean_row, std_row = bw_stats
        labels.append("Backward\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("backward_avg")

    total_stats = collect_recent4_percentile_stats(records, backend)
    if total_stats is not None:
        mean_row, std_row = total_stats
        labels.append("Total\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("total_avg")

    return labels, means, stds, group_types


def lighten_color(color: str, white_fraction: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    white = np.ones(3, dtype=np.float64)
    mixed = rgb * (1.0 - white_fraction) + white * white_fraction
    return mcolors.to_hex(mixed)


def recent_frame_colors(group_type: str) -> list[str]:
    base_color = COLOR_MAP[group_type]
    return [
        lighten_color(base_color, white_fraction)
        for white_fraction in (0.62, 0.42, 0.22, 0.0)
    ]


def add_group_separators(ax: Any, group_types: list[str]) -> None:
    separator_positions = []
    for i in range(len(group_types) - 1):
        if group_types[i] != group_types[i + 1]:
            separator_positions.append(i + 0.5)
    for pos in separator_positions:
        ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)


def plot_mean_percentile_bar(
    summary: dict[str, Any],
    backend: str,
    plots_dir: Path,
) -> None:
    labels, means, stds, group_types = extract_bar_data(summary, backend)
    if not labels:
        print("No data to plot.")
        return

    n = len(labels)
    x = np.arange(n)
    colors = [COLOR_MAP[g] for g in group_types]
    edge_colors = [COLOR_MAP[g] for g in group_types]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    ax.bar(
        x, means,
        color=colors, edgecolor=edge_colors,
        linewidth=1.2,
    )
    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        ax.text(i, mean_val + 0.01, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(i, mean_val + 0.01 + 0.04, f"std={std_val:.3f}", ha="center", va="bottom", fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Percentile", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Recent-4 Frame Similarity: Mean Percentile by Task", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    # Add vertical separators between groups
    add_group_separators(ax, group_types)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "similarity_mean_percentile.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_recent4_per_frame_mean_percentile_bar(
    records: list[dict[str, Any]],
    backend: str,
    plots_dir: Path,
) -> None:
    labels, means, stds, group_types = extract_recent4_per_frame_bar_data(records, backend)
    if not labels:
        print("No recent-4 per-frame data to plot.")
        return

    n = len(labels)
    x = np.arange(n)
    means_matrix = np.vstack(means)
    stds_matrix = np.vstack(stds)
    bar_width = 0.18
    offsets = (np.arange(4, dtype=np.float64) - 1.5) * bar_width

    fig, ax = plt.subplots(figsize=(max(12, n * 1.25), 6.4))
    for frame_idx, frame_label in enumerate(RECENT_FRAME_LABELS):
        positions = x + offsets[frame_idx]
        colors = [recent_frame_colors(group_type)[frame_idx] for group_type in group_types]
        edge_colors = [COLOR_MAP[group_type] for group_type in group_types]
        ax.bar(
            positions,
            means_matrix[:, frame_idx],
            width=bar_width,
            color=colors,
            edgecolor=edge_colors,
            linewidth=0.9,
            label=frame_label,
        )

        for xpos, mean_val, std_val in zip(positions, means_matrix[:, frame_idx], stds_matrix[:, frame_idx]):
            ax.text(
                xpos,
                mean_val + 0.01,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                fontweight="bold",
                rotation=90,
            )
            ax.text(
                xpos,
                mean_val + 0.055,
                f"std={std_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=4.8,
                color="#555555",
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Percentile", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Recent-4 Frame Similarity: Per-Frame Mean Percentile by Task", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.legend(title="Recent frame", ncol=4, fontsize=8, title_fontsize=9)
    add_group_separators(ax, group_types)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "similarity_recent4_per_frame_mean_percentile.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    summary = load_summary(result_dir)
    backend = detect_similarity_backend(summary)
    records = filter_excluded_tasks(load_jsonl(result_dir / "records.jsonl"))
    if records:
        summary = build_experiment_summary(records, config=summary.get("config", {}))
    else:
        summary = filtered_summary_fallback(summary, backend)
    plots_dir = result_dir / "plots"
    plot_mean_percentile_bar(summary, backend, plots_dir)
    plot_recent4_per_frame_mean_percentile_bar(records, backend, plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SigLIP similarity saliency analysis results.")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
