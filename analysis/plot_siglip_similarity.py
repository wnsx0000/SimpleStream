from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import build_experiment_summary
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

EXCLUDED_PLOT_TASKS = frozenset({"HLD"})


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
    bars = ax.bar(
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
    separator_positions = []
    for i in range(n - 1):
        if group_types[i] != group_types[i + 1]:
            separator_positions.append(i + 0.5)
    for pos in separator_positions:
        ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "similarity_mean_percentile.png"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SigLIP similarity saliency analysis results.")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
