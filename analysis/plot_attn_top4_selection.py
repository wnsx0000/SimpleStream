from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

EXCLUDED_PLOT_TASKS = frozenset({"HLD"})
SPLIT_BY_TASK: dict[str, str] = {
    **{task: "backward" for task in BACKWARD_TASKS},
    **{task: "realtime" for task in REAL_TIME_TASKS},
}


def load_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "summary.json"
    if not path.exists():
        return {}
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


def record_mean_relative_position(record: dict[str, Any]) -> float | None:
    if record.get("error"):
        return None
    value = record.get("selected_frame_mean_relative_position")
    if value is not None:
        return float(value)

    positions = record.get("selected_frame_relative_positions")
    if isinstance(positions, list) and positions:
        return float(np.mean([float(p) for p in positions]))

    selected = record.get("selected_frame_indices_for_inference")
    num_sampled = record.get("num_sampled_frames")
    if isinstance(selected, list) and selected and isinstance(num_sampled, int) and num_sampled > 0:
        denom = max(1, num_sampled - 1)
        return float(np.mean([float(idx) / float(denom) for idx in selected]))
    return None


def collect_per_task_means(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    per_task: dict[str, list[float]] = {}
    for record in records:
        task = str(record.get("task", ""))
        if not task or task in EXCLUDED_PLOT_TASKS:
            continue
        mean_pos = record_mean_relative_position(record)
        if mean_pos is None:
            continue
        per_task.setdefault(task, []).append(mean_pos)
    return per_task


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


COLOR_MAP = {
    "backward": "#6baed6",
    "backward_avg": "#2171b5",
    "realtime": "#fc9272",
    "realtime_avg": "#cb181d",
    "total_avg": "#525252",
}


def extract_bar_data(
    per_task_means: dict[str, list[float]],
) -> tuple[list[str], list[float], list[float], list[int], list[str]]:
    """Returns (labels, means, stds, counts, group_types) ordered:
    realtime tasks -> Real-time avg -> backward tasks -> Backward avg -> Total avg
    """
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    group_types: list[str] = []

    realtime_pool: list[float] = []
    backward_pool: list[float] = []

    for task in REAL_TIME_TASKS:
        values = per_task_means.get(task, [])
        if not values:
            continue
        m, s = mean_std(values)
        labels.append(task)
        means.append(m)
        stds.append(s)
        counts.append(len(values))
        group_types.append("realtime")
        realtime_pool.extend(values)

    if realtime_pool:
        m, s = mean_std(realtime_pool)
        labels.append("Real-time\nAvg")
        means.append(m)
        stds.append(s)
        counts.append(len(realtime_pool))
        group_types.append("realtime_avg")

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        values = per_task_means.get(task, [])
        if not values:
            continue
        m, s = mean_std(values)
        labels.append(task)
        means.append(m)
        stds.append(s)
        counts.append(len(values))
        group_types.append("backward")
        backward_pool.extend(values)

    if backward_pool:
        m, s = mean_std(backward_pool)
        labels.append("Backward\nAvg")
        means.append(m)
        stds.append(s)
        counts.append(len(backward_pool))
        group_types.append("backward_avg")

    total_pool = realtime_pool + backward_pool
    if total_pool:
        m, s = mean_std(total_pool)
        labels.append("Total\nAvg")
        means.append(m)
        stds.append(s)
        counts.append(len(total_pool))
        group_types.append("total_avg")

    return labels, means, stds, counts, group_types


def plot_selected_position_bar(
    per_task_means: dict[str, list[float]],
    plots_dir: Path,
    layer_number: int | None,
) -> None:
    labels, means, stds, counts, group_types = extract_bar_data(per_task_means)
    if not labels:
        print("No data to plot.")
        return

    n = len(labels)
    x = np.arange(n)
    colors = [COLOR_MAP[g] for g in group_types]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    ax.bar(x, means, color=colors, edgecolor=colors, linewidth=1.2)
    for i, (mean_val, std_val, count_val) in enumerate(zip(means, stds, counts)):
        ax.text(i, mean_val + 0.01, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(i, mean_val + 0.01 + 0.04, f"std={std_val:.3f}", ha="center", va="bottom", fontsize=7, color="#555555")
        ax.text(i, mean_val + 0.01 + 0.08, f"n={count_val}", ha="center", va="bottom", fontsize=7, color="#777777")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Relative Position (in sampled video)", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    title_layer = f" (layer={layer_number})" if layer_number is not None else ""
    ax.set_title(f"Attention Top-4 Selected Frame Position by Task{title_layer}", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    separator_positions: list[float] = []
    for i in range(n - 1):
        if group_types[i] != group_types[i + 1]:
            separator_positions.append(i + 0.5)
    for pos in separator_positions:
        ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "attn_top4_selected_position.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    summary = load_summary(result_dir)
    layer_number_value = summary.get("config", {}).get("layer_number")
    layer_number = int(layer_number_value) if isinstance(layer_number_value, (int, float)) else None

    records = load_jsonl(result_dir / "results_incremental.jsonl")
    if not records:
        print(f"No records found at {result_dir / 'results_incremental.jsonl'}")
        return

    per_task_means = collect_per_task_means(records)
    if not per_task_means:
        print("No usable records (after task filtering / position recovery).")
        return

    plots_dir = result_dir / "plots"
    plot_selected_position_bar(per_task_means, plots_dir, layer_number)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot attention top-4 selected frame relative positions per OVO subset / split / total."
    )
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
