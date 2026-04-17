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


NUM_BINS = 20


def record_relative_positions(record: dict[str, Any]) -> list[float] | None:
    """Extract all selected frame relative positions from a record."""
    if record.get("error"):
        return None
    positions = record.get("selected_frame_relative_positions")
    if isinstance(positions, list) and positions:
        return [float(p) for p in positions]

    selected = record.get("selected_frame_indices_for_inference")
    num_sampled = record.get("num_sampled_frames")
    if isinstance(selected, list) and selected and isinstance(num_sampled, int) and num_sampled > 0:
        denom = max(1, num_sampled - 1)
        return [float(idx) / float(denom) for idx in selected]
    return None


def collect_per_task_positions(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Collect raw relative positions per task (all selected frames, not means)."""
    per_task: dict[str, list[float]] = {}
    for record in records:
        task = str(record.get("task", ""))
        if not task or task in EXCLUDED_PLOT_TASKS:
            continue
        positions = record_relative_positions(record)
        if positions is None:
            continue
        per_task.setdefault(task, []).extend(positions)
    return per_task


def line_styles() -> dict[str, dict[str, Any]]:
    return {
        "total": {"color": "black", "linewidth": 3.0, "linestyle": "-", "marker": "o"},
        "backward": {"color": "#1f77b4", "linewidth": 2.4, "linestyle": "--", "marker": "o"},
        "realtime": {"color": "#d62728", "linewidth": 2.4, "linestyle": "-.", "marker": "o"},
    }


def histogram_line(
    positions: list[float], num_bins: int = NUM_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin positions into a histogram and return (bin_centers, counts)."""
    counts, bin_edges = np.histogram(positions, bins=num_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, counts.astype(float)


def plot_selected_position_line(
    per_task_positions: dict[str, list[float]],
    plots_dir: Path,
    layer_number: int | None,
) -> None:
    if not per_task_positions:
        print("No data to plot.")
        return

    # Collect task lines and per-task histograms for averaging
    task_lines: list[tuple[str, np.ndarray, np.ndarray]] = []  # (label, x, y)
    realtime_hists: list[np.ndarray] = []
    backward_hists: list[np.ndarray] = []

    for task in REAL_TIME_TASKS:
        positions = per_task_positions.get(task, [])
        if not positions:
            continue
        x, y = histogram_line(positions)
        task_lines.append((task, x, y))
        realtime_hists.append(y)

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        positions = per_task_positions.get(task, [])
        if not positions:
            continue
        x, y = histogram_line(positions)
        task_lines.append((task, x, y))
        backward_hists.append(y)

    # Pooled lines use mean of per-task histograms
    bin_centers = histogram_line([], NUM_BINS)[0]  # just get bin centers
    pooled_lines: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_hists: list[np.ndarray] = []
    if realtime_hists:
        mean_y = np.mean(np.stack(realtime_hists), axis=0)
        pooled_lines.append(("realtime", bin_centers, mean_y))
        all_hists.extend(realtime_hists)
    if backward_hists:
        mean_y = np.mean(np.stack(backward_hists), axis=0)
        pooled_lines.append(("backward", bin_centers, mean_y))
        all_hists.extend(backward_hists)
    if all_hists:
        mean_y = np.mean(np.stack(all_hists), axis=0)
        pooled_lines.append(("total", bin_centers, mean_y))

    # Plot
    task_colors = plt.get_cmap("tab20", max(len(task_lines), 1))
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (label, x, y) in enumerate(task_lines):
        ax.plot(x, y, label=label, color=task_colors(idx), linewidth=1.8, marker="o", alpha=0.9)

    style_map = line_styles()
    pooled_label_map = {
        "total": "Total (average)",
        "backward": "Backward Tracing Subset (average)",
        "realtime": "Real-time Subset (average)",
    }
    for label, x, y in pooled_lines:
        ax.plot(x, y, label=pooled_label_map[label], **style_map[label])

    title_layer = f" (layer={layer_number})" if layer_number is not None else ""
    ax.set_title(f"Attention Top-4 Selected Frame Position Distribution{title_layer}", fontsize=13)
    ax.set_xlabel("Relative Position in Video", fontsize=11)
    ax.set_ylabel("Selection Count (average for pooled)", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, fontsize=9)
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

    per_task_positions = collect_per_task_positions(records)
    if not per_task_positions:
        print("No usable records (after task filtering / position recovery).")
        return

    plots_dir = result_dir / "plots"
    plot_selected_position_line(per_task_positions, plots_dir, layer_number)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot attention top-4 selected frame relative positions per OVO subset / split / total."
    )
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
