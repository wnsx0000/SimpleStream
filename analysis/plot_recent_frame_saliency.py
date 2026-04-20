from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import build_experiment_summary, question_prefill_layer_indices
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

QUESTION_PREFILL_FRAME_FRAME_AVERAGE_SHAPE = (64, 64)
QUESTION_PREFILL_QUESTION_FRAME_AVERAGE_SHAPE = (64, 64)
EXCLUDED_PLOT_TASKS = frozenset({"HLD"})
TASK_PLOT_ORDER = [
    *[task for task in BACKWARD_TASKS if task not in EXCLUDED_PLOT_TASKS],
    *REAL_TIME_TASKS,
]
ATTENTION_METRIC_SPECS = (
    ("question_prefill_attention", "question_prefill", "Question Prefill Attention"),
)
LINE_PLOT_SPECS = (
    ("percentile_mean", "layer_recent4_mean_percentile_mean", "Recent4 Mean Percentile", "Mean Percentile"),
    ("percentile_std", "layer_recent4_mean_percentile_std", "Recent4 Mean Percentile Std", "Std"),
)

BAR_PLOT_COLOR_MAP = {
    "backward": "#6baed6",
    "backward_avg": "#2171b5",
    "realtime": "#fc9272",
    "realtime_avg": "#cb181d",
    "total_avg": "#525252",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if not record.get("error")]


def records_for_split(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("split", "")) == split_name]


def filter_excluded_tasks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("task", "")) not in EXCLUDED_PLOT_TASKS]


def normalized_metric_layer_array(metric: dict[str, Any], field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if field not in metric:
        return None, None

    values = np.asarray(metric[field], dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[0] < 1:
        return None, None

    saved_indices = np.asarray(metric.get("display_layer_indices", []), dtype=np.int64)
    if saved_indices.size == values.shape[0]:
        return saved_indices, values

    sampled_indices = np.asarray(question_prefill_layer_indices(values.shape[0]), dtype=np.int64)
    if sampled_indices.size < 1:
        return None, None
    return sampled_indices, values[sampled_indices]


def ordered_task_names(summary: dict[str, Any]) -> list[str]:
    task_summary = summary.get("tasks", {})
    known_tasks = [task for task in TASK_PLOT_ORDER if task in task_summary]
    extra_tasks = sorted(task for task in task_summary if task not in TASK_PLOT_ORDER)
    return [*known_tasks, *extra_tasks]


def extract_metric_series(metric: dict[str, Any], field_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    return normalized_metric_layer_array(metric, field_name)


def metric_series_groups(summary: dict[str, Any], metric_name: str) -> list[tuple[str, dict[str, Any]]]:
    groups: list[tuple[str, dict[str, Any]]] = []

    total_metric = summary.get("metrics", {}).get(metric_name)
    if isinstance(total_metric, dict):
        groups.append(("total", total_metric))

    split_labels = {
        "backward": "backward",
        "realtime": "realtime",
    }
    for split_name, label in split_labels.items():
        split_metric = summary.get("splits", {}).get(split_name, {}).get("metrics", {}).get(metric_name)
        if isinstance(split_metric, dict):
            groups.append((label, split_metric))

    for task_name in ordered_task_names(summary):
        task_metric = summary.get("tasks", {}).get(task_name, {}).get("metrics", {}).get(metric_name)
        if isinstance(task_metric, dict):
            groups.append((task_name, task_metric))

    return groups


def line_styles() -> dict[str, dict[str, Any]]:
    return {
        "total": {"color": "black", "linewidth": 3.0, "linestyle": "-", "marker": "o"},
        "backward": {"color": "#1f77b4", "linewidth": 2.4, "linestyle": "--", "marker": "o"},
        "realtime": {"color": "#d62728", "linewidth": 2.4, "linestyle": "-.", "marker": "o"},
    }


def collect_metric_lines(
    summary: dict[str, Any],
    metric_name: str,
    field_name: str,
) -> tuple[np.ndarray | None, list[tuple[str, np.ndarray]]]:
    groups = metric_series_groups(summary, metric_name)
    if not groups:
        return None, []

    base_indices: np.ndarray | None = None
    lines: list[tuple[str, np.ndarray]] = []
    for label, metric in groups:
        layer_indices, values = extract_metric_series(metric, field_name)
        if layer_indices is None or values is None or values.ndim != 1:
            continue
        if base_indices is None:
            base_indices = layer_indices
        if not np.array_equal(layer_indices, base_indices):
            print(
                f"Skipping {metric_name} {field_name} line for {label}: "
                "display_layer_indices do not match the pooled metric."
            )
            continue
        lines.append((label, values))
    return base_indices, lines


def plot_attention_metric_line(
    summary: dict[str, Any],
    metric_name: str,
    metric_title: str,
    file_prefix: str,
    field_suffix: str,
    field_name: str,
    plot_title: str,
    y_label: str,
    plots_dir: Path,
) -> None:
    layer_indices, lines = collect_metric_lines(summary, metric_name, field_name)
    if layer_indices is None or not lines:
        return

    pooled_names = {"total", "backward", "realtime"}
    task_lines = [(label, values) for label, values in lines if label not in pooled_names]
    pooled_lines = [(label, values) for label, values in lines if label in pooled_names]
    task_colors = plt.get_cmap("tab20", max(len(task_lines), 1))

    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (label, values) in enumerate(task_lines):
        ax.plot(
            layer_indices,
            values,
            label=label,
            color=task_colors(idx),
            linewidth=1.8,
            marker="o",
            alpha=0.9,
        )

    style_map = line_styles()
    pooled_label_map = {
        "total": "Total",
        "backward": "Backward Tracing Subset",
        "realtime": "Real-time Subset",
    }
    for label, values in pooled_lines:
        ax.plot(
            layer_indices,
            values,
            label=pooled_label_map[label],
            **style_map[label],
        )

    ax.set_title(f"{metric_title}: {plot_title}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(y_label)
    ax.set_xticks(layer_indices)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{file_prefix}_{field_suffix}.png", dpi=200)
    plt.close(fig)


def plot_attention_metric_line_pooled_only(
    summary: dict[str, Any],
    metric_name: str,
    metric_title: str,
    file_prefix: str,
    field_suffix: str,
    field_name: str,
    plot_title: str,
    y_label: str,
    plots_dir: Path,
) -> None:
    layer_indices, lines = collect_metric_lines(summary, metric_name, field_name)
    if layer_indices is None or not lines:
        return

    pooled_order = ("backward", "realtime", "total")
    pooled_lines = {label: values for label, values in lines if label in {"total", "backward", "realtime"}}
    if not pooled_lines:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    style_map = line_styles()
    pooled_label_map = {
        "total": "Total",
        "backward": "Backward",
        "realtime": "Real-time",
    }
    for label in pooled_order:
        values = pooled_lines.get(label)
        if values is None:
            continue
        ax.plot(
            layer_indices,
            values,
            label=pooled_label_map[label],
            **style_map[label],
        )

    ax.set_title(f"{metric_title}: {plot_title} (Total / Backward / Real-time)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(y_label)
    ax.set_xticks(layer_indices)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{file_prefix}_{field_suffix}_pooled.png", dpi=200)
    plt.close(fig)


def plot_attention_line_plots(summary: dict[str, Any], plots_dir: Path) -> None:
    plots_dir = ensure_dir(plots_dir)
    for metric_name, file_prefix, metric_title in ATTENTION_METRIC_SPECS:
        if metric_name not in summary.get("metrics", {}):
            continue
        for field_suffix, field_name, plot_title, y_label in LINE_PLOT_SPECS:
            plot_attention_metric_line(
                summary,
                metric_name=metric_name,
                metric_title=metric_title,
                file_prefix=file_prefix,
                field_suffix=field_suffix,
                field_name=field_name,
                plot_title=plot_title,
                y_label=y_label,
                plots_dir=plots_dir,
            )
            plot_attention_metric_line_pooled_only(
                summary,
                metric_name=metric_name,
                metric_title=metric_title,
                file_prefix=file_prefix,
                field_suffix=field_suffix,
                field_name=field_name,
                plot_title=plot_title,
                y_label=y_label,
                plots_dir=plots_dir,
            )


def to_numpy_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return None


def question_prefill_map_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("question_prefill_attention_maps")
    if not isinstance(value, dict):
        return None
    frame_frame = to_numpy_array(value.get("frame_frame_maps"))
    question_frame = to_numpy_array(value.get("question_frame_maps"))
    if frame_frame is None or question_frame is None:
        return None
    if frame_frame.ndim != 3 or question_frame.ndim != 3:
        return None
    if frame_frame.shape[0] != question_frame.shape[0] or frame_frame.shape[0] < 1:
        return None
    return value


def question_prefill_sink_bin_token_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("question_prefill_sink_bin_token_attention")
    if not isinstance(value, dict):
        return None
    maps = to_numpy_array(value.get("maps"))
    if maps is None or maps.ndim != 3:
        return None
    if maps.shape[0] < 1 or maps.shape[1] < 1 or maps.shape[2] < 1:
        return None
    return value


def frame_slice_centers(frame_bin_slices: list[list[int]] | list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(
        [0.5 * (int(start) + int(end) - 1) for start, end in frame_bin_slices],
        dtype=np.float64,
    )


def square_marker_size(matrix_shape: tuple[int, int]) -> float:
    return max(30.0, 12000.0 / float(max(matrix_shape)))


def scatter_square_heatmap(
    ax: Any,
    matrix: np.ndarray,
    norm: mcolors.Normalize,
    cmap: str = "viridis",
) -> Any:
    rows, cols = matrix.shape
    mesh = ax.pcolormesh(
        np.arange(cols + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    return mesh


def annotate_frame_boundaries(
    ax: Any,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]],
    *,
    draw_x: bool,
    draw_y: bool,
) -> None:
    for start, end in frame_bin_slices[:-1]:
        boundary = int(end) - 0.5
        if draw_x:
            ax.axvline(boundary, color="white", linewidth=0.5, alpha=0.5)
        if draw_y:
            ax.axhline(boundary, color="white", linewidth=0.5, alpha=0.5)


def apply_frame_ticks(
    ax: Any,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]],
    frame_bin_labels: list[str],
    *,
    axis: str,
) -> None:
    centers = frame_slice_centers(frame_bin_slices)
    if axis in {"x", "both"}:
        ax.set_xticks(centers)
        ax.set_xticklabels(frame_bin_labels, rotation=90, fontsize=7)
    if axis in {"y", "both"}:
        ax.set_yticks(centers)
        ax.set_yticklabels(frame_bin_labels, fontsize=7)


def apply_question_ticks(ax: Any, question_bin_labels: list[str]) -> None:
    ax.set_yticks(np.arange(len(question_bin_labels)))
    ax.set_yticklabels(question_bin_labels, fontsize=7)


def apply_token_ticks(ax: Any, token_labels: list[str], max_ticks: int = 12) -> None:
    if not token_labels:
        return
    num_tokens = len(token_labels)
    if num_tokens <= max_ticks:
        positions = np.arange(num_tokens)
    else:
        positions = np.unique(np.linspace(0, num_tokens - 1, max_ticks, dtype=np.int64))
    ax.set_xticks(positions)
    ax.set_xticklabels([token_labels[int(index)] for index in positions], rotation=90, fontsize=7)


def apply_relative_ticks(ax: Any, *, width: int, height: int, mode: str) -> None:
    x_positions = np.linspace(0, width - 1, 5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, 5)])
    if mode == "frame_frame":
        y_positions = np.linspace(0, height - 1, 5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, 5)])
    else:
        y_positions = np.linspace(0, height - 1, min(5, height))
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, len(y_positions))])


def build_emphasized_heatmap_norm(values: np.ndarray) -> mcolors.Normalize:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 1:
        return mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    robust_vmax = float(np.percentile(finite_values, 99.5))
    if robust_vmax > vmin + 1e-12:
        vmax = min(vmax, robust_vmax)

    if np.all(finite_values >= 0.0):
        return mcolors.PowerNorm(gamma=0.35, vmin=max(0.0, vmin), vmax=vmax, clip=True)

    linthresh = max((vmax - vmin) / 100.0, 1e-8)
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10.0, clip=True)


def make_example_label(payload: dict[str, Any], fallback: str) -> str:
    task = payload.get("task")
    sample_id = payload.get("id")
    split = payload.get("split")
    label_parts = []
    if task is not None and sample_id is not None:
        label_parts.append(f"{task}:{sample_id}")
    elif task is not None:
        label_parts.append(str(task))
    elif sample_id is not None:
        label_parts.append(str(sample_id))
    else:
        label_parts.append(fallback)
    if split:
        label_parts.append(f"[{split}]")
    return " ".join(label_parts)


def render_question_prefill_map_panels(
    maps: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    x_label: str,
    y_label: str,
    mode: str,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]] | None = None,
    frame_bin_labels: list[str] | None = None,
    question_bin_labels: list[str] | None = None,
) -> None:
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] < 1:
        return

    finite_values = maps[np.isfinite(maps)]
    if finite_values.size < 1:
        return
    norm = build_emphasized_heatmap_norm(finite_values)

    max_cols = 4
    num_panels = maps.shape[0]
    num_rows = (num_panels + max_cols - 1) // max_cols
    num_cols = min(num_panels, max_cols)
    panel_width = 4.6
    panel_height = panel_width
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )
    mappable = None
    all_axes: list[Any] = []
    for panel_idx, (matrix, layer_idx) in enumerate(zip(maps, display_layer_indices)):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        all_axes.append(ax)
        mappable = scatter_square_heatmap(ax, matrix, norm=norm)
        ax.set_title(f"Layer {int(layer_idx)}")
        ax.set_xlabel(x_label)
        if mode == "frame_frame":
            if frame_bin_slices is not None and frame_bin_labels is not None:
                annotate_frame_boundaries(ax, frame_bin_slices, draw_x=True, draw_y=True)
                apply_frame_ticks(ax, frame_bin_slices, frame_bin_labels, axis="both")
            else:
                apply_relative_ticks(ax, width=matrix.shape[1], height=matrix.shape[0], mode=mode)
        else:
            if frame_bin_slices is not None and frame_bin_labels is not None:
                annotate_frame_boundaries(ax, frame_bin_slices, draw_x=True, draw_y=False)
                apply_frame_ticks(ax, frame_bin_slices, frame_bin_labels, axis="x")
            else:
                apply_relative_ticks(ax, width=matrix.shape[1], height=matrix.shape[0], mode=mode)
            if question_bin_labels is not None:
                apply_question_ticks(ax, question_bin_labels)
            else:
                apply_relative_ticks(ax, width=matrix.shape[1], height=matrix.shape[0], mode=mode)
        if col_idx == 0:
            ax.set_ylabel(y_label)

    # Hide unused axes in the last row
    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    fig.suptitle(figure_title, y=0.98)
    if mode == "question_frame":
        fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.10, wspace=0.30, hspace=0.8)
    else:
        fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.10, wspace=0.30, hspace=0.45)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.82])
        fig.colorbar(mappable, cax=cbar_ax, label="Mean Attention Score")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_question_prefill_sink_bin_token_panels(
    maps: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    question_bin_labels: list[str] | None = None,
    token_labels: list[str] | None = None,
) -> None:
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] < 1:
        return

    finite_values = maps[np.isfinite(maps)]
    if finite_values.size < 1:
        return
    norm = build_emphasized_heatmap_norm(finite_values)

    max_cols = 4
    num_panels = maps.shape[0]
    num_rows = (num_panels + max_cols - 1) // max_cols
    num_cols = min(num_panels, max_cols)
    panel_width = 5.0
    panel_height = 3.8 if maps.shape[1] <= 12 else 4.8
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    if token_labels is None or len(token_labels) != maps.shape[2]:
        token_labels = [f"tok{index}" for index in range(maps.shape[2])]

    mappable = None
    for panel_idx, (matrix, layer_idx) in enumerate(zip(maps, display_layer_indices)):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        mappable = scatter_square_heatmap(ax, matrix, norm=norm)
        ax.set_title(f"Layer {int(layer_idx)}")
        ax.set_xlabel("Token Offset in Sink Bin")
        apply_token_ticks(ax, token_labels)
        if question_bin_labels is not None and len(question_bin_labels) == matrix.shape[0]:
            apply_question_ticks(ax, question_bin_labels)
        else:
            y_positions = np.linspace(0, matrix.shape[0] - 1, min(5, matrix.shape[0]))
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, len(y_positions))])
        if col_idx == 0:
            ax.set_ylabel("Question Token Bin")

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    fig.suptitle(figure_title, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.14, wspace=0.30, hspace=0.75)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.90, 0.14, 0.015, 0.76])
        fig.colorbar(mappable, cax=cbar_ax, label="Mean Attention Score")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def resample_attention_map(matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = torch.as_tensor(matrix, dtype=torch.float32)[None, None]
    resized = F.interpolate(source, size=target_shape, mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def compute_average_question_prefill_maps(
    payloads: list[dict[str, Any]],
    *,
    field_name: str,
    target_shape: tuple[int, int],
) -> tuple[list[int] | None, np.ndarray | None]:
    display_layer_indices: list[int] | None = None
    stacked: list[np.ndarray] = []
    for payload in payloads:
        map_payload = question_prefill_map_payload(payload)
        if map_payload is None:
            continue
        display_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
        if not display_indices:
            continue
        values = to_numpy_array(map_payload.get(field_name))
        if values is None or values.ndim != 3:
            continue
        if display_layer_indices is None:
            display_layer_indices = display_indices
        if display_indices != display_layer_indices:
            continue
        resized_layers = np.stack(
            [resample_attention_map(np.asarray(layer, dtype=np.float32), target_shape) for layer in values],
            axis=0,
        )
        stacked.append(resized_layers)
    if not stacked or display_layer_indices is None:
        return None, None
    return display_layer_indices, np.stack(stacked, axis=0).mean(axis=0)


def plot_question_prefill_attention_map_averages(payloads: list[dict[str, Any]], plots_dir: Path) -> None:
    available_payloads = [payload for payload in payloads if question_prefill_map_payload(payload) is not None]
    if not available_payloads:
        print("Skipping question-prefill average map heatmaps: no saved example payload includes question_prefill_attention_maps.")
        return

    display_layers, frame_frame_average = compute_average_question_prefill_maps(
        available_payloads,
        field_name="frame_frame_maps",
        target_shape=QUESTION_PREFILL_FRAME_FRAME_AVERAGE_SHAPE,
    )
    if display_layers is not None and frame_frame_average is not None:
        render_question_prefill_map_panels(
            frame_frame_average,
            display_layers,
            plots_dir / "question_prefill_frame_frame_maps_average.png",
            figure_title="Question Prefill Frame↔Frame Maps (Saved Examples Average)",
            x_label="Relative Frame-Bin Position",
            y_label="Relative Frame-Bin Position",
            mode="frame_frame",
        )

    display_layers, question_frame_average = compute_average_question_prefill_maps(
        available_payloads,
        field_name="question_frame_maps",
        target_shape=QUESTION_PREFILL_QUESTION_FRAME_AVERAGE_SHAPE,
    )
    if display_layers is not None and question_frame_average is not None:
        render_question_prefill_map_panels(
            question_frame_average,
            display_layers,
            plots_dir / "question_prefill_question_frame_maps_average.png",
            figure_title="Question Prefill Question→Frame Maps (Saved Examples Average)",
            x_label="Relative Frame-Bin Position",
            y_label="Relative Question-Bin Position",
            mode="question_frame",
        )


def render_per_layer_attention_overlay(
    frames: list[np.ndarray],
    per_frame_attention: list[np.ndarray],
    layer_idx: int,
    output_path: Path,
    *,
    figure_title: str,
    frame_indices: list[int] | None = None,
    recent_frame_indices: list[int] | None = None,
) -> None:
    if not frames or not per_frame_attention:
        return
    num_panels = min(len(frames), len(per_frame_attention))
    if num_panels < 1:
        return
    max_cols = 4
    num_cols = min(num_panels, max_cols)
    num_rows = (num_panels + max_cols - 1) // max_cols
    panel_width = 4.6
    panel_height = panel_width
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    finite_values = np.concatenate([
        np.asarray(matrix, dtype=np.float64).ravel()
        for matrix in per_frame_attention[:num_panels]
        if np.asarray(matrix).size > 0
    ]) if per_frame_attention else np.empty(0, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    norm = build_emphasized_heatmap_norm(finite_values)
    recent_set = {int(idx) for idx in (recent_frame_indices or [])}

    mappable = None
    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        frame_image = np.asarray(frames[panel_idx])
        if frame_image.ndim != 3 or frame_image.shape[-1] not in (3, 4):
            ax.set_visible(False)
            continue
        image_h, image_w = int(frame_image.shape[0]), int(frame_image.shape[1])
        ax.imshow(frame_image)

        attention_matrix = np.asarray(per_frame_attention[panel_idx], dtype=np.float32)
        if attention_matrix.ndim == 2 and attention_matrix.size > 0:
            mappable = ax.imshow(
                attention_matrix,
                extent=(-0.5, image_w - 0.5, image_h - 0.5, -0.5),
                cmap="magma",
                norm=norm,
                alpha=0.55,
                interpolation="nearest",
            )
        ax.set_xlim(-0.5, image_w - 0.5)
        ax.set_ylim(image_h - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        title_parts: list[str] = []
        if frame_indices is not None and panel_idx < len(frame_indices):
            absolute_idx = int(frame_indices[panel_idx])
            tag = "*" if absolute_idx in recent_set else ""
            title_parts.append(f"frame {absolute_idx}{tag}")
        else:
            title_parts.append(f"panel {panel_idx}")
        ax.set_title(" ".join(title_parts), fontsize=9)

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    fig.suptitle(figure_title, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.06, wspace=0.10, hspace=0.20)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.10, 0.012, 0.80])
        fig.colorbar(mappable, cax=cbar_ax, label=f"Layer {int(layer_idx)} Mean Attention")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_value_norm_bar_panels(
    value_norms: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]],
    frame_bin_labels: list[str],
    recent_frame_indices_within_attention: list[int] | None = None,
) -> None:
    value_norms = np.asarray(value_norms, dtype=np.float32)
    if value_norms.ndim != 2 or value_norms.shape[0] < 1 or value_norms.shape[1] < 1:
        return
    if not frame_bin_slices or len(frame_bin_slices) != value_norms.shape[1]:
        return

    num_panels = value_norms.shape[0]
    max_cols = 4
    num_cols = min(num_panels, max_cols)
    num_rows = (num_panels + max_cols - 1) // max_cols
    panel_width = 4.6
    panel_height = panel_width
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    centers = frame_slice_centers(frame_bin_slices)
    widths = np.asarray(
        [max(1, int(end) - int(start)) for start, end in frame_bin_slices],
        dtype=np.float64,
    )
    total_bins = int(frame_bin_slices[-1][1])
    recent_set = {int(idx) for idx in (recent_frame_indices_within_attention or [])}

    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        layer_values = value_norms[panel_idx]
        bar_colors = [
            "#cb181d" if frame_idx in recent_set else "#4f81bd"
            for frame_idx in range(layer_values.shape[0])
        ]
        ax.bar(centers, layer_values, width=widths, align="center", color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.set_xlim(-0.5, total_bins - 0.5)
        annotate_frame_boundaries(ax, frame_bin_slices, draw_x=True, draw_y=False)
        apply_frame_ticks(ax, frame_bin_slices, frame_bin_labels, axis="x")
        ax.set_title(f"Layer {int(display_layer_indices[panel_idx])}")
        ax.set_xlabel("Frame Index")
        if col_idx == 0:
            ax.set_ylabel("Mean V Norm")

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    fig.suptitle(figure_title, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.10, wspace=0.30, hspace=0.55)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_payload(example_path: Path, plots_dir: Path, payload: dict[str, Any] | None = None) -> None:
    payload = torch.load(example_path, map_location="cpu") if payload is None else payload
    if str(payload.get("task", "")) in EXCLUDED_PLOT_TASKS:
        return
    example_key = example_path.stem
    example_dir = ensure_dir(plots_dir / "examples" / example_key)
    metrics = payload.get("metrics", {})

    for metric_name, title in (
        ("question_prefill_attention", "Question Prefill"),
    ):
        if metric_name not in metrics:
            continue
        layer_indices, matrix = normalized_metric_layer_array(metrics[metric_name], "layer_attention_scores")
        if layer_indices is None or matrix is None:
            continue

        x_positions = np.asarray(
            metrics[metric_name].get("attention_frame_indices", list(range(matrix.shape[1]))),
            dtype=np.int64,
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="magma",
            norm=build_emphasized_heatmap_norm(matrix),
        )
        ax.set_title(f"{title} Attention Heatmap: {example_key}")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Layer")
        tick_positions = np.linspace(0, matrix.shape[1] - 1, min(6, matrix.shape[1]))
        tick_labels = [str(int(x_positions[int(round(pos))])) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(np.arange(layer_indices.shape[0]))
        ax.set_yticklabels([str(int(index)) for index in layer_indices])
        # Draw boundary lines between each frame column
        for col_idx in range(matrix.shape[1] - 1):
            ax.axvline(col_idx + 0.5, color="white", linewidth=0.4, alpha=0.35)
        selected_recent = set(int(index) for index in metrics[metric_name].get("recent_frame_indices_within_attention", []))
        for local_index in selected_recent:
            ax.axvline(local_index, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, label="Attention Score")
        fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.12)
        fig.savefig(example_dir / f"{metric_name}_heatmap.png", dpi=200)
        plt.close(fig)

    map_payload = question_prefill_map_payload(payload)
    example_label = make_example_label(payload, example_key)
    if map_payload is None:
        print(
            f"Skipping question-prefill map heatmaps for {example_key}: "
            "saved example payload does not contain question_prefill_attention_maps."
        )
    else:
        display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
        frame_bin_slices = [
            [int(start), int(end)]
            for start, end in map_payload.get("frame_bin_slices", [])
        ]
        frame_bin_labels = [str(label) for label in map_payload.get("frame_bin_labels", [])]
        question_bin_labels = [str(label) for label in map_payload.get("question_bin_labels", [])]

        frame_frame_maps = to_numpy_array(map_payload.get("frame_frame_maps"))
        if frame_frame_maps is not None:
            render_question_prefill_map_panels(
                frame_frame_maps,
                display_layer_indices,
                example_dir / "question_prefill_frame_frame_maps.png",
                figure_title=f"Question Prefill Frame↔Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Frame Index",
                mode="frame_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
            )

        question_frame_maps = to_numpy_array(map_payload.get("question_frame_maps"))
        if question_frame_maps is not None:
            render_question_prefill_map_panels(
                question_frame_maps,
                display_layer_indices,
                example_dir / "question_prefill_question_frame_maps.png",
                figure_title=f"Question Prefill Question→Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Question Token Bin",
                mode="question_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
                question_bin_labels=question_bin_labels,
            )

        analysis_frames = payload.get("analysis_frames")
        per_patch_attention = payload.get("question_prefill_per_patch_attention")
        analysis_frame_indices_payload = payload.get("analysis_frame_indices")
        recent_attention_indices = (
            payload.get("metrics", {})
            .get("question_prefill_attention", {})
            .get("recent_frame_indices_within_attention", [])
        )
        recent_absolute_indices: list[int] = []
        if analysis_frame_indices_payload is not None and recent_attention_indices:
            for local_idx in recent_attention_indices:
                local_idx = int(local_idx)
                if 0 <= local_idx < len(analysis_frame_indices_payload):
                    recent_absolute_indices.append(int(analysis_frame_indices_payload[local_idx]))
        if analysis_frames and isinstance(per_patch_attention, dict) and display_layer_indices:
            frame_arrays = [np.asarray(frame) for frame in analysis_frames]
            absolute_frame_indices = (
                [int(idx) for idx in analysis_frame_indices_payload]
                if analysis_frame_indices_payload is not None
                else list(range(len(frame_arrays)))
            )
            for layer_idx in display_layer_indices:
                per_frame_tensors = per_patch_attention.get(int(layer_idx))
                if not per_frame_tensors:
                    continue
                per_frame_arrays = [
                    np.asarray(to_numpy_array(matrix), dtype=np.float32)
                    for matrix in per_frame_tensors
                ]
                render_per_layer_attention_overlay(
                    frame_arrays,
                    per_frame_arrays,
                    int(layer_idx),
                    example_dir / f"question_prefill_attention_overlay_layer{int(layer_idx)}.png",
                    figure_title=(
                        f"Question Prefill Attention Overlay (Layer {int(layer_idx)}): {example_label}"
                    ),
                    frame_indices=absolute_frame_indices,
                    recent_frame_indices=recent_absolute_indices,
                )
        elif display_layer_indices and (analysis_frames is None or per_patch_attention is None):
            print(
                f"Skipping per-layer attention overlay for {example_key}: "
                "saved example payload is missing analysis_frames or "
                "question_prefill_per_patch_attention."
            )

        value_norms = to_numpy_array(payload.get("question_prefill_value_norms"))
        if value_norms is not None and value_norms.ndim == 2 and display_layer_indices:
            render_value_norm_bar_panels(
                value_norms,
                display_layer_indices,
                example_dir / "question_prefill_value_norms.png",
                figure_title=f"Question Prefill V-Norms: {example_label}",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
                recent_frame_indices_within_attention=recent_attention_indices,
            )
        elif display_layer_indices and value_norms is None:
            print(
                f"Skipping value-norm bar panels for {example_key}: "
                "saved example payload is missing question_prefill_value_norms."
            )

    sink_payload = question_prefill_sink_bin_token_payload(payload)
    if sink_payload is None:
        print(
            f"Skipping question-prefill sink-bin token heatmap for {example_key}: "
            "saved example payload does not contain question_prefill_sink_bin_token_attention."
        )
        return

    sink_maps = to_numpy_array(sink_payload.get("maps"))
    if sink_maps is None:
        return
    sink_display_layer_indices = [int(index) for index in sink_payload.get("display_layer_indices", [])]
    if len(sink_display_layer_indices) != sink_maps.shape[0]:
        sink_display_layer_indices = list(range(sink_maps.shape[0]))
    sink_question_bin_labels = [str(label) for label in sink_payload.get("question_bin_labels", [])]
    sink_token_labels = [str(label) for label in sink_payload.get("token_labels", [])]
    sink_bin_index = sink_payload.get("sink_bin_index", "n/a")
    sink_frame_label = sink_payload.get("sink_frame_label", "n/a")
    sink_token_start = sink_payload.get("sink_token_start", "n/a")
    sink_token_end = sink_payload.get("sink_token_end", "n/a")
    render_question_prefill_sink_bin_token_panels(
        sink_maps,
        sink_display_layer_indices,
        example_dir / "question_prefill_sink_bin_token_attention.png",
        figure_title=(
            f"Question Prefill Sink-Bin Token Attention: {example_label} "
            f"(bin={sink_bin_index}, frame={sink_frame_label}, tokens={sink_token_start}:{sink_token_end})"
        ),
        question_bin_labels=sink_question_bin_labels,
        token_labels=sink_token_labels,
    )


def load_example_payloads(examples_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for example_path in sorted(examples_dir.glob("*.pt")):
        payload = torch.load(example_path, map_location="cpu")
        if str(payload.get("task", "")) in EXCLUDED_PLOT_TASKS:
            continue
        payloads.append((example_path, payload))
    return payloads


def extract_bar_plot_data(
    summary: dict[str, Any],
    metric_name: str,
) -> tuple[list[str], list[float], list[float], list[str]]:
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    group_types: list[str] = []

    task_data = summary.get("tasks", {})

    for task in REAL_TIME_TASKS:
        if task in EXCLUDED_PLOT_TASKS or task not in task_data:
            continue
        m = task_data[task].get("metrics", {}).get(metric_name, {})
        if not m:
            continue
        labels.append(task)
        means.append(float(m.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(m.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("realtime")

    rt_metric = summary.get("splits", {}).get("realtime", {}).get("metrics", {}).get(metric_name, {})
    if rt_metric:
        labels.append("Real-time\nAvg")
        means.append(float(rt_metric.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(rt_metric.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("realtime_avg")

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS or task not in task_data:
            continue
        m = task_data[task].get("metrics", {}).get(metric_name, {})
        if not m:
            continue
        labels.append(task)
        means.append(float(m.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(m.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("backward")

    bw_metric = summary.get("splits", {}).get("backward", {}).get("metrics", {}).get(metric_name, {})
    if bw_metric:
        labels.append("Backward\nAvg")
        means.append(float(bw_metric.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(bw_metric.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("backward_avg")

    total_metric = summary.get("metrics", {}).get(metric_name, {})
    if total_metric:
        labels.append("Total\nAvg")
        means.append(float(total_metric.get("recent4_mean_percentile_mean", 0.0)))
        stds.append(float(total_metric.get("recent4_mean_percentile_std", 0.0)))
        group_types.append("total_avg")

    return labels, means, stds, group_types


def plot_attention_mean_percentile_bar(
    summary: dict[str, Any],
    metric_name: str,
    file_prefix: str,
    metric_title: str,
    plots_dir: Path,
) -> None:
    labels, means, stds, group_types = extract_bar_plot_data(summary, metric_name)
    if not labels:
        return

    n = len(labels)
    x = np.arange(n)
    colors = [BAR_PLOT_COLOR_MAP[g] for g in group_types]
    edge_colors = [BAR_PLOT_COLOR_MAP[g] for g in group_types]

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
    ax.set_title(f"Recent-4 {metric_title}: Mean Percentile by Task", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    separator_positions = [i + 0.5 for i in range(n - 1) if group_types[i] != group_types[i + 1]]
    for pos in separator_positions:
        ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{file_prefix}_recent4_mean_percentile_bar.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_attention_mean_percentile_bars(summary: dict[str, Any], plots_dir: Path) -> None:
    plots_dir = ensure_dir(plots_dir)
    for metric_name, file_prefix, metric_title in ATTENTION_METRIC_SPECS:
        if metric_name not in summary.get("metrics", {}):
            continue
        plot_attention_mean_percentile_bar(
            summary,
            metric_name=metric_name,
            file_prefix=file_prefix,
            metric_title=metric_title,
            plots_dir=plots_dir,
        )


def generate_top_level_plots(records: list[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir = ensure_dir(plots_dir)
    if not records:
        return

    summary = build_experiment_summary(records, config={})
    plot_attention_line_plots(summary, plots_dir)
    plot_attention_mean_percentile_bars(summary, plots_dir)


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    records = filter_excluded_tasks(load_jsonl(result_dir / "records.jsonl"))
    plots_dir = result_dir / "plots"
    if not records:
        return

    generate_top_level_plots(records, plots_dir)

    examples_dir = result_dir / "examples"
    if not examples_dir.exists():
        return

    example_payloads = load_example_payloads(examples_dir)
    for example_path, payload in example_payloads:
        plot_example_payload(example_path, ensure_dir(plots_dir), payload=payload)
    plot_question_prefill_attention_map_averages(
        [payload for _, payload in example_payloads],
        ensure_dir(plots_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SimpleStream recent4 saliency analysis results.")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
