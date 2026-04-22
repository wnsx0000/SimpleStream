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
ROBUST_HEATMAP_PERCENTILE = 99.5
EXCLUDED_PLOT_TASKS = frozenset({"HLD"})
TASK_PLOT_ORDER = [
    *[task for task in BACKWARD_TASKS if task not in EXCLUDED_PLOT_TASKS],
    *REAL_TIME_TASKS,
]
RECENT_FRAME_LABELS = ("oldest", "older", "newer", "current")
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
ATTENTION_TOKEN_COUNT_BIN_PLOT_FILENAME = "question_prefill_attention_score_token_count_by_bin.png"
ATTENTION_TOKEN_COUNT_BIN_AVERAGE_PLOT_FILENAME = "question_prefill_attention_score_token_count_by_bin_average.png"
ATTENTION_TOKEN_COUNT_NORM_PLOT_FILENAME = "question_prefill_attention_score_token_count_by_norm.png"
ATTENTION_TOKEN_COUNT_NORM_AVERAGE_PLOT_FILENAME = "question_prefill_attention_score_token_count_by_norm_average.png"
ATTENTION_TOPK_TOKEN_COUNT = 20
ATTENTION_TOPK_TOKEN_VALUE_NORM_PLOT_FILENAME = "question_prefill_attention_score_top20_token_value_norm.png"
ATTENTION_TOPK_TOKEN_VALUE_NORM_POOLED_PLOT_FILENAME = (
    "question_prefill_attention_score_top20_token_value_norm_pooled.png"
)
ATTENTION_TOKEN_COUNT_NORM_BIN_COUNT = 40
ATTENTION_TOKEN_COUNT_PANEL_HSPACE = 0.42
ATTENTION_TOKEN_COUNT_FRAME_SEPARATOR_COLOR = "#c7c7c7"


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


def load_summary(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def skip_existing_plot(output_path: str | Path) -> bool:
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Skipping existing plot: {output_path}")
        return True
    return False


def skip_existing_plot_set(output_paths: list[Path] | tuple[Path, ...]) -> bool:
    if not all(Path(path).exists() for path in output_paths):
        return False
    for path in output_paths:
        skip_existing_plot(path)
    return True


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
    output_path = plots_dir / f"{file_prefix}_{field_suffix}.png"
    if skip_existing_plot(output_path):
        return

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
    fig.savefig(output_path, dpi=200)
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
    output_path = plots_dir / f"{file_prefix}_{field_suffix}_pooled.png"
    if skip_existing_plot(output_path):
        return

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
    fig.savefig(output_path, dpi=200)
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


def to_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
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


def frame_index_for_bin(
    bin_index: int,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]],
) -> int | None:
    bin_index = int(bin_index)
    for frame_idx, (start, end) in enumerate(frame_bin_slices):
        if int(start) <= bin_index < int(end):
            return int(frame_idx)
    return None


def mean_pool_value_norms_by_spans(
    value_norms: np.ndarray,
    spans: list[list[int]] | list[tuple[int, int]],
) -> np.ndarray | None:
    value_norms = np.asarray(value_norms, dtype=np.float32)
    if value_norms.ndim != 2 or not spans:
        return None
    pooled_columns: list[np.ndarray] = []
    for start, end in spans:
        start = int(start)
        end = int(end)
        if end <= start or start < 0 or end > value_norms.shape[1]:
            return None
        pooled_columns.append(value_norms[:, start:end].mean(axis=1))
    if not pooled_columns:
        return None
    return np.stack(pooled_columns, axis=1)


def normalize_frame_bin_slices(value: Any) -> list[list[int]]:
    slices: list[list[int]] = []
    if not isinstance(value, (list, tuple)):
        return slices
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return []
        start, end = int(item[0]), int(item[1])
        if end <= start:
            return []
        slices.append([start, end])
    return slices


def frame_bin_token_counts(
    frame_local_bin_spans: list[list[int]] | list[tuple[int, int]],
) -> np.ndarray | None:
    spans = normalize_frame_bin_slices(frame_local_bin_spans)
    if not spans:
        return None
    counts = np.asarray([end - start for start, end in spans], dtype=np.float64)
    if counts.size < 1 or not np.all(np.isfinite(counts)) or np.any(counts <= 0.0):
        return None
    return counts


def extract_attention_score_token_count_by_bin(
    payload: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray] | None:
    map_payload = question_prefill_map_payload(payload)
    if map_payload is None:
        return None

    question_frame_maps = to_numpy_array(map_payload.get("question_frame_maps"))
    if question_frame_maps is None or question_frame_maps.ndim != 3:
        return None
    if question_frame_maps.shape[0] < 1 or question_frame_maps.shape[1] < 1 or question_frame_maps.shape[2] < 1:
        return None

    token_counts = frame_bin_token_counts(map_payload.get("frame_local_bin_spans", []))
    if token_counts is None or token_counts.shape[0] != question_frame_maps.shape[2]:
        return None

    display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
    if len(display_layer_indices) != question_frame_maps.shape[0]:
        display_layer_indices = list(range(question_frame_maps.shape[0]))

    attention_scores = np.asarray(question_frame_maps, dtype=np.float64).mean(axis=1)
    if attention_scores.shape != (len(display_layer_indices), token_counts.shape[0]):
        return None
    return display_layer_indices, attention_scores, token_counts


def pad_2d_width(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.full((values.shape[0], int(width)), np.nan, dtype=np.float64)
    padded[:, : values.shape[1]] = values
    return padded


def pad_1d_width(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.full(int(width), np.nan, dtype=np.float64)
    padded[: values.shape[0]] = values
    return padded


def compute_average_attention_score_token_count_by_bin(
    payloads: list[dict[str, Any]],
) -> tuple[list[int] | None, np.ndarray | None, np.ndarray | None, list[list[int]] | None, list[str] | None]:
    display_layer_indices: list[int] | None = None
    attention_arrays: list[np.ndarray] = []
    token_count_arrays: list[np.ndarray] = []
    common_frame_bin_slices: list[list[int]] | None = None
    common_frame_bin_labels: list[str] | None = None
    has_common_frame_axis = True
    frame_axis_layout_counts: dict[tuple[tuple[int, int], ...], int] = {}
    frame_axis_layout_labels: dict[tuple[tuple[int, int], ...], list[str]] = {}

    for payload in payloads:
        extracted = extract_attention_score_token_count_by_bin(payload)
        if extracted is None:
            continue

        payload_display_indices, attention_scores, token_counts = extracted
        if display_layer_indices is None:
            display_layer_indices = payload_display_indices
        elif payload_display_indices != display_layer_indices:
            label = make_example_label(payload, "unknown")
            print(
                f"Skipping attention score/token count average for {label}: "
                "display_layer_indices do not match the first valid payload."
            )
            continue

        map_payload = question_prefill_map_payload(payload)
        frame_bin_slices = normalize_frame_bin_slices(map_payload.get("frame_bin_slices", [])) if map_payload else []
        frame_bin_labels = [str(label) for label in map_payload.get("frame_bin_labels", [])] if map_payload else []
        if frame_bin_slices and int(frame_bin_slices[-1][1]) == attention_scores.shape[1]:
            layout_key = tuple((int(start), int(end)) for start, end in frame_bin_slices)
            frame_axis_layout_counts[layout_key] = frame_axis_layout_counts.get(layout_key, 0) + 1
            if len(frame_bin_labels) == len(frame_bin_slices):
                frame_axis_layout_labels.setdefault(layout_key, frame_bin_labels)
            if common_frame_bin_slices is None:
                common_frame_bin_slices = frame_bin_slices
                common_frame_bin_labels = frame_bin_labels
            elif frame_bin_slices != common_frame_bin_slices or frame_bin_labels != common_frame_bin_labels:
                has_common_frame_axis = False
        else:
            has_common_frame_axis = False

        attention_arrays.append(attention_scores)
        token_count_arrays.append(token_counts)

    if display_layer_indices is None or not attention_arrays:
        return None, None, None, None, None

    max_bins = max(array.shape[1] for array in attention_arrays)
    padded_attention = np.stack([pad_2d_width(array, max_bins) for array in attention_arrays], axis=0)
    padded_token_counts = np.stack([pad_1d_width(array, max_bins) for array in token_count_arrays], axis=0)

    average_attention = np.nanmean(padded_attention, axis=0)
    average_token_counts = np.nanmean(padded_token_counts, axis=0)
    if not has_common_frame_axis:
        common_frame_bin_slices = None
        common_frame_bin_labels = None
        modal_layout_key: tuple[tuple[int, int], ...] | None = None
        modal_layout_count = 0
        for layout_key, count in frame_axis_layout_counts.items():
            if not layout_key or int(layout_key[-1][1]) != max_bins:
                continue
            if count > modal_layout_count:
                modal_layout_key = layout_key
                modal_layout_count = count
        if modal_layout_key is not None and modal_layout_count * 2 > len(attention_arrays):
            common_frame_bin_slices = [[int(start), int(end)] for start, end in modal_layout_key]
            common_frame_bin_labels = frame_axis_layout_labels.get(modal_layout_key, [])
    return display_layer_indices, average_attention, average_token_counts, common_frame_bin_slices, common_frame_bin_labels


def tight_data_axis_limits(
    values: np.ndarray,
    *,
    fallback: tuple[float, float] = (0.0, 1.0),
    min_span: float = 1e-12,
    constant_pad_fraction: float = 0.05,
    clamp_nonnegative: bool = False,
    robust_percentile: float | None = None,
) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 1:
        return fallback
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if robust_percentile is not None and finite_values.size > 1:
        robust_vmax = float(np.percentile(finite_values, robust_percentile))
        if np.isfinite(robust_vmax) and robust_vmax > vmin:
            vmax = robust_vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return fallback
    if vmax <= vmin:
        padding = max(abs(vmax) * constant_pad_fraction, float(min_span))
        lower = vmin - padding
        upper = vmax + padding
    else:
        lower = vmin
        upper = vmax
    if clamp_nonnegative and vmin >= 0.0 and lower < 0.0:
        lower = 0.0
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return fallback
    return lower, upper


def tight_position_axis_limits(
    centers: np.ndarray,
    widths: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    fallback: tuple[float, float] = (0.0, 1.0),
    min_span: float = 1e-6,
) -> tuple[float, float]:
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if centers.ndim != 1 or widths.ndim != 1 or centers.shape != widths.shape:
        return fallback
    valid = np.isfinite(centers) & np.isfinite(widths) & (widths > 0.0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != centers.shape:
            return fallback
        valid &= mask
    if not np.any(valid):
        return fallback

    lower = float(np.min(centers[valid] - 0.5 * widths[valid]))
    upper = float(np.max(centers[valid] + 0.5 * widths[valid]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return fallback
    if upper <= lower:
        center = 0.5 * (lower + upper)
        padding = max(abs(center) * 0.05, float(min_span))
        lower = center - padding
        upper = center + padding
    return lower, upper


def norm_bin_edges_from_values(
    values: np.ndarray,
    *,
    num_bins: int = ATTENTION_TOKEN_COUNT_NORM_BIN_COUNT,
) -> np.ndarray | None:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 1:
        return None

    num_bins = max(1, int(num_bins))
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        padding = max(abs(vmax) * 0.05, 1.0)
        vmin -= padding
        vmax += padding
        if vmin < 0.0 <= float(np.min(finite_values)):
            vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, num_bins + 1, dtype=np.float64)


def norm_bin_edges_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    num_bins: int = ATTENTION_TOKEN_COUNT_NORM_BIN_COUNT,
) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for payload in payloads:
        map_payload = question_prefill_map_payload(payload)
        if map_payload is None:
            continue
        value_norms = to_numpy_array(payload.get("question_prefill_value_norms"))
        if value_norms is None:
            continue
        value_norms = np.asarray(value_norms, dtype=np.float64)
        if value_norms.ndim != 2 or value_norms.shape[0] < 1 or value_norms.shape[1] < 1:
            continue
        display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
        if len(display_layer_indices) != value_norms.shape[0]:
            display_layer_indices = list(range(value_norms.shape[0]))
        attention_vectors = extract_per_patch_attention_vectors(
            payload,
            display_layer_indices,
            expected_token_count=int(value_norms.shape[1]),
        )
        if attention_vectors is None:
            continue
        arrays.append(value_norms.reshape(-1))
    if not arrays:
        return None
    return norm_bin_edges_from_values(np.concatenate(arrays), num_bins=num_bins)


def get_layer_mapping_value(mapping: dict[Any, Any], layer_idx: int) -> Any:
    if int(layer_idx) in mapping:
        return mapping[int(layer_idx)]
    return mapping.get(str(int(layer_idx)))


def extract_per_patch_attention_vectors(
    payload: dict[str, Any],
    display_layer_indices: list[int],
    *,
    expected_token_count: int,
) -> np.ndarray | None:
    per_patch_payload = payload.get("question_prefill_per_patch_attention")
    if not isinstance(per_patch_payload, dict):
        return None

    rows: list[np.ndarray] = []
    for layer_idx in display_layer_indices:
        layer_items = get_layer_mapping_value(per_patch_payload, int(layer_idx))
        if not isinstance(layer_items, (list, tuple)) or not layer_items:
            return None

        frame_arrays: list[np.ndarray] = []
        for item in layer_items:
            array = to_numpy_array(item)
            if array is None:
                return None
            frame_arrays.append(np.asarray(array, dtype=np.float64).reshape(-1))
        if not frame_arrays:
            return None

        row = np.concatenate(frame_arrays)
        if row.shape[0] != int(expected_token_count):
            return None
        rows.append(row)

    if not rows:
        return None
    return np.stack(rows, axis=0)


def select_topk_attention_token_value_norm(
    attention_vectors: np.ndarray,
    value_norms: np.ndarray,
    *,
    top_k: int = ATTENTION_TOPK_TOKEN_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    attention_vectors = np.asarray(attention_vectors, dtype=np.float64)
    value_norms = np.asarray(value_norms, dtype=np.float64)
    top_k = max(1, int(top_k))
    if attention_vectors.ndim != 2 or value_norms.ndim != 2:
        return None
    if attention_vectors.shape != value_norms.shape or attention_vectors.shape[0] < 1:
        return None

    token_indices = np.full((attention_vectors.shape[0], top_k), np.nan, dtype=np.float64)
    top_attention = np.full((attention_vectors.shape[0], top_k), np.nan, dtype=np.float64)
    top_norms = np.full((attention_vectors.shape[0], top_k), np.nan, dtype=np.float64)

    for layer_pos in range(attention_vectors.shape[0]):
        layer_attention = attention_vectors[layer_pos]
        layer_norms = value_norms[layer_pos]
        valid = np.isfinite(layer_attention) & np.isfinite(layer_norms)
        if not np.any(valid):
            continue

        valid_token_indices = np.nonzero(valid)[0].astype(np.int64)
        valid_attention = layer_attention[valid]
        order = np.lexsort((valid_token_indices, -valid_attention))
        selected = order[:top_k]
        count = int(selected.shape[0])
        selected_token_indices = valid_token_indices[selected]
        token_indices[layer_pos, :count] = selected_token_indices
        top_attention[layer_pos, :count] = valid_attention[selected]
        top_norms[layer_pos, :count] = layer_norms[selected_token_indices]

    return token_indices, top_attention, top_norms


def extract_attention_score_topk_token_value_norm(
    payload: dict[str, Any],
    *,
    top_k: int = ATTENTION_TOPK_TOKEN_COUNT,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray] | None:
    map_payload = question_prefill_map_payload(payload)
    if map_payload is None:
        return None

    value_norms = to_numpy_array(payload.get("question_prefill_value_norms"))
    if value_norms is None:
        return None
    value_norms = np.asarray(value_norms, dtype=np.float64)
    if value_norms.ndim != 2 or value_norms.shape[0] < 1 or value_norms.shape[1] < 1:
        return None

    display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
    if len(display_layer_indices) != value_norms.shape[0]:
        display_layer_indices = list(range(value_norms.shape[0]))

    attention_vectors = extract_per_patch_attention_vectors(
        payload,
        display_layer_indices,
        expected_token_count=int(value_norms.shape[1]),
    )
    if attention_vectors is None or attention_vectors.shape != value_norms.shape:
        return None

    topk_values = select_topk_attention_token_value_norm(
        attention_vectors,
        value_norms,
        top_k=top_k,
    )
    if topk_values is None:
        return None
    token_indices, attention_scores, top_value_norms = topk_values
    return display_layer_indices, token_indices, attention_scores, top_value_norms


def compute_pooled_attention_score_topk_token_value_norm(
    payloads: list[dict[str, Any]],
    *,
    top_k: int = ATTENTION_TOPK_TOKEN_COUNT,
) -> tuple[list[int] | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    top_k = max(1, int(top_k))
    display_layer_indices: list[int] | None = None
    candidates_by_layer: list[list[tuple[float, float, float]]] = []

    for payload in payloads:
        extracted = extract_attention_score_topk_token_value_norm(payload, top_k=top_k)
        if extracted is None:
            continue
        payload_display_indices, token_indices, attention_scores, value_norms = extracted
        if display_layer_indices is None:
            display_layer_indices = payload_display_indices
            candidates_by_layer = [[] for _ in display_layer_indices]
        elif payload_display_indices != display_layer_indices:
            label = make_example_label(payload, "unknown")
            print(
                f"Skipping attention score top-{top_k} token/value-norm pooled plot for {label}: "
                "display_layer_indices do not match the first valid payload."
            )
            continue

        for layer_pos in range(len(display_layer_indices)):
            valid = (
                np.isfinite(token_indices[layer_pos])
                & np.isfinite(attention_scores[layer_pos])
                & np.isfinite(value_norms[layer_pos])
            )
            for token_num, attention_score, value_norm in zip(
                token_indices[layer_pos][valid],
                attention_scores[layer_pos][valid],
                value_norms[layer_pos][valid],
            ):
                candidates_by_layer[layer_pos].append(
                    (float(token_num), float(attention_score), float(value_norm))
                )

    if display_layer_indices is None or not candidates_by_layer or not any(candidates_by_layer):
        return None, None, None, None

    pooled_token_indices = np.full((len(display_layer_indices), top_k), np.nan, dtype=np.float64)
    pooled_attention = np.full((len(display_layer_indices), top_k), np.nan, dtype=np.float64)
    pooled_norms = np.full((len(display_layer_indices), top_k), np.nan, dtype=np.float64)

    for layer_pos, candidates in enumerate(candidates_by_layer):
        if not candidates:
            continue
        candidate_array = np.asarray(candidates, dtype=np.float64)
        token_values = candidate_array[:, 0]
        attention_values = candidate_array[:, 1]
        order = np.lexsort((token_values, -attention_values))
        selected = order[:top_k]
        count = int(selected.shape[0])
        pooled_token_indices[layer_pos, :count] = candidate_array[selected, 0]
        pooled_attention[layer_pos, :count] = candidate_array[selected, 1]
        pooled_norms[layer_pos, :count] = candidate_array[selected, 2]

    return display_layer_indices, pooled_token_indices, pooled_attention, pooled_norms


def expand_bin_attention_to_patch_tokens(
    attention_scores: np.ndarray,
    frame_local_bin_spans: list[list[int]] | list[tuple[int, int]],
    *,
    expected_token_count: int,
) -> np.ndarray | None:
    spans = normalize_frame_bin_slices(frame_local_bin_spans)
    if not spans or len(spans) != attention_scores.shape[1]:
        return None
    if int(spans[-1][1]) != int(expected_token_count):
        return None

    expanded = np.full(
        (attention_scores.shape[0], int(expected_token_count)),
        np.nan,
        dtype=np.float64,
    )
    for bin_idx, (start, end) in enumerate(spans):
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > int(expected_token_count):
            return None
        expanded[:, start:end] = attention_scores[:, bin_idx : bin_idx + 1]
    return expanded


def extract_attention_score_token_count_by_norm(
    payload: dict[str, Any],
    *,
    norm_bin_edges: np.ndarray | None = None,
    num_bins: int = ATTENTION_TOKEN_COUNT_NORM_BIN_COUNT,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    map_payload = question_prefill_map_payload(payload)
    if map_payload is None:
        return None

    value_norms = to_numpy_array(payload.get("question_prefill_value_norms"))
    if value_norms is None:
        return None
    value_norms = np.asarray(value_norms, dtype=np.float64)
    if value_norms.ndim != 2 or value_norms.shape[0] < 1 or value_norms.shape[1] < 1:
        return None

    display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
    if len(display_layer_indices) != value_norms.shape[0]:
        display_layer_indices = list(range(value_norms.shape[0]))

    if norm_bin_edges is None:
        norm_bin_edges = norm_bin_edges_from_values(value_norms, num_bins=num_bins)
    if norm_bin_edges is None:
        return None
    norm_bin_edges = np.asarray(norm_bin_edges, dtype=np.float64)
    if norm_bin_edges.ndim != 1 or norm_bin_edges.shape[0] < 2:
        return None
    if not np.all(np.isfinite(norm_bin_edges)) or np.any(np.diff(norm_bin_edges) <= 0.0):
        return None

    expected_token_count = int(value_norms.shape[1])
    attention_vectors = extract_per_patch_attention_vectors(
        payload,
        display_layer_indices,
        expected_token_count=expected_token_count,
    )
    if attention_vectors is None:
        return None
    if attention_vectors is None or attention_vectors.shape != value_norms.shape:
        return None

    bin_count = int(norm_bin_edges.shape[0] - 1)
    binned_attention = np.full((value_norms.shape[0], bin_count), np.nan, dtype=np.float64)
    token_counts = np.zeros((value_norms.shape[0], bin_count), dtype=np.float64)

    for layer_idx in range(value_norms.shape[0]):
        layer_norms = value_norms[layer_idx]
        layer_attention = attention_vectors[layer_idx]
        valid = np.isfinite(layer_norms) & np.isfinite(layer_attention)
        if not np.any(valid):
            continue

        bin_indices = np.searchsorted(norm_bin_edges, layer_norms[valid], side="right") - 1
        bin_indices = np.where(
            layer_norms[valid] == norm_bin_edges[-1],
            bin_count - 1,
            bin_indices,
        )
        inside = (bin_indices >= 0) & (bin_indices < bin_count)
        if not np.any(inside):
            continue
        bin_indices = bin_indices[inside].astype(np.int64)
        attention_values = layer_attention[valid][inside]

        counts = np.bincount(bin_indices, minlength=bin_count).astype(np.float64)
        sums = np.bincount(bin_indices, weights=attention_values, minlength=bin_count).astype(np.float64)
        token_counts[layer_idx] = counts
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
        means[counts <= 0.0] = np.nan
        binned_attention[layer_idx] = means

    norm_bin_centers = 0.5 * (norm_bin_edges[:-1] + norm_bin_edges[1:])
    norm_bin_widths = np.diff(norm_bin_edges)
    return display_layer_indices, norm_bin_centers, norm_bin_widths, binned_attention, token_counts


def compute_average_attention_score_token_count_by_norm(
    payloads: list[dict[str, Any]],
    *,
    num_bins: int = ATTENTION_TOKEN_COUNT_NORM_BIN_COUNT,
) -> tuple[list[int] | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    norm_bin_edges = norm_bin_edges_from_payloads(payloads, num_bins=num_bins)
    if norm_bin_edges is None:
        return None, None, None, None, None

    display_layer_indices: list[int] | None = None
    weighted_attention_sum: np.ndarray | None = None
    attention_weight_count: np.ndarray | None = None
    token_count_arrays: list[np.ndarray] = []
    norm_bin_centers: np.ndarray | None = None
    norm_bin_widths: np.ndarray | None = None

    for payload in payloads:
        extracted = extract_attention_score_token_count_by_norm(
            payload,
            norm_bin_edges=norm_bin_edges,
            num_bins=num_bins,
        )
        if extracted is None:
            continue
        payload_display_indices, centers, widths, attention_scores, token_counts = extracted
        if display_layer_indices is None:
            display_layer_indices = payload_display_indices
            weighted_attention_sum = np.zeros_like(attention_scores, dtype=np.float64)
            attention_weight_count = np.zeros_like(token_counts, dtype=np.float64)
            norm_bin_centers = centers
            norm_bin_widths = widths
        elif payload_display_indices != display_layer_indices:
            label = make_example_label(payload, "unknown")
            print(
                f"Skipping attention score/token count norm average for {label}: "
                "display_layer_indices do not match the first valid payload."
            )
            continue

        assert weighted_attention_sum is not None
        assert attention_weight_count is not None
        valid = np.isfinite(attention_scores) & np.isfinite(token_counts) & (token_counts > 0.0)
        weighted_attention_sum[valid] += attention_scores[valid] * token_counts[valid]
        attention_weight_count[valid] += token_counts[valid]
        token_count_arrays.append(token_counts)

    if display_layer_indices is None or weighted_attention_sum is None or attention_weight_count is None:
        return None, None, None, None, None
    if not token_count_arrays or norm_bin_centers is None or norm_bin_widths is None:
        return None, None, None, None, None

    average_attention = np.full_like(weighted_attention_sum, np.nan, dtype=np.float64)
    valid_average = attention_weight_count > 0.0
    average_attention[valid_average] = weighted_attention_sum[valid_average] / attention_weight_count[valid_average]
    average_token_counts = np.mean(np.stack(token_count_arrays, axis=0), axis=0)
    return display_layer_indices, norm_bin_centers, norm_bin_widths, average_attention, average_token_counts


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
    color: str = "white",
    linewidth: float = 0.5,
    alpha: float = 0.5,
    zorder: float | None = None,
) -> None:
    line_kwargs: dict[str, Any] = {
        "color": color,
        "linewidth": linewidth,
        "alpha": alpha,
    }
    if zorder is not None:
        line_kwargs["zorder"] = zorder
    for start, end in frame_bin_slices[:-1]:
        boundary = int(end) - 0.5
        if draw_x:
            ax.axvline(boundary, **line_kwargs)
        if draw_y:
            ax.axhline(boundary, **line_kwargs)


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


def apply_bin_index_ticks(ax: Any, num_bins: int, max_ticks: int = 12) -> None:
    if num_bins < 1:
        return
    if num_bins <= max_ticks:
        positions = np.arange(num_bins)
    else:
        positions = np.unique(np.linspace(0, num_bins - 1, max_ticks, dtype=np.int64))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(index)) for index in positions], rotation=90, fontsize=7)


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


def rawscale_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_rawscale{output_path.suffix}")


def heatmap_scale_label(robust_percentile: float | None) -> str:
    if robust_percentile is None:
        return "raw max scale"
    return f"p{robust_percentile:g} scale"


def title_with_scale(figure_title: str, robust_percentile: float | None) -> str:
    if robust_percentile is None:
        return f"{figure_title} (raw max scale)"
    return figure_title


def build_emphasized_heatmap_norm(
    values: np.ndarray,
    *,
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> mcolors.Normalize:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 1:
        return mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    if robust_percentile is not None:
        robust_vmax = float(np.percentile(finite_values, robust_percentile))
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
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] < 1:
        return

    finite_values = maps[np.isfinite(maps)]
    if finite_values.size < 1:
        return
    norm = build_emphasized_heatmap_norm(finite_values, robust_percentile=robust_percentile)

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

    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.98)
    if mode == "question_frame":
        fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.10, wspace=0.30, hspace=0.8)
    else:
        fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.10, wspace=0.30, hspace=0.45)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.82])
        fig.colorbar(
            mappable,
            cax=cbar_ax,
            label=f"Mean Attention Score ({heatmap_scale_label(robust_percentile)})",
        )
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
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 3 or maps.shape[0] < 1:
        return

    finite_values = maps[np.isfinite(maps)]
    if finite_values.size < 1:
        return
    norm = build_emphasized_heatmap_norm(finite_values, robust_percentile=robust_percentile)

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

    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.14, wspace=0.30, hspace=0.75)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.90, 0.14, 0.015, 0.76])
        fig.colorbar(
            mappable,
            cax=cbar_ax,
            label=f"Mean Attention Score ({heatmap_scale_label(robust_percentile)})",
        )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def padded_axis_limits(
    values: np.ndarray,
    *,
    fallback: tuple[float, float] = (0.0, 1.0),
    pad_fraction: float = 0.12,
    min_padding: float = 1e-12,
    constant_pad_fraction: float = 0.05,
    clamp_nonnegative: bool = False,
) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 1:
        return fallback
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    value_range = vmax - vmin
    if abs(value_range) < 1e-15:
        padding = max(abs(vmax) * constant_pad_fraction, min_padding)
    else:
        padding = max(value_range * pad_fraction, min_padding)
    lower = vmin - padding
    upper = vmax + padding
    if clamp_nonnegative and vmin >= 0.0 and lower < 0.0:
        lower = 0.0
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return fallback
    return lower, upper


def render_attention_score_token_count_by_bin_panels(
    attention_scores: np.ndarray,
    token_counts: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    frame_bin_slices: list[list[int]] | list[tuple[int, int]] | None = None,
    frame_bin_labels: list[str] | None = None,
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
    show_token_count_line: bool = True,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    attention_scores = np.asarray(attention_scores, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.float64)
    if attention_scores.ndim != 2 or token_counts.ndim != 1:
        return
    if attention_scores.shape[0] < 1 or attention_scores.shape[1] < 1:
        return
    if token_counts.shape[0] != attention_scores.shape[1]:
        return
    if len(display_layer_indices) != attention_scores.shape[0]:
        display_layer_indices = list(range(attention_scores.shape[0]))

    num_bins = int(attention_scores.shape[1])
    x_positions = np.arange(num_bins, dtype=np.float64)
    usable_frame_slices = normalize_frame_bin_slices(frame_bin_slices) if frame_bin_slices is not None else []
    if not usable_frame_slices or int(usable_frame_slices[-1][1]) != num_bins:
        usable_frame_slices = []
    if frame_bin_labels is None or len(frame_bin_labels) != len(usable_frame_slices):
        frame_bin_labels = []

    max_cols = 4
    num_panels = attention_scores.shape[0]
    num_cols = min(num_panels, max_cols)
    num_rows = (num_panels + max_cols - 1) // max_cols
    panel_width = 4.8
    panel_height = 3.7
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    line_handle = None
    bar_handle = None
    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        layer_values = attention_scores[panel_idx]

        bar_ax = ax.twinx() if show_token_count_line else ax
        bars = bar_ax.bar(
            x_positions,
            layer_values,
            width=0.92,
            align="center",
            color="#a98bd0",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.95,
            label="Average Attention Score",
        )
        if bar_handle is None and len(bars) > 0:
            bar_handle = bars[0]
        bar_ax.set_ylim(
            *tight_data_axis_limits(
                layer_values,
                fallback=(0.0, 1.0),
                min_span=1e-12,
                clamp_nonnegative=True,
                robust_percentile=robust_percentile,
            )
        )
        if show_token_count_line:
            if col_idx == num_cols - 1:
                bar_ax.set_ylabel("Average Attention Score")

            ax.set_zorder(bar_ax.get_zorder() + 1)
            ax.patch.set_visible(False)
            (line,) = ax.plot(
                x_positions,
                token_counts,
                color="#e6ad00",
                linewidth=1.8,
                marker="o",
                markersize=3.0,
                label="Avg # of Tokens",
                zorder=5,
            )
            if line_handle is None:
                line_handle = line
            ax.set_ylim(
                *tight_data_axis_limits(
                    token_counts,
                    fallback=(0.0, 1.0),
                    min_span=0.5,
                    constant_pad_fraction=0.0,
                    robust_percentile=robust_percentile,
                )
            )
            if col_idx == 0:
                ax.set_ylabel("Avg # of Tokens")
        elif col_idx == 0:
            ax.set_ylabel("Average Attention Score")
        ax.set_xlim(-0.5, num_bins - 0.5)
        ax.set_title(f"Layer {int(display_layer_indices[panel_idx])}")
        ax.set_xlabel("Frame Token Bin")
        apply_bin_index_ticks(ax, num_bins)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if usable_frame_slices:
            annotate_frame_boundaries(
                ax,
                usable_frame_slices,
                draw_x=True,
                draw_y=False,
                color=ATTENTION_TOKEN_COUNT_FRAME_SEPARATOR_COLOR,
                linewidth=0.7,
                alpha=0.85,
                zorder=4,
            )

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    if show_token_count_line and line_handle is not None and bar_handle is not None:
        fig.legend(
            [line_handle, bar_handle],
            ["Avg # of Tokens", "Average Attention Score"],
            loc="upper center",
            ncol=2,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.965),
        )
    elif not show_token_count_line and bar_handle is not None:
        fig.legend(
            [bar_handle],
            ["Average Attention Score"],
            loc="upper center",
            ncol=1,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.965),
        )
    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.995)
    fig.subplots_adjust(
        left=0.06,
        right=0.92,
        top=0.88,
        bottom=0.10,
        wspace=0.36,
        hspace=ATTENTION_TOKEN_COUNT_PANEL_HSPACE,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_attention_score_token_count_by_norm_panels(
    attention_scores: np.ndarray,
    token_counts: np.ndarray,
    norm_bin_centers: np.ndarray,
    norm_bin_widths: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    attention_scores = np.asarray(attention_scores, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.float64)
    norm_bin_centers = np.asarray(norm_bin_centers, dtype=np.float64)
    norm_bin_widths = np.asarray(norm_bin_widths, dtype=np.float64)
    if attention_scores.ndim != 2:
        return
    if token_counts.ndim == 1:
        token_counts = np.repeat(token_counts[None, :], attention_scores.shape[0], axis=0)
    if token_counts.ndim != 2:
        return
    if attention_scores.shape[0] < 1 or attention_scores.shape[1] < 1:
        return
    if token_counts.shape != attention_scores.shape:
        return
    if norm_bin_centers.shape[0] != attention_scores.shape[1]:
        return
    if norm_bin_widths.shape[0] != attention_scores.shape[1]:
        return
    if len(display_layer_indices) != attention_scores.shape[0]:
        display_layer_indices = list(range(attention_scores.shape[0]))

    max_cols = 4
    num_panels = attention_scores.shape[0]
    num_cols = min(num_panels, max_cols)
    num_rows = (num_panels + max_cols - 1) // max_cols
    panel_width = 4.8
    panel_height = 3.7
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    x_axis_limits = tight_position_axis_limits(norm_bin_centers, norm_bin_widths)
    line_handle = None
    bar_handle = None
    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        layer_attention = attention_scores[panel_idx]
        layer_counts = token_counts[panel_idx]

        ax2 = ax.twinx()
        valid_bar = np.isfinite(layer_attention)
        bars = ax2.bar(
            norm_bin_centers[valid_bar],
            layer_attention[valid_bar],
            width=norm_bin_widths[valid_bar] * 0.94,
            align="center",
            color="#a98bd0",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.95,
            label="Average Attention Score",
        )
        if bar_handle is None and len(bars) > 0:
            bar_handle = bars[0]
        ax2.set_ylim(
            *tight_data_axis_limits(
                layer_attention[valid_bar],
                fallback=(0.0, 1.0),
                min_span=1e-12,
                clamp_nonnegative=True,
                robust_percentile=robust_percentile,
            )
        )
        if col_idx == num_cols - 1:
            ax2.set_ylabel("Average Attention Score")

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        (line,) = ax.plot(
            norm_bin_centers,
            layer_counts,
            color="#e6ad00",
            linewidth=1.8,
            marker="o",
            markersize=3.0,
            label="Avg # of Tokens",
            zorder=5,
        )
        if line_handle is None:
            line_handle = line
        ax.set_xlim(*x_axis_limits)
        ax.set_ylim(
            *tight_data_axis_limits(
                layer_counts,
                fallback=(0.0, 1.0),
                min_span=0.5,
                clamp_nonnegative=True,
                robust_percentile=robust_percentile,
            )
        )
        ax.set_title(f"Layer {int(display_layer_indices[panel_idx])}")
        ax.set_xlabel("L2 Norm of ViT Token")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if col_idx == 0:
            ax.set_ylabel("Avg # of Tokens")

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    if line_handle is not None and bar_handle is not None:
        fig.legend(
            [line_handle, bar_handle],
            ["Avg # of Tokens", "Average Attention Score"],
            loc="upper center",
            ncol=2,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.965),
        )
    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.995)
    fig.subplots_adjust(
        left=0.06,
        right=0.92,
        top=0.88,
        bottom=0.10,
        wspace=0.36,
        hspace=ATTENTION_TOKEN_COUNT_PANEL_HSPACE,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_attention_score_topk_token_value_norm_panels(
    token_indices: np.ndarray,
    attention_scores: np.ndarray,
    value_norms: np.ndarray,
    display_layer_indices: list[int],
    output_path: Path,
    *,
    figure_title: str,
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    token_indices = np.asarray(token_indices, dtype=np.float64)
    attention_scores = np.asarray(attention_scores, dtype=np.float64)
    value_norms = np.asarray(value_norms, dtype=np.float64)
    if token_indices.ndim != 2 or attention_scores.ndim != 2 or value_norms.ndim != 2:
        return
    if token_indices.shape != attention_scores.shape or token_indices.shape != value_norms.shape:
        return
    if attention_scores.shape[0] < 1 or attention_scores.shape[1] < 1:
        return
    if len(display_layer_indices) != attention_scores.shape[0]:
        display_layer_indices = list(range(attention_scores.shape[0]))

    max_cols = 4
    num_panels = attention_scores.shape[0]
    num_cols = min(num_panels, max_cols)
    num_rows = (num_panels + max_cols - 1) // max_cols
    panel_width = 4.8
    panel_height = 3.9
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(panel_width * num_cols, panel_height * num_rows + 1.0),
        squeeze=False,
    )

    x_positions = np.arange(attention_scores.shape[1], dtype=np.float64)
    line_handle = None
    bar_handle = None
    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        layer_tokens = token_indices[panel_idx]
        layer_attention = attention_scores[panel_idx]
        layer_norms = value_norms[panel_idx]

        ax2 = ax.twinx()
        valid_bar = np.isfinite(layer_tokens) & np.isfinite(layer_attention)
        bars = ax2.bar(
            x_positions[valid_bar],
            layer_attention[valid_bar],
            width=0.78,
            align="center",
            color="#a98bd0",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.95,
            label="Attention Score",
        )
        if bar_handle is None and len(bars) > 0:
            bar_handle = bars[0]
        ax2.set_ylim(
            *tight_data_axis_limits(
                layer_attention[valid_bar],
                fallback=(0.0, 1.0),
                min_span=1e-12,
                clamp_nonnegative=True,
                robust_percentile=robust_percentile,
            )
        )
        if col_idx == num_cols - 1:
            ax2.set_ylabel("Attention Score")

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        valid_line = np.isfinite(layer_tokens) & np.isfinite(layer_norms)
        (line,) = ax.plot(
            x_positions[valid_line],
            layer_norms[valid_line],
            color="#008c8c",
            linewidth=1.8,
            marker="o",
            markersize=3.0,
            label="Value Norm",
            zorder=5,
        )
        if line_handle is None:
            line_handle = line
        ax.set_xlim(-0.5, attention_scores.shape[1] - 0.5)
        ax.set_ylim(
            *tight_data_axis_limits(
                layer_norms[valid_line],
                fallback=(0.0, 1.0),
                min_span=1e-6,
                clamp_nonnegative=True,
                robust_percentile=robust_percentile,
            )
        )
        ax.set_title(f"Layer {int(display_layer_indices[panel_idx])}")
        ax.set_xlabel("Token Num (sorted by attention score)")
        tick_labels = [
            str(int(token_num)) if np.isfinite(token_num) else ""
            for token_num in layer_tokens
        ]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if col_idx == 0:
            ax.set_ylabel("Value Norm")

    for col_idx in range(num_panels % max_cols or max_cols, max_cols):
        if num_rows > 1 or num_panels < max_cols:
            last_row = num_rows - 1
            if last_row < axes.shape[0] and col_idx < axes.shape[1]:
                axes[last_row][col_idx].set_visible(False)

    if line_handle is not None and bar_handle is not None:
        fig.legend(
            [line_handle, bar_handle],
            ["Value Norm", "Attention Score"],
            loc="upper center",
            ncol=2,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.965),
        )
    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.995)
    fig.subplots_adjust(
        left=0.06,
        right=0.92,
        top=0.88,
        bottom=0.14,
        wspace=0.36,
        hspace=ATTENTION_TOKEN_COUNT_PANEL_HSPACE,
    )
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

    frame_frame_path = plots_dir / "question_prefill_frame_frame_maps_average.png"
    frame_frame_paths = (frame_frame_path, rawscale_output_path(frame_frame_path))
    if not skip_existing_plot_set(frame_frame_paths):
        display_layers, frame_frame_average = compute_average_question_prefill_maps(
            available_payloads,
            field_name="frame_frame_maps",
            target_shape=QUESTION_PREFILL_FRAME_FRAME_AVERAGE_SHAPE,
        )
        if display_layers is not None and frame_frame_average is not None:
            render_question_prefill_map_panels(
                frame_frame_average,
                display_layers,
                frame_frame_path,
                figure_title="Question Prefill Frame↔Frame Maps (Saved Examples Average)",
                x_label="Relative Frame-Bin Position",
                y_label="Relative Frame-Bin Position",
                mode="frame_frame",
            )
            render_question_prefill_map_panels(
                frame_frame_average,
                display_layers,
                rawscale_output_path(frame_frame_path),
                figure_title="Question Prefill Frame↔Frame Maps (Saved Examples Average)",
                x_label="Relative Frame-Bin Position",
                y_label="Relative Frame-Bin Position",
                mode="frame_frame",
                robust_percentile=None,
            )

    question_frame_path = plots_dir / "question_prefill_question_frame_maps_average.png"
    question_frame_paths = (question_frame_path, rawscale_output_path(question_frame_path))
    if not skip_existing_plot_set(question_frame_paths):
        display_layers, question_frame_average = compute_average_question_prefill_maps(
            available_payloads,
            field_name="question_frame_maps",
            target_shape=QUESTION_PREFILL_QUESTION_FRAME_AVERAGE_SHAPE,
        )
        if display_layers is not None and question_frame_average is not None:
            render_question_prefill_map_panels(
                question_frame_average,
                display_layers,
                question_frame_path,
                figure_title="Question Prefill Question→Frame Maps (Saved Examples Average)",
                x_label="Relative Frame-Bin Position",
                y_label="Relative Question-Bin Position",
                mode="question_frame",
            )
            render_question_prefill_map_panels(
                question_frame_average,
                display_layers,
                rawscale_output_path(question_frame_path),
                figure_title="Question Prefill Question→Frame Maps (Saved Examples Average)",
                x_label="Relative Frame-Bin Position",
                y_label="Relative Question-Bin Position",
                mode="question_frame",
                robust_percentile=None,
            )


def plot_question_prefill_attention_score_token_count_by_bin_average(
    payloads: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    plots_dir = ensure_dir(plots_dir)
    output_path = plots_dir / ATTENTION_TOKEN_COUNT_BIN_AVERAGE_PLOT_FILENAME
    output_paths = (output_path, rawscale_output_path(output_path))
    if skip_existing_plot_set(output_paths):
        return

    (
        display_layer_indices,
        average_attention_scores,
        average_token_counts,
        frame_bin_slices,
        frame_bin_labels,
    ) = compute_average_attention_score_token_count_by_bin(payloads)
    if display_layer_indices is None or average_attention_scores is None or average_token_counts is None:
        print(
            "Skipping question-prefill attention score/token count average plot: "
            "no saved example payload includes question_frame_maps with frame_local_bin_spans."
        )
        return

    for path, robust_percentile in ((output_path, ROBUST_HEATMAP_PERCENTILE), (rawscale_output_path(output_path), None)):
        render_attention_score_token_count_by_bin_panels(
            average_attention_scores,
            average_token_counts,
            display_layer_indices,
            path,
            figure_title="Question Prefill Attention Score and Patch Tokens by Bin (Saved Examples Average)",
            frame_bin_slices=frame_bin_slices,
            frame_bin_labels=frame_bin_labels,
            robust_percentile=robust_percentile,
            show_token_count_line=False,
        )


def plot_question_prefill_attention_score_token_count_by_norm_average(
    payloads: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    plots_dir = ensure_dir(plots_dir)
    output_path = plots_dir / ATTENTION_TOKEN_COUNT_NORM_AVERAGE_PLOT_FILENAME
    output_paths = (output_path, rawscale_output_path(output_path))
    if skip_existing_plot_set(output_paths):
        return

    (
        display_layer_indices,
        norm_bin_centers,
        norm_bin_widths,
        average_attention_scores,
        average_token_counts,
    ) = compute_average_attention_score_token_count_by_norm(payloads)
    if (
        display_layer_indices is None
        or norm_bin_centers is None
        or norm_bin_widths is None
        or average_attention_scores is None
        or average_token_counts is None
    ):
        print(
            "Skipping question-prefill attention score/token count norm plot: "
            "no saved example payload includes compatible value norms and attention scores."
        )
        return

    for path, robust_percentile in ((output_path, ROBUST_HEATMAP_PERCENTILE), (rawscale_output_path(output_path), None)):
        render_attention_score_token_count_by_norm_panels(
            average_attention_scores,
            average_token_counts,
            norm_bin_centers,
            norm_bin_widths,
            display_layer_indices,
            path,
            figure_title="Question Prefill Attention Score and Tokens by L2 Norm (Saved Examples Average)",
            robust_percentile=robust_percentile,
        )


def plot_question_prefill_attention_score_topk_token_value_norm_pooled(
    payloads: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    plots_dir = ensure_dir(plots_dir)
    output_path = plots_dir / ATTENTION_TOPK_TOKEN_VALUE_NORM_POOLED_PLOT_FILENAME
    output_paths = (output_path, rawscale_output_path(output_path))
    if skip_existing_plot_set(output_paths):
        return

    (
        display_layer_indices,
        token_indices,
        attention_scores,
        value_norms,
    ) = compute_pooled_attention_score_topk_token_value_norm(payloads)
    if (
        display_layer_indices is None
        or token_indices is None
        or attention_scores is None
        or value_norms is None
    ):
        print(
            "Skipping question-prefill top-20 attention token/value-norm pooled plot: "
            "no saved example payload includes compatible value norms and token-wise attention scores."
        )
        return

    for path, robust_percentile in ((output_path, ROBUST_HEATMAP_PERCENTILE), (rawscale_output_path(output_path), None)):
        render_attention_score_topk_token_value_norm_panels(
            token_indices,
            attention_scores,
            value_norms,
            display_layer_indices,
            path,
            figure_title="Question Prefill Top-20 Attention Tokens and Value Norms (Saved Examples Pooled)",
            robust_percentile=robust_percentile,
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
    robust_percentile: float | None = ROBUST_HEATMAP_PERCENTILE,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

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
        figsize=(panel_width * num_cols, panel_height * num_rows + 0.5),
        squeeze=False,
    )

    finite_values = np.concatenate([
        np.asarray(matrix, dtype=np.float64).ravel()
        for matrix in per_frame_attention[:num_panels]
        if np.asarray(matrix).size > 0
    ]) if per_frame_attention else np.empty(0, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    norm = build_emphasized_heatmap_norm(finite_values, robust_percentile=robust_percentile)
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

    fig.suptitle(title_with_scale(figure_title, robust_percentile), y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.94, bottom=0.04, wspace=0.10, hspace=0.05)
    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.10, 0.012, 0.80])
        fig.colorbar(
            mappable,
            cax=cbar_ax,
            label=f"Layer {int(layer_idx)} Mean Attention ({heatmap_scale_label(robust_percentile)})",
        )
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
    sink_bin_index: int | None = None,
) -> None:
    output_path = Path(output_path)
    if skip_existing_plot(output_path):
        return

    value_norms = np.asarray(value_norms, dtype=np.float32)
    if value_norms.ndim != 2 or value_norms.shape[0] < 1 or value_norms.shape[1] < 1:
        return
    if not frame_bin_slices:
        return
    total_bins = int(frame_bin_slices[-1][1])
    if total_bins < 1:
        return
    is_bin_level = value_norms.shape[1] == total_bins
    is_legacy_frame_level = value_norms.shape[1] == len(frame_bin_slices)
    if not is_bin_level and not is_legacy_frame_level:
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

    if is_bin_level:
        centers = np.arange(total_bins, dtype=np.float64)
        widths = np.ones(total_bins, dtype=np.float64) * 0.95
    else:
        centers = frame_slice_centers(frame_bin_slices)
        widths = np.asarray(
            [max(1, int(end) - int(start)) for start, end in frame_bin_slices],
            dtype=np.float64,
        )
    recent_set = {int(idx) for idx in (recent_frame_indices_within_attention or [])}
    sink_bin_index = sink_bin_index if sink_bin_index is not None and sink_bin_index >= 0 else None

    for panel_idx in range(num_panels):
        row_idx, col_idx = divmod(panel_idx, max_cols)
        ax = axes[row_idx][col_idx]
        layer_values = value_norms[panel_idx]
        if is_bin_level:
            bar_colors = []
            for bin_idx in range(layer_values.shape[0]):
                frame_idx = frame_index_for_bin(bin_idx, frame_bin_slices)
                bar_colors.append("#cb181d" if frame_idx in recent_set else "#4f81bd")
            bars = ax.bar(
                centers,
                layer_values,
                width=widths,
                align="center",
                color=bar_colors,
                edgecolor="white",
                linewidth=0.25,
            )
            if sink_bin_index is not None and sink_bin_index < len(bars):
                bars[sink_bin_index].set_color("#31a354")
                bars[sink_bin_index].set_edgecolor("black")
                bars[sink_bin_index].set_linewidth(0.9)
                ax.axvline(sink_bin_index, color="black", linewidth=0.6, alpha=0.75)
        else:
            bar_colors = [
                "#cb181d" if frame_idx in recent_set else "#4f81bd"
                for frame_idx in range(layer_values.shape[0])
            ]
            ax.bar(
                centers,
                layer_values,
                width=widths,
                align="center",
                color=bar_colors,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlim(-0.5, total_bins - 0.5)
        annotate_frame_boundaries(ax, frame_bin_slices, draw_x=True, draw_y=False)
        apply_frame_ticks(ax, frame_bin_slices, frame_bin_labels, axis="x")
        ax.set_title(f"Layer {int(display_layer_indices[panel_idx])}")
        ax.set_xlabel("Frame Token Bin" if is_bin_level else "Frame Index")
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
        base_heatmap_path = example_dir / f"{metric_name}_heatmap.png"
        for output_path, robust_percentile in (
            (base_heatmap_path, ROBUST_HEATMAP_PERCENTILE),
            (rawscale_output_path(base_heatmap_path), None),
        ):
            if skip_existing_plot(output_path):
                continue

            fig, ax = plt.subplots(figsize=(11, 5))
            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap="magma",
                norm=build_emphasized_heatmap_norm(matrix, robust_percentile=robust_percentile),
            )
            ax.set_title(title_with_scale(f"{title} Attention Heatmap: {example_key}", robust_percentile))
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
            selected_recent = set(
                int(index) for index in metrics[metric_name].get("recent_frame_indices_within_attention", [])
            )
            for local_index in selected_recent:
                ax.axvline(local_index, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
            fig.colorbar(
                im,
                ax=ax,
                fraction=0.025,
                pad=0.04,
                label=f"Attention Score ({heatmap_scale_label(robust_percentile)})",
            )
            fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.12)
            fig.savefig(output_path, dpi=200)
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
        frame_local_bin_spans = [
            [int(start), int(end)]
            for start, end in map_payload.get("frame_local_bin_spans", [])
        ]
        frame_bin_labels = [str(label) for label in map_payload.get("frame_bin_labels", [])]
        question_bin_labels = [str(label) for label in map_payload.get("question_bin_labels", [])]

        frame_frame_maps = to_numpy_array(map_payload.get("frame_frame_maps"))
        if frame_frame_maps is not None:
            frame_frame_path = example_dir / "question_prefill_frame_frame_maps.png"
            render_question_prefill_map_panels(
                frame_frame_maps,
                display_layer_indices,
                frame_frame_path,
                figure_title=f"Question Prefill Frame↔Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Frame Index",
                mode="frame_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
            )
            render_question_prefill_map_panels(
                frame_frame_maps,
                display_layer_indices,
                rawscale_output_path(frame_frame_path),
                figure_title=f"Question Prefill Frame↔Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Frame Index",
                mode="frame_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
                robust_percentile=None,
            )

        question_frame_maps = to_numpy_array(map_payload.get("question_frame_maps"))
        if question_frame_maps is not None:
            question_frame_path = example_dir / "question_prefill_question_frame_maps.png"
            render_question_prefill_map_panels(
                question_frame_maps,
                display_layer_indices,
                question_frame_path,
                figure_title=f"Question Prefill Question→Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Question Token Bin",
                mode="question_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
                question_bin_labels=question_bin_labels,
            )
            render_question_prefill_map_panels(
                question_frame_maps,
                display_layer_indices,
                rawscale_output_path(question_frame_path),
                figure_title=f"Question Prefill Question→Frame Maps: {example_label}",
                x_label="Frame Index",
                y_label="Question Token Bin",
                mode="question_frame",
                frame_bin_slices=frame_bin_slices,
                frame_bin_labels=frame_bin_labels,
                question_bin_labels=question_bin_labels,
                robust_percentile=None,
            )

        bin_plot_path = example_dir / ATTENTION_TOKEN_COUNT_BIN_PLOT_FILENAME
        if not skip_existing_plot_set((bin_plot_path, rawscale_output_path(bin_plot_path))):
            attention_token_count_plot = extract_attention_score_token_count_by_bin(payload)
            if attention_token_count_plot is None:
                if not frame_local_bin_spans:
                    print(
                        f"Skipping attention score/token count bin plot for {example_key}: "
                        "saved example payload is missing frame_local_bin_spans."
                    )
                else:
                    print(
                        f"Skipping attention score/token count bin plot for {example_key}: "
                        "question_frame_maps and frame_local_bin_spans are incompatible."
                    )
            else:
                bin_display_layer_indices, attention_scores, token_counts = attention_token_count_plot
                for path, robust_percentile in (
                    (bin_plot_path, ROBUST_HEATMAP_PERCENTILE),
                    (rawscale_output_path(bin_plot_path), None),
                ):
                    render_attention_score_token_count_by_bin_panels(
                        attention_scores,
                        token_counts,
                        bin_display_layer_indices,
                        path,
                        figure_title=f"Question Prefill Attention Score and Patch Tokens by Bin: {example_label}",
                        frame_bin_slices=frame_bin_slices,
                        frame_bin_labels=frame_bin_labels,
                        robust_percentile=robust_percentile,
                        show_token_count_line=False,
                    )

        norm_plot_path = example_dir / ATTENTION_TOKEN_COUNT_NORM_PLOT_FILENAME
        if not skip_existing_plot_set((norm_plot_path, rawscale_output_path(norm_plot_path))):
            attention_token_count_norm_plot = extract_attention_score_token_count_by_norm(payload)
            if attention_token_count_norm_plot is None:
                if to_numpy_array(payload.get("question_prefill_value_norms")) is None:
                    print(
                        f"Skipping attention score/token count norm plot for {example_key}: "
                        "saved example payload is missing question_prefill_value_norms."
                    )
                else:
                    print(
                        f"Skipping attention score/token count norm plot for {example_key}: "
                        "value norms and attention scores are incompatible."
                    )
            else:
                (
                    norm_display_layer_indices,
                    norm_bin_centers,
                    norm_bin_widths,
                    norm_attention_scores,
                    norm_token_counts,
                ) = attention_token_count_norm_plot
                for path, robust_percentile in (
                    (norm_plot_path, ROBUST_HEATMAP_PERCENTILE),
                    (rawscale_output_path(norm_plot_path), None),
                ):
                    render_attention_score_token_count_by_norm_panels(
                        norm_attention_scores,
                        norm_token_counts,
                        norm_bin_centers,
                        norm_bin_widths,
                        norm_display_layer_indices,
                        path,
                        figure_title=f"Question Prefill Attention Score and Tokens by L2 Norm: {example_label}",
                        robust_percentile=robust_percentile,
                    )

        topk_plot_path = example_dir / ATTENTION_TOPK_TOKEN_VALUE_NORM_PLOT_FILENAME
        if not skip_existing_plot_set((topk_plot_path, rawscale_output_path(topk_plot_path))):
            attention_topk_plot = extract_attention_score_topk_token_value_norm(payload)
            if attention_topk_plot is None:
                if to_numpy_array(payload.get("question_prefill_value_norms")) is None:
                    print(
                        f"Skipping top-20 attention token/value-norm plot for {example_key}: "
                        "saved example payload is missing question_prefill_value_norms."
                    )
                elif not isinstance(payload.get("question_prefill_per_patch_attention"), dict):
                    print(
                        f"Skipping top-20 attention token/value-norm plot for {example_key}: "
                        "saved example payload is missing question_prefill_per_patch_attention."
                    )
                else:
                    print(
                        f"Skipping top-20 attention token/value-norm plot for {example_key}: "
                        "value norms and attention scores are incompatible."
                    )
            else:
                (
                    topk_display_layer_indices,
                    topk_token_indices,
                    topk_attention_scores,
                    topk_value_norms,
                ) = attention_topk_plot
                for path, robust_percentile in (
                    (topk_plot_path, ROBUST_HEATMAP_PERCENTILE),
                    (rawscale_output_path(topk_plot_path), None),
                ):
                    render_attention_score_topk_token_value_norm_panels(
                        topk_token_indices,
                        topk_attention_scores,
                        topk_value_norms,
                        topk_display_layer_indices,
                        path,
                        figure_title=f"Question Prefill Top-20 Attention Tokens and Value Norms: {example_label}",
                        robust_percentile=robust_percentile,
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
                overlay_path = example_dir / f"question_prefill_attention_overlay_layer{int(layer_idx)}.png"
                if skip_existing_plot_set((overlay_path, rawscale_output_path(overlay_path))):
                    continue
                per_frame_arrays = [
                    np.asarray(to_numpy_array(matrix), dtype=np.float32)
                    for matrix in per_frame_tensors
                ]
                overlay_title = f"Question Prefill Attention Overlay (Layer {int(layer_idx)}): {example_label}"
                render_per_layer_attention_overlay(
                    frame_arrays,
                    per_frame_arrays,
                    int(layer_idx),
                    overlay_path,
                    figure_title=overlay_title,
                    frame_indices=absolute_frame_indices,
                    recent_frame_indices=recent_absolute_indices,
                )
                render_per_layer_attention_overlay(
                    frame_arrays,
                    per_frame_arrays,
                    int(layer_idx),
                    rawscale_output_path(overlay_path),
                    figure_title=overlay_title,
                    frame_indices=absolute_frame_indices,
                    recent_frame_indices=recent_absolute_indices,
                    robust_percentile=None,
                )
        elif display_layer_indices and (analysis_frames is None or per_patch_attention is None):
            print(
                f"Skipping per-layer attention overlay for {example_key}: "
                "saved example payload is missing analysis_frames or "
                "question_prefill_per_patch_attention."
            )

        sink_payload = question_prefill_sink_bin_token_payload(payload)
        sink_bin_index = to_int_or_none(sink_payload.get("sink_bin_index")) if sink_payload is not None else None

        value_norm_path = example_dir / "question_prefill_value_norms.png"
        if not skip_existing_plot(value_norm_path):
            value_norms = to_numpy_array(payload.get("question_prefill_value_norms"))
            if value_norms is not None and value_norms.ndim == 2 and display_layer_indices:
                plot_value_norms = value_norms
                title_suffix = ""
                if frame_local_bin_spans:
                    total_patch_tokens = int(frame_local_bin_spans[-1][1])
                    total_frame_bins = int(frame_bin_slices[-1][1]) if frame_bin_slices else 0
                    if value_norms.shape[1] == total_patch_tokens and total_patch_tokens != total_frame_bins:
                        pooled = mean_pool_value_norms_by_spans(value_norms, frame_local_bin_spans)
                        if pooled is not None:
                            plot_value_norms = pooled
                            title_suffix = " (patch values pooled to bins)"
                render_value_norm_bar_panels(
                    plot_value_norms,
                    display_layer_indices,
                    value_norm_path,
                    figure_title=f"Question Prefill V-Norms{title_suffix}: {example_label}",
                    frame_bin_slices=frame_bin_slices,
                    frame_bin_labels=frame_bin_labels,
                    recent_frame_indices_within_attention=recent_attention_indices,
                    sink_bin_index=sink_bin_index,
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

    sink_path = example_dir / "question_prefill_sink_bin_token_attention.png"
    if skip_existing_plot_set((sink_path, rawscale_output_path(sink_path))):
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
    sink_title = (
        f"Question Prefill Sink-Bin Token Attention: {example_label} "
        f"(bin={sink_bin_index}, frame={sink_frame_label}, tokens={sink_token_start}:{sink_token_end})"
    )
    render_question_prefill_sink_bin_token_panels(
        sink_maps,
        sink_display_layer_indices,
        sink_path,
        figure_title=sink_title,
        question_bin_labels=sink_question_bin_labels,
        token_labels=sink_token_labels,
    )
    render_question_prefill_sink_bin_token_panels(
        sink_maps,
        sink_display_layer_indices,
        rawscale_output_path(sink_path),
        figure_title=sink_title,
        question_bin_labels=sink_question_bin_labels,
        token_labels=sink_token_labels,
        robust_percentile=None,
    )


def load_example_payloads(examples_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for example_path in sorted(examples_dir.glob("*.pt")):
        try:
            payload = torch.load(example_path, map_location="cpu", weights_only=False)
        except TypeError:
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


def record_recent4_percentile_vector(record: dict[str, Any], metric_name: str) -> np.ndarray | None:
    metric = record.get("metrics", {}).get(metric_name, {})
    if not isinstance(metric, dict):
        return None
    return recent4_percentile_vector(
        metric,
        percentile_field="mean_percentile",
        index_field="recent_frame_indices_within_attention",
    )


def collect_recent4_percentile_stats(
    records: list[dict[str, Any]],
    metric_name: str,
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
        values = record_recent4_percentile_vector(record, metric_name)
        if values is not None:
            rows.append(values)

    if not rows:
        return None

    stacked = np.vstack(rows).astype(np.float64, copy=False)
    return stacked.mean(axis=0), stacked.std(axis=0)


def extract_recent4_per_frame_bar_data(
    records: list[dict[str, Any]],
    metric_name: str,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    labels: list[str] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    group_types: list[str] = []

    for task in REAL_TIME_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        stats = collect_recent4_percentile_stats(records, metric_name, task_name=task)
        if stats is None:
            continue
        mean_row, std_row = stats
        labels.append(task)
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("realtime")

    rt_stats = collect_recent4_percentile_stats(records, metric_name, split_name="realtime")
    if rt_stats is not None:
        mean_row, std_row = rt_stats
        labels.append("Real-time\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("realtime_avg")

    for task in BACKWARD_TASKS:
        if task in EXCLUDED_PLOT_TASKS:
            continue
        stats = collect_recent4_percentile_stats(records, metric_name, task_name=task)
        if stats is None:
            continue
        mean_row, std_row = stats
        labels.append(task)
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("backward")

    bw_stats = collect_recent4_percentile_stats(records, metric_name, split_name="backward")
    if bw_stats is not None:
        mean_row, std_row = bw_stats
        labels.append("Backward\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("backward_avg")

    total_stats = collect_recent4_percentile_stats(records, metric_name)
    if total_stats is not None:
        mean_row, std_row = total_stats
        labels.append("Total\nAvg")
        means.append(mean_row)
        stds.append(std_row)
        group_types.append("total_avg")

    return labels, means, stds, group_types


def lighten_bar_color(color: str, white_fraction: float) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    white = np.ones(3, dtype=np.float64)
    mixed = rgb * (1.0 - white_fraction) + white * white_fraction
    return mcolors.to_hex(mixed)


def recent_frame_colors(group_type: str) -> list[str]:
    base_color = BAR_PLOT_COLOR_MAP[group_type]
    return [
        lighten_bar_color(base_color, white_fraction)
        for white_fraction in (0.62, 0.42, 0.22, 0.0)
    ]


def add_bar_group_separators(ax: Any, group_types: list[str]) -> None:
    separator_positions = [i + 0.5 for i in range(len(group_types) - 1) if group_types[i] != group_types[i + 1]]
    for pos in separator_positions:
        ax.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)


def plot_attention_mean_percentile_bar(
    summary: dict[str, Any],
    metric_name: str,
    file_prefix: str,
    metric_title: str,
    plots_dir: Path,
) -> None:
    out_path = plots_dir / f"{file_prefix}_recent4_mean_percentile_bar.png"
    if skip_existing_plot(out_path):
        return

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
    ax.set_ylim(0.0, 1.18)
    ax.set_title(f"Recent-4 {metric_title}: Mean Percentile by Task", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    add_bar_group_separators(ax, group_types)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_attention_recent4_per_frame_mean_percentile_bar(
    records: list[dict[str, Any]],
    metric_name: str,
    file_prefix: str,
    metric_title: str,
    plots_dir: Path,
) -> None:
    out_path = plots_dir / f"{file_prefix}_recent4_per_frame_mean_percentile_bar.png"
    if skip_existing_plot(out_path):
        return

    labels, means, stds, group_types = extract_recent4_per_frame_bar_data(records, metric_name)
    if not labels:
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
        edge_colors = [BAR_PLOT_COLOR_MAP[group_type] for group_type in group_types]
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
    ax.set_ylim(0.0, 1.18)
    ax.set_title(f"Recent-4 {metric_title}: Per-Frame Mean Percentile by Task", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.legend(title="Recent frame", ncol=4, fontsize=8, title_fontsize=9)
    add_bar_group_separators(ax, group_types)

    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
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


def plot_attention_recent4_per_frame_mean_percentile_bars(
    records: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    plots_dir = ensure_dir(plots_dir)
    for metric_name, file_prefix, metric_title in ATTENTION_METRIC_SPECS:
        plot_attention_recent4_per_frame_mean_percentile_bar(
            records,
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
    plot_attention_recent4_per_frame_mean_percentile_bars(records, plots_dir)


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    records = filter_excluded_tasks(load_jsonl(result_dir / "records.jsonl"))
    plots_dir = result_dir / "plots"
    if not records:
        summary = load_summary(result_dir / "summary.json")
        if summary is not None:
            plot_attention_line_plots(summary, plots_dir)
            plot_attention_mean_percentile_bars(summary, plots_dir)
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
    plot_question_prefill_attention_score_token_count_by_bin_average(
        [payload for _, payload in example_payloads],
        ensure_dir(plots_dir),
    )
    plot_question_prefill_attention_score_token_count_by_norm_average(
        [payload for _, payload in example_payloads],
        ensure_dir(plots_dir),
    )
    plot_question_prefill_attention_score_topk_token_value_norm_pooled(
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
