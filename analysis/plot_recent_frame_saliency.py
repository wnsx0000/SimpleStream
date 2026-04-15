from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F

DISPLAY_LAYER_COUNT = 5
QUESTION_PREFILL_FRAME_FRAME_AVERAGE_SHAPE = (64, 64)
QUESTION_PREFILL_QUESTION_FRAME_AVERAGE_SHAPE = (32, 64)


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


def scalar_metric_recent_indices(record: dict[str, Any], metric: dict[str, Any]) -> list[int]:
    subset_recent = metric.get("recent_frame_indices_within_analysis")
    if subset_recent is not None:
        return [int(index) for index in subset_recent]
    return [int(index) for index in record.get("recent_frame_indices", [])]


def scalar_metric_frame_indices(metric: dict[str, Any], field: str) -> np.ndarray:
    metric_frame_indices = metric.get("analysis_frame_indices")
    if metric_frame_indices is not None:
        return np.asarray(metric_frame_indices, dtype=np.int64)
    return np.arange(len(metric.get(field, [])), dtype=np.int64)


def metric_recent_frame_percentiles(records: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if not metric:
            continue
        if metric_name.endswith("_attention"):
            frame_values = metric.get("mean_percentile", [])
            recent_indices = metric.get("recent_frame_indices_within_attention", [])
        else:
            frame_values = metric.get("frame_percentiles", [])
            recent_indices = scalar_metric_recent_indices(record, metric)
        values.extend(float(frame_values[idx]) for idx in recent_indices if 0 <= idx < len(frame_values))
    return values


def metric_recent_sample_scores(records: list[dict[str, Any]], metric_name: str, field: str) -> list[float]:
    values: list[float] = []
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if metric and field in metric:
            values.append(float(metric[field]))
    return values


def uniform_center_indices(total_count: int, target_count: int = DISPLAY_LAYER_COUNT) -> list[int]:
    total_count = int(total_count)
    target_count = int(target_count)
    if total_count <= 0 or target_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    return [int(round(x)) for x in np.linspace(0, total_count - 1, target_count)]


def normalized_metric_layer_array(metric: dict[str, Any], field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if field not in metric:
        return None, None

    values = np.asarray(metric[field], dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[0] < 1:
        return None, None

    saved_indices = np.asarray(metric.get("display_layer_indices", []), dtype=np.int64)
    if saved_indices.size == values.shape[0]:
        return saved_indices, values

    sampled_indices = np.asarray(uniform_center_indices(values.shape[0], DISPLAY_LAYER_COUNT), dtype=np.int64)
    if sampled_indices.size < 1:
        return None, None
    return sampled_indices, values[sampled_indices]


def attention_layer_means(records: list[dict[str, Any]], metric_name: str, field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    rows = []
    display_indices: np.ndarray | None = None
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if not metric:
            continue
        layer_indices, values = normalized_metric_layer_array(metric, field)
        if layer_indices is None or values is None:
            continue
        if display_indices is None:
            display_indices = layer_indices
        if values.shape[0] != display_indices.shape[0] or not np.array_equal(layer_indices, display_indices):
            continue
        rows.append(values)
    if not rows:
        return None, None
    return display_indices, np.vstack(rows).mean(axis=0)


def interpolate_layer_percentiles(
    records: list[dict[str, Any]],
    metric_name: str,
    bins: int = 20,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    stacked = []
    display_indices: np.ndarray | None = None
    target_x = np.linspace(0.0, 1.0, bins)
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if not metric:
            continue
        layer_indices, layer_percentiles = normalized_metric_layer_array(metric, "layer_attention_percentiles")
        if layer_indices is None or layer_percentiles is None:
            continue
        if layer_percentiles.ndim != 2 or layer_percentiles.shape[1] < 1:
            continue
        if display_indices is None:
            display_indices = layer_indices
        if layer_percentiles.shape[0] != display_indices.shape[0] or not np.array_equal(layer_indices, display_indices):
            continue
        source_x = np.linspace(0.0, 1.0, layer_percentiles.shape[1])
        interpolated = np.stack(
            [np.interp(target_x, source_x, layer_values) for layer_values in layer_percentiles],
            axis=0,
        )
        stacked.append(interpolated)
    if not stacked:
        return None, None
    return display_indices, np.stack(stacked, axis=0).mean(axis=0)


def plot_violin_distribution(records: list[dict[str, Any]], plots_dir: Path) -> None:
    metric_map = {
        "siglip_similarity": "SigLIP Frame-Question Similarity",
        "question_prefill_attention": "Question Prefill Attn",
        "first_token_attention": "First Token Attn",
    }
    data = []
    labels = []
    for metric_name, label in metric_map.items():
        values = metric_recent_frame_percentiles(records, metric_name)
        if values:
            data.append(values)
            labels.append(label)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_alpha(0.35)
    ax.boxplot(data, widths=0.18, positions=np.arange(1, len(data) + 1))
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Recent Frame Percentile Rank")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Recent4 Percentile Distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "recent_frame_percentile_distribution.png", dpi=200)
    plt.close(fig)


def plot_sample_histograms(records: list[dict[str, Any]], plots_dir: Path) -> None:
    metric_map = {
        "siglip_similarity": "SigLIP Frame-Question Similarity",
        "question_prefill_attention": "Question Prefill Attn",
        "first_token_attention": "First Token Attn",
    }
    available = [
        (metric_name, label)
        for metric_name, label in metric_map.items()
        if metric_recent_sample_scores(records, metric_name, "recent4_mean_percentile")
    ]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.5 * len(available)), squeeze=False)
    for ax, (metric_name, label) in zip(axes[:, 0], available):
        values = metric_recent_sample_scores(records, metric_name, "recent4_mean_percentile")
        ax.hist(values, bins=15, range=(0.0, 1.0), alpha=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: Recent4 Mean Percentile")
    axes[-1, 0].set_xlabel("Mean Percentile")
    fig.tight_layout()
    fig.savefig(plots_dir / "recent4_mean_percentile_histograms.png", dpi=200)
    plt.close(fig)


def plot_overlap_bars(records: list[dict[str, Any]], plots_dir: Path) -> None:
    metric_map = {
        "siglip_similarity": ("SigLIP Frame-Question Similarity", "recent4_top4_overlap"),
        "question_prefill_attention": ("Question Prefill Attn", "recent4_top4_overlap"),
        "first_token_attention": ("First Token Attn", "recent4_top4_overlap"),
    }
    labels = []
    values = []
    for metric_name, (label, field) in metric_map.items():
        metric_values = metric_recent_sample_scores(records, metric_name, field)
        if metric_values:
            labels.append(label)
            values.append(float(np.mean(metric_values)))
    if not values:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top4 Overlap")
    ax.set_title("Recent4 vs Metric Top4 Overlap")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "recent4_top4_overlap.png", dpi=200)
    plt.close(fig)


def plot_layer_lines(records: list[dict[str, Any]], plots_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), squeeze=False)
    plotted = False
    for metric_name, label in (
        ("question_prefill_attention", "Question Prefill"),
        ("first_token_attention", "First Token"),
    ):
        percentile_layers, mean_percentiles = attention_layer_means(records, metric_name, "layer_recent4_mean_percentile")
        overlap_layers, mean_overlap = attention_layer_means(records, metric_name, "layer_recent4_top4_overlap")
        if percentile_layers is not None and mean_percentiles is not None:
            axes[0, 0].plot(percentile_layers, mean_percentiles, label=label)
            axes[0, 0].set_xticks(percentile_layers)
            plotted = True
        if overlap_layers is not None and mean_overlap is not None:
            axes[1, 0].plot(overlap_layers, mean_overlap, label=label)
            axes[1, 0].set_xticks(overlap_layers)
            plotted = True
    if not plotted:
        plt.close(fig)
        return

    axes[0, 0].set_title("Layer-wise Recent4 Mean Percentile")
    axes[0, 0].set_ylabel("Mean Percentile")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].legend()

    axes[1, 0].set_title("Layer-wise Recent4 Top4 Overlap")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Overlap")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "layerwise_recent4_curves.png", dpi=200)
    plt.close(fig)


def plot_layer_heatmaps(records: list[dict[str, Any]], plots_dir: Path) -> None:
    matrices = []
    labels = []
    layer_index_sets = []
    for metric_name, label in (
        ("question_prefill_attention", "Question Prefill"),
        ("first_token_attention", "First Token"),
    ):
        layer_indices, matrix = interpolate_layer_percentiles(records, metric_name)
        if layer_indices is not None and matrix is not None:
            matrices.append(matrix)
            labels.append(label)
            layer_index_sets.append(layer_indices)
    if not matrices:
        return

    fig, axes = plt.subplots(len(matrices), 1, figsize=(12, 4.5 * len(matrices)), squeeze=False)
    for ax, matrix, label, layer_indices in zip(axes[:, 0], matrices, labels, layer_index_sets):
        im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{label}: Layer-wise Percentile Heatmap")
        ax.set_ylabel("Layer")
        ax.set_xlabel("Relative Frame Position")
        ax.set_xticks(np.linspace(0, matrix.shape[1] - 1, 5))
        ax.set_xticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, 5)])
        ax.set_yticks(np.arange(layer_indices.shape[0]))
        ax.set_yticklabels([str(int(index)) for index in layer_indices])
        recent_start = matrix.shape[1] * 0.75
        ax.axvspan(recent_start, matrix.shape[1] - 0.5, facecolor="white", alpha=0.12, edgecolor="none")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "layerwise_percentile_heatmaps.png", dpi=200)
    plt.close(fig)


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
    yy, xx = np.indices((rows, cols))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=matrix.ravel(),
        cmap=cmap,
        norm=norm,
        marker="s",
        s=square_marker_size((rows, cols)),
        linewidths=0.0,
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    return scatter


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
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, maps.shape[0], figsize=(4.6 * maps.shape[0], 5.2), squeeze=False)
    axes_row = list(axes[0])
    mappable = None
    for ax, matrix, layer_idx in zip(axes_row, maps, display_layer_indices):
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

    axes_row[0].set_ylabel(y_label)
    fig.suptitle(figure_title)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes_row, fraction=0.025, pad=0.02, label="Mean Attention Score")
    fig.subplots_adjust(left=0.05, right=0.92, top=0.84, bottom=0.24, wspace=0.28)
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
    display_layers, frame_frame_average = compute_average_question_prefill_maps(
        payloads,
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
        payloads,
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


def plot_example_payload(example_path: Path, plots_dir: Path, payload: dict[str, Any] | None = None) -> None:
    payload = torch.load(example_path, map_location="cpu") if payload is None else payload
    example_key = example_path.stem
    recent_indices = set(int(index) for index in payload.get("recent_frame_indices", []))
    example_dir = ensure_dir(plots_dir / "examples" / example_key)

    metrics = payload.get("metrics", {})
    line_plotted = False
    fig, ax = plt.subplots(figsize=(11, 5))
    if "siglip_similarity" in metrics:
        siglip_x = scalar_metric_frame_indices(metrics["siglip_similarity"], "frame_scores")
        ax.plot(
            siglip_x,
            metrics["siglip_similarity"]["frame_scores"],
            label="SigLIP Frame-Question Similarity",
        )
        line_plotted = True
    if "question_prefill_attention" in metrics:
        attn_x = np.asarray(
            metrics["question_prefill_attention"].get(
                "attention_frame_indices",
                list(range(len(metrics["question_prefill_attention"]["mean_attention_score"]))),
            ),
            dtype=np.int64,
        )
        ax.plot(
            attn_x,
            metrics["question_prefill_attention"]["mean_attention_score"],
            label="Question Prefill Attn",
        )
        line_plotted = True
    if "first_token_attention" in metrics:
        attn_x = np.asarray(
            metrics["first_token_attention"].get(
                "attention_frame_indices",
                list(range(len(metrics["first_token_attention"]["mean_attention_score"]))),
            ),
            dtype=np.int64,
        )
        ax.plot(
            attn_x,
            metrics["first_token_attention"]["mean_attention_score"],
            label="First Token Attn",
        )
        line_plotted = True
    if line_plotted:
        for index in recent_indices:
            ax.axvline(index, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_title(f"Frame-wise Scores: {example_key}")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(example_dir / "frame_score_lines.png", dpi=200)
    plt.close(fig)

    for metric_name, title in (
        ("question_prefill_attention", "Question Prefill"),
        ("first_token_attention", "First Token"),
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
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{title} Attention Heatmap: {example_key}")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Layer")
        tick_positions = np.linspace(0, matrix.shape[1] - 1, min(6, matrix.shape[1]))
        tick_labels = [str(int(x_positions[int(round(pos))])) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(np.arange(layer_indices.shape[0]))
        ax.set_yticklabels([str(int(index)) for index in layer_indices])
        selected_recent = set(int(index) for index in metrics[metric_name].get("recent_frame_indices_within_attention", []))
        for local_index in selected_recent:
            ax.axvline(local_index, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(example_dir / f"{metric_name}_heatmap.png", dpi=200)
        plt.close(fig)

    map_payload = question_prefill_map_payload(payload)
    if map_payload is None:
        return

    display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
    frame_bin_slices = [
        [int(start), int(end)]
        for start, end in map_payload.get("frame_bin_slices", [])
    ]
    frame_bin_labels = [str(label) for label in map_payload.get("frame_bin_labels", [])]
    question_bin_labels = [str(label) for label in map_payload.get("question_bin_labels", [])]
    example_label = make_example_label(payload, example_key)

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


def load_example_payloads(examples_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for example_path in sorted(examples_dir.glob("*.pt")):
        payloads.append((example_path, torch.load(example_path, map_location="cpu")))
    return payloads


def generate_aggregate_plots(records: list[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir = ensure_dir(plots_dir)
    if not records:
        return

    plot_violin_distribution(records, plots_dir)
    plot_sample_histograms(records, plots_dir)
    plot_overlap_bars(records, plots_dir)
    plot_layer_lines(records, plots_dir)
    plot_layer_heatmaps(records, plots_dir)


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    records = load_jsonl(result_dir / "records.jsonl")
    plots_dir = result_dir / "plots"
    if not records:
        return

    generate_aggregate_plots(records, plots_dir)
    for split_name in ("backward", "realtime"):
        generate_aggregate_plots(records_for_split(records, split_name), plots_dir / split_name)

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
