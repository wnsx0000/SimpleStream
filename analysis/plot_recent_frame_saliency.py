from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def metric_recent_frame_percentiles(records: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        recent_indices = record.get("recent_frame_indices", [])
        if not metric:
            continue
        if metric_name.endswith("_attention"):
            frame_values = metric.get("last_k_layers_mean_percentile", [])
        else:
            frame_values = metric.get("frame_percentiles", [])
        values.extend(float(frame_values[idx]) for idx in recent_indices if 0 <= idx < len(frame_values))
    return values


def metric_recent_sample_scores(records: list[dict[str, Any]], metric_name: str, field: str) -> list[float]:
    values: list[float] = []
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if metric and field in metric:
            values.append(float(metric[field]))
    return values


def attention_layer_means(records: list[dict[str, Any]], metric_name: str, field: str) -> np.ndarray | None:
    rows = []
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if metric and field in metric:
            rows.append(np.asarray(metric[field], dtype=np.float64))
    if not rows:
        return None
    return np.vstack(rows).mean(axis=0)


def interpolate_layer_percentiles(records: list[dict[str, Any]], metric_name: str, bins: int = 20) -> np.ndarray | None:
    stacked = []
    target_x = np.linspace(0.0, 1.0, bins)
    for record in valid_records(records):
        metric = record.get("metrics", {}).get(metric_name)
        if not metric:
            continue
        layer_percentiles = np.asarray(metric.get("layer_attention_percentiles", []), dtype=np.float64)
        if layer_percentiles.ndim != 2 or layer_percentiles.shape[1] < 1:
            continue
        source_x = np.linspace(0.0, 1.0, layer_percentiles.shape[1])
        interpolated = np.stack(
            [np.interp(target_x, source_x, layer_values) for layer_values in layer_percentiles],
            axis=0,
        )
        stacked.append(interpolated)
    if not stacked:
        return None
    return np.stack(stacked, axis=0).mean(axis=0)


def plot_violin_distribution(records: list[dict[str, Any]], plots_dir: Path) -> None:
    metric_map = {
        "qwen_similarity": "Qwen Similarity",
        "siglip_similarity": "SigLIP Similarity",
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
        "qwen_similarity": "Qwen Similarity",
        "siglip_similarity": "SigLIP Similarity",
        "question_prefill_attention": "Question Prefill Attn",
        "first_token_attention": "First Token Attn",
    }
    available = [
        (metric_name, label)
        for metric_name, label in metric_map.items()
        if metric_recent_sample_scores(records, metric_name, "recent4_mean_percentile" if not metric_name.endswith("_attention") else "last_k_layers_recent4_mean_percentile")
    ]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3.5 * len(available)), squeeze=False)
    for ax, (metric_name, label) in zip(axes[:, 0], available):
        field = "recent4_mean_percentile" if not metric_name.endswith("_attention") else "last_k_layers_recent4_mean_percentile"
        values = metric_recent_sample_scores(records, metric_name, field)
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
        "qwen_similarity": ("Qwen Similarity", "recent4_top4_overlap"),
        "siglip_similarity": ("SigLIP Similarity", "recent4_top4_overlap"),
        "question_prefill_attention": ("Question Prefill Attn", "last_k_layers_recent4_top4_overlap"),
        "first_token_attention": ("First Token Attn", "last_k_layers_recent4_top4_overlap"),
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
        mean_percentiles = attention_layer_means(records, metric_name, "layer_recent4_mean_percentile")
        mean_overlap = attention_layer_means(records, metric_name, "layer_recent4_top4_overlap")
        if mean_percentiles is not None:
            x = np.arange(mean_percentiles.shape[0])
            axes[0, 0].plot(x, mean_percentiles, label=label)
            plotted = True
        if mean_overlap is not None:
            x = np.arange(mean_overlap.shape[0])
            axes[1, 0].plot(x, mean_overlap, label=label)
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
    for metric_name, label in (
        ("question_prefill_attention", "Question Prefill"),
        ("first_token_attention", "First Token"),
    ):
        matrix = interpolate_layer_percentiles(records, metric_name)
        if matrix is not None:
            matrices.append(matrix)
            labels.append(label)
    if not matrices:
        return

    fig, axes = plt.subplots(len(matrices), 1, figsize=(12, 4.5 * len(matrices)), squeeze=False)
    for ax, matrix, label in zip(axes[:, 0], matrices, labels):
        im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{label}: Layer-wise Percentile Heatmap")
        ax.set_ylabel("Layer")
        ax.set_xlabel("Relative Frame Position")
        ax.set_xticks(np.linspace(0, matrix.shape[1] - 1, 5))
        ax.set_xticklabels([f"{value:.2f}" for value in np.linspace(0.0, 1.0, 5)])
        recent_start = matrix.shape[1] * 0.75
        ax.axvspan(recent_start, matrix.shape[1] - 0.5, facecolor="white", alpha=0.12, edgecolor="none")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "layerwise_percentile_heatmaps.png", dpi=200)
    plt.close(fig)


def plot_example_payload(example_path: Path, plots_dir: Path) -> None:
    payload = torch.load(example_path, map_location="cpu")
    example_key = example_path.stem
    frame_rows = payload.get("frame_rows", [])
    recent_indices = set(int(index) for index in payload.get("recent_frame_indices", []))
    example_dir = ensure_dir(plots_dir / "examples" / example_key)

    qwen_matrix = payload.get("qwen_similarity_matrix")
    siglip_matrix = payload.get("siglip_similarity_matrix")
    if qwen_matrix is not None or siglip_matrix is not None:
        matrices = []
        titles = []
        if qwen_matrix is not None:
            matrices.append(np.asarray(qwen_matrix, dtype=np.float64))
            titles.append("Qwen Similarity")
        if siglip_matrix is not None:
            matrices.append(np.asarray(siglip_matrix, dtype=np.float64))
            titles.append("SigLIP Similarity")
        fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5), squeeze=False)
        for ax, matrix, title in zip(axes[0], matrices, titles):
            im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax.set_title(title)
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Frame Index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(example_dir / "similarity_heatmaps.png", dpi=200)
        plt.close(fig)

    metrics = payload.get("metrics", {})
    frame_indices = np.arange(len(frame_rows))
    line_plotted = False
    fig, ax = plt.subplots(figsize=(11, 5))
    if "qwen_similarity" in metrics:
        ax.plot(frame_indices, metrics["qwen_similarity"]["frame_scores"], label="Qwen Similarity")
        line_plotted = True
    if "siglip_similarity" in metrics:
        ax.plot(frame_indices, metrics["siglip_similarity"]["frame_scores"], label="SigLIP Similarity")
        line_plotted = True
    if "question_prefill_attention" in metrics:
        ax.plot(
            frame_indices,
            metrics["question_prefill_attention"]["last_k_layers_mean_attention_score"],
            label="Question Prefill Attn",
        )
        line_plotted = True
    if "first_token_attention" in metrics:
        ax.plot(
            frame_indices,
            metrics["first_token_attention"]["last_k_layers_mean_attention_score"],
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
        matrix = np.asarray(metrics[metric_name]["layer_attention_scores"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{title} Attention Heatmap: {example_key}")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Layer")
        for index in recent_indices:
            ax.axvline(index, color="white", linestyle="--", linewidth=0.8, alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(example_dir / f"{metric_name}_heatmap.png", dpi=200)
        plt.close(fig)


def generate_plots(result_dir: str | Path) -> None:
    result_dir = Path(result_dir)
    records = load_jsonl(result_dir / "records.jsonl")
    plots_dir = ensure_dir(result_dir / "plots")
    if not records:
        return

    plot_violin_distribution(records, plots_dir)
    plot_sample_histograms(records, plots_dir)
    plot_overlap_bars(records, plots_dir)
    plot_layer_lines(records, plots_dir)
    plot_layer_heatmaps(records, plots_dir)

    examples_dir = result_dir / "examples"
    if examples_dir.exists():
        for example_path in sorted(examples_dir.glob("*.pt")):
            plot_example_payload(example_path, plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SimpleStream recent4 saliency analysis results.")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    generate_plots(args.result_dir)


if __name__ == "__main__":
    main()
