"""Render question-prefill attention heatmaps for saved V-RAG examples.

Given a ``--result-dir`` produced by ``main_experiments/eval_qwen3vl_ovo_test4.py``
(which writes ``.pt`` payloads under ``<result_dir>/examples/``), this script
emits per-example frame-frame and question-frame attention heatmap PNGs under
``<result_dir>/plots/examples/<example_key>/``. No aggregate bar/line plots
are produced; only the attention heatmaps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.plot_recent_frame_saliency import (
    ensure_dir,
    make_example_label,
    question_prefill_map_payload,
    rawscale_output_path,
    render_question_prefill_map_panels,
    to_numpy_array,
)

EXCLUDED_PLOT_TASKS = frozenset({"HLD"})


def plot_example_heatmaps(example_path: Path, plots_dir: Path) -> bool:
    payload: dict[str, Any] = torch.load(example_path, map_location="cpu")
    if str(payload.get("task", "")) in EXCLUDED_PLOT_TASKS:
        return False

    map_payload = question_prefill_map_payload(payload)
    if map_payload is None:
        print(
            f"Skipping {example_path.name}: payload has no question_prefill_attention_maps."
        )
        return False

    example_key = example_path.stem
    example_dir = ensure_dir(plots_dir / "examples" / example_key)
    example_label = make_example_label(payload, example_key)

    display_layer_indices = [int(index) for index in map_payload.get("display_layer_indices", [])]
    frame_bin_slices = [
        [int(start), int(end)]
        for start, end in map_payload.get("frame_bin_slices", [])
    ]
    frame_bin_labels = [str(label) for label in map_payload.get("frame_bin_labels", [])]
    question_bin_labels = [str(label) for label in map_payload.get("question_bin_labels", [])]

    rendered_any = False

    frame_frame_maps = to_numpy_array(map_payload.get("frame_frame_maps"))
    if frame_frame_maps is not None:
        frame_frame_path = example_dir / "question_prefill_frame_frame_maps.png"
        render_question_prefill_map_panels(
            frame_frame_maps,
            display_layer_indices,
            frame_frame_path,
            figure_title=f"Question Prefill Frame\u2194Frame Maps: {example_label}",
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
            figure_title=f"Question Prefill Frame\u2194Frame Maps: {example_label}",
            x_label="Frame Index",
            y_label="Frame Index",
            mode="frame_frame",
            frame_bin_slices=frame_bin_slices,
            frame_bin_labels=frame_bin_labels,
            robust_percentile=None,
        )
        rendered_any = True

    question_frame_maps = to_numpy_array(map_payload.get("question_frame_maps"))
    if question_frame_maps is not None:
        question_frame_path = example_dir / "question_prefill_question_frame_maps.png"
        render_question_prefill_map_panels(
            question_frame_maps,
            display_layer_indices,
            question_frame_path,
            figure_title=f"Question Prefill Question\u2192Frame Maps: {example_label}",
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
            figure_title=f"Question Prefill Question\u2192Frame Maps: {example_label}",
            x_label="Frame Index",
            y_label="Question Token Bin",
            mode="question_frame",
            frame_bin_slices=frame_bin_slices,
            frame_bin_labels=frame_bin_labels,
            question_bin_labels=question_bin_labels,
            robust_percentile=None,
        )
        rendered_any = True

    return rendered_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render V-RAG question-prefill attention heatmaps from saved examples "
            "(.pt payloads written by eval_qwen3vl_ovo_test4.py)."
        )
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Path to the V-RAG result directory (contains examples/*.pt).",
    )
    parser.add_argument(
        "--examples-subdir",
        default="examples",
        help="Subdirectory under --result-dir that holds *.pt example payloads.",
    )
    parser.add_argument(
        "--plots-subdir",
        default="plots",
        help="Subdirectory under --result-dir where heatmap PNGs are written.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    examples_dir = result_dir / args.examples_subdir
    plots_dir = result_dir / args.plots_subdir

    if not examples_dir.exists():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

    example_paths = sorted(examples_dir.glob("*.pt"))
    if not example_paths:
        print(f"No .pt example payloads found under {examples_dir}.")
        return

    ensure_dir(plots_dir)
    rendered = 0
    for example_path in example_paths:
        try:
            if plot_example_heatmaps(example_path, plots_dir):
                rendered += 1
        except Exception as exc:
            print(f"Failed to render {example_path.name}: {type(exc).__name__}: {exc}")

    print(f"Rendered attention heatmaps for {rendered}/{len(example_paths)} example(s).")
    print(f"Output: {plots_dir / 'examples'}")


if __name__ == "__main__":
    main()
