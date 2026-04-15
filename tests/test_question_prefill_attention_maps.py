from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plot_recent_frame_saliency import generate_plots  # noqa: E402
from lib.frame_saliency_qwen3 import (  # noqa: E402
    allocate_proportional_bin_counts,
    build_question_prefill_attention_map_metadata,
    build_question_prefill_attention_maps,
)


class QuestionPrefillAttentionMapTests(unittest.TestCase):
    @staticmethod
    def _synthetic_attention_metric(offset: float) -> dict[str, object]:
        base = np.asarray(
            [
                [1.0e-10, 2.0e-10, 4.0e-10],
                [1.5e-10, 3.0e-10, 6.0e-10],
                [2.0e-10, 4.0e-10, 8.0e-10],
            ],
            dtype=np.float32,
        )
        return {
            "recent4_mean_percentile": 0.25 + offset,
            "recent4_top4_overlap": 0.10 + offset,
            "layer_recent4_mean_percentile": [0.20 + offset, 0.30 + offset, 0.40 + offset],
            "layer_recent4_top4_overlap": [0.05 + offset, 0.10 + offset, 0.15 + offset],
            "display_layer_indices": [0, 12, 24],
            "layer_attention_scores": (base + np.float32(offset * 1.0e-4)).tolist(),
            "attention_frame_indices": [10, 20, 30],
            "recent_frame_indices_within_attention": [1, 2],
        }

    def test_allocate_proportional_bin_counts(self) -> None:
        counts = [4, 8, 12]
        bins = allocate_proportional_bin_counts(counts, max_bins=9)
        self.assertEqual(bins, [2, 3, 4])
        self.assertEqual(sum(bins), 9)
        self.assertEqual(len(bins), len(counts))

    def test_question_prefill_attention_map_pooling_shapes_and_means(self) -> None:
        metadata = build_question_prefill_attention_map_metadata(
            frame_token_spans=[(0, 2), (2, 6)],
            query_positions=[6, 7, 8, 9],
            attention_frame_indices=[10, 11],
            frame_axis_max_bins=3,
            question_axis_max_bins=2,
        )
        self.assertEqual(metadata.frame_bin_slices, [(0, 1), (1, 3)])
        self.assertEqual(metadata.frame_bin_labels, ["10", "11"])
        self.assertEqual(metadata.question_bin_labels, ["q0-1", "q2-3"])

        base = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        attn_weights = torch.stack([base, base], dim=0)
        maps = build_question_prefill_attention_maps(attn_weights, metadata)

        frame_frame = maps["frame_frame_map"].numpy()
        question_frame = maps["question_frame_map"].numpy()
        self.assertEqual(frame_frame.shape, (3, 3))
        self.assertEqual(question_frame.shape, (2, 3))

        frame_source = base[:6, :6].numpy()
        question_source = base[6:10, :6].numpy()
        frame_spans = [(0, 2), (2, 4), (4, 6)]
        question_spans = [(0, 2), (2, 4)]
        expected_frame = np.array(
            [
                [
                    frame_source[row_start:row_end, col_start:col_end].mean()
                    for col_start, col_end in frame_spans
                ]
                for row_start, row_end in frame_spans
            ],
            dtype=np.float32,
        )
        expected_question = np.array(
            [
                [
                    question_source[row_start:row_end, col_start:col_end].mean()
                    for col_start, col_end in frame_spans
                ]
                for row_start, row_end in question_spans
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(frame_frame, expected_frame)
        np.testing.assert_allclose(question_frame, expected_question)

    def test_plotting_smoke_generates_question_prefill_map_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_dir = Path(tmp_dir)
            examples_dir = result_dir / "examples"
            examples_dir.mkdir(parents=True, exist_ok=True)

            records = [
                {
                    "_key": "SYN:1",
                    "split": "backward",
                    "task": "SYN",
                    "id": 1,
                    "metrics": {},
                }
            ]
            with (result_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            payload = {
                "split": "backward",
                "task": "SYN",
                "id": 1,
                "question": "synthetic question",
                "recent_frame_indices": [1, 2],
                "metrics": {},
                "question_prefill_attention_maps": {
                    "frame_frame_maps": torch.linspace(0.0, 1.0, steps=5 * 6 * 6, dtype=torch.float32).reshape(5, 6, 6),
                    "question_frame_maps": torch.linspace(
                        0.0,
                        1.0,
                        steps=5 * 3 * 6,
                        dtype=torch.float32,
                    ).reshape(5, 3, 6),
                    "display_layer_indices": [0, 9, 18, 27, 35],
                    "attention_frame_indices": [10, 20, 30],
                    "frame_bin_slices": [[0, 2], [2, 4], [4, 6]],
                    "frame_bin_labels": ["10", "20", "30"],
                    "question_bin_labels": ["q0", "q1", "q2"],
                },
            }
            torch.save(payload, examples_dir / "sample.pt")

            generate_plots(result_dir)

            self.assertTrue((result_dir / "plots" / "question_prefill_frame_frame_maps_average.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_question_frame_maps_average.png").exists())
            self.assertTrue(
                (result_dir / "plots" / "examples" / "sample" / "question_prefill_frame_frame_maps.png").exists()
            )
            self.assertTrue(
                (result_dir / "plots" / "examples" / "sample" / "question_prefill_question_frame_maps.png").exists()
            )

    def test_plotting_filters_hld_and_generates_pooled_line_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_dir = Path(tmp_dir)
            examples_dir = result_dir / "examples"
            examples_dir.mkdir(parents=True, exist_ok=True)

            records = [
                {
                    "_key": "ASI:1",
                    "split": "backward",
                    "task": "ASI",
                    "id": 1,
                    "metrics": {
                        "question_prefill_attention": self._synthetic_attention_metric(0.00),
                    },
                },
                {
                    "_key": "OCR:2",
                    "split": "realtime",
                    "task": "OCR",
                    "id": 2,
                    "metrics": {
                        "question_prefill_attention": self._synthetic_attention_metric(0.05),
                    },
                },
                {
                    "_key": "HLD:3",
                    "split": "backward",
                    "task": "HLD",
                    "id": 3,
                    "metrics": {
                        "question_prefill_attention": self._synthetic_attention_metric(0.50),
                    },
                },
            ]
            with (result_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            allowed_payload = {
                "split": "backward",
                "task": "ASI",
                "id": 1,
                "question": "allowed sample",
                "recent_frame_indices": [1, 2],
                "metrics": {
                    "question_prefill_attention": self._synthetic_attention_metric(0.0),
                },
                "question_prefill_attention_maps": {
                    "frame_frame_maps": torch.linspace(
                        1.0e-10,
                        9.0e-10,
                        steps=3 * 3 * 3,
                        dtype=torch.float32,
                    ).reshape(3, 3, 3),
                    "question_frame_maps": torch.linspace(
                        1.0e-10,
                        6.0e-10,
                        steps=3 * 2 * 3,
                        dtype=torch.float32,
                    ).reshape(3, 2, 3),
                    "display_layer_indices": [0, 12, 24],
                    "attention_frame_indices": [10, 20, 30],
                    "frame_bin_slices": [[0, 1], [1, 2], [2, 3]],
                    "frame_bin_labels": ["10", "20", "30"],
                    "question_bin_labels": ["q0", "q1"],
                },
            }
            filtered_payload = {
                **allowed_payload,
                "task": "HLD",
                "id": 3,
            }
            torch.save(allowed_payload, examples_dir / "asi_sample.pt")
            torch.save(filtered_payload, examples_dir / "hld_sample.pt")

            generate_plots(result_dir)

            self.assertTrue((result_dir / "plots" / "question_prefill_percentile_mean.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_percentile_mean_pooled.png").exists())
            self.assertFalse((result_dir / "plots" / "layerwise_percentile_heatmaps.png").exists())
            self.assertTrue(
                (result_dir / "plots" / "examples" / "asi_sample" / "question_prefill_attention_heatmap.png").exists()
            )
            self.assertFalse((result_dir / "plots" / "examples" / "hld_sample").exists())


if __name__ == "__main__":
    unittest.main()
