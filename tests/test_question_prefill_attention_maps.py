from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plot_recent_frame_saliency import (  # noqa: E402
    compute_average_attention_score_token_count_by_bin,
    compute_average_attention_score_token_count_by_norm,
    compute_pooled_attention_score_topk_token_value_norm,
    extract_attention_score_token_count_by_bin,
    extract_attention_score_token_count_by_norm,
    extract_attention_score_topk_token_value_norm,
    generate_plots as generate_attention_plots,
)
from analysis.plot_siglip_similarity import (  # noqa: E402
    extract_bar_data as extract_siglip_bar_data,
    filter_excluded_tasks as filter_siglip_excluded_tasks,
    generate_plots as generate_siglip_plots,
)
from lib.frame_saliency_qwen3 import (  # noqa: E402
    LayerwiseFrameAttentionCollector,
    allocate_proportional_bin_counts,
    build_experiment_summary,
    build_question_prefill_attention_map_metadata,
    build_question_prefill_attention_maps,
    question_prefill_layer_indices,
    resolve_siglip_device,
)
from lib.recent_window_eval import select_attention_frame_indices  # noqa: E402


class QuestionPrefillAttentionMapTests(unittest.TestCase):
    @staticmethod
    def _synthetic_attention_metric(offset: float) -> dict[str, object]:
        layer_indices = question_prefill_layer_indices(36)
        base = np.asarray(
            [[(layer_pos + 1) * 1.0e-10, (layer_pos + 2) * 1.0e-10, (layer_pos + 4) * 1.0e-10]
             for layer_pos in range(len(layer_indices))],
            dtype=np.float32,
        )
        return {
            "recent4_mean_percentile": 0.25 + offset,
            "layer_recent4_mean_percentile": [
                0.20 + offset + float(layer_pos) * 0.01
                for layer_pos in range(len(layer_indices))
            ],
            "display_layer_indices": layer_indices,
            "layer_attention_scores": (base + np.float32(offset * 1.0e-4)).tolist(),
            "attention_frame_indices": [10, 20, 30],
            "recent_frame_indices_within_attention": [1, 2],
        }

    def test_question_prefill_layer_indices_for_qwen3vl_8b(self) -> None:
        self.assertEqual(
            question_prefill_layer_indices(36),
            [0, 9, 18, 26, 28, 29, 30, 31, 32, 33, 34, 35],
        )

    def test_build_experiment_summary_excludes_hld(self) -> None:
        records = [
            {
                "split": "backward",
                "task": "ASI",
                "metrics": {"question_prefill_attention": self._synthetic_attention_metric(0.0)},
            },
            {
                "split": "backward",
                "task": "HLD",
                "metrics": {"question_prefill_attention": self._synthetic_attention_metric(0.5)},
            },
        ]
        summary = build_experiment_summary(records, config={})
        self.assertEqual(summary["total_records"], 1)
        self.assertIn("ASI", summary["tasks"])
        self.assertNotIn("HLD", summary["tasks"])

    def test_uniform_with_recent_anchor_includes_recent_frames(self) -> None:
        selected, strategy = select_attention_frame_indices(
            total_frames=100,
            recent_indices=[96, 97, 98, 99],
            max_analysis_frames=12,
        )
        self.assertEqual(strategy, "uniform_with_recent_anchor")
        self.assertTrue({96, 97, 98, 99}.issubset(set(selected)))
        self.assertEqual(len(selected), 12)

    def test_resolve_siglip_device_selects_most_free_visible_cuda(self) -> None:
        def fake_mem_get_info(index: int) -> tuple[int, int]:
            free_by_index = {0: 8, 1: 24, 2: 16}
            return free_by_index[index] * 1024**3, 32 * 1024**3

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=3),
            mock.patch.object(torch.cuda, "mem_get_info", side_effect=fake_mem_get_info),
        ):
            self.assertEqual(resolve_siglip_device(), torch.device("cuda:1"))

    def test_resolve_siglip_device_honors_explicit_override(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=3),
        ):
            self.assertEqual(resolve_siglip_device("cuda:2"), torch.device("cuda:2"))

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

    def test_value_norm_hook_stores_frame_patch_tokens(self) -> None:
        metadata = build_question_prefill_attention_map_metadata(
            frame_token_spans=[(0, 4), (4, 8)],
            query_positions=[8, 9],
            attention_frame_indices=[10, 11],
            frame_axis_max_bins=4,
            question_axis_max_bins=1,
        )
        collector = LayerwiseFrameAttentionCollector(
            frame_token_spans=[(0, 4), (4, 8)],
            query_positions=[8, 9],
            num_layers=1,
            map_layer_indices=[0],
            question_prefill_map_metadata=metadata,
            capture_per_patch=True,
            value_head_dim=2,
            value_num_kv_heads=1,
        )

        value_tensor = torch.zeros(1, 10, 2, dtype=torch.float32)
        value_tensor[0, :, 0] = torch.arange(1, 11, dtype=torch.float32)
        collector.make_value_hook(0, head_dim=2, num_kv_heads=1)(None, None, value_tensor)

        self.assertIn(0, collector.layer_value_norms)
        self.assertEqual(tuple(collector.layer_value_norms[0].shape), (8,))
        np.testing.assert_allclose(
            collector.layer_value_norms[0].numpy(),
            np.arange(1, 9, dtype=np.float32),
        )
        self.assertIn(0, collector.layer_per_patch_value_norms)
        self.assertEqual([tuple(row.shape) for row in collector.layer_per_patch_value_norms[0]], [(4,), (4,)])

    def test_sink_bin_payload_selects_single_frame_axis_bin(self) -> None:
        metadata = build_question_prefill_attention_map_metadata(
            frame_token_spans=[(0, 4), (4, 8)],
            query_positions=[8, 9],
            attention_frame_indices=[10, 11],
            frame_axis_max_bins=4,
            question_axis_max_bins=1,
        )
        collector = LayerwiseFrameAttentionCollector(
            frame_token_spans=[(0, 4), (4, 8)],
            query_positions=[8, 9],
            num_layers=1,
            map_layer_indices=[0],
            question_prefill_map_metadata=metadata,
        )
        collector.layer_attention_maps[0] = {
            "question_frame_map": torch.tensor([[0.1, 0.2, 0.9, 0.3]], dtype=torch.float32),
            "question_frame_token_map": torch.arange(8, dtype=torch.float32).reshape(1, 8),
        }

        payload = collector.export_question_prefill_sink_bin_token_attention([0])

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["sink_bin_index"], 2)
        self.assertEqual(payload["sink_frame_local_index"], 1)
        self.assertEqual(payload["sink_frame_label"], "11")
        self.assertEqual(payload["sink_token_start"], 4)
        self.assertEqual(payload["sink_token_end"], 6)
        self.assertEqual(tuple(payload["maps"].shape), (1, 1, 2))

    def test_attention_score_token_count_by_bin_uses_question_mean_and_nan_padding(self) -> None:
        payload_a = {
            "task": "SYN",
            "id": "a",
            "question_prefill_attention_maps": {
                "frame_frame_maps": torch.zeros(2, 3, 3, dtype=torch.float32),
                "question_frame_maps": torch.tensor(
                    [
                        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                        [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
                    ],
                    dtype=torch.float32,
                ),
                "display_layer_indices": [0, 9],
                "frame_bin_slices": [[0, 1], [1, 3]],
                "frame_local_bin_spans": [[0, 1], [1, 3], [3, 6]],
                "frame_bin_labels": ["10", "11"],
                "question_bin_labels": ["q0", "q1"],
            },
        }
        payload_b = {
            "task": "SYN",
            "id": "b",
            "question_prefill_attention_maps": {
                "frame_frame_maps": torch.zeros(2, 5, 5, dtype=torch.float32),
                "question_frame_maps": torch.tensor(
                    [
                        [[3.0, 5.0, 7.0, 9.0, 11.0], [5.0, 7.0, 9.0, 11.0, 13.0]],
                        [[30.0, 50.0, 70.0, 90.0, 110.0], [50.0, 70.0, 90.0, 110.0, 130.0]],
                    ],
                    dtype=torch.float32,
                ),
                "display_layer_indices": [0, 9],
                "frame_bin_slices": [[0, 2], [2, 5]],
                "frame_local_bin_spans": [[0, 2], [2, 5], [5, 9], [9, 14], [14, 20]],
                "frame_bin_labels": ["10", "11"],
                "question_bin_labels": ["q0", "q1"],
            },
        }

        extracted = extract_attention_score_token_count_by_bin(payload_a)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        display_layers, attention_scores, token_counts = extracted
        self.assertEqual(display_layers, [0, 9])
        np.testing.assert_allclose(attention_scores, [[1.5, 3.5, 5.5], [15.0, 35.0, 55.0]])
        np.testing.assert_allclose(token_counts, [1.0, 2.0, 3.0])

        averaged = compute_average_attention_score_token_count_by_bin([payload_a, payload_b])
        avg_layers, avg_attention, avg_token_counts, common_slices, common_labels = averaged
        self.assertEqual(avg_layers, [0, 9])
        self.assertIsNone(common_slices)
        self.assertIsNone(common_labels)
        assert avg_attention is not None and avg_token_counts is not None
        np.testing.assert_allclose(
            avg_attention,
            [
                [2.75, 4.75, 6.75, 10.0, 12.0],
                [27.5, 47.5, 67.5, 100.0, 120.0],
            ],
        )
        np.testing.assert_allclose(avg_token_counts, [1.5, 2.5, 3.5, 5.0, 6.0])

        payload_missing_spans = {
            "question_prefill_attention_maps": {
                "frame_frame_maps": torch.zeros(1, 1, 1),
                "question_frame_maps": torch.zeros(1, 1, 1),
                "display_layer_indices": [0],
            }
        }
        self.assertIsNone(extract_attention_score_token_count_by_bin(payload_missing_spans))

    def test_attention_score_token_count_by_norm_uses_patch_attention_and_weighted_average(self) -> None:
        payload_a = {
            "task": "SYN",
            "id": "a",
            "question_prefill_attention_maps": {
                "frame_frame_maps": torch.zeros(2, 3, 3, dtype=torch.float32),
                "question_frame_maps": torch.zeros(2, 2, 3, dtype=torch.float32),
                "display_layer_indices": [0, 9],
                "frame_bin_slices": [[0, 1], [1, 3]],
                "frame_local_bin_spans": [[0, 2], [2, 4], [4, 6]],
                "frame_bin_labels": ["10", "11"],
                "question_bin_labels": ["q0", "q1"],
            },
            "question_prefill_value_norms": torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                ],
                dtype=torch.float32,
            ),
            "question_prefill_per_patch_attention": {
                0: [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0, 6.0])],
                9: [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0, 50.0, 60.0])],
            },
        }
        payload_b = {
            **payload_a,
            "id": "b",
            "question_prefill_per_patch_attention": {
                0: [torch.tensor([2.0, 4.0]), torch.tensor([6.0, 8.0, 10.0, 12.0])],
                9: [torch.tensor([20.0, 40.0]), torch.tensor([60.0, 80.0, 100.0, 120.0])],
            },
        }
        norm_edges = np.asarray([0.0, 3.0, 6.0], dtype=np.float64)

        extracted = extract_attention_score_token_count_by_norm(payload_a, norm_bin_edges=norm_edges)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        display_layers, centers, widths, attention_scores, token_counts = extracted
        self.assertEqual(display_layers, [0, 9])
        np.testing.assert_allclose(centers, [1.5, 4.5])
        np.testing.assert_allclose(widths, [3.0, 3.0])
        np.testing.assert_allclose(attention_scores, [[2.0, 5.0], [20.0, 50.0]])
        np.testing.assert_allclose(token_counts, [[3.0, 3.0], [3.0, 3.0]])
        self.assertIsNone(
            extract_attention_score_token_count_by_norm(
                {key: value for key, value in payload_a.items() if key != "question_prefill_per_patch_attention"},
                norm_bin_edges=norm_edges,
            )
        )

        averaged = compute_average_attention_score_token_count_by_norm([payload_a, payload_b], num_bins=2)
        avg_layers, _avg_centers, _avg_widths, avg_attention, avg_token_counts = averaged
        self.assertEqual(avg_layers, [0, 9])
        assert avg_attention is not None and avg_token_counts is not None
        np.testing.assert_allclose(avg_attention, [[3.0, 7.5], [30.0, 75.0]])
        np.testing.assert_allclose(avg_token_counts, [[3.0, 3.0], [3.0, 3.0]])

    def test_attention_score_topk_token_value_norm_sorts_attention_and_uses_matching_norms(self) -> None:
        payload = {
            "task": "SYN",
            "id": "a",
            "question_prefill_attention_maps": {
                "frame_frame_maps": torch.zeros(2, 3, 3, dtype=torch.float32),
                "question_frame_maps": torch.zeros(2, 2, 3, dtype=torch.float32),
                "display_layer_indices": [0, 9],
            },
            "question_prefill_value_norms": torch.tensor(
                [
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                ],
                dtype=torch.float32,
            ),
            "question_prefill_per_patch_attention": {
                0: [torch.tensor([0.2, 0.5]), torch.tensor([0.5, float("nan"), 0.9, 0.1])],
                9: [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 2.0, 4.0, 0.5])],
            },
        }

        extracted = extract_attention_score_topk_token_value_norm(payload, top_k=4)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        display_layers, token_indices, attention_scores, value_norms = extracted
        self.assertEqual(display_layers, [0, 9])
        np.testing.assert_allclose(token_indices[0], [4.0, 1.0, 2.0, 0.0])
        np.testing.assert_allclose(attention_scores[0], [0.9, 0.5, 0.5, 0.2])
        np.testing.assert_allclose(value_norms[0], [14.0, 11.0, 12.0, 10.0])
        np.testing.assert_allclose(token_indices[1], [4.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(attention_scores[1], [4.0, 3.0, 3.0, 2.0])
        np.testing.assert_allclose(value_norms[1], [24.0, 21.0, 22.0, 23.0])

    def test_pooled_attention_score_topk_token_value_norm_reselects_from_example_topk(self) -> None:
        base_maps = {
            "frame_frame_maps": torch.zeros(2, 2, 2, dtype=torch.float32),
            "question_frame_maps": torch.zeros(2, 1, 2, dtype=torch.float32),
            "display_layer_indices": [0, 9],
        }
        payload_a = {
            "task": "SYN",
            "id": "a",
            "question_prefill_attention_maps": base_maps,
            "question_prefill_value_norms": torch.tensor(
                [[90.0, 70.0, 40.0, 10.0], [9.0, 7.0, 4.0, 1.0]],
                dtype=torch.float32,
            ),
            "question_prefill_per_patch_attention": {
                0: [torch.tensor([0.9, 0.7]), torch.tensor([0.4, 0.1])],
                9: [torch.tensor([0.09, 0.07]), torch.tensor([0.04, 0.01])],
            },
        }
        payload_b = {
            **payload_a,
            "id": "b",
            "question_prefill_value_norms": torch.tensor(
                [[85.0, 80.0, 30.0, 20.0], [8.5, 8.0, 3.0, 2.0]],
                dtype=torch.float32,
            ),
            "question_prefill_per_patch_attention": {
                0: [torch.tensor([0.85, 0.8]), torch.tensor([0.3, 0.2])],
                9: [torch.tensor([0.085, 0.08]), torch.tensor([0.03, 0.02])],
            },
        }
        mismatched_payload = {
            **payload_a,
            "id": "mismatch",
            "question_prefill_attention_maps": {
                **base_maps,
                "display_layer_indices": [1, 9],
            },
            "question_prefill_per_patch_attention": {
                1: [torch.tensor([99.0, 98.0]), torch.tensor([97.0, 96.0])],
                9: [torch.tensor([99.0, 98.0]), torch.tensor([97.0, 96.0])],
            },
        }

        pooled = compute_pooled_attention_score_topk_token_value_norm(
            [payload_a, payload_b, mismatched_payload],
            top_k=3,
        )
        display_layers, token_indices, attention_scores, value_norms = pooled
        self.assertEqual(display_layers, [0, 9])
        assert token_indices is not None and attention_scores is not None and value_norms is not None
        np.testing.assert_allclose(token_indices[0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(attention_scores[0], [0.9, 0.85, 0.8])
        np.testing.assert_allclose(value_norms[0], [90.0, 85.0, 80.0])
        np.testing.assert_allclose(token_indices[1], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(attention_scores[1], [0.09, 0.085, 0.08])
        np.testing.assert_allclose(value_norms[1], [9.0, 8.5, 8.0])

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
                    "frame_frame_maps": torch.linspace(0.0, 1.0, steps=12 * 6 * 6, dtype=torch.float32).reshape(12, 6, 6),
                    "question_frame_maps": torch.linspace(
                        0.0,
                        1.0,
                        steps=12 * 3 * 6,
                        dtype=torch.float32,
                    ).reshape(12, 3, 6),
                    "display_layer_indices": question_prefill_layer_indices(36),
                    "attention_frame_indices": [10, 20, 30],
                    "frame_bin_slices": [[0, 2], [2, 4], [4, 6]],
                    "frame_local_bin_spans": [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12]],
                    "frame_token_range": [0, 12],
                    "frame_bin_labels": ["10", "20", "30"],
                    "question_bin_labels": ["q0", "q1", "q2"],
                },
                "question_prefill_sink_bin_token_attention": {
                    "maps": torch.linspace(
                        0.0,
                        1.0,
                        steps=12 * 3 * 2,
                        dtype=torch.float32,
                    ).reshape(12, 3, 2),
                    "display_layer_indices": question_prefill_layer_indices(36),
                    "question_bin_labels": ["q0", "q1", "q2"],
                    "sink_bin_index": 2,
                    "sink_bin_score": 0.75,
                    "sink_frame_label": "20",
                    "sink_frame_index": 20,
                    "sink_frame_local_index": 1,
                    "sink_frame_bin_start": 2,
                    "sink_frame_bin_end": 4,
                    "sink_bin_start": 2,
                    "sink_bin_end": 3,
                    "sink_token_start": 0,
                    "sink_token_end": 2,
                    "sink_global_token_start": 100,
                    "sink_global_token_end": 102,
                    "token_labels": ["tok0", "tok1"],
                    "selection_rule": "sample_common_max_question_frame_bin_mean",
                },
                "question_prefill_value_norms": torch.linspace(
                    1.0,
                    2.0,
                    steps=12 * 12,
                    dtype=torch.float32,
                ).reshape(12, 12),
                "question_prefill_per_patch_attention": {
                    int(layer_idx): [
                        torch.linspace(0.0, 0.3, steps=4, dtype=torch.float32),
                        torch.linspace(0.4, 0.7, steps=4, dtype=torch.float32),
                        torch.linspace(0.8, 1.1, steps=4, dtype=torch.float32),
                    ]
                    for layer_idx in question_prefill_layer_indices(36)
                },
                "question_prefill_value_norm_unit": "frame_patch_token",
            }
            torch.save(payload, examples_dir / "sample.pt")
            shorter_payload = {
                **payload,
                "id": 2,
                "question_prefill_attention_maps": {
                    "frame_frame_maps": torch.linspace(
                        0.0,
                        1.0,
                        steps=12 * 4 * 4,
                        dtype=torch.float32,
                    ).reshape(12, 4, 4),
                    "question_frame_maps": torch.linspace(
                        0.0,
                        1.0,
                        steps=12 * 3 * 4,
                        dtype=torch.float32,
                    ).reshape(12, 3, 4),
                    "display_layer_indices": question_prefill_layer_indices(36),
                    "attention_frame_indices": [10, 20],
                    "frame_bin_slices": [[0, 2], [2, 4]],
                    "frame_local_bin_spans": [[0, 1], [1, 3], [3, 6], [6, 10]],
                    "frame_token_range": [0, 10],
                    "frame_bin_labels": ["10", "20"],
                    "question_bin_labels": ["q0", "q1", "q2"],
                },
            }
            torch.save(shorter_payload, examples_dir / "sample_shorter.pt")

            generate_attention_plots(result_dir)

            self.assertTrue((result_dir / "plots" / "question_prefill_frame_frame_maps_average.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_frame_frame_maps_average_rawscale.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_question_frame_maps_average.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_question_frame_maps_average_rawscale.png").exists())
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_token_count_by_bin_average.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_token_count_by_bin_average_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_token_count_by_norm_average.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_token_count_by_norm_average_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_top20_token_value_norm_pooled.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "question_prefill_attention_score_top20_token_value_norm_pooled_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (result_dir / "plots" / "examples" / "sample" / "question_prefill_frame_frame_maps.png").exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_frame_frame_maps_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (result_dir / "plots" / "examples" / "sample" / "question_prefill_question_frame_maps.png").exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_question_frame_maps_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (result_dir / "plots" / "examples" / "sample" / "question_prefill_value_norms.png").exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_token_count_by_bin.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_token_count_by_bin_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_token_count_by_norm.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_token_count_by_norm_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_top20_token_value_norm.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_attention_score_top20_token_value_norm_rawscale.png"
                ).exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "sample"
                    / "question_prefill_sink_bin_token_attention.png"
                ).exists()
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
                        steps=12 * 3 * 3,
                        dtype=torch.float32,
                    ).reshape(12, 3, 3),
                    "question_frame_maps": torch.linspace(
                        1.0e-10,
                        6.0e-10,
                        steps=12 * 2 * 3,
                        dtype=torch.float32,
                    ).reshape(12, 2, 3),
                    "display_layer_indices": question_prefill_layer_indices(36),
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

            generate_attention_plots(result_dir)

            self.assertTrue((result_dir / "plots" / "question_prefill_percentile_mean.png").exists())
            self.assertTrue((result_dir / "plots" / "question_prefill_percentile_mean_pooled.png").exists())
            self.assertFalse((result_dir / "plots" / "layerwise_percentile_heatmaps.png").exists())
            self.assertTrue(
                (result_dir / "plots" / "examples" / "asi_sample" / "question_prefill_attention_heatmap.png").exists()
            )
            self.assertTrue(
                (
                    result_dir
                    / "plots"
                    / "examples"
                    / "asi_sample"
                    / "question_prefill_attention_heatmap_rawscale.png"
                ).exists()
            )
            self.assertFalse((result_dir / "plots" / "examples" / "hld_sample").exists())

    def test_siglip_plotting_filters_hld_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_dir = Path(tmp_dir)
            records = [
                {
                    "_key": "ASI:1",
                    "split": "backward",
                    "task": "ASI",
                    "metrics": {
                        "siglip_similarity": {
                            "recent4_mean_percentile": 0.25,
                        },
                    },
                },
                {
                    "_key": "HLD:2",
                    "split": "backward",
                    "task": "HLD",
                    "metrics": {
                        "siglip_similarity": {
                            "recent4_mean_percentile": 0.99,
                        },
                    },
                },
            ]
            with (result_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config": {"similarity_backends": ["siglip"]},
                        "metrics": {"siglip_similarity": {}},
                    },
                    handle,
                )

            generate_siglip_plots(result_dir)

            self.assertTrue((result_dir / "plots" / "similarity_mean_percentile.png").exists())
            summary = build_experiment_summary(
                filter_siglip_excluded_tasks(records),
                config={"similarity_backends": ["siglip"]},
            )
            labels, means, _stds, _group_types = extract_siglip_bar_data(summary, "siglip_similarity")
            self.assertNotIn("HLD", labels)
            self.assertIn("ASI", labels)
            self.assertEqual(means[labels.index("Backward\nAvg")], 0.25)
            self.assertEqual(means[labels.index("Total\nAvg")], 0.25)


if __name__ == "__main__":
    unittest.main()
