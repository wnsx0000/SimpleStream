from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.recent_window_eval import decode_video_to_chunks_qwen, select_attention_frame_indices
from lib.recent_window_eval_qwen3 import RecentWindowQAModel as _BaseQwen3RecentWindowQAModel

DISPLAY_LAYER_COUNT = 10
QUESTION_PREFILL_DISPLAY_LAYER_COUNT = 5
QUESTION_PREFILL_FRAME_MAP_MAX_BINS = 96
QUESTION_PREFILL_QUESTION_MAP_MAX_BINS = 32


def parse_csv_options(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text)).strip("-")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, ensure_ascii=False)


def release_unused_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tie_aware_percentiles(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 1:
        return [1.0]
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(arr.size, dtype=np.float64)
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return (ranks / float(arr.size - 1)).tolist()


def topk_overlap(scores: list[float], selected_indices: list[int], top_k: int | None = None) -> float:
    if not scores or not selected_indices:
        return 0.0
    k = int(top_k or len(selected_indices))
    k = max(1, min(k, len(scores)))
    top_indices = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")[:k]
    overlap = len(set(int(idx) for idx in top_indices.tolist()) & set(int(idx) for idx in selected_indices))
    return float(overlap / max(len(selected_indices), 1))


def summarize_scalar_metric(scores: list[float], recent_indices: list[int]) -> dict[str, Any]:
    percentiles = tie_aware_percentiles(scores)
    recent_percentiles = [float(percentiles[idx]) for idx in recent_indices if 0 <= idx < len(percentiles)]
    return {
        "frame_scores": [float(score) for score in scores],
        "frame_percentiles": [float(value) for value in percentiles],
        "recent4_mean_percentile": float(np.mean(recent_percentiles)) if recent_percentiles else 0.0,
        "recent4_top4_overlap": topk_overlap(scores, recent_indices, top_k=len(recent_indices)),
    }


def uniform_center_indices(total_count: int, target_count: int = DISPLAY_LAYER_COUNT) -> list[int]:
    total_count = int(total_count)
    target_count = int(target_count)
    if total_count <= 0 or target_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    if target_count == 1:
        return [0]

    # Keep the selection monotonic while still spreading indices uniformly and
    # always including the first and last decoder layers.
    selected: list[int] = []
    last_index = -1
    for position in range(target_count):
        remaining_slots = target_count - position - 1
        min_index = last_index + 1
        max_index = total_count - remaining_slots - 1
        raw_index = int(round(position * (total_count - 1) / float(target_count - 1)))
        clamped_index = min(max(raw_index, min_index), max_index)
        selected.append(clamped_index)
        last_index = clamped_index
    return selected


def summarize_layerwise_metric(
    layer_scores: torch.Tensor,
    recent_indices: list[int],
    display_layer_count: int = DISPLAY_LAYER_COUNT,
    display_layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    if layer_scores.ndim != 2:
        raise ValueError(f"Expected [layers, frames] scores, got shape={tuple(layer_scores.shape)}")

    layer_scores = layer_scores.detach().float().cpu()
    layer_percentiles = torch.tensor(
        [tie_aware_percentiles(row.tolist()) for row in layer_scores],
        dtype=torch.float32,
    )
    if display_layer_indices is None:
        display_layer_indices = uniform_center_indices(layer_scores.shape[0], display_layer_count)
    else:
        display_layer_indices = [int(index) for index in display_layer_indices]
    display_layer_scores = layer_scores[display_layer_indices]
    display_layer_percentiles = layer_percentiles[display_layer_indices]
    display_layer_recent_mean = [
        float(layer_percentiles[idx, recent_indices].mean().item()) if recent_indices else 0.0
        for idx in display_layer_indices
    ]
    display_layer_overlap = [
        topk_overlap(layer_scores[idx].tolist(), recent_indices, top_k=len(recent_indices))
        for idx in display_layer_indices
    ]

    mean_scores = layer_scores.mean(dim=0)
    mean_summary = summarize_scalar_metric(mean_scores.tolist(), recent_indices)

    num_layers = int(layer_scores.shape[0])
    tail_layer_indices = sorted(set(range(max(0, num_layers - 8), num_layers)) - set(display_layer_indices))
    tail_layer_scores = layer_scores[tail_layer_indices] if tail_layer_indices else torch.empty(0, layer_scores.shape[1])

    return {
        "num_layers_total": num_layers,
        "display_layer_indices": [int(index) for index in display_layer_indices],
        "layer_attention_scores": display_layer_scores.tolist(),
        "layer_attention_percentiles": display_layer_percentiles.tolist(),
        "layer_recent4_mean_percentile": [float(value) for value in display_layer_recent_mean],
        "layer_recent4_top4_overlap": [float(value) for value in display_layer_overlap],
        "tail_layer_indices": [int(index) for index in tail_layer_indices],
        "tail_layer_attention_scores": tail_layer_scores.tolist(),
        "mean_attention_score": mean_summary["frame_scores"],
        "mean_percentile": mean_summary["frame_percentiles"],
        "recent4_mean_percentile": mean_summary["recent4_mean_percentile"],
        "recent4_top4_overlap": mean_summary["recent4_top4_overlap"],
    }


def sample_metric_layer_field(metric: dict[str, Any], field: str) -> tuple[np.ndarray | None, list[int]]:
    if field not in metric:
        return None, []

    values = np.asarray(metric[field], dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[0] < 1:
        return None, []

    saved_indices = [int(index) for index in metric.get("display_layer_indices", [])]
    if len(saved_indices) == values.shape[0]:
        return values, saved_indices

    display_indices = uniform_center_indices(values.shape[0], DISPLAY_LAYER_COUNT)
    if not display_indices:
        return None, []
    return values[display_indices], display_indices


def cosine_scores_against_query(features: torch.Tensor, query_feature: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected [frames, dim] features, got shape={tuple(features.shape)}")
    if query_feature.ndim != 1:
        raise ValueError(f"Expected [dim] query feature, got shape={tuple(query_feature.shape)}")
    normalized_features = F.normalize(features.float(), dim=-1)
    normalized_query = F.normalize(query_feature.float(), dim=-1)
    return normalized_features @ normalized_query


def frame_token_counts_from_grid(grid_thw: torch.Tensor, merge_size: int) -> list[int]:
    rows = grid_thw.detach().cpu().to(torch.long)
    counts: list[int] = []
    denom = int(max(1, merge_size) ** 2)
    for row in rows:
        count = int(torch.prod(row).item()) // denom
        if count < 1:
            raise ValueError(f"Invalid frame token count from grid row={row.tolist()} merge_size={merge_size}")
        counts.append(count)
    return counts


def allocate_proportional_bin_counts(token_counts: list[int], max_bins: int) -> list[int]:
    counts = np.asarray(token_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size < 1:
        return []
    if np.any(counts < 1):
        raise ValueError(f"Token counts must be >= 1, got {counts.tolist()}")

    target_bins = int(min(max_bins, int(counts.sum())))
    if target_bins < int(counts.size):
        target_bins = int(counts.size)

    desired = counts.astype(np.float64) / float(counts.sum()) * float(target_bins)
    bins = np.floor(desired).astype(np.int64)
    bins = np.maximum(bins, 1)
    bins = np.minimum(bins, counts)

    current_total = int(bins.sum())
    if current_total > target_bins:
        removable = bins > 1
        order = np.argsort(desired - bins, kind="mergesort")
        for idx in order:
            while current_total > target_bins and removable[idx]:
                bins[idx] -= 1
                current_total -= 1
                removable[idx] = bins[idx] > 1
            if current_total == target_bins:
                break
    elif current_total < target_bins:
        order = np.argsort(-(desired - bins), kind="mergesort")
        remaining_capacity = counts - bins
        for idx in order:
            while current_total < target_bins and remaining_capacity[idx] > 0:
                bins[idx] += 1
                remaining_capacity[idx] -= 1
                current_total += 1
            if current_total == target_bins:
                break

    if int(bins.sum()) != target_bins:
        raise RuntimeError(
            "Failed to allocate proportional bin counts: "
            f"counts={counts.tolist()} bins={bins.tolist()} target_bins={target_bins}"
        )
    return [int(value) for value in bins.tolist()]


def build_uniform_token_bin_spans(
    total_tokens: int,
    num_bins: int,
    offset: int = 0,
) -> list[tuple[int, int]]:
    total_tokens = int(total_tokens)
    num_bins = int(num_bins)
    offset = int(offset)
    if total_tokens < 1:
        raise ValueError(f"total_tokens must be >= 1, got {total_tokens}")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    if num_bins > total_tokens:
        raise ValueError(f"num_bins must be <= total_tokens, got bins={num_bins} total_tokens={total_tokens}")

    base_size = total_tokens // num_bins
    remainder = total_tokens % num_bins
    spans: list[tuple[int, int]] = []
    cursor = offset
    for bin_idx in range(num_bins):
        bin_size = base_size + (1 if bin_idx < remainder else 0)
        spans.append((cursor, cursor + bin_size))
        cursor += bin_size
    return spans


def format_token_bin_label(prefix: str, start_index: int, end_index: int) -> str:
    start_index = int(start_index)
    end_index = int(end_index)
    if end_index <= start_index:
        raise ValueError(f"Invalid bin label range: start={start_index}, end={end_index}")
    if end_index - start_index == 1:
        return f"{prefix}{start_index}"
    return f"{prefix}{start_index}-{end_index - 1}"


@dataclass(frozen=True)
class QuestionPrefillAttentionMapMetadata:
    frame_token_start: int
    frame_token_end: int
    question_token_start: int
    question_token_end: int
    frame_local_bin_spans: list[tuple[int, int]]
    question_local_bin_spans: list[tuple[int, int]]
    frame_bin_slices: list[tuple[int, int]]
    frame_bin_labels: list[str]
    question_bin_labels: list[str]
    attention_frame_indices: list[int]


def build_question_prefill_attention_map_metadata(
    frame_token_spans: list[tuple[int, int]],
    query_positions: list[int],
    attention_frame_indices: list[int],
    frame_axis_max_bins: int = QUESTION_PREFILL_FRAME_MAP_MAX_BINS,
    question_axis_max_bins: int = QUESTION_PREFILL_QUESTION_MAP_MAX_BINS,
) -> QuestionPrefillAttentionMapMetadata:
    if not frame_token_spans:
        raise ValueError("At least one frame token span is required for question-prefill attention maps.")
    if len(frame_token_spans) != len(attention_frame_indices):
        raise ValueError(
            "Frame token spans and attention frame indices must have the same length: "
            f"{len(frame_token_spans)} vs {len(attention_frame_indices)}"
        )
    if not query_positions:
        raise ValueError("At least one query token position is required for question-prefill attention maps.")

    frame_token_start = int(frame_token_spans[0][0])
    frame_token_end = int(frame_token_spans[-1][1])
    expected_frame_start = frame_token_start
    frame_token_counts: list[int] = []
    for start, end in frame_token_spans:
        start = int(start)
        end = int(end)
        if start != expected_frame_start:
            raise ValueError(
                "Frame token spans must be contiguous for pooling: "
                f"expected_start={expected_frame_start} got_start={start}"
            )
        if end <= start:
            raise ValueError(f"Invalid frame token span: start={start} end={end}")
        frame_token_counts.append(end - start)
        expected_frame_start = end

    question_token_start = int(query_positions[0])
    question_token_end = int(query_positions[-1]) + 1
    expected_query_positions = list(range(question_token_start, question_token_end))
    if [int(position) for position in query_positions] != expected_query_positions:
        raise ValueError("Query positions must form one contiguous block for question-prefill attention maps.")

    frame_bin_counts = allocate_proportional_bin_counts(frame_token_counts, max_bins=frame_axis_max_bins)
    frame_local_bin_spans: list[tuple[int, int]] = []
    frame_bin_slices: list[tuple[int, int]] = []
    pooled_cursor = 0
    local_cursor = 0
    for frame_idx, (token_count, bin_count) in enumerate(zip(frame_token_counts, frame_bin_counts)):
        local_spans = build_uniform_token_bin_spans(token_count, bin_count, offset=local_cursor)
        frame_local_bin_spans.extend(local_spans)
        frame_bin_slices.append((pooled_cursor, pooled_cursor + bin_count))
        pooled_cursor += bin_count
        local_cursor += token_count

    question_token_count = question_token_end - question_token_start
    question_bin_count = min(question_axis_max_bins, question_token_count)
    question_global_bin_spans = build_uniform_token_bin_spans(
        question_token_count,
        question_bin_count,
        offset=question_token_start,
    )
    question_local_bin_spans = [
        (start - question_token_start, end - question_token_start)
        for start, end in question_global_bin_spans
    ]

    return QuestionPrefillAttentionMapMetadata(
        frame_token_start=frame_token_start,
        frame_token_end=frame_token_end,
        question_token_start=question_token_start,
        question_token_end=question_token_end,
        frame_local_bin_spans=frame_local_bin_spans,
        question_local_bin_spans=question_local_bin_spans,
        frame_bin_slices=frame_bin_slices,
        frame_bin_labels=[str(int(index)) for index in attention_frame_indices],
        question_bin_labels=[
            format_token_bin_label(
                "q",
                start - question_token_start,
                end - question_token_start,
            )
            for start, end in question_global_bin_spans
        ],
        attention_frame_indices=[int(index) for index in attention_frame_indices],
    )


def mean_pool_2d_by_spans(
    matrix: torch.Tensor,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor for pooling, got shape={tuple(matrix.shape)}")
    if not row_spans or not col_spans:
        return torch.empty((len(row_spans), len(col_spans)), dtype=matrix.dtype, device=matrix.device)

    prefix = matrix.cumsum(dim=0).cumsum(dim=1)
    prefix = F.pad(prefix, (1, 0, 1, 0))

    row_starts = torch.tensor([int(start) for start, _ in row_spans], dtype=torch.long, device=matrix.device)
    row_ends = torch.tensor([int(end) for _, end in row_spans], dtype=torch.long, device=matrix.device)
    col_starts = torch.tensor([int(start) for start, _ in col_spans], dtype=torch.long, device=matrix.device)
    col_ends = torch.tensor([int(end) for _, end in col_spans], dtype=torch.long, device=matrix.device)

    pooled_sum = (
        prefix[row_ends[:, None], col_ends[None, :]]
        - prefix[row_starts[:, None], col_ends[None, :]]
        - prefix[row_ends[:, None], col_starts[None, :]]
        + prefix[row_starts[:, None], col_starts[None, :]]
    )
    areas = (
        (row_ends - row_starts).to(dtype=matrix.dtype)[:, None]
        * (col_ends - col_starts).to(dtype=matrix.dtype)[None, :]
    )
    return pooled_sum / areas


def build_question_prefill_attention_maps(
    attn_weights: torch.Tensor,
    metadata: QuestionPrefillAttentionMapMetadata,
) -> dict[str, torch.Tensor]:
    if attn_weights.ndim != 3:
        raise ValueError(f"Expected [heads, seq_len, seq_len] attention weights, got shape={tuple(attn_weights.shape)}")

    mean_attention = attn_weights.float().mean(dim=0)
    frame_attention = mean_attention[
        metadata.frame_token_start : metadata.frame_token_end,
        metadata.frame_token_start : metadata.frame_token_end,
    ]
    question_attention = mean_attention[
        metadata.question_token_start : metadata.question_token_end,
        metadata.frame_token_start : metadata.frame_token_end,
    ]
    return {
        "frame_frame_map": mean_pool_2d_by_spans(
            frame_attention,
            metadata.frame_local_bin_spans,
            metadata.frame_local_bin_spans,
        ).detach().cpu(),
        "question_frame_map": mean_pool_2d_by_spans(
            question_attention,
            metadata.question_local_bin_spans,
            metadata.frame_local_bin_spans,
        ).detach().cpu(),
    }


def flatten_chunks(
    chunks: list[Any],
    recent_frames_only: int,
) -> tuple[list[Image.Image], list[dict[str, Any]], list[int], list[int]]:
    window_size = max(1, int(recent_frames_only))
    recent_chunks = list(chunks[-window_size:])
    recent_chunk_ids = [int(chunk.chunk_index) for chunk in recent_chunks]
    recent_chunk_set = set(recent_chunk_ids)

    frames: list[Image.Image] = []
    frame_rows: list[dict[str, Any]] = []
    recent_indices: list[int] = []

    for chunk in chunks:
        for frame, timestamp in zip(chunk.frames, chunk.frame_timestamps):
            frame_index = len(frames)
            is_recent = int(chunk.chunk_index) in recent_chunk_set
            frames.append(frame)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "chunk_index": int(chunk.chunk_index),
                    "timestamp": float(timestamp),
                    "is_recent": bool(is_recent),
                }
            )
            if is_recent:
                recent_indices.append(frame_index)

    return frames, frame_rows, recent_indices, recent_chunk_ids


@dataclass
class LayerwiseFrameAttentionCollector:
    frame_token_spans: list[tuple[int, int]]
    query_positions: list[int]
    num_layers: int
    save_raw: bool = False
    map_layer_indices: list[int] | None = None
    question_prefill_map_metadata: QuestionPrefillAttentionMapMetadata | None = None

    def __post_init__(self) -> None:
        self.layer_scores: list[torch.Tensor | None] = [None] * self.num_layers
        self.layer_raw_attentions: dict[int, torch.Tensor] = {}
        self.map_layer_index_set = set(int(index) for index in (self.map_layer_indices or []))
        self.layer_attention_maps: dict[int, dict[str, torch.Tensor]] = {}

    def make_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any):
            if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    "Qwen3 text self-attention did not return attention weights. "
                    "Run with --attn-implementation eager."
                )

            attn_weights = output[1]
            if attn_weights.ndim != 4:
                raise RuntimeError(f"Unexpected attention shape from layer {layer_idx}: {tuple(attn_weights.shape)}")

            selected = attn_weights[0]
            if self.question_prefill_map_metadata is not None and layer_idx in self.map_layer_index_set:
                self.layer_attention_maps[layer_idx] = build_question_prefill_attention_maps(
                    selected,
                    metadata=self.question_prefill_map_metadata,
                )
            if self.query_positions:
                selected = selected[:, self.query_positions, :]
            if selected.ndim == 2:
                selected = selected.unsqueeze(1)

            frame_scores = []
            for start, end in self.frame_token_spans:
                frame_scores.append(selected[..., start:end].sum(dim=-1))
            stacked = torch.stack(frame_scores, dim=-1)
            self.layer_scores[layer_idx] = stacked.mean(dim=1).mean(dim=0).detach().float().cpu()

            if self.save_raw:
                self.layer_raw_attentions[layer_idx] = attn_weights[0].detach().float().cpu()

            # Return modified output with attention weights replaced by None to
            # immediately free the large [batch, heads, seq_len, seq_len] GPU tensor
            # instead of keeping it alive through the rest of the decoder layer.
            return (output[0], None) + output[2:] if len(output) > 2 else (output[0], None)

        return hook

    def as_tensor(self) -> torch.Tensor:
        missing = [str(index) for index, value in enumerate(self.layer_scores) if value is None]
        if missing:
            raise RuntimeError(f"Missing captured attentions for layers: {', '.join(missing)}")
        return torch.stack([value for value in self.layer_scores if value is not None], dim=0)

    def export_question_prefill_attention_maps(
        self,
        display_layer_indices: list[int],
    ) -> dict[str, Any] | None:
        metadata = self.question_prefill_map_metadata
        if metadata is None or not display_layer_indices:
            return None

        frame_frame_maps: list[torch.Tensor] = []
        question_frame_maps: list[torch.Tensor] = []
        for layer_idx in display_layer_indices:
            layer_payload = self.layer_attention_maps.get(int(layer_idx))
            if layer_payload is None:
                return None
            frame_frame_maps.append(layer_payload["frame_frame_map"])
            question_frame_maps.append(layer_payload["question_frame_map"])

        return {
            "frame_frame_maps": torch.stack(frame_frame_maps, dim=0),
            "question_frame_maps": torch.stack(question_frame_maps, dim=0),
            "display_layer_indices": [int(index) for index in display_layer_indices],
            "attention_frame_indices": [int(index) for index in metadata.attention_frame_indices],
            "frame_bin_slices": [[int(start), int(end)] for start, end in metadata.frame_bin_slices],
            "frame_bin_labels": list(metadata.frame_bin_labels),
            "question_bin_labels": list(metadata.question_bin_labels),
        }


class SiglipFrameEncoder:
    def __init__(self, model_name: str, device: str | torch.device) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _as_feature_tensor(self, value: Any, source: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        pooled = getattr(value, "pooler_output", None)
        if isinstance(pooled, torch.Tensor):
            return pooled
        hidden = getattr(value, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden
        if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
            return value[0]
        raise TypeError(f"Unexpected SigLIP {source} feature type: {type(value)}")

    @torch.inference_mode()
    def encode_frames(self, frames: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            inputs = self.processor(images=batch_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            features = self._as_feature_tensor(
                self.model.get_image_features(pixel_values=pixel_values),
                source="image",
            )
            batches.append(F.normalize(features.float(), dim=-1).cpu())
            del inputs, pixel_values, features
        frame_features = torch.cat(batches, dim=0)
        del batches
        return frame_features

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        query_text = str(text).strip()
        if not query_text:
            raise ValueError("SigLIP frame-question similarity requires a non-empty question.")

        inputs = self.processor(text=[query_text], padding=True, truncation=True, return_tensors="pt")
        text_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
        }
        if not text_inputs:
            raise RuntimeError("SigLIP processor did not produce text inputs.")

        features = self._as_feature_tensor(
            self.model.get_text_features(**text_inputs),
            source="text",
        )
        text_feature = F.normalize(features.float(), dim=-1)[0].cpu()
        del inputs, text_inputs, features
        return text_feature


def resolve_siglip_device(fallback_device: str | torch.device | None = None) -> torch.device:
    if torch.cuda.device_count() > 1:
        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    if fallback_device is not None:
        return torch.device(fallback_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_analysis_subset(
    total_frames: int,
    recent_indices: list[int],
    max_analysis_frames: int,
) -> tuple[list[int], list[int], str]:
    analysis_frame_indices, analysis_sampling_strategy = select_attention_frame_indices(
        total_frames=total_frames,
        recent_indices=recent_indices,
        max_analysis_frames=max_analysis_frames,
    )
    analysis_index_lookup = {frame_idx: local_idx for local_idx, frame_idx in enumerate(analysis_frame_indices)}
    analysis_recent_indices = [
        analysis_index_lookup[frame_idx]
        for frame_idx in recent_indices
        if frame_idx in analysis_index_lookup
    ]
    return analysis_frame_indices, analysis_recent_indices, analysis_sampling_strategy


class SiglipOnlyRecent4FrameSaliencyAnalyzer:
    def __init__(
        self,
        siglip_model_name: str = "google/siglip-so400m-patch14-384",
        device: str | torch.device = "auto",
    ) -> None:
        self.siglip_model_name = siglip_model_name
        self.device = device
        self._siglip_encoder: SiglipFrameEncoder | None = None

    def get_siglip_encoder(self) -> SiglipFrameEncoder:
        if self._siglip_encoder is None:
            fallback_device = None if str(self.device) == "auto" else self.device
            self._siglip_encoder = SiglipFrameEncoder(
                self.siglip_model_name,
                resolve_siglip_device(fallback_device=fallback_device),
            )
        return self._siglip_encoder

    @torch.inference_mode()
    def analyze_sample(
        self,
        video_path: str,
        prompt: str,
        similarity_text: str | None,
        chunk_duration: float,
        fps: float,
        recent_frames_only: int,
        similarity_backends: list[str],
        attention_modes: list[str],
        max_analysis_frames: int,
        save_example_matrices: bool = False,
        save_raw_attentions: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        del prompt
        if attention_modes:
            raise ValueError("SigLIP-only analyzer does not support attention modes.")

        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
        )
        if not chunks:
            raise ValueError(f"No chunks decoded from video: {video_path}")

        frames, frame_rows, recent_indices, recent_chunk_ids = flatten_chunks(chunks, recent_frames_only)
        num_sampled_frames = len(frames)
        del chunks

        metrics: dict[str, Any] = {}
        example_payload: dict[str, Any] = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "frame_rows": frame_rows,
            "recent_frame_indices": recent_indices,
            "recent_chunk_ids": recent_chunk_ids,
        } if (save_example_matrices or save_raw_attentions) else {}

        analysis_frame_indices, analysis_recent_indices, analysis_sampling_strategy = build_analysis_subset(
            total_frames=num_sampled_frames,
            recent_indices=recent_indices,
            max_analysis_frames=max_analysis_frames,
        )

        if "siglip" in similarity_backends:
            siglip_encoder = self.get_siglip_encoder()
            analysis_frames = [frames[frame_idx] for frame_idx in analysis_frame_indices]
            siglip_features = siglip_encoder.encode_frames(analysis_frames)
            del analysis_frames
            siglip_question_feature = siglip_encoder.encode_text(similarity_text or "")
            siglip_scores = cosine_scores_against_query(siglip_features, siglip_question_feature).tolist()
            metrics["siglip_similarity"] = summarize_scalar_metric(siglip_scores, analysis_recent_indices)
            metrics["siglip_similarity"]["analysis_frame_indices"] = analysis_frame_indices
            metrics["siglip_similarity"]["recent_frame_indices_within_analysis"] = analysis_recent_indices
            metrics["siglip_similarity"]["analysis_sampling_strategy"] = analysis_sampling_strategy
            if save_example_matrices:
                example_payload["siglip_similarity_query"] = str(similarity_text or "")
            del siglip_features, siglip_question_feature, siglip_scores

        del frames

        record = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "num_sampled_frames": num_sampled_frames,
            "recent_chunk_ids": recent_chunk_ids,
            "recent_frame_indices": recent_indices,
            "attention_frame_indices": list(range(num_sampled_frames)),
            "attention_sampling_strategy": "all_frames",
            "frames": frame_rows,
            "metrics": metrics,
            "attention_skipped_reason": None,
        }

        if save_example_matrices or save_raw_attentions:
            example_payload["metrics"] = metrics
        return record, (example_payload if save_example_matrices or save_raw_attentions else None)


class Qwen3Recent4FrameSaliencyAnalyzer(_BaseQwen3RecentWindowQAModel):
    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "eager",
        siglip_model_name: str = "google/siglip-so400m-patch14-384",
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
        self.siglip_model_name = siglip_model_name
        self._siglip_encoder: SiglipFrameEncoder | None = None

    def get_siglip_encoder(self) -> SiglipFrameEncoder:
        if self._siglip_encoder is None:
            siglip_device = resolve_siglip_device(fallback_device=self._get_visual_device())
            self._siglip_encoder = SiglipFrameEncoder(self.siglip_model_name, siglip_device)
        return self._siglip_encoder

    def _get_text_layers(self) -> list[Any]:
        pending = [self._get_text_model()]
        visited: set[int] = set()
        while pending:
            module = pending.pop(0)
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)

            layers = getattr(module, "layers", None)
            if layers is not None:
                return list(layers)

            for attr_name in ("language_model", "model", "decoder"):
                child = getattr(module, attr_name, None)
                if child is not None:
                    pending.append(child)

        text_model = self._get_text_model()
        raise RuntimeError(
            "Unable to locate Qwen3 text decoder layers for attention capture. "
            f"text_model_type={type(text_model).__name__}"
        )

    def _build_cached_multimodal_inputs(
        self,
        cached_embeds: torch.Tensor,
        cached_grid_thw: torch.Tensor,
        frame_token_counts: list[int],
        question: str,
    ) -> dict[str, Any]:
        text_device = self._get_text_input_device()
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        if not question_ids:
            raise ValueError("Question must tokenize to at least one token for attention analysis.")

        image_token_start = 0
        input_ids_list: list[int] = [self.im_start_id]
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        input_ids_list.append(self.vision_start_id)
        image_token_start = len(input_ids_list)
        input_ids_list.extend([self.image_token_id] * int(cached_embeds.shape[0]))
        input_ids_list.append(self.vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        question_start = len(input_ids_list)
        input_ids_list.extend(question_ids)
        question_end = len(input_ids_list)
        input_ids_list.append(self.im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        frame_token_spans: list[tuple[int, int]] = []
        offset = image_token_start
        for count in frame_token_counts:
            frame_token_spans.append((offset, offset + count))
            offset += count

        if offset != image_token_start + int(cached_embeds.shape[0]):
            raise ValueError(
                "Frame token span construction did not match total vision tokens: "
                f"offset={offset} expected={image_token_start + int(cached_embeds.shape[0])}"
            )

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=text_device)
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(inputs_embeds), cached_embeds)
        position_ids, _ = text_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=cached_grid_thw.to(text_device),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "question_token_positions": list(range(question_start, question_end)),
            "frame_token_spans": frame_token_spans,
        }

    def _run_with_collector(
        self,
        collector: LayerwiseFrameAttentionCollector | None,
        use_cache: bool = True,
        **model_kwargs: Any,
    ) -> Any:
        handles = []
        if collector is not None:
            for layer_idx, layer in enumerate(self._get_text_layers()):
                handles.append(layer.self_attn.register_forward_hook(collector.make_hook(layer_idx)))
        try:
            return self.model(
                use_cache=use_cache,
                return_dict=True,
                **model_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def analyze_sample(
        self,
        video_path: str,
        prompt: str,
        similarity_text: str | None,
        chunk_duration: float,
        fps: float,
        recent_frames_only: int,
        similarity_backends: list[str],
        attention_modes: list[str],
        max_analysis_frames: int,
        save_example_matrices: bool = False,
        save_raw_attentions: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        # Decode the video into chunk-level frame samples and fail fast if nothing is available.
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
        )
        if not chunks:
            raise ValueError(f"No chunks decoded from video: {video_path}")

        # Flatten the chunked output into frame-level structures and collect analysis metadata.
        frames, frame_rows, recent_indices, recent_chunk_ids = flatten_chunks(chunks, recent_frames_only)
        num_sampled_frames = len(frames)
        del chunks

        # Initialize the metric store and the optional example payload scaffold.
        metrics: dict[str, Any] = {}
        example_payload: dict[str, Any] = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "frame_rows": frame_rows,
            "recent_frame_indices": recent_indices,
            "recent_chunk_ids": recent_chunk_ids,
        } if (save_example_matrices or save_raw_attentions) else {}

        # Compute one shared analysis subset so SigLIP and attention use the same
        # frame selection policy when max_analysis_frames is active.
        analysis_frame_indices, analysis_recent_indices, analysis_sampling_strategy = build_analysis_subset(
            total_frames=num_sampled_frames,
            recent_indices=recent_indices,
            max_analysis_frames=max_analysis_frames,
        )
        analysis_index_lookup = {frame_idx: local_idx for local_idx, frame_idx in enumerate(analysis_frame_indices)}

        # If requested, compute SigLIP frame-text similarity and summarize it over recent frames.
        if "siglip" in similarity_backends:
            siglip_encoder = self.get_siglip_encoder()
            analysis_frames = [frames[frame_idx] for frame_idx in analysis_frame_indices]
            siglip_features = siglip_encoder.encode_frames(analysis_frames)
            del analysis_frames
            siglip_question_feature = siglip_encoder.encode_text(similarity_text or "")
            siglip_scores = cosine_scores_against_query(siglip_features, siglip_question_feature).tolist()
            metrics["siglip_similarity"] = summarize_scalar_metric(siglip_scores, analysis_recent_indices)
            metrics["siglip_similarity"]["analysis_frame_indices"] = analysis_frame_indices
            metrics["siglip_similarity"]["recent_frame_indices_within_analysis"] = analysis_recent_indices
            metrics["siglip_similarity"]["analysis_sampling_strategy"] = analysis_sampling_strategy
            if save_example_matrices:
                example_payload["siglip_similarity_query"] = str(similarity_text or "")
            del siglip_features, siglip_question_feature, siglip_scores

        # Initialize default attention metadata so the output schema stays stable even when skipped.
        attention_skipped_reason: str | None = None
        attention_frame_indices = list(range(num_sampled_frames))
        attention_recent_indices: list[int] = recent_indices
        attention_sampling_strategy = "all_frames"
        if attention_modes:
            attention_frame_indices = analysis_frame_indices
            attention_recent_indices = analysis_recent_indices
            attention_sampling_strategy = analysis_sampling_strategy
            attention_index_lookup = analysis_index_lookup

            # Encode only the selected attention frames so max_analysis_frames reduces vision memory too.
            attention_frames = [frames[frame_idx] for frame_idx in attention_frame_indices]
            attention_embeds, attention_grid = self.encode_vision(attention_frames)
            del attention_frames
            attention_frame_token_counts = frame_token_counts_from_grid(attention_grid, self.merge_size)
            for row in frame_rows:
                row["used_for_attention"] = bool(int(row["frame_index"]) in attention_index_lookup)

            # Collect layer-wise prefill attention showing which frames the question tokens focus on.
            if "question_prefill" in attention_modes:
                text_layers = self._get_text_layers()
                num_text_layers = len(text_layers)
                question_prefill_display_layers = uniform_center_indices(
                    num_text_layers,
                    QUESTION_PREFILL_DISPLAY_LAYER_COUNT,
                )
                question_prefill_tail_layers = sorted(
                    set(range(max(0, num_text_layers - 8), num_text_layers)) - set(question_prefill_display_layers)
                )
                question_only_inputs = self._build_cached_multimodal_inputs(
                    cached_embeds=attention_embeds,
                    cached_grid_thw=attention_grid,
                    frame_token_counts=attention_frame_token_counts,
                    question=similarity_text or "",
                )
                map_metadata = (
                    build_question_prefill_attention_map_metadata(
                        frame_token_spans=question_only_inputs["frame_token_spans"],
                        query_positions=question_only_inputs["question_token_positions"],
                        attention_frame_indices=attention_frame_indices,
                    )
                    if save_example_matrices
                    else None
                )
                prefill_collector = LayerwiseFrameAttentionCollector(
                    frame_token_spans=question_only_inputs["frame_token_spans"],
                    query_positions=question_only_inputs["question_token_positions"],
                    num_layers=len(text_layers),
                    save_raw=save_raw_attentions,
                    map_layer_indices=(
                        sorted(set(question_prefill_display_layers + question_prefill_tail_layers))
                        if map_metadata is not None
                        else None
                    ),
                    question_prefill_map_metadata=map_metadata,
                )
                _prefill_output = self._run_with_collector(
                    prefill_collector,
                    use_cache=False,
                    input_ids=None,
                    inputs_embeds=question_only_inputs["inputs_embeds"],
                    attention_mask=question_only_inputs["attention_mask"],
                    position_ids=question_only_inputs["position_ids"],
                )
                del _prefill_output
                prefill_scores = prefill_collector.as_tensor()
                del question_only_inputs
                metrics["question_prefill_attention"] = summarize_layerwise_metric(
                    prefill_scores,
                    recent_indices=attention_recent_indices,
                    display_layer_count=QUESTION_PREFILL_DISPLAY_LAYER_COUNT,
                    display_layer_indices=question_prefill_display_layers,
                )
                metrics["question_prefill_attention"]["attention_frame_indices"] = attention_frame_indices
                metrics["question_prefill_attention"]["recent_frame_indices_within_attention"] = attention_recent_indices
                metrics["question_prefill_attention"]["attention_sampling_strategy"] = attention_sampling_strategy
                if save_example_matrices:
                    display_layer_indices = metrics["question_prefill_attention"]["display_layer_indices"]
                    example_payload["question_prefill_attention_scores"] = prefill_scores[display_layer_indices]
                    export_layer_indices = sorted(set(display_layer_indices) | set(question_prefill_tail_layers))
                    attention_map_payload = prefill_collector.export_question_prefill_attention_maps(export_layer_indices)
                    if attention_map_payload is not None:
                        example_payload["question_prefill_attention_maps"] = attention_map_payload
                if save_raw_attentions:
                    display_layer_indices = metrics["question_prefill_attention"]["display_layer_indices"]
                    example_payload["raw_question_prefill_attentions"] = {
                        int(layer_idx): prefill_collector.layer_raw_attentions[int(layer_idx)]
                        for layer_idx in display_layer_indices
                        if int(layer_idx) in prefill_collector.layer_raw_attentions
                    }
                del prefill_scores, prefill_collector
                release_unused_cuda_memory()

            # Collect frame-level decode attention for the first generated token and summarize it.
            if "first_token" in attention_modes:
                full_prompt_inputs = self._build_cached_multimodal_inputs(
                    cached_embeds=attention_embeds,
                    cached_grid_thw=attention_grid,
                    frame_token_counts=attention_frame_token_counts,
                    question=prompt,
                )
                prefill_outputs = self._run_with_collector(
                    None,
                    input_ids=None,
                    inputs_embeds=full_prompt_inputs["inputs_embeds"],
                    attention_mask=full_prompt_inputs["attention_mask"],
                    position_ids=full_prompt_inputs["position_ids"],
                )
                prefill_past_key_values = prefill_outputs.past_key_values
                first_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1)
                del prefill_outputs
                release_unused_cuda_memory()
                decode_collector = LayerwiseFrameAttentionCollector(
                    frame_token_spans=full_prompt_inputs["frame_token_spans"],
                    query_positions=[0],
                    num_layers=len(self._get_text_layers()),
                    save_raw=save_raw_attentions,
                )
                _decode_output = self._run_with_collector(
                    decode_collector,
                    use_cache=False,
                    input_ids=first_token[:, None],
                    past_key_values=prefill_past_key_values,
                )
                del _decode_output, full_prompt_inputs, prefill_past_key_values
                decode_scores = decode_collector.as_tensor()
                metrics["first_token_attention"] = summarize_layerwise_metric(
                    decode_scores,
                    recent_indices=attention_recent_indices,
                )
                metrics["first_token_attention"]["attention_frame_indices"] = attention_frame_indices
                metrics["first_token_attention"]["recent_frame_indices_within_attention"] = attention_recent_indices
                metrics["first_token_attention"]["attention_sampling_strategy"] = attention_sampling_strategy
                if save_example_matrices:
                    display_layer_indices = metrics["first_token_attention"]["display_layer_indices"]
                    example_payload["first_token_attention_scores"] = decode_scores[display_layer_indices]
                    example_payload["first_token_id"] = int(first_token.item())
                if save_raw_attentions:
                    example_payload["raw_first_token_attentions"] = decode_collector.layer_raw_attentions
                del decode_scores, decode_collector, first_token
                release_unused_cuda_memory()

            del attention_embeds, attention_grid, attention_frame_token_counts, attention_index_lookup
            release_unused_cuda_memory()

        del frames

        # Assemble the sample-level output record with frame metadata and computed metrics.
        record = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "num_sampled_frames": num_sampled_frames,
            "recent_chunk_ids": recent_chunk_ids,
            "recent_frame_indices": recent_indices,
            "attention_frame_indices": attention_frame_indices,
            "attention_sampling_strategy": attention_sampling_strategy,
            "frames": frame_rows,
            "metrics": metrics,
            "attention_skipped_reason": attention_skipped_reason,
        }

        # Attach metrics to the example payload only when example export is enabled.
        if save_example_matrices or save_raw_attentions:
            example_payload["metrics"] = metrics
        return record, (example_payload if save_example_matrices or save_raw_attentions else None)


def build_siglip_frame_saliency_analyzer(
    siglip_model_name: str = "google/siglip-so400m-patch14-384",
    device: str | torch.device = "auto",
) -> SiglipOnlyRecent4FrameSaliencyAnalyzer:
    return SiglipOnlyRecent4FrameSaliencyAnalyzer(
        siglip_model_name=siglip_model_name,
        device=device,
    )


def build_qwen3_attention_frame_saliency_analyzer(
    model_name: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 256,
    attn_implementation: str = "eager",
) -> Qwen3Recent4FrameSaliencyAnalyzer:
    return Qwen3Recent4FrameSaliencyAnalyzer(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        attn_implementation=attn_implementation,
    )


def build_frame_saliency_analyzer(
    model_name: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 256,
    attn_implementation: str = "eager",
    siglip_model_name: str = "google/siglip-so400m-patch14-384",
    attention_modes: list[str] | None = None,
) -> SiglipOnlyRecent4FrameSaliencyAnalyzer | Qwen3Recent4FrameSaliencyAnalyzer:
    warnings.warn(
        "build_frame_saliency_analyzer() is deprecated. "
        "Use build_siglip_frame_saliency_analyzer() or "
        "build_qwen3_attention_frame_saliency_analyzer() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if attention_modes:
        return build_qwen3_attention_frame_saliency_analyzer(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
    return build_siglip_frame_saliency_analyzer(
        siglip_model_name=siglip_model_name,
        device=device,
    )


def summarize_record_slice(records: list[dict[str, Any]], include_task_mean_metrics: bool = False) -> dict[str, Any]:
    scalar_metric_names = ["siglip_similarity"]
    attention_metric_names = ["question_prefill_attention", "first_token_attention"]
    valid_records = [record for record in records if not record.get("error")]
    summary: dict[str, Any] = {
        "total_records": len(records),
        "valid_records": len(valid_records),
        "error_records": len(records) - len(valid_records),
        "metrics": {},
        "tasks": {},
    }

    def summarize_stats(values: list[float], prefix: str) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 1:
            return {}
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
        }

    def summarize_metrics(metric_records: list[dict[str, Any]]) -> dict[str, Any]:
        metrics_summary: dict[str, Any] = {}

        for name in scalar_metric_names:
            values = [record["metrics"][name] for record in metric_records if name in record.get("metrics", {})]
            if values:
                metrics_summary[name] = {
                    "count": len(values),
                    **summarize_stats(
                        [item["recent4_mean_percentile"] for item in values],
                        "recent4_mean_percentile",
                    ),
                    **summarize_stats(
                        [item["recent4_top4_overlap"] for item in values],
                        "recent4_top4_overlap",
                    ),
                }

        for name in attention_metric_names:
            values = [record["metrics"][name] for record in metric_records if name in record.get("metrics", {})]
            if values:
                layer_recent_rows: list[np.ndarray] = []
                layer_overlap_rows: list[np.ndarray] = []
                display_layer_indices: list[int] = []
                num_layers_total: int | None = None
                for item in values:
                    recent_row, recent_indices = sample_metric_layer_field(item, "layer_recent4_mean_percentile")
                    overlap_row, overlap_indices = sample_metric_layer_field(item, "layer_recent4_top4_overlap")
                    if recent_row is None or overlap_row is None or recent_row.shape != overlap_row.shape:
                        continue
                    if display_layer_indices and recent_indices != display_layer_indices:
                        continue
                    if not display_layer_indices:
                        display_layer_indices = list(recent_indices)
                    layer_recent_rows.append(recent_row)
                    layer_overlap_rows.append(overlap_row)
                    if num_layers_total is None and item.get("num_layers_total") is not None:
                        num_layers_total = int(item["num_layers_total"])

                summary_payload = {
                    "count": len(values),
                    **summarize_stats(
                        [item["recent4_mean_percentile"] for item in values],
                        "recent4_mean_percentile",
                    ),
                    **summarize_stats(
                        [item["recent4_top4_overlap"] for item in values],
                        "recent4_top4_overlap",
                    ),
                }
                if layer_recent_rows and layer_overlap_rows:
                    stacked_recent = np.vstack(layer_recent_rows).astype(np.float64, copy=False)
                    stacked_overlap = np.vstack(layer_overlap_rows).astype(np.float64, copy=False)
                    summary_payload["display_layer_indices"] = display_layer_indices
                    summary_payload["layer_recent4_mean_percentile_mean"] = stacked_recent.mean(axis=0).tolist()
                    summary_payload["layer_recent4_mean_percentile_std"] = stacked_recent.std(axis=0).tolist()
                    summary_payload["layer_recent4_top4_overlap_mean"] = stacked_overlap.mean(axis=0).tolist()
                    summary_payload["layer_recent4_top4_overlap_std"] = stacked_overlap.std(axis=0).tolist()
                if num_layers_total is not None:
                    summary_payload["num_layers_total"] = num_layers_total
                metrics_summary[name] = summary_payload

        return metrics_summary

    def summarize_task_mean_metrics(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
        task_mean_metrics: dict[str, Any] = {}

        for name in scalar_metric_names:
            values = [
                task_summary["metrics"][name]
                for task_summary in task_summaries.values()
                if name in task_summary.get("metrics", {})
            ]
            if values:
                task_mean_metrics[name] = {
                    "task_count": len(values),
                    **summarize_stats(
                        [item["recent4_mean_percentile_mean"] for item in values],
                        "recent4_mean_percentile",
                    ),
                    **summarize_stats(
                        [item["recent4_top4_overlap_mean"] for item in values],
                        "recent4_top4_overlap",
                    ),
                }

        for name in attention_metric_names:
            values = [
                task_summary["metrics"][name]
                for task_summary in task_summaries.values()
                if name in task_summary.get("metrics", {})
            ]
            if values:
                layer_recent_rows: list[np.ndarray] = []
                layer_overlap_rows: list[np.ndarray] = []
                display_layer_indices: list[int] = []
                num_layers_total: int | None = None
                for item in values:
                    recent_row, recent_indices = sample_metric_layer_field(item, "layer_recent4_mean_percentile_mean")
                    overlap_row, overlap_indices = sample_metric_layer_field(item, "layer_recent4_top4_overlap_mean")
                    if recent_row is None or overlap_row is None or recent_row.shape != overlap_row.shape:
                        continue
                    if display_layer_indices and recent_indices != display_layer_indices:
                        continue
                    if not display_layer_indices:
                        display_layer_indices = list(recent_indices)
                    layer_recent_rows.append(recent_row)
                    layer_overlap_rows.append(overlap_row)
                    if num_layers_total is None and item.get("num_layers_total") is not None:
                        num_layers_total = int(item["num_layers_total"])

                summary_payload = {
                    "task_count": len(values),
                    **summarize_stats(
                        [item["recent4_mean_percentile_mean"] for item in values],
                        "recent4_mean_percentile",
                    ),
                    **summarize_stats(
                        [item["recent4_top4_overlap_mean"] for item in values],
                        "recent4_top4_overlap",
                    ),
                }
                if layer_recent_rows and layer_overlap_rows:
                    stacked_recent = np.vstack(layer_recent_rows).astype(np.float64, copy=False)
                    stacked_overlap = np.vstack(layer_overlap_rows).astype(np.float64, copy=False)
                    summary_payload["display_layer_indices"] = display_layer_indices
                    summary_payload["layer_recent4_mean_percentile_mean"] = stacked_recent.mean(axis=0).tolist()
                    summary_payload["layer_recent4_mean_percentile_std"] = stacked_recent.std(axis=0).tolist()
                    summary_payload["layer_recent4_top4_overlap_mean"] = stacked_overlap.mean(axis=0).tolist()
                    summary_payload["layer_recent4_top4_overlap_std"] = stacked_overlap.std(axis=0).tolist()
                if num_layers_total is not None:
                    summary_payload["num_layers_total"] = num_layers_total
                task_mean_metrics[name] = summary_payload

        return task_mean_metrics

    task_groups: dict[str, list[dict[str, Any]]] = {}
    for record in valid_records:
        task_groups.setdefault(str(record.get("task", "unknown")), []).append(record)

    task_summaries: dict[str, dict[str, Any]] = {}
    for task, task_records in sorted(task_groups.items()):
        task_payload: dict[str, Any] = {"count": len(task_records)}
        task_metrics = summarize_metrics(task_records)
        if task_metrics:
            task_payload["metrics"] = task_metrics
        task_summaries[task] = task_payload
    summary["tasks"] = task_summaries

    summary["metrics"] = summarize_metrics(valid_records)
    if include_task_mean_metrics and task_summaries:
        summary["task_mean_metrics"] = summarize_task_mean_metrics(task_summaries)

    return summary


def build_experiment_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "config": config,
        **summarize_record_slice(records),
    }
    summary["splits"] = {
        split_name: summarize_record_slice(
            [record for record in records if str(record.get("split", "")) == split_name],
            include_task_mean_metrics=True,
        )
        for split_name in ("backward", "realtime")
    }
    return summary
