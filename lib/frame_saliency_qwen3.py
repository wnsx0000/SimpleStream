from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.recent_window_eval import decode_video_to_chunks_qwen
from lib.recent_window_eval_qwen3 import RecentWindowQAModel as _BaseQwen3RecentWindowQAModel


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


def summarize_layerwise_metric(layer_scores: torch.Tensor, recent_indices: list[int], last_k_layers: int) -> dict[str, Any]:
    if layer_scores.ndim != 2:
        raise ValueError(f"Expected [layers, frames] scores, got shape={tuple(layer_scores.shape)}")

    layer_scores = layer_scores.detach().float().cpu()
    layer_percentiles = torch.tensor(
        [tie_aware_percentiles(row.tolist()) for row in layer_scores],
        dtype=torch.float32,
    )
    layer_recent_mean = [
        float(layer_percentiles[layer_idx, recent_indices].mean().item()) if recent_indices else 0.0
        for layer_idx in range(layer_percentiles.shape[0])
    ]
    layer_overlap = [
        topk_overlap(layer_scores[layer_idx].tolist(), recent_indices, top_k=len(recent_indices))
        for layer_idx in range(layer_scores.shape[0])
    ]

    last_k = max(1, min(int(last_k_layers), int(layer_scores.shape[0])))
    last_k_mean_scores = layer_scores[-last_k:].mean(dim=0)
    last_k_summary = summarize_scalar_metric(last_k_mean_scores.tolist(), recent_indices)

    return {
        "layer_attention_scores": layer_scores.tolist(),
        "layer_attention_percentiles": layer_percentiles.tolist(),
        "layer_recent4_mean_percentile": [float(value) for value in layer_recent_mean],
        "layer_recent4_top4_overlap": [float(value) for value in layer_overlap],
        "last_k_layers_used": last_k,
        "last_k_layers_mean_attention_score": last_k_summary["frame_scores"],
        "last_k_layers_mean_percentile": last_k_summary["frame_percentiles"],
        "last_k_layers_recent4_mean_percentile": last_k_summary["recent4_mean_percentile"],
        "last_k_layers_recent4_top4_overlap": last_k_summary["recent4_top4_overlap"],
    }


def pairwise_cosine_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected [frames, dim] features, got shape={tuple(features.shape)}")
    normalized = F.normalize(features.float(), dim=-1)
    return normalized @ normalized.transpose(0, 1)


def centrality_from_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={tuple(matrix.shape)}")
    if matrix.shape[0] == 1:
        return torch.ones(1, dtype=torch.float32, device=matrix.device)
    total = matrix.sum(dim=-1) - torch.diag(matrix)
    return total / float(matrix.shape[0] - 1)


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


def pool_frame_features(token_features: torch.Tensor, frame_token_counts: list[int]) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    offset = 0
    for count in frame_token_counts:
        pooled.append(token_features[offset : offset + count].mean(dim=0))
        offset += count
    if offset != int(token_features.shape[0]):
        raise ValueError(
            f"Frame token counts do not cover all vision tokens: total={token_features.shape[0]} covered={offset}"
        )
    return torch.stack(pooled, dim=0)


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

    def __post_init__(self) -> None:
        self.layer_scores: list[torch.Tensor | None] = [None] * self.num_layers
        self.layer_raw_attentions: dict[int, torch.Tensor] = {}

    def make_hook(self, layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
                raise RuntimeError(
                    "Qwen3 text self-attention did not return attention weights. "
                    "Run with --attn-implementation eager."
                )

            attn_weights = output[1]
            if attn_weights.ndim != 4:
                raise RuntimeError(f"Unexpected attention shape from layer {layer_idx}: {tuple(attn_weights.shape)}")

            selected = attn_weights[0]
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

        return hook

    def as_tensor(self) -> torch.Tensor:
        missing = [str(index) for index, value in enumerate(self.layer_scores) if value is None]
        if missing:
            raise RuntimeError(f"Missing captured attentions for layers: {', '.join(missing)}")
        return torch.stack([value for value in self.layer_scores if value is not None], dim=0)


class SiglipFrameEncoder:
    def __init__(self, model_name: str, device: str | torch.device) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_frames(self, frames: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            inputs = self.processor(images=batch_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            features = self.model.get_image_features(pixel_values=pixel_values)
            batches.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(batches, dim=0)


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
            visual_device = self._get_visual_device()
            self._siglip_encoder = SiglipFrameEncoder(self.siglip_model_name, visual_device)
        return self._siglip_encoder

    def _get_text_layers(self) -> list[Any]:
        text_model = self._get_text_model()
        layers = getattr(text_model, "layers", None)
        if layers is None:
            raise RuntimeError("Unable to locate Qwen3 text decoder layers for attention capture.")
        return list(layers)

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
        position_ids, rope_deltas = text_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=cached_grid_thw.to(text_device),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "question_token_positions": list(range(question_start, question_end)),
            "frame_token_spans": frame_token_spans,
            "prompt_length": int(input_ids.shape[1]),
        }

    def _run_with_collector(
        self,
        collector: LayerwiseFrameAttentionCollector | None,
        **model_kwargs: Any,
    ) -> Any:
        handles = []
        if collector is not None:
            for layer_idx, layer in enumerate(self._get_text_layers()):
                handles.append(layer.self_attn.register_forward_hook(collector.make_hook(layer_idx)))
        try:
            return self.model(
                use_cache=True,
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
        chunk_duration: float,
        fps: float,
        recent_frames_only: int,
        similarity_backends: list[str],
        attention_modes: list[str],
        attention_last_k_layers: int,
        max_analysis_frames: int,
        save_example_matrices: bool = False,
        save_raw_attentions: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
        )
        if not chunks:
            raise ValueError(f"No chunks decoded from video: {video_path}")

        frames, frame_rows, recent_indices, recent_chunk_ids = flatten_chunks(chunks, recent_frames_only)
        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        frame_token_counts = frame_token_counts_from_grid(cached_grid_thw, self.merge_size)
        qwen_frame_features = pool_frame_features(cached_embeds, frame_token_counts)

        metrics: dict[str, Any] = {}
        example_payload: dict[str, Any] = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "frame_rows": frame_rows,
            "recent_frame_indices": recent_indices,
            "recent_chunk_ids": recent_chunk_ids,
        } if (save_example_matrices or save_raw_attentions) else {}

        if "qwen" in similarity_backends:
            qwen_matrix = pairwise_cosine_matrix(qwen_frame_features)
            qwen_scores = centrality_from_similarity_matrix(qwen_matrix).tolist()
            metrics["qwen_similarity"] = summarize_scalar_metric(qwen_scores, recent_indices)
            if save_example_matrices:
                example_payload["qwen_similarity_matrix"] = qwen_matrix.detach().cpu()

        if "siglip" in similarity_backends:
            siglip_features = self.get_siglip_encoder().encode_frames(frames)
            siglip_matrix = pairwise_cosine_matrix(siglip_features)
            siglip_scores = centrality_from_similarity_matrix(siglip_matrix).tolist()
            metrics["siglip_similarity"] = summarize_scalar_metric(siglip_scores, recent_indices)
            if save_example_matrices:
                example_payload["siglip_similarity_matrix"] = siglip_matrix.detach().cpu()

        attention_skipped_reason: str | None = None
        if attention_modes:
            if len(frames) > int(max_analysis_frames):
                attention_skipped_reason = (
                    f"Skipped attention analysis because num_frames={len(frames)} exceeds "
                    f"--max-analysis-frames={max_analysis_frames}."
                )
            else:
                inputs = self._build_cached_multimodal_inputs(
                    cached_embeds=cached_embeds,
                    cached_grid_thw=cached_grid_thw,
                    frame_token_counts=frame_token_counts,
                    question=prompt,
                )
                prefill_collector = None
                if "question_prefill" in attention_modes:
                    prefill_collector = LayerwiseFrameAttentionCollector(
                        frame_token_spans=inputs["frame_token_spans"],
                        query_positions=inputs["question_token_positions"],
                        num_layers=len(self._get_text_layers()),
                        save_raw=save_raw_attentions,
                    )

                prefill_outputs = self._run_with_collector(
                    prefill_collector,
                    input_ids=None,
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs["attention_mask"],
                    position_ids=inputs["position_ids"],
                )

                if prefill_collector is not None:
                    prefill_scores = prefill_collector.as_tensor()
                    metrics["question_prefill_attention"] = summarize_layerwise_metric(
                        prefill_scores,
                        recent_indices=recent_indices,
                        last_k_layers=attention_last_k_layers,
                    )
                    if save_example_matrices:
                        example_payload["question_prefill_attention_scores"] = prefill_scores
                    if save_raw_attentions:
                        example_payload["raw_question_prefill_attentions"] = prefill_collector.layer_raw_attentions

                if "first_token" in attention_modes:
                    first_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1)
                    decode_collector = LayerwiseFrameAttentionCollector(
                        frame_token_spans=inputs["frame_token_spans"],
                        query_positions=[0],
                        num_layers=len(self._get_text_layers()),
                        save_raw=save_raw_attentions,
                    )
                    self._run_with_collector(
                        decode_collector,
                        input_ids=first_token[:, None],
                        past_key_values=prefill_outputs.past_key_values,
                    )
                    decode_scores = decode_collector.as_tensor()
                    metrics["first_token_attention"] = summarize_layerwise_metric(
                        decode_scores,
                        recent_indices=recent_indices,
                        last_k_layers=attention_last_k_layers,
                    )
                    if save_example_matrices:
                        example_payload["first_token_attention_scores"] = decode_scores
                        example_payload["first_token_id"] = int(first_token.item())
                    if save_raw_attentions:
                        example_payload["raw_first_token_attentions"] = decode_collector.layer_raw_attentions

        record = {
            "video_path": video_path,
            "decode_backend": decode_backend,
            "num_sampled_frames": len(frames),
            "recent_chunk_ids": recent_chunk_ids,
            "recent_frame_indices": recent_indices,
            "frames": frame_rows,
            "metrics": metrics,
            "attention_skipped_reason": attention_skipped_reason,
        }

        if save_example_matrices or save_raw_attentions:
            example_payload["metrics"] = metrics
        return record, (example_payload if save_example_matrices or save_raw_attentions else None)


def build_experiment_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    valid_records = [record for record in records if not record.get("error")]
    summary: dict[str, Any] = {
        "config": config,
        "total_records": len(records),
        "valid_records": len(valid_records),
        "error_records": len(records) - len(valid_records),
        "metrics": {},
        "tasks": {},
    }

    task_groups: dict[str, list[dict[str, Any]]] = {}
    for record in valid_records:
        task_groups.setdefault(str(record.get("task", "unknown")), []).append(record)
    for task, task_records in sorted(task_groups.items()):
        summary["tasks"][task] = {"count": len(task_records)}

    scalar_metric_names = ["qwen_similarity", "siglip_similarity"]
    attention_metric_names = ["question_prefill_attention", "first_token_attention"]

    for name in scalar_metric_names:
        values = [record["metrics"][name] for record in valid_records if name in record.get("metrics", {})]
        if values:
            summary["metrics"][name] = {
                "count": len(values),
                "recent4_mean_percentile_mean": float(
                    np.mean([item["recent4_mean_percentile"] for item in values], dtype=np.float64)
                ),
                "recent4_top4_overlap_mean": float(
                    np.mean([item["recent4_top4_overlap"] for item in values], dtype=np.float64)
                ),
            }

    for name in attention_metric_names:
        values = [record["metrics"][name] for record in valid_records if name in record.get("metrics", {})]
        if values:
            layer_recent = np.asarray([item["layer_recent4_mean_percentile"] for item in values], dtype=np.float64)
            layer_overlap = np.asarray([item["layer_recent4_top4_overlap"] for item in values], dtype=np.float64)
            summary["metrics"][name] = {
                "count": len(values),
                "last_k_layers_recent4_mean_percentile_mean": float(
                    np.mean([item["last_k_layers_recent4_mean_percentile"] for item in values], dtype=np.float64)
                ),
                "last_k_layers_recent4_top4_overlap_mean": float(
                    np.mean([item["last_k_layers_recent4_top4_overlap"] for item in values], dtype=np.float64)
                ),
                "layer_recent4_mean_percentile_mean": layer_recent.mean(axis=0).tolist(),
                "layer_recent4_top4_overlap_mean": layer_overlap.mean(axis=0).tolist(),
            }

    return summary
