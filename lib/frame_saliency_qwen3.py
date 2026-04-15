from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.recent_window_eval import decode_video_to_chunks_qwen, evenly_spaced_indices, select_attention_frame_indices
from lib.recent_window_eval_qwen3 import RecentWindowQAModel as _BaseQwen3RecentWindowQAModel

DISPLAY_LAYER_COUNT = 10


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
    return [int(round(x)) for x in np.linspace(0, total_count - 1, target_count)]


def summarize_layerwise_metric(layer_scores: torch.Tensor, recent_indices: list[int]) -> dict[str, Any]:
    if layer_scores.ndim != 2:
        raise ValueError(f"Expected [layers, frames] scores, got shape={tuple(layer_scores.shape)}")

    layer_scores = layer_scores.detach().float().cpu()
    layer_percentiles = torch.tensor(
        [tie_aware_percentiles(row.tolist()) for row in layer_scores],
        dtype=torch.float32,
    )
    display_layer_indices = uniform_center_indices(layer_scores.shape[0], DISPLAY_LAYER_COUNT)
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

    return {
        "num_layers_total": int(layer_scores.shape[0]),
        "display_layer_indices": [int(index) for index in display_layer_indices],
        "layer_attention_scores": display_layer_scores.tolist(),
        "layer_attention_percentiles": display_layer_percentiles.tolist(),
        "layer_recent4_mean_percentile": [float(value) for value in display_layer_recent_mean],
        "layer_recent4_top4_overlap": [float(value) for value in display_layer_overlap],
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
            if torch.cuda.device_count() > 1:
                siglip_device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
            else:
                siglip_device = self._get_visual_device()
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

        # If requested, compute SigLIP frame-text similarity and summarize it over recent frames.
        if "siglip" in similarity_backends:
            siglip_encoder = self.get_siglip_encoder()
            siglip_features = siglip_encoder.encode_frames(frames)
            siglip_question_feature = siglip_encoder.encode_text(similarity_text or "")
            siglip_scores = cosine_scores_against_query(siglip_features, siglip_question_feature).tolist()
            metrics["siglip_similarity"] = summarize_scalar_metric(siglip_scores, recent_indices)
            if save_example_matrices:
                example_payload["siglip_similarity_query"] = str(similarity_text or "")
            del siglip_features, siglip_question_feature, siglip_scores

        # Initialize default attention metadata so the output schema stays stable even when skipped.
        attention_skipped_reason: str | None = None
        attention_frame_indices = list(range(num_sampled_frames))
        attention_recent_indices: list[int] = recent_indices
        attention_sampling_strategy = "all_frames"
        if attention_modes:
            attention_frame_indices, attention_sampling_strategy = select_attention_frame_indices(
                total_frames=num_sampled_frames,
                recent_indices=recent_indices,
                max_analysis_frames=max_analysis_frames,
            )
            attention_index_lookup = {frame_idx: local_idx for local_idx, frame_idx in enumerate(attention_frame_indices)}
            attention_recent_indices = [
                attention_index_lookup[frame_idx]
                for frame_idx in recent_indices
                if frame_idx in attention_index_lookup
            ]

            # Encode only the selected attention frames so max_analysis_frames reduces vision memory too.
            attention_frames = [frames[frame_idx] for frame_idx in attention_frame_indices]
            attention_embeds, attention_grid = self.encode_vision(attention_frames)
            del attention_frames
            attention_frame_token_counts = frame_token_counts_from_grid(attention_grid, self.merge_size)
            for row in frame_rows:
                row["used_for_attention"] = bool(int(row["frame_index"]) in attention_index_lookup)

            # Collect layer-wise prefill attention showing which frames the question tokens focus on.
            if "question_prefill" in attention_modes:
                question_only_inputs = self._build_cached_multimodal_inputs(
                    cached_embeds=attention_embeds,
                    cached_grid_thw=attention_grid,
                    frame_token_counts=attention_frame_token_counts,
                    question=similarity_text or "",
                )
                prefill_collector = LayerwiseFrameAttentionCollector(
                    frame_token_spans=question_only_inputs["frame_token_spans"],
                    query_positions=question_only_inputs["question_token_positions"],
                    num_layers=len(self._get_text_layers()),
                    save_raw=save_raw_attentions,
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
                )
                metrics["question_prefill_attention"]["attention_frame_indices"] = attention_frame_indices
                metrics["question_prefill_attention"]["recent_frame_indices_within_attention"] = attention_recent_indices
                metrics["question_prefill_attention"]["attention_sampling_strategy"] = attention_sampling_strategy
                if save_example_matrices:
                    display_layer_indices = metrics["question_prefill_attention"]["display_layer_indices"]
                    example_payload["question_prefill_attention_scores"] = prefill_scores[display_layer_indices]
                if save_raw_attentions:
                    example_payload["raw_question_prefill_attentions"] = prefill_collector.layer_raw_attentions
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
