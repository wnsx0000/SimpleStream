"""
OVO-Bench layer-wise per-token Q·K retrieval evaluation for Qwen3-VL.

For each sample this script performs a SINGLE multimodal prefill over the
candidate frame pool (recent + uniform context up to ``--max_analysis_frames``)
with ``use_cache=True``. A forward pre-hook on each text decoder layer's
``self_attn`` captures only the Q slice at question-token positions (and, in
``pre_rope`` mode, also the K slice at vision-token positions), while the actual
attention forward runs with the user-selected ``attn_implementation`` (default
``flash_attention_2``). Per layer we then rank vision tokens by raw
``mean_heads mean_query (Q · K^T) / sqrt(d)``; this is ReKV §3's internal
retrieval score and is rank-equivalent per-row to the softmax-normalised signal.
The layer's KV cache is sliced to keep only the non-vision tokens plus the top-K
vision tokens worth 4 frames.

Two independent CLI options control positional behaviour:
  * ``--retrieval_rope_mode {post_rope, pre_rope}`` -- whether the retrieval
    Q·K^T score is computed on post-RoPE Q/K (cache K reused) or pre-RoPE Q/K
    (both captured fresh in the hook).
  * ``--decode_pe_mode {original, reindex}`` -- whether the compressed KV keeps
    each retained token's real M-RoPE position or is re-rotated into a
    contiguous 0..N-1 layout before decoding.

Key differences vs ``eval_qwen3vl_ovo_test3.py``:
  * ONE prefill per sample (not two).
  * Per-token, per-layer retrieval granularity (test3 was a single-layer frame-
    level top-4).
  * Every decoder layer keeps a potentially DIFFERENT subset of vision tokens.
  * No longer depends on eager attention / post-softmax attention weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from PIL import Image

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS
from lib.frame_saliency_qwen3 import (
    Qwen3Recent4FrameSaliencyAnalyzer,
    flatten_chunks,
    frame_token_counts_from_grid,
    release_unused_cuda_memory,
)
from lib.recent_window_eval import (
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    load_jsonl_results,
    score_ovo_br,
    select_attention_frame_indices,
)
from main_experiments.eval_qwen3vl_ovo_saliency_common import (
    format_task_counts,
    select_split_annotations,
    smoke_cap_or_default,
)

MODEL_LABEL = "Qwen3-VL-AttnTokenRetrieval"
EXCLUDED_FORWARD_TASKS = ("REC", "SSR", "CRR")
EXCLUDED_BACKWARD_TASKS = frozenset({"HLD"})
EVAL_BACKWARD_TASKS = [task for task in BACKWARD_TASKS if task not in EXCLUDED_BACKWARD_TASKS]
EVAL_TASK_SET = frozenset([*EVAL_BACKWARD_TASKS, *REAL_TIME_TASKS])
FRAMES_WORTH_BUDGET = 4


RETRIEVAL_ROPE_MODES = ("post_rope", "pre_rope")
DECODE_PE_MODES = ("original", "reindex")


EvictSpec = tuple[str, float]
EvictLayerRange = tuple[int, int]


def _parse_evict_spec(raw: str) -> EvictSpec:
    """Parse ``"N"`` (absolute count) or ``"N%"`` (percent of candidate pool).

    Returns a tuple ``("absolute", int)`` or ``("percent", float)``.
    Raises ``argparse.ArgumentTypeError`` on malformed input.
    """
    if raw is None:
        raise argparse.ArgumentTypeError("evict spec must not be None.")
    s = str(raw).strip()
    if not s:
        raise argparse.ArgumentTypeError("evict spec must be a non-empty string.")
    if s.endswith("%"):
        num_part = s[:-1].strip()
        try:
            pct = float(num_part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"evict percentage {s!r} is not a valid number."
            ) from exc
        if not (0.0 < pct <= 100.0):
            raise argparse.ArgumentTypeError(
                f"evict percentage must be in (0, 100]; got {pct}."
            )
        return ("percent", pct)
    try:
        count = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"evict spec {s!r} must be an integer or a percentage like '10%'."
        ) from exc
    if count <= 0:
        raise argparse.ArgumentTypeError(
            f"evict absolute count must be positive; got {count}."
        )
    return ("absolute", float(count))


def _resolve_evict_count(spec: EvictSpec | None, num_vision_tokens: int) -> int:
    """Convert a parsed evict spec into an absolute token count for the sample."""
    if spec is None:
        return 0
    mode, value = spec
    if mode == "absolute":
        return int(value)
    if mode == "percent":
        return int((num_vision_tokens * float(value)) // 100)
    raise ValueError(f"Unknown evict spec mode: {mode!r}")


def _format_evict_spec(spec: EvictSpec | None) -> str | None:
    """Round-trip an evict spec back to the CLI-style string form."""
    if spec is None:
        return None
    mode, value = spec
    if mode == "absolute":
        return str(int(value))
    return f"{value:g}%"


def _parse_evict_layer_range(raw: str) -> EvictLayerRange:
    """Parse an inclusive decoder-layer range in ``START-END`` form."""
    if raw is None:
        raise argparse.ArgumentTypeError("evict layer range must not be None.")
    s = str(raw).strip()
    parts = s.split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise argparse.ArgumentTypeError(
            f"evict layer range {s!r} must use START-END form, e.g. '20-34'."
        )
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise argparse.ArgumentTypeError(
            f"evict layer range start must be <= end; got {start}-{end}."
        )
    return (start, end)


def _format_evict_layer_range(layer_range: EvictLayerRange | None) -> str | None:
    if layer_range is None:
        return None
    start, end = layer_range
    return f"{int(start)}-{int(end)}"


@dataclass
class LayerwiseQKRetrievalCollector:
    """Capture Q (and, in pre_rope, K) slices needed for Q·K^T retrieval scoring.

    In ``post_rope`` mode the hook saves only the post-RoPE Q slice at question
    positions per layer; K for the vision range is read from the KV cache after
    prefill. In ``pre_rope`` mode the hook also computes the pre-RoPE K slice
    at vision positions and finalises the per-vision-token score inline, so no
    K copy outlives the hook.
    """

    vision_start: int
    vision_end: int
    query_positions: list[int]
    num_layers: int
    rope_mode: str
    query_states: list[torch.Tensor | None] = field(default_factory=list)
    layer_token_scores: list[torch.Tensor | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rope_mode not in RETRIEVAL_ROPE_MODES:
            raise ValueError(
                f"Unknown retrieval_rope_mode={self.rope_mode!r}; "
                f"expected one of {RETRIEVAL_ROPE_MODES}."
            )
        self.query_states = [None] * int(self.num_layers)
        self.layer_token_scores = [None] * int(self.num_layers)
        if self.vision_end <= self.vision_start:
            raise ValueError(
                f"Invalid vision span: start={self.vision_start} end={self.vision_end}"
            )
        if not self.query_positions:
            raise ValueError("query_positions must be non-empty for retrieval scoring.")

    def make_pre_hook(self, layer_idx: int, self_attn: Any):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        qpos_list = list(self.query_positions)
        vstart = int(self.vision_start)
        vend = int(self.vision_end)
        head_dim = int(self_attn.head_dim)
        num_q_heads = int(self_attn.config.num_attention_heads)
        num_kv_heads = int(self_attn.config.num_key_value_heads)
        groups = num_q_heads // num_kv_heads
        scaling = float(self_attn.scaling)
        mode = self.rope_mode

        def _project_q(hs: torch.Tensor) -> torch.Tensor:
            shape = (*hs.shape[:-1], -1, head_dim)
            return self_attn.q_norm(self_attn.q_proj(hs).view(shape)).transpose(1, 2)

        def _project_k(hs: torch.Tensor) -> torch.Tensor:
            shape = (*hs.shape[:-1], -1, head_dim)
            return self_attn.k_norm(self_attn.k_proj(hs).view(shape)).transpose(1, 2)

        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            if hs is None:
                raise RuntimeError("self_attn pre-hook received no hidden_states.")
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(
                    "self_attn pre-hook received no position_embeddings; "
                    "cannot compute RoPE-aware retrieval."
                )
            cos, sin = position_embeddings

            idx_q = torch.tensor(qpos_list, dtype=torch.long, device=hs.device)
            hs_q = hs.index_select(dim=1, index=idx_q)
            q = _project_q(hs_q)

            if mode == "post_rope":
                cos_q = cos.index_select(dim=-2, index=idx_q)
                sin_q = sin.index_select(dim=-2, index=idx_q)
                q_rot, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
                self.query_states[layer_idx] = q_rot.detach()[0]
                return None

            hs_v = hs[:, vstart:vend, :]
            k = _project_k(hs_v)
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scaling
            token_score = scores[0].mean(dim=0).mean(dim=0)
            self.layer_token_scores[layer_idx] = token_score.detach().cpu()
            return None

        return pre_hook


class AttentionTokenRetrievalQA(Qwen3Recent4FrameSaliencyAnalyzer):
    """Qwen3-VL wrapper: single prefill + per-layer token KV compression + decode."""

    def __init__(
        self,
        *args: Any,
        rope_mode: str = "post_rope",
        decode_pe_mode: str = "original",
        shared_scoring_layer: int | None = None,
        evict_spec: EvictSpec | None = None,
        evict_layer_range: EvictLayerRange | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if rope_mode not in RETRIEVAL_ROPE_MODES:
            raise ValueError(
                f"rope_mode must be one of {RETRIEVAL_ROPE_MODES}; got {rope_mode!r}."
            )
        if decode_pe_mode not in DECODE_PE_MODES:
            raise ValueError(
                f"decode_pe_mode must be one of {DECODE_PE_MODES}; got {decode_pe_mode!r}."
            )
        if shared_scoring_layer is not None and int(shared_scoring_layer) < 0:
            raise ValueError(
                f"shared_scoring_layer must be >= 0 when set; got {shared_scoring_layer}."
            )
        if evict_spec is not None:
            mode, value = evict_spec
            if mode not in ("absolute", "percent"):
                raise ValueError(f"Unknown evict spec mode: {mode!r}")
            if float(value) <= 0:
                raise ValueError(f"evict spec value must be positive; got {value}.")
        if evict_layer_range is not None:
            start, end = evict_layer_range
            if int(start) < 0 or int(end) < 0 or int(start) > int(end):
                raise ValueError(
                    f"evict_layer_range must be an inclusive non-negative START-END range; "
                    f"got {_format_evict_layer_range(evict_layer_range)!r}."
                )
            if evict_spec is None:
                raise ValueError("--evict_layer_range requires --pre_retrieval_evict.")
            if shared_scoring_layer is not None:
                raise ValueError(
                    "--evict_layer_range is not compatible with --shared_scoring_layer."
                )
        self.rope_mode = rope_mode
        self.decode_pe_mode = decode_pe_mode
        self.shared_scoring_layer = (
            int(shared_scoring_layer) if shared_scoring_layer is not None else None
        )
        self.evict_spec: EvictSpec | None = evict_spec
        self.evict_layer_range: EvictLayerRange | None = evict_layer_range
        self._per_frame_vision_tokens: int | None = None
        self._target_vision_token_budget: int | None = None

    def _ensure_token_budget(self, grid_thw: torch.Tensor) -> tuple[int, int]:
        counts = frame_token_counts_from_grid(grid_thw, self.merge_size)
        if not counts:
            raise ValueError("No frame token counts derived from grid_thw.")
        per_frame = int(round(sum(counts) / len(counts)))
        if self._per_frame_vision_tokens is None:
            self._per_frame_vision_tokens = per_frame
            self._target_vision_token_budget = per_frame * FRAMES_WORTH_BUDGET
            print(
                f"[test6] Token budget fixed: per_frame_vision_tokens={per_frame}, "
                f"target_budget={self._target_vision_token_budget} "
                f"(= {FRAMES_WORTH_BUDGET} frames worth).",
                flush=True,
            )
        return int(self._per_frame_vision_tokens), int(self._target_vision_token_budget)

    def _register_token_hooks(
        self,
        collector: LayerwiseQKRetrievalCollector,
    ) -> list[Any]:
        handles: list[Any] = []
        text_layers = self._get_text_layers()
        if len(text_layers) != collector.num_layers:
            raise RuntimeError(
                f"Layer count mismatch: text_layers={len(text_layers)} "
                f"collector.num_layers={collector.num_layers}"
            )
        for layer_idx, layer in enumerate(text_layers):
            pre_hook = collector.make_pre_hook(layer_idx, layer.self_attn)
            handles.append(
                layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
        return handles

    @staticmethod
    def _score_one_layer(
        *,
        collector: "LayerwiseQKRetrievalCollector",
        past_key_values: Any,
        layer_idx: int,
        rope_mode: str,
        vision_start: int,
        vision_end: int,
        groups: int,
        scaling: float,
    ) -> torch.Tensor:
        """Return the per-vision-token retrieval score for a single layer (CPU)."""
        if rope_mode == "post_rope":
            q_slice = collector.query_states[layer_idx]
            if q_slice is None:
                raise RuntimeError(f"Missing Q slice for layer {layer_idx}.")
            k_all = past_key_values.layers[layer_idx].keys
            k_vis = k_all[0, :, vision_start:vision_end, :]
            if groups > 1:
                k_vis = k_vis.repeat_interleave(groups, dim=0)
            scores = torch.matmul(
                q_slice.float(), k_vis.float().transpose(-1, -2)
            ) * scaling
            token_score = scores.mean(dim=0).mean(dim=0).cpu()
            collector.query_states[layer_idx] = None
            del q_slice, k_vis, scores
            return token_score
        token_score = collector.layer_token_scores[layer_idx]
        if token_score is None:
            raise RuntimeError(f"Missing pre-RoPE score for layer {layer_idx}.")
        collector.layer_token_scores[layer_idx] = None
        return token_score

    @staticmethod
    def _topk_local_with_eviction(
        token_score: torch.Tensor,
        effective_k: int,
        evict_count: int,
    ) -> torch.Tensor:
        """Mask the ``evict_count`` highest-scoring positions then return top-K indices.

        Returns local vision-space indices (within ``[0, num_vision_tokens)``).
        """
        if effective_k <= 0:
            raise RuntimeError(
                "effective token budget is non-positive; evict_count may exceed "
                "the candidate vision token pool."
            )
        if evict_count <= 0:
            return torch.topk(token_score, k=effective_k).indices
        if evict_count >= token_score.numel():
            raise RuntimeError(
                f"evict_count={evict_count} >= num_vision_tokens={token_score.numel()}; "
                "no tokens remain for retrieval."
            )
        evict_local = torch.topk(token_score, k=evict_count).indices
        masked = token_score.clone()
        masked[evict_local] = float("-inf")
        return torch.topk(masked, k=effective_k).indices

    def _evict_count_for_layer(self, layer_idx: int, evict_count: int) -> int:
        if evict_count <= 0:
            return 0
        if self.evict_layer_range is None:
            return int(evict_count)
        start, end = self.evict_layer_range
        return int(evict_count) if int(start) <= int(layer_idx) <= int(end) else 0

    @staticmethod
    def _compress_layer_cache(
        past_key_values: Any,
        layer_idx: int,
        kept_indices: torch.Tensor,
    ) -> None:
        layer = past_key_values.layers[layer_idx]
        key_indices = kept_indices.to(device=layer.keys.device, dtype=torch.long)
        layer.keys = layer.keys.index_select(dim=2, index=key_indices)

        value_indices = (
            key_indices
            if layer.values.device == layer.keys.device
            else kept_indices.to(device=layer.values.device, dtype=torch.long)
        )
        layer.values = layer.values.index_select(dim=2, index=value_indices)

    @staticmethod
    def _truncate_cache_to_length(past_key_values: Any, cache_len: int) -> None:
        """Truncate every layer cache to ``cache_len`` tokens along sequence dim."""
        target_len = int(cache_len)
        if target_len < 1:
            raise RuntimeError(f"Cannot truncate KV cache to non-positive length: {target_len}")
        for layer_idx, layer in enumerate(past_key_values.layers):
            key_len = int(layer.keys.shape[2])
            value_len = int(layer.values.shape[2])
            if key_len != value_len:
                raise RuntimeError(
                    f"KV cache length mismatch before truncation at layer {layer_idx}: "
                    f"keys={key_len}, values={value_len}."
                )
            if target_len > key_len:
                raise RuntimeError(
                    f"Cannot extend KV cache during truncation at layer {layer_idx}: "
                    f"target={target_len}, current={key_len}."
                )
            layer.keys = layer.keys[:, :, :target_len, :]
            layer.values = layer.values[:, :, :target_len, :]

    @staticmethod
    def _first_token_diagnostics(tokenizer: Any, logits: torch.Tensor) -> dict[str, Any]:
        """Return compact first-token diagnostics without storing full vocab logits."""
        logits_1d = logits[0].detach().float()
        log_probs = torch.log_softmax(logits_1d, dim=-1)
        argmax_token_id = int(torch.argmax(logits_1d, dim=-1).item())

        choice_token_ids: dict[str, int | None] = {}
        choice_logits: dict[str, float | None] = {}
        choice_logprobs: dict[str, float | None] = {}
        choice_ranks: dict[str, int | None] = {}
        valid_choice_logits: dict[str, float] = {}

        for choice in ("A", "B", "C", "D"):
            token_ids = tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                choice_token_ids[choice] = None
                choice_logits[choice] = None
                choice_logprobs[choice] = None
                choice_ranks[choice] = None
                continue

            token_id = int(token_ids[0])
            token_logit = float(logits_1d[token_id].item())
            choice_token_ids[choice] = token_id
            choice_logits[choice] = token_logit
            choice_logprobs[choice] = float(log_probs[token_id].item())
            choice_ranks[choice] = int((logits_1d > logits_1d[token_id]).sum().item() + 1)
            valid_choice_logits[choice] = token_logit

        choice_argmax: str | None = None
        choice_logit_margin: float | None = None
        if valid_choice_logits:
            sorted_choices = sorted(
                valid_choice_logits.items(), key=lambda item: item[1], reverse=True
            )
            choice_argmax = sorted_choices[0][0]
            if len(sorted_choices) > 1:
                choice_logit_margin = float(sorted_choices[0][1] - sorted_choices[1][1])

        del log_probs
        return {
            "argmax_token_id": argmax_token_id,
            "argmax_token_text": tokenizer.decode(
                [argmax_token_id], skip_special_tokens=False
            ),
            "choice_token_ids": choice_token_ids,
            "choice_logits": choice_logits,
            "choice_logprobs": choice_logprobs,
            "choice_ranks": choice_ranks,
            "choice_argmax": choice_argmax,
            "choice_logit_margin": choice_logit_margin,
        }

    @staticmethod
    def _diagnostic_delta(
        after: dict[str, Any],
        before: dict[str, Any],
        field: str,
    ) -> dict[str, float | None]:
        delta: dict[str, float | None] = {}
        after_values = after.get(field, {})
        before_values = before.get(field, {})
        for choice in ("A", "B", "C", "D"):
            after_value = after_values.get(choice)
            before_value = before_values.get(choice)
            delta[choice] = (
                None
                if after_value is None or before_value is None
                else float(after_value - before_value)
            )
        return delta

    def _get_text_rotary_emb(self) -> Any:
        pending = [self._get_text_model()]
        visited: set[int] = set()
        while pending:
            module = pending.pop(0)
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)
            rotary = getattr(module, "rotary_emb", None)
            if rotary is not None:
                return rotary
            for attr_name in ("language_model", "model", "decoder"):
                child = getattr(module, attr_name, None)
                if child is not None:
                    pending.append(child)
        raise RuntimeError("Unable to locate rotary_emb on the Qwen3-VL text backbone.")

    def _reindex_cache_positions(
        self,
        *,
        past_key_values: Any,
        kept_indices_by_layer: list[torch.Tensor],
        prefill_position_ids: torch.Tensor,
        compressed_cache_len: int,
        text_device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> None:
        """Re-rotate the compressed K-cache so retained tokens occupy positions 0..N-1.

        Values are RoPE-invariant. Only K tensors are re-rotated: we invert RoPE at
        the original positions (applying RoPE with sin negated), then apply RoPE at
        the new contiguous positions.
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        rotary = self._get_text_rotary_emb()
        dummy_full = torch.zeros((1, 1, 1), device=text_device, dtype=hidden_dtype)
        cos_full, sin_full = rotary(dummy_full, prefill_position_ids)

        new_positions = torch.arange(
            int(compressed_cache_len), device=text_device, dtype=torch.long
        )
        new_position_ids = new_positions.view(1, 1, -1).expand(3, 1, -1).contiguous()
        cos_new_full, sin_new_full = rotary(dummy_full, new_position_ids)

        num_layers = len(past_key_values.layers)
        if len(kept_indices_by_layer) != num_layers:
            raise RuntimeError(
                f"kept_indices_by_layer length {len(kept_indices_by_layer)} does not match "
                f"num_layers {num_layers}."
            )

        for layer_idx in range(num_layers):
            layer = past_key_values.layers[layer_idx]
            k = layer.keys
            kept = kept_indices_by_layer[layer_idx].to(device=k.device, dtype=torch.long)
            if kept.numel() != k.shape[2]:
                raise RuntimeError(
                    f"Kept index count {kept.numel()} does not match compressed cache "
                    f"length {k.shape[2]} at layer {layer_idx}."
                )
            cos_old = cos_full.to(k.device).index_select(dim=-2, index=kept)
            sin_old = sin_full.to(k.device).index_select(dim=-2, index=kept)
            cos_new = cos_new_full.to(k.device)
            sin_new = sin_new_full.to(k.device)

            orig_dtype = k.dtype
            k_f = k.to(cos_old.dtype)
            k_unrot, _ = apply_rotary_pos_emb(k_f, k_f, cos_old, -sin_old)
            k_rot, _ = apply_rotary_pos_emb(k_unrot, k_unrot, cos_new, sin_new)
            layer.keys = k_rot.to(orig_dtype)

    def _prompt_substring_token_positions(
        self,
        *,
        prompt: str,
        query: str,
        prompt_token_offset: int,
    ) -> list[int]:
        query_text = str(query).strip()
        if not query_text:
            raise ValueError("retrieval_query must be non-empty.")

        start_char = str(prompt).find(query_text)
        if start_char < 0:
            raise ValueError("retrieval_query was not found inside the generation prompt.")
        end_char = start_char + len(query_text)

        encoded = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise RuntimeError("Tokenizer did not return offset_mapping for retrieval query span.")

        positions: list[int] = []
        for token_idx, (token_start, token_end) in enumerate(offsets):
            token_start = int(token_start)
            token_end = int(token_end)
            if token_end <= token_start:
                continue
            if token_start < end_char and token_end > start_char:
                positions.append(int(prompt_token_offset) + token_idx)

        if not positions:
            raise RuntimeError("retrieval_query produced no token positions inside the prompt.")
        return positions

    @torch.inference_mode()
    def score_and_generate(
        self,
        frames: list[Image.Image],
        prompt: str,
        retrieval_query: str | None = None,
    ) -> dict[str, Any]:
        text_layers = self._get_text_layers()
        num_layers = len(text_layers)
        if self.evict_layer_range is not None:
            start, end = self.evict_layer_range
            if int(end) >= num_layers:
                raise ValueError(
                    f"evict_layer_range={_format_evict_layer_range(self.evict_layer_range)} "
                    f"is out of range for {num_layers} decoder layers."
                )

        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        per_frame_tokens, budget = self._ensure_token_budget(cached_grid_thw)
        frame_token_counts = frame_token_counts_from_grid(cached_grid_thw, self.merge_size)

        prefill_inputs = self._build_cached_multimodal_inputs(
            cached_embeds=cached_embeds,
            cached_grid_thw=cached_grid_thw,
            frame_token_counts=frame_token_counts,
            question=prompt,
        )

        frame_token_spans = prefill_inputs["frame_token_spans"]
        prompt_positions = prefill_inputs["question_token_positions"]
        query_positions = (
            self._prompt_substring_token_positions(
                prompt=prompt,
                query=retrieval_query,
                prompt_token_offset=int(prompt_positions[0]),
            )
            if retrieval_query is not None
            else prompt_positions
        )
        inputs_embeds = prefill_inputs["inputs_embeds"]
        attention_mask = prefill_inputs["attention_mask"]
        input_ids = prefill_inputs["input_ids"]
        position_ids = prefill_inputs["position_ids"]
        tokenizer = self.processor.tokenizer

        vision_start = int(frame_token_spans[0][0])
        vision_end = int(frame_token_spans[-1][1])
        num_vision_tokens = vision_end - vision_start
        total_seq_len = int(inputs_embeds.shape[1])
        non_vision_positions = (
            list(range(0, vision_start)) + list(range(vision_end, total_seq_len))
        )

        self._last_num_vision_tokens = int(num_vision_tokens)
        self._last_num_vision_frames = int(len(frames))

        collector = LayerwiseQKRetrievalCollector(
            vision_start=vision_start,
            vision_end=vision_end,
            query_positions=query_positions,
            num_layers=num_layers,
            rope_mode=self.rope_mode,
        )

        # -- Phase 1: single prefill with cache retention and Q/K capture.
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_prefill = time.perf_counter()
        handles = self._register_token_hooks(collector)
        try:
            outputs = self.model(
                use_cache=True,
                return_dict=True,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        prefill_time = time.perf_counter() - t_prefill

        full_prefill_first_token_diagnostics = self._first_token_diagnostics(
            tokenizer, outputs.logits[:, -1, :]
        )
        past_key_values = outputs.past_key_values
        del outputs

        # -- Phase 2: per-layer top-K vision token selection + KV compression.
        evict_count = _resolve_evict_count(self.evict_spec, num_vision_tokens)
        if evict_count >= num_vision_tokens:
            raise RuntimeError(
                f"evict_count={evict_count} >= num_vision_tokens={num_vision_tokens}; "
                "no tokens remain for retrieval."
            )
        effective_k = int(min(budget, num_vision_tokens - evict_count))
        non_vision_tensor = torch.tensor(non_vision_positions, dtype=torch.long)

        sample_attn = text_layers[0].self_attn
        num_q_heads = int(sample_attn.config.num_attention_heads)
        num_kv_heads = int(sample_attn.config.num_key_value_heads)
        groups = num_q_heads // num_kv_heads
        scaling = float(sample_attn.scaling)

        shared_topk_global: torch.Tensor | None = None
        if self.shared_scoring_layer is not None:
            if not (0 <= self.shared_scoring_layer < num_layers):
                raise ValueError(
                    f"shared_scoring_layer={self.shared_scoring_layer} is out of range "
                    f"[0, {num_layers})."
                )
            shared_score = self._score_one_layer(
                collector=collector,
                past_key_values=past_key_values,
                layer_idx=self.shared_scoring_layer,
                rope_mode=self.rope_mode,
                vision_start=vision_start,
                vision_end=vision_end,
                groups=groups,
                scaling=scaling,
            )
            shared_topk_local = self._topk_local_with_eviction(
                shared_score, effective_k, evict_count
            )
            shared_topk_global = (shared_topk_local + vision_start).to(torch.long)

        layer_top_vision_global: list[list[int]] = []
        layer_selected_frame_token_counts: list[list[int]] = []
        kept_indices_by_layer: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            if shared_topk_global is not None:
                # Drop any per-layer score still held by the collector; the
                # shared layer's score already drove token selection.
                collector.query_states[layer_idx] = None
                collector.layer_token_scores[layer_idx] = None
                topk_global = shared_topk_global
            else:
                token_score = self._score_one_layer(
                    collector=collector,
                    past_key_values=past_key_values,
                    layer_idx=layer_idx,
                    rope_mode=self.rope_mode,
                    vision_start=vision_start,
                    vision_end=vision_end,
                    groups=groups,
                    scaling=scaling,
                )
                topk_local = self._topk_local_with_eviction(
                    token_score,
                    effective_k,
                    self._evict_count_for_layer(layer_idx, evict_count),
                )
                topk_global = (topk_local + vision_start).to(torch.long)

            topk_sorted_cpu = topk_global.detach().to("cpu").sort().values
            layer_frame_counts: list[int] = []
            for frame_start, frame_end in frame_token_spans:
                selected_in_frame = (
                    (topk_sorted_cpu >= int(frame_start))
                    & (topk_sorted_cpu < int(frame_end))
                )
                layer_frame_counts.append(int(selected_in_frame.sum().item()))

            kept = torch.cat([non_vision_tensor, topk_global], dim=0)
            kept, _ = torch.sort(kept)
            self._compress_layer_cache(past_key_values, layer_idx, kept)
            kept_indices_by_layer.append(kept)
            layer_top_vision_global.append([int(v) for v in topk_sorted_cpu.tolist()])
            layer_selected_frame_token_counts.append(layer_frame_counts)

        compressed_cache_len = past_key_values.layers[0].keys.shape[2]
        expected_len = len(non_vision_positions) + effective_k
        if compressed_cache_len != expected_len:
            raise RuntimeError(
                f"Compressed cache length mismatch: got {compressed_cache_len}, "
                f"expected {expected_len}"
            )
        for layer_idx in range(1, num_layers):
            if past_key_values.layers[layer_idx].keys.shape[2] != compressed_cache_len:
                raise RuntimeError(
                    f"Inconsistent compressed cache length at layer {layer_idx}."
                )

        # -- Phase 2b: optional position-embedding reindex of the compressed cache.
        text_device = inputs_embeds.device
        if self.decode_pe_mode == "reindex":
            self._reindex_cache_positions(
                past_key_values=past_key_values,
                kept_indices_by_layer=kept_indices_by_layer,
                prefill_position_ids=position_ids,
                compressed_cache_len=compressed_cache_len,
                text_device=text_device,
                hidden_dtype=inputs_embeds.dtype,
            )

        # -- Phase 3: replay the final prompt token on top of compressed KV, then
        # continue manual decode. This makes the first answer token depend on the
        # compressed/reindexed cache instead of the full-prefill logits.
        prefix_cache_len = int(compressed_cache_len) - 1
        self._truncate_cache_to_length(past_key_values, prefix_cache_len)
        if past_key_values.layers[0].keys.shape[2] != prefix_cache_len:
            raise RuntimeError(
                f"Prefix cache truncation failed: got {past_key_values.layers[0].keys.shape[2]}, "
                f"expected {prefix_cache_len}."
            )

        if self.decode_pe_mode == "reindex":
            prompt_token_position = torch.full(
                (3, 1),
                fill_value=int(prefix_cache_len),
                dtype=torch.long,
                device=text_device,
            )
            current_position = torch.full(
                (3, 1),
                fill_value=int(compressed_cache_len),
                dtype=torch.long,
                device=text_device,
            )
        else:
            last_position_col = position_ids[..., -1]  # M-RoPE: shape [3, 1]
            prompt_token_position = last_position_col.clone()
            current_position = (last_position_col + 1).clone()

        eos_candidates = {
            int(self.im_end_id),
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1,
        }

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_generate = time.perf_counter()

        last_prompt_token = input_ids[:, -1:].to(device=text_device, dtype=torch.long)
        cur_cache_len = past_key_values.layers[0].keys.shape[2]
        attn_mask = torch.ones(
            (last_prompt_token.shape[0], cur_cache_len + 1),
            device=text_device,
            dtype=attention_mask.dtype,
        )
        first_outputs = self.model(
            input_ids=last_prompt_token,
            past_key_values=past_key_values,
            attention_mask=attn_mask,
            position_ids=prompt_token_position.view(3, 1, 1),
            use_cache=True,
            return_dict=True,
        )
        past_key_values = first_outputs.past_key_values
        replayed_cache_len = int(past_key_values.layers[0].keys.shape[2])
        if replayed_cache_len != compressed_cache_len:
            raise RuntimeError(
                f"Prompt-token replay cache length mismatch: got {replayed_cache_len}, "
                f"expected {compressed_cache_len}."
            )
        first_logits = first_outputs.logits[:, -1, :]
        compressed_first_token_diagnostics = self._first_token_diagnostics(
            tokenizer, first_logits
        )
        first_token_id = int(compressed_first_token_diagnostics["argmax_token_id"])
        generated_ids: list[int] = [first_token_id]
        del first_outputs, first_logits

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        ttft_seconds = time.perf_counter() - t_prefill

        max_new_tokens = int(self.max_new_tokens)
        step_idx = 0
        while len(generated_ids) < max_new_tokens and generated_ids[-1] not in eos_candidates:
            next_token = torch.tensor(
                [[generated_ids[-1]]], device=text_device, dtype=torch.long
            )
            cur_cache_len = past_key_values.layers[0].keys.shape[2]
            attn_mask = torch.ones(
                (1, cur_cache_len + 1), device=text_device, dtype=attention_mask.dtype
            )
            pos_ids_step = current_position.view(3, 1, 1)
            step_outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                attention_mask=attn_mask,
                position_ids=pos_ids_step,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = step_outputs.past_key_values
            step_logits = step_outputs.logits[:, -1, :]
            next_id = int(torch.argmax(step_logits, dim=-1).item())
            generated_ids.append(next_id)
            current_position = current_position + 1
            step_idx += 1

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        generate_time = time.perf_counter() - t_generate

        trimmed = []
        for token_id in generated_ids:
            if token_id in eos_candidates:
                break
            trimmed.append(int(token_id))
        response = tokenizer.decode(trimmed, skip_special_tokens=True).strip()

        self._last_ttft_seconds = float(ttft_seconds)

        del past_key_values, collector, cached_embeds, cached_grid_thw, prefill_inputs
        release_unused_cuda_memory()

        return {
            "response": response,
            "per_frame_vision_tokens": int(per_frame_tokens),
            "vision_token_budget": int(budget),
            "effective_token_budget": int(effective_k),
            "num_vision_tokens": int(num_vision_tokens),
            "total_seq_len": int(total_seq_len),
            "vision_token_start": int(vision_start),
            "vision_token_end": int(vision_end),
            "frame_token_counts": [int(count) for count in frame_token_counts],
            "frame_token_spans": [
                [int(frame_start), int(frame_end)]
                for frame_start, frame_end in frame_token_spans
            ],
            "retrieval_query_token_count": int(len(query_positions)),
            "num_non_vision_tokens": int(len(non_vision_positions)),
            "compressed_cache_len": int(compressed_cache_len),
            "layer_top_vision_global_indices": layer_top_vision_global,
            "layer_selected_frame_token_counts": layer_selected_frame_token_counts,
            "first_token_full_prefill": full_prefill_first_token_diagnostics,
            "first_token_compressed": compressed_first_token_diagnostics,
            "first_token_choice_logit_delta": self._diagnostic_delta(
                compressed_first_token_diagnostics,
                full_prefill_first_token_diagnostics,
                "choice_logits",
            ),
            "first_token_choice_logprob_delta": self._diagnostic_delta(
                compressed_first_token_diagnostics,
                full_prefill_first_token_diagnostics,
                "choice_logprobs",
            ),
            "first_token_argmax_changed": bool(
                compressed_first_token_diagnostics["argmax_token_id"]
                != full_prefill_first_token_diagnostics["argmax_token_id"]
            ),
            "first_token_choice_argmax_changed": bool(
                compressed_first_token_diagnostics["choice_argmax"]
                != full_prefill_first_token_diagnostics["choice_argmax"]
            ),
            "shared_scoring_layer": (
                int(self.shared_scoring_layer)
                if self.shared_scoring_layer is not None
                else None
            ),
            "evict_spec_raw": _format_evict_spec(self.evict_spec),
            "evict_count_absolute": int(evict_count),
            "evict_layer_range": _format_evict_layer_range(self.evict_layer_range),
            "prefill_time": float(prefill_time),
            "generate_time": float(generate_time),
            "ttft_seconds": float(ttft_seconds),
        }


def make_ovo_key(item: dict[str, Any]) -> str:
    return f"{item.get('task', '')}:{item.get('id')}"


def infer_split_name(task: str) -> str:
    if task in EVAL_BACKWARD_TASKS:
        return "backward"
    if task in REAL_TIME_TASKS:
        return "realtime"
    return "unknown"


def get_checkpoint_path(result_dir: str, process_index: int, num_processes: int) -> str:
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "results_incremental.jsonl")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "results_incremental.jsonl")


def get_done_path(result_dir: str, process_index: int, num_processes: int) -> str:
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "done")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "done")


def strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "_key"}


def load_checkpoint_state(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    records, done_keys = load_jsonl_results(path)
    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []

    for raw in records:
        item = strip_internal_fields(raw)
        if str(item.get("task", "")) not in EVAL_TASK_SET:
            continue
        key = raw.get("_key")
        if not isinstance(key, str) or not key:
            key = make_ovo_key(item)
        done_keys.add(key)
        split_name = str(item.get("split") or infer_split_name(str(item.get("task", ""))))
        if split_name == "backward":
            backward_results.append(item)
        elif split_name == "realtime":
            realtime_results.append(item)

    return backward_results, realtime_results, forward_results, done_keys


def append_checkpoint_row(handle: Any, item: dict[str, Any]) -> None:
    record = dict(item)
    record["_key"] = make_ovo_key(item)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def merge_shard_results(result_dir: str, num_processes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_paths = (
        [os.path.join(result_dir, "results_incremental.jsonl")]
        if num_processes == 1
        else [os.path.join(result_dir, f"rank_{rank}", "results_incremental.jsonl") for rank in range(num_processes)]
    )

    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for path in checkpoint_paths:
        records, _ = load_jsonl_results(path)
        for raw in records:
            item = strip_internal_fields(raw)
            if str(item.get("task", "")) not in EVAL_TASK_SET:
                continue
            key = raw.get("_key")
            if not isinstance(key, str) or not key:
                key = make_ovo_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            split_name = str(item.get("split") or infer_split_name(str(item.get("task", ""))))
            if split_name == "backward":
                backward_results.append(item)
            elif split_name == "realtime":
                realtime_results.append(item)

    return backward_results, realtime_results, forward_results


def write_done_marker(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(datetime.now().isoformat() + "\n")


def wait_for_done_markers(result_dir: str, num_processes: int) -> None:
    if num_processes <= 1:
        return

    timeout_seconds = float(os.environ.get("FILE_SYNC_TIMEOUT_SECONDS", "43200"))
    poll_interval = float(os.environ.get("FILE_SYNC_POLL_SECONDS", "10"))
    done_paths = [os.path.join(result_dir, f"rank_{rank}", "done") for rank in range(num_processes)]
    deadline = time.time() + timeout_seconds

    while True:
        missing = [path for path in done_paths if not os.path.exists(path)]
        if not missing:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for rank completion markers: {missing}")
        time.sleep(poll_interval)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def decode_full_video_to_chunks(
    *,
    video_path: str,
    chunk_duration: float,
    fps: float,
) -> tuple[list[Any], str]:
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        return decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=None,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent


def score_record(record: dict[str, Any]) -> int:
    if record.get("correct") is not None:
        return int(record["correct"])
    return int(score_ovo_br(record.get("response"), str(record.get("ground_truth", ""))))


def build_eval_summary(
    *,
    backward_results: list[dict[str, Any]],
    realtime_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    split_specs = (
        ("backward", "Backward Tracing", EVAL_BACKWARD_TASKS, backward_results),
        ("realtime", "Real-time Perception", REAL_TIME_TASKS, realtime_results),
    )

    split_payloads: dict[str, Any] = {}
    official_split_avgs: list[float] = []
    total_correct = 0
    total_count = 0

    for split_name, split_title, task_order, records in split_specs:
        subset_rows: list[dict[str, Any]] = []
        subsets: dict[str, Any] = {}
        split_correct = 0
        split_total = 0

        for task in task_order:
            task_records = [record for record in records if str(record.get("task", "")) == task]
            if not task_records:
                continue

            task_correct = sum(score_record(record) for record in task_records)
            task_total = len(task_records)
            task_accuracy = 100.0 * float(task_correct) / float(task_total)
            row = {
                "task": task,
                "correct": task_correct,
                "total": task_total,
                "accuracy": task_accuracy,
            }
            subset_rows.append(row)
            subsets[task] = row
            split_correct += task_correct
            split_total += task_total

        official_split_avg = (
            sum(row["accuracy"] for row in subset_rows) / len(subset_rows)
            if subset_rows
            else 0.0
        )
        pooled_accuracy = 100.0 * float(split_correct) / float(split_total) if split_total else 0.0
        split_payloads[split_name] = {
            "title": split_title,
            "subset_accuracy_list": subset_rows,
            "subsets": subsets,
            "official_split_avg": official_split_avg,
            "pooled_accuracy": pooled_accuracy,
            "correct": split_correct,
            "total": split_total,
        }

        if subset_rows:
            official_split_avgs.append(official_split_avg)
        total_correct += split_correct
        total_count += split_total

    official_total_avg = (
        sum(official_split_avgs) / len(official_split_avgs)
        if official_split_avgs
        else 0.0
    )
    pooled_overall_accuracy = 100.0 * float(total_correct) / float(total_count) if total_count else 0.0

    return {
        "config": config,
        "splits": split_payloads,
        "overall": {
            "official_total_avg": official_total_avg,
            "pooled_overall_accuracy": pooled_overall_accuracy,
            "correct": total_correct,
            "total": total_count,
        },
    }


def print_eval_summary(model_label: str, summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"OVO-Bench Per-Token Attention Retrieval Results ({model_label})")
    print("=" * 60)

    for split_name in ("backward", "realtime"):
        split_summary = summary.get("splits", {}).get(split_name, {})
        subset_rows = split_summary.get("subset_accuracy_list", [])
        if not subset_rows:
            continue

        print(f"\n{split_summary.get('title', split_name.title())}:")
        for row in subset_rows:
            print(f"  {row['task']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
        print(f"  Official Avg.: {split_summary['official_split_avg']:.2f}%")
        print(
            f"  Pooled Acc.: {split_summary['pooled_accuracy']:.2f}% "
            f"({split_summary['correct']}/{split_summary['total']})"
        )

    overall = summary.get("overall", {})
    print(f"\n{'=' * 60}")
    print(f"Official Total Avg.: {float(overall.get('official_total_avg', 0.0)):.2f}%")
    print(
        f"Pooled Overall Acc.: {float(overall.get('pooled_overall_accuracy', 0.0)):.2f}% "
        f"({int(overall.get('correct', 0))}/{int(overall.get('total', 0))})"
    )
    print("=" * 60)


def evaluate_token_retrieval_backward_realtime(
    *,
    anno: dict[str, Any],
    split_name: str,
    chunked_dir: str,
    qa: AttentionTokenRetrievalQA,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    max_analysis_frames: int,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    prompt = build_ovo_prompt(anno["task"], anno)
    ground_truth = chr(65 + int(anno["gt"]))

    chunks, decode_backend = decode_full_video_to_chunks(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    frames, frame_rows, recent_indices, recent_chunk_ids = flatten_chunks(chunks, recent_frames_only)
    num_sampled_frames = len(frames)
    del chunks

    analysis_frame_indices, analysis_sampling_strategy = select_attention_frame_indices(
        total_frames=num_sampled_frames,
        recent_indices=recent_indices,
        max_analysis_frames=max_analysis_frames,
    )
    if not analysis_frame_indices:
        raise ValueError(f"No analysis frames selected for video: {video_path}")

    analysis_frames = [frames[frame_index] for frame_index in analysis_frame_indices]
    retrieval_query = str(anno.get("question", "")).strip()
    generation_output = qa.score_and_generate(
        analysis_frames,
        prompt,
        retrieval_query=retrieval_query,
    )
    del analysis_frames

    response = generation_output["response"]
    correct = int(score_ovo_br(response, ground_truth))

    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "split": split_name,
        "question": anno["question"],
        "response": response,
        "ground_truth": ground_truth,
        "correct": correct,
        "video_path": video_path,
        "decode_backend": decode_backend,
        "recent_chunk_ids": recent_chunk_ids,
        "recent_frame_indices": recent_indices,
        "num_sampled_frames": num_sampled_frames,
        "num_candidate_frames": len(analysis_frame_indices),
        "analysis_frame_indices": [int(index) for index in analysis_frame_indices],
        "analysis_sampling_strategy": analysis_sampling_strategy,
        "per_frame_vision_tokens": generation_output["per_frame_vision_tokens"],
        "vision_token_budget": generation_output["vision_token_budget"],
        "effective_token_budget": generation_output["effective_token_budget"],
        "num_vision_tokens": generation_output["num_vision_tokens"],
        "vision_token_start": generation_output["vision_token_start"],
        "vision_token_end": generation_output["vision_token_end"],
        "frame_token_counts": generation_output["frame_token_counts"],
        "frame_token_spans": generation_output["frame_token_spans"],
        "num_non_vision_tokens": generation_output["num_non_vision_tokens"],
        "total_seq_len": generation_output["total_seq_len"],
        "retrieval_query": retrieval_query,
        "retrieval_query_token_count": generation_output["retrieval_query_token_count"],
        "compressed_cache_len": generation_output["compressed_cache_len"],
        "layer_top_vision_global_indices": generation_output["layer_top_vision_global_indices"],
        "layer_selected_frame_token_counts": generation_output[
            "layer_selected_frame_token_counts"
        ],
        "first_token_full_prefill": generation_output["first_token_full_prefill"],
        "first_token_compressed": generation_output["first_token_compressed"],
        "first_token_choice_logit_delta": generation_output[
            "first_token_choice_logit_delta"
        ],
        "first_token_choice_logprob_delta": generation_output[
            "first_token_choice_logprob_delta"
        ],
        "first_token_argmax_changed": generation_output["first_token_argmax_changed"],
        "first_token_choice_argmax_changed": generation_output[
            "first_token_choice_argmax_changed"
        ],
        "shared_scoring_layer": generation_output["shared_scoring_layer"],
        "evict_spec_raw": generation_output["evict_spec_raw"],
        "evict_count_absolute": generation_output["evict_count_absolute"],
        "evict_layer_range": generation_output["evict_layer_range"],
        "prefill_time": generation_output["prefill_time"],
        "score_time": generation_output["prefill_time"],
        "generate_time": generation_output["generate_time"],
        "ttft_seconds": generation_output["ttft_seconds"],
        "num_vision_tokens_before": int(generation_output["num_vision_tokens"]),
        "num_vision_tokens_after": int(generation_output["effective_token_budget"]),
        "num_frames": int(len(analysis_frame_indices)),
        "retrieval_rope_mode": qa.rope_mode,
        "decode_pe_mode": qa.decode_pe_mode,
    }


def build_error_record(anno: dict[str, Any], split_name: str, chunked_dir: str, error: Exception) -> dict[str, Any]:
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "split": split_name,
        "question": anno["question"],
        "response": None,
        "ground_truth": chr(65 + int(anno["gt"])),
        "correct": 0,
        "video_path": os.path.join(chunked_dir, f"{anno['id']}.mp4"),
        "error": str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OVO-Bench layer-wise per-token attention retrieval evaluation for Qwen3-VL"
    )
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument(
        "--result_dir", default="results/ovo_bench_attn_token_retrieval_qwen3vl"
    )
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument(
        "--max_analysis_frames",
        "--max_frames",
        dest="max_analysis_frames",
        type=int,
        default=12,
        help=(
            "Candidate frame pool size. Recent frames are always included and "
            "the remaining budget is filled uniformly (uniform_with_recent_anchor)."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help=(
            "Attention kernel used for prefill and decode. Default 'flash_attention_2'. "
            "Attention weights are no longer needed: retrieval scoring uses Q and K "
            "captured separately via a forward pre-hook."
        ),
    )
    parser.add_argument(
        "--retrieval_rope_mode",
        choices=list(RETRIEVAL_ROPE_MODES),
        default="post_rope",
        help=(
            "Whether to apply RoPE to Q and K before the retrieval Q·K^T score. "
            "'post_rope' reuses K from the KV cache (matches the original attention "
            "signal); 'pre_rope' captures K pre-rotation in the hook "
            "(content-only similarity)."
        ),
    )
    parser.add_argument(
        "--decode_pe_mode",
        choices=list(DECODE_PE_MODES),
        default="original",
        help=(
            "Position-embedding layout used for decoding after KV compression. "
            "'original' keeps each retained token's real M-RoPE position; "
            "'reindex' re-rotates the cache so retained tokens occupy positions "
            "0..N-1 and decode continues from N."
        ),
    )
    parser.add_argument(
        "--shared_scoring_layer",
        type=int,
        default=None,
        help=(
            "If set, score vision tokens at this single decoder layer and apply "
            "the same top-K indices to every layer's KV cache (test7). When "
            "omitted, test6's per-layer independent selection is used."
        ),
    )
    parser.add_argument(
        "--pre_retrieval_evict",
        type=_parse_evict_spec,
        default=None,
        metavar="N|N%",
        help=(
            "Evict the highest-scoring vision tokens before top-K retrieval "
            "(test8). Accepts an integer count (e.g. '32') or a percentage of "
            "the candidate vision tokens (e.g. '10%%'). Omit to disable. "
            "Compatible with --shared_scoring_layer."
        ),
    )
    parser.add_argument(
        "--evict_layer_range",
        type=_parse_evict_layer_range,
        default=None,
        metavar="START-END",
        help=(
            "Apply --pre_retrieval_evict only to this inclusive decoder-layer range "
            "(e.g. '20-34'). Incompatible with --shared_scoring_layer."
        ),
    )
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument("--analysis_scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--max_samples_per_subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_device",
        choices=["local_process", "auto"],
        default="local_process",
        help=(
            "Model placement mode. "
            "'local_process' keeps one full model replica per accelerate process. "
            "'auto' uses Hugging Face device_map='auto' and requires --num_processes=1."
        ),
    )
    args = parser.parse_args()

    if args.max_analysis_frames < 1:
        raise ValueError("--max_analysis_frames must be >= 1")
    if args.max_samples_per_split is not None and args.max_samples_per_split < 1:
        raise ValueError("--max_samples_per_split must be >= 1 when provided.")
    if args.max_samples_per_subset is not None and args.max_samples_per_subset < 1:
        raise ValueError("--max_samples_per_subset must be >= 1 when provided.")
    if args.max_samples_per_split is not None and args.max_samples_per_subset is not None:
        raise ValueError("Use either --max_samples_per_split or --max_samples_per_subset, not both.")
    if args.shared_scoring_layer is not None and args.shared_scoring_layer < 0:
        raise ValueError("--shared_scoring_layer must be >= 0 when provided.")
    if args.evict_layer_range is not None and args.pre_retrieval_evict is None:
        raise ValueError("--evict_layer_range requires --pre_retrieval_evict.")
    if args.evict_layer_range is not None and args.shared_scoring_layer is not None:
        raise ValueError("--evict_layer_range is not compatible with --shared_scoring_layer.")

    accelerator = Accelerator()

    if args.model_device == "auto" and accelerator.num_processes != 1:
        raise ValueError("--model_device=auto requires accelerate --num_processes=1.")

    split_sample_cap = (
        None
        if args.max_samples_per_subset is not None
        else smoke_cap_or_default(args.analysis_scope, args.max_samples_per_split)
    )

    with open(args.anno_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    rng = random.Random(args.seed)
    backward_anno, backward_available_counts, backward_selected_counts = select_split_annotations(
        annotations,
        EVAL_BACKWARD_TASKS,
        rng,
        max_samples_per_split=split_sample_cap,
        max_samples_per_subset=args.max_samples_per_subset,
    )
    realtime_anno, realtime_available_counts, realtime_selected_counts = select_split_annotations(
        annotations,
        REAL_TIME_TASKS,
        rng,
        max_samples_per_split=split_sample_cap,
        max_samples_per_subset=args.max_samples_per_subset,
    )

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"OVO-Bench Per-Token Attention Retrieval Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}")
    if EXCLUDED_BACKWARD_TASKS:
        accelerator.print(f"Excluded backward tasks: {', '.join(sorted(EXCLUDED_BACKWARD_TASKS))}")
    accelerator.print(f"Excluded forward tasks: {', '.join(EXCLUDED_FORWARD_TASKS)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Window: recent_frames_only={args.recent_frames_only}, "
        f"max_analysis_frames={args.max_analysis_frames}, "
        f"frames_worth_budget={FRAMES_WORTH_BUDGET}, "
        f"attn_implementation={args.attn_implementation}, "
        f"retrieval_rope_mode={args.retrieval_rope_mode}, "
        f"decode_pe_mode={args.decode_pe_mode}, "
        f"shared_scoring_layer={args.shared_scoring_layer}, "
        f"pre_retrieval_evict={_format_evict_spec(args.pre_retrieval_evict)}, "
        f"evict_layer_range={_format_evict_layer_range(args.evict_layer_range)}, "
        f"chunk_duration={args.chunk_duration}, fps={args.fps}"
    )
    accelerator.print(f"Scope: {args.analysis_scope}")
    if args.max_samples_per_subset is not None:
        accelerator.print(f"Sampling: up to {args.max_samples_per_subset} per subset/task")
    elif split_sample_cap is not None:
        accelerator.print(f"Sampling: up to {split_sample_cap} per split")
    else:
        accelerator.print("Sampling: full split")
    accelerator.print(
        f"Backward subsets: {format_task_counts(EVAL_BACKWARD_TASKS, backward_selected_counts, backward_available_counts)}"
    )
    accelerator.print(
        f"Realtime subsets: {format_task_counts(REAL_TIME_TASKS, realtime_selected_counts, realtime_available_counts)}"
    )
    accelerator.print(f"{'=' * 60}\n")

    evaluator = AttentionTokenRetrievalQA(
        model_name=args.model_path,
        device="auto" if args.model_device == "auto" else accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
        rope_mode=args.retrieval_rope_mode,
        decode_pe_mode=args.decode_pe_mode,
        shared_scoring_layer=args.shared_scoring_layer,
        evict_spec=args.pre_retrieval_evict,
        evict_layer_range=args.evict_layer_range,
    )

    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)

    checkpoint_path = get_checkpoint_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    done_path = get_done_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    if os.path.exists(done_path):
        os.remove(done_path)
    backward_results, realtime_results, forward_results, done_keys = load_checkpoint_state(checkpoint_path)

    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_file:
        split_jobs = (
            ("backward", local_backward, backward_results),
            ("realtime", local_realtime, realtime_results),
        )
        for split_name, local_annos, result_sink in split_jobs:
            for anno in tqdm(
                local_annos,
                desc=f"[GPU{accelerator.process_index}] {split_name.title()}",
                disable=not accelerator.is_local_main_process,
            ):
                key = make_ovo_key(anno)
                if key in done_keys:
                    continue
                try:
                    result = evaluate_token_retrieval_backward_realtime(
                        anno=anno,
                        split_name=split_name,
                        chunked_dir=args.chunked_dir,
                        qa=evaluator,
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        recent_frames_only=args.recent_frames_only,
                        max_analysis_frames=args.max_analysis_frames,
                    )
                except Exception as exc:
                    result = build_error_record(anno, split_name, args.chunked_dir, exc)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                result_sink.append(result)
                done_keys.add(key)
                append_checkpoint_row(checkpoint_file, result)

    write_done_marker(done_path)

    if accelerator.is_main_process:
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)

        summary_config = {
            "generated_at": datetime.now().isoformat(),
            "model_label": MODEL_LABEL,
            "model_path": args.model_path,
            "anno_path": args.anno_path,
            "chunked_dir": args.chunked_dir,
            "result_dir": args.result_dir,
            "analysis_scope": args.analysis_scope,
            "recent_frames_only": args.recent_frames_only,
            "max_analysis_frames": args.max_analysis_frames,
            "frames_worth_budget": FRAMES_WORTH_BUDGET,
            "retrieval_granularity": "per_layer_per_token",
            "attention_score_policy": "raw_qk_mean_over_heads_then_mean_over_question_tokens",
            "attn_implementation": args.attn_implementation,
            "retrieval_rope_mode": args.retrieval_rope_mode,
            "decode_pe_mode": args.decode_pe_mode,
            "shared_scoring_layer": args.shared_scoring_layer,
            "pre_retrieval_evict": _format_evict_spec(args.pre_retrieval_evict),
            "evict_layer_range": _format_evict_layer_range(args.evict_layer_range),
            "chunk_duration": args.chunk_duration,
            "fps": args.fps,
            "max_qa_tokens": args.max_qa_tokens,
            "max_samples_per_split": split_sample_cap,
            "max_samples_per_subset": args.max_samples_per_subset,
            "seed": args.seed,
            "excluded_backward_tasks": sorted(EXCLUDED_BACKWARD_TASKS),
            "excluded_forward_tasks": list(EXCLUDED_FORWARD_TASKS),
            "sampling_counts": {
                "backward": {
                    "available": backward_available_counts,
                    "selected": backward_selected_counts,
                },
                "realtime": {
                    "available": realtime_available_counts,
                    "selected": realtime_selected_counts,
                },
            },
        }
        summary = build_eval_summary(
            backward_results=all_backward,
            realtime_results=all_realtime,
            config=summary_config,
        )
        print_eval_summary(MODEL_LABEL, summary)

        os.makedirs(args.result_dir, exist_ok=True)
        summary_path = os.path.join(args.result_dir, "summary.json")
        write_json(summary_path, summary)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.result_dir, f"qwen3vl_attn_token_retrieval_results_{timestamp}.json"
        )
        write_json(
            output_path,
            {
                "config": summary_config,
                "summary": summary,
                "backward": all_backward,
                "realtime": all_realtime,
                "forward": all_forward,
            },
        )
        print(f"\nSummary saved to: {summary_path}")
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
