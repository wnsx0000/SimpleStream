"""
OVO-Bench Visual-RAG with chunked attention for Qwen3-VL (Test 5).

This script extends Test 4 (V-RAG) with a *chunked attention* restriction:
frame selection is identical to Test 4 (recent N frames + top-K SigLIP
retrieved historical frames), but during Qwen3-VL prefill the self-attention
is restricted so that retrieved historical frames are split into
``--chunk_size``-sized chunks in temporal order, recent frames form a single
separate chunk, and cross-chunk attention between frame tokens is blocked in
every decoder layer. Non-frame tokens (system prompt, question, generation
tokens) remain unaffected and can attend to every preceding frame token
causally.

The chunked mask is implemented as a 4D additive float attention mask
(shape ``[1, 1, seq_len, seq_len]``) and injected during prefill only by
monkey-patching ``transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask``.
``create_causal_mask`` early-returns when given a 4D mask, so the mask is
consumed as-is by every decoder layer. Decode steps (seq_len == 1) fall
through to the original function. The SDPA attention backend is the default
because flash_attention_2 does not support arbitrary 4D masks.

When ``--save_example_matrices`` > 0, the script captures Qwen3-VL
question-prefill self-attention under the chunked mask and writes per-example
``.pt`` payloads under ``<result_dir>/examples/`` for offline heatmap
plotting with ``analysis/plot_vrag_attention_heatmap.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS
from lib.frame_saliency_qwen3 import (
    LayerwiseFrameAttentionCollector,
    Qwen3Recent4FrameSaliencyAnalyzer,
    SiglipFrameEncoder,
    build_question_prefill_attention_map_metadata,
    cosine_scores_against_query,
    format_siglip_device_for_log,
    flatten_chunks,
    frame_token_counts_from_grid,
    question_prefill_layer_indices,
    slugify,
)
from lib.recent_window_eval import (
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    load_jsonl_results,
    score_ovo_br,
)
from main_experiments.eval_qwen3vl_ovo_saliency_common import (
    format_task_counts,
    select_split_annotations,
    smoke_cap_or_default,
)

EXCLUDED_FORWARD_TASKS = ("REC", "SSR", "CRR")
EXCLUDED_BACKWARD_TASKS = frozenset({"HLD"})
EVAL_BACKWARD_TASKS = [task for task in BACKWARD_TASKS if task not in EXCLUDED_BACKWARD_TASKS]
EVAL_TASK_SET = frozenset([*EVAL_BACKWARD_TASKS, *REAL_TIME_TASKS])
DEFAULT_TOP_K_HISTORICAL_FRAMES = 5
DEFAULT_CHUNK_SIZE = 2
DECODE_MAX_FRAMES = 768
CANDIDATE_FRAME_POLICY = "non_recent_frames"
ANALYSIS_SAMPLING_STRATEGY_PREFIX = "non_recent_frames_qwen_cap"
VISION_ATTN_IMPLEMENTATION = "flash_attention_2"


def format_model_label(top_k_historical: int, chunk_size: int) -> str:
    return f"Qwen3-VL-V-RAG-Top{int(top_k_historical)}-Chunk{int(chunk_size)}"


def format_analysis_sampling_strategy(decode_max_frames: int) -> str:
    return f"{ANALYSIS_SAMPLING_STRATEGY_PREFIX}{int(decode_max_frames)}"


def assign_frame_chunks(
    combined_frame_indices: list[int],
    recent_index_set: set[int],
    chunk_size: int,
) -> tuple[list[int], list[int], int]:
    """Assign a chunk id to every frame in ``combined_frame_indices``.

    Retrieved (non-recent) frames are grouped in temporal order with at most
    ``chunk_size`` frames per chunk (chunk ids 0..R-1). Recent frames form a
    single final chunk (chunk id R) regardless of ``chunk_size``.

    Returns a tuple of:
      - chunk_id_per_frame: list aligned with ``combined_frame_indices``
      - retrieved_chunk_sizes: size of each retrieved chunk
      - recent_chunk_id: the chunk id assigned to the recent-frame block
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    retrieved_positions_in_combined: list[int] = [
        position
        for position, frame_index in enumerate(combined_frame_indices)
        if int(frame_index) not in recent_index_set
    ]
    retrieved_chunk_id_by_position: dict[int, int] = {}
    retrieved_chunk_sizes: list[int] = []
    for local_index, position in enumerate(retrieved_positions_in_combined):
        chunk_id = local_index // int(chunk_size)
        retrieved_chunk_id_by_position[position] = chunk_id
        if chunk_id >= len(retrieved_chunk_sizes):
            retrieved_chunk_sizes.append(0)
        retrieved_chunk_sizes[chunk_id] += 1

    recent_chunk_id = len(retrieved_chunk_sizes)

    chunk_id_per_frame: list[int] = []
    for position, frame_index in enumerate(combined_frame_indices):
        if int(frame_index) in recent_index_set:
            chunk_id_per_frame.append(recent_chunk_id)
        else:
            chunk_id_per_frame.append(retrieved_chunk_id_by_position[position])
    return chunk_id_per_frame, retrieved_chunk_sizes, recent_chunk_id


def build_per_chunk_position_ids(
    *,
    position_ids: torch.Tensor,
    frame_token_spans: list[tuple[int, int]],
    chunk_id_per_frame: list[int],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-number mRoPE ``position_ids`` so each frame chunk starts at the same base.

    The goal is PCW-style positional encoding: every frame chunk sees mRoPE
    coordinates as if it were the only video input. Concretely, each chunk's
    first frame token is shifted to land on ``position_ids[:, 0, image_token_start]``
    (the original position of the first frame's first token), preserving the
    chunk's internal relative offsets. Non-frame tokens (system prompt,
    ``<vision_start>``) keep their original positions; post-frame tokens
    (``<vision_end>``, question, assistant prompt) are shifted to start right
    after the max position produced by any chunk.

    ``rope_deltas`` is recomputed as ``new_max_position + 1 - seq_len`` to
    match HF's convention so the decode-step position calculation in
    ``Qwen3VLModel.forward`` (modeling_qwen3_vl.py:1266) continues to line up.

    Inputs:
      position_ids: original mRoPE position_ids of shape ``[3, 1, seq_len]``.
      frame_token_spans: per-frame ``(start, end_exclusive)`` token ranges,
        aligned with ``chunk_id_per_frame``.
      chunk_id_per_frame: chunk assignment per frame, from ``assign_frame_chunks``.
      seq_len: total prefill sequence length.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            f"Expected mRoPE position_ids of shape [3, 1, seq_len], got {tuple(position_ids.shape)}"
        )
    if len(frame_token_spans) != len(chunk_id_per_frame):
        raise ValueError(
            "frame_token_spans and chunk_id_per_frame must have the same length: "
            f"{len(frame_token_spans)} vs {len(chunk_id_per_frame)}"
        )
    if not frame_token_spans:
        new_max = int(position_ids.max().item())
        new_rope_deltas = torch.tensor(
            [[new_max + 1 - int(seq_len)]],
            dtype=torch.long,
            device=position_ids.device,
        )
        return position_ids.clone(), new_rope_deltas

    device = position_ids.device
    new_position_ids = position_ids.clone()

    image_token_start = int(frame_token_spans[0][0])
    p_start = position_ids[:, 0, image_token_start].clone()  # shape [3]

    chunk_to_range: dict[int, tuple[int, int]] = {}
    for (span_start, span_end), chunk_id in zip(frame_token_spans, chunk_id_per_frame):
        span_start = int(span_start)
        span_end = int(span_end)
        if chunk_id in chunk_to_range:
            existing_start, existing_end = chunk_to_range[chunk_id]
            chunk_to_range[chunk_id] = (
                min(existing_start, span_start),
                max(existing_end, span_end),
            )
        else:
            chunk_to_range[chunk_id] = (span_start, span_end)

    # Sanity: the union of chunk ranges must cover exactly
    # [image_token_start, last_frame_end) with no gaps and no overlaps, because
    # combined_frame_indices is temporally sorted and chunks are contiguous.
    sorted_ranges = sorted(chunk_to_range.values(), key=lambda pair: pair[0])
    prev_end = sorted_ranges[0][0]
    for start, end in sorted_ranges:
        if start != prev_end:
            raise ValueError(
                "Chunks are expected to form a contiguous frame-token range, "
                f"but found gap/overlap at start={start} prev_end={prev_end}"
            )
        prev_end = end

    max_new_pos = int(position_ids[:, 0, :image_token_start].max().item())
    for chunk_start, chunk_end in sorted_ranges:
        chunk_first_pos = position_ids[:, 0, chunk_start]  # [3]
        shift_per_dim = (chunk_first_pos - p_start).view(3, 1)  # [3, 1]
        new_position_ids[:, 0, chunk_start:chunk_end] = (
            position_ids[:, 0, chunk_start:chunk_end] - shift_per_dim
        )
        chunk_max = int(new_position_ids[:, 0, chunk_start:chunk_end].max().item())
        if chunk_max > max_new_pos:
            max_new_pos = chunk_max

    last_frame_end = int(frame_token_spans[-1][1])
    if last_frame_end < int(seq_len):
        orig_post_first = int(position_ids[:, 0, last_frame_end].max().item())
        new_post_first = max_new_pos + 1
        shift_post = orig_post_first - new_post_first
        if shift_post != 0:
            new_position_ids[:, 0, last_frame_end:] = (
                position_ids[:, 0, last_frame_end:] - shift_post
            )

    new_max = int(new_position_ids.max().item())
    new_rope_deltas = torch.tensor(
        [[new_max + 1 - int(seq_len)]],
        dtype=torch.long,
        device=device,
    )
    return new_position_ids, new_rope_deltas


def build_chunked_attention_mask_4d(
    *,
    seq_len: int,
    frame_token_spans: list[tuple[int, int]],
    chunk_id_per_frame: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Construct a 4D additive attention mask that blocks cross-chunk frame attention.

    Shape ``[1, 1, seq_len, seq_len]``. Entries are 0 where attention is
    allowed and ``finfo.min`` where it is blocked. Non-frame tokens keep the
    standard causal behaviour (lower triangular zeros).
    """
    if len(frame_token_spans) != len(chunk_id_per_frame):
        raise ValueError(
            "frame_token_spans and chunk_id_per_frame must have the same length: "
            f"{len(frame_token_spans)} vs {len(chunk_id_per_frame)}"
        )

    neg_inf = torch.finfo(dtype).min
    # Start with a standard causal mask: upper-triangle (strict) = -inf.
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    causal_block = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )
    mask.masked_fill_(causal_block, neg_inf)

    # Per-token chunk id. -1 indicates a non-frame token (unchanged semantics).
    chunk_id_per_token = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for (start, end), chunk_id in zip(frame_token_spans, chunk_id_per_frame):
        chunk_id_per_token[int(start):int(end)] = int(chunk_id)

    q_ids = chunk_id_per_token.view(-1, 1)
    k_ids = chunk_id_per_token.view(1, -1)
    # Block only when BOTH q and k are frame tokens AND they belong to different chunks.
    cross_chunk_block = (q_ids >= 0) & (k_ids >= 0) & (q_ids != k_ids)
    mask.masked_fill_(cross_chunk_block, neg_inf)

    return mask.view(1, 1, seq_len, seq_len)


@contextmanager
def chunked_causal_mask_override(custom_4d_mask: torch.Tensor):
    """Temporarily route Qwen3VL's ``create_causal_mask`` through our 4D mask.

    The override fires only during prefill (query length > 1 and no pre-existing
    KV cache). Decode steps fall through to the original implementation so the
    standard causal behaviour is retained for generated tokens.
    """
    original_create_causal_mask = qwen3vl_mod.create_causal_mask

    def patched_create_causal_mask(**kwargs: Any):
        input_embeds = kwargs.get("input_embeds")
        past_key_values = kwargs.get("past_key_values")
        cache_len = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            try:
                cache_len = int(past_key_values.get_seq_length())
            except Exception:
                cache_len = 0
        is_prefill = (
            input_embeds is not None
            and input_embeds.shape[1] > 1
            and cache_len == 0
        )
        if is_prefill:
            return custom_4d_mask.to(device=input_embeds.device, dtype=input_embeds.dtype)
        return original_create_causal_mask(**kwargs)

    qwen3vl_mod.create_causal_mask = patched_create_causal_mask
    try:
        yield
    finally:
        qwen3vl_mod.create_causal_mask = original_create_causal_mask


@contextmanager
def temporary_visual_attn_implementation(
    qa: Qwen3Recent4FrameSaliencyAnalyzer,
    attn_implementation: str,
):
    """Temporarily switch only the Qwen3-VL vision encoder attention backend.

    Qwen3-VL's non-FA2 vision path currently hits a length mismatch for
    multi-image inputs. The chunked 4D mask is needed only in the text decoder,
    so vision features can still be encoded with FA2 before the text prefill
    runs under ``sdpa`` or ``eager``.
    """
    config_candidates: list[Any] = []
    visual = qa._get_visual_module()
    if hasattr(visual, "config"):
        config_candidates.append(visual.config)
    for module in visual.modules():
        module_config = getattr(module, "config", None)
        if module_config is not None:
            config_candidates.append(module_config)
    model_config = getattr(qa.model, "config", None)
    if model_config is not None and hasattr(model_config, "vision_config"):
        config_candidates.append(model_config.vision_config)
    hf_config = getattr(qa._get_hf_model(), "config", None)
    if hf_config is not None and hasattr(hf_config, "vision_config"):
        config_candidates.append(hf_config.vision_config)

    configs: list[Any] = []
    seen: set[int] = set()
    for config in config_candidates:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        configs.append(config)

    previous_values = [
        (
            config,
            getattr(config, "_attn_implementation", None),
            hasattr(config, "_attn_implementation"),
        )
        for config in configs
    ]
    for config, _, _ in previous_values:
        config._attn_implementation = attn_implementation
    try:
        yield
    finally:
        for config, previous_value, had_attr in previous_values:
            if had_attr:
                config._attn_implementation = previous_value
            elif hasattr(config, "_attn_implementation"):
                delattr(config, "_attn_implementation")


def _eos_token_id_set(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, torch.Tensor):
        return {int(token_id) for token_id in eos_token_id.detach().cpu().flatten().tolist()}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


@torch.inference_mode()
def greedy_generate_with_chunked_prefill(
    qa: Qwen3Recent4FrameSaliencyAnalyzer,
    *,
    inputs: dict[str, Any],
    chunked_mask_4d: torch.Tensor,
) -> str:
    """Run a masked prefill directly, then continue deterministic greedy decode.

    HF ``generate()`` may reshape/slice ``inputs_embeds`` before the first model
    call. Test 5 needs the prefill sequence length to match the custom 4D mask
    exactly, so the first forward pass is driven explicitly and subsequent
    single-token steps use the returned KV cache with the normal causal mask.
    """
    inputs_embeds = inputs["inputs_embeds"]
    attention_mask = inputs["attention_mask"]
    position_ids = inputs["position_ids"]
    rope_deltas = inputs["rope_deltas"]
    seq_len = int(inputs_embeds.shape[1])

    hf_model = qa._get_hf_model()
    previous_rope_deltas = getattr(hf_model, "rope_deltas", None)
    hf_model.rope_deltas = rope_deltas.to(inputs_embeds.device)
    eos_token_ids = _eos_token_id_set(getattr(qa.model.generation_config, "eos_token_id", None))

    t0 = time.perf_counter()
    generated_tokens: list[torch.Tensor] = []
    try:
        with chunked_causal_mask_override(chunked_mask_4d):
            outputs = qa.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.arange(seq_len, device=inputs_embeds.device),
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        qa._last_ttft_seconds = time.perf_counter() - t0
        past_key_values = outputs.past_key_values
        current_attention_mask = attention_mask

        for _ in range(int(qa.max_new_tokens)):
            token_for_decode = next_token.to(current_attention_mask.device)
            generated_tokens.append(token_for_decode.detach().cpu())
            if eos_token_ids and int(token_for_decode.item()) in eos_token_ids:
                break

            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            cache_position = torch.tensor(
                [int(current_attention_mask.shape[1] - 1)],
                dtype=torch.long,
                device=current_attention_mask.device,
            )
            outputs = qa.model(
                input_ids=token_for_decode,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        if not generated_tokens:
            return ""
        generated_ids = torch.cat(generated_tokens, dim=1)
        return qa.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    finally:
        hf_model.rope_deltas = previous_rope_deltas


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


def load_checkpoint_state(
    path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str], Counter]:
    records, done_keys = load_jsonl_results(path)
    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    saved_examples_by_task: Counter = Counter()

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
        if item.get("matrix_example_saved"):
            saved_examples_by_task[str(item.get("task", ""))] += 1

    return backward_results, realtime_results, forward_results, done_keys, saved_examples_by_task


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


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_full_video_to_chunks(
    *,
    video_path: str,
    chunk_duration: float,
    fps: float,
    decode_max_frames: int,
) -> tuple[list[Any], str]:
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        return decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=None,
            max_frames=int(decode_max_frames),
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
    print(f"OVO-Bench Visual-RAG Results ({model_label})")
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


def generate_with_chunked_attention(
    *,
    qa: Qwen3Recent4FrameSaliencyAnalyzer,
    combined_frames: list[Any],
    combined_frame_indices: list[int],
    recent_index_set: set[int],
    chunk_size: int,
    prompt: str,
    reset_chunk_position_ids: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run Qwen3-VL generation with chunked attention over frame tokens.

    ``combined_frames`` / ``combined_frame_indices`` must be in temporal order and
    aligned. ``prompt`` is the full OVO prompt passed to the assistant head as
    the user turn text.
    """
    with temporary_visual_attn_implementation(qa, VISION_ATTN_IMPLEMENTATION):
        cached_embeds, cached_grid_thw = qa.encode_vision(combined_frames)
    try:
        frame_token_counts = frame_token_counts_from_grid(cached_grid_thw, qa.merge_size)
        inputs = qa._build_cached_multimodal_inputs(
            cached_embeds=cached_embeds,
            cached_grid_thw=cached_grid_thw,
            frame_token_counts=frame_token_counts,
            question=prompt,
        )

        frame_token_spans = inputs["frame_token_spans"]
        chunk_id_per_frame, retrieved_chunk_sizes, recent_chunk_id = assign_frame_chunks(
            combined_frame_indices=combined_frame_indices,
            recent_index_set=recent_index_set,
            chunk_size=chunk_size,
        )

        inputs_embeds = inputs["inputs_embeds"]
        chunked_mask_4d = build_chunked_attention_mask_4d(
            seq_len=int(inputs_embeds.shape[1]),
            frame_token_spans=frame_token_spans,
            chunk_id_per_frame=chunk_id_per_frame,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        if reset_chunk_position_ids:
            new_position_ids, new_rope_deltas = build_per_chunk_position_ids(
                position_ids=inputs["position_ids"],
                frame_token_spans=frame_token_spans,
                chunk_id_per_frame=chunk_id_per_frame,
                seq_len=int(inputs_embeds.shape[1]),
            )
            inputs["position_ids"] = new_position_ids
            inputs["rope_deltas"] = new_rope_deltas

        # The analyzer normally sets these inside ``generate_from_frames``. Because
        # we bypass the chat-template path, set them manually so downstream fields
        # ``num_vision_tokens`` / ``num_frames`` remain populated.
        qa._last_num_vision_tokens = int(cached_embeds.shape[0])
        qa._last_num_vision_frames = int(len(combined_frames))

        response = greedy_generate_with_chunked_prefill(
            qa,
            inputs=inputs,
            chunked_mask_4d=chunked_mask_4d,
        )

        chunk_metadata = {
            "chunk_size": int(chunk_size),
            "chunk_id_per_frame": [int(c) for c in chunk_id_per_frame],
            "retrieved_chunk_sizes": [int(s) for s in retrieved_chunk_sizes],
            "recent_chunk_id": int(recent_chunk_id),
            "num_retrieved_chunks": len(retrieved_chunk_sizes),
            "num_chunks_total": len(retrieved_chunk_sizes) + (1 if recent_index_set else 0),
        }
        return response, chunk_metadata
    finally:
        del cached_embeds, cached_grid_thw
        release_cuda_cache()


def capture_question_prefill_attention_maps(
    *,
    qa: Qwen3Recent4FrameSaliencyAnalyzer,
    combined_frames: list[Any],
    combined_frame_indices: list[int],
    recent_index_set: set[int],
    chunk_size: int,
    question_text: str,
    reset_chunk_position_ids: bool = False,
) -> dict[str, Any] | None:
    if not combined_frames or not question_text:
        return None

    cached_embeds = None
    cached_grid_thw = None
    question_only_inputs = None
    collector = None
    prefill_output = None
    prefill_scores = None

    try:
        with temporary_visual_attn_implementation(qa, VISION_ATTN_IMPLEMENTATION):
            cached_embeds, cached_grid_thw = qa.encode_vision(combined_frames)
        frame_token_counts = frame_token_counts_from_grid(cached_grid_thw, qa.merge_size)

        text_layers = qa._get_text_layers()
        num_text_layers = len(text_layers)
        display_layer_indices = question_prefill_layer_indices(num_text_layers)

        question_only_inputs = qa._build_cached_multimodal_inputs(
            cached_embeds=cached_embeds,
            cached_grid_thw=cached_grid_thw,
            frame_token_counts=frame_token_counts,
            question=question_text,
        )
        map_metadata = build_question_prefill_attention_map_metadata(
            frame_token_spans=question_only_inputs["frame_token_spans"],
            query_positions=question_only_inputs["question_token_positions"],
            attention_frame_indices=combined_frame_indices,
        )
        collector = LayerwiseFrameAttentionCollector(
            frame_token_spans=question_only_inputs["frame_token_spans"],
            query_positions=question_only_inputs["question_token_positions"],
            num_layers=num_text_layers,
            save_raw=False,
            map_layer_indices=display_layer_indices,
            question_prefill_map_metadata=map_metadata,
        )

        chunk_id_per_frame, _, _ = assign_frame_chunks(
            combined_frame_indices=combined_frame_indices,
            recent_index_set=recent_index_set,
            chunk_size=chunk_size,
        )
        inputs_embeds_q = question_only_inputs["inputs_embeds"]
        chunked_mask_4d = build_chunked_attention_mask_4d(
            seq_len=int(inputs_embeds_q.shape[1]),
            frame_token_spans=question_only_inputs["frame_token_spans"],
            chunk_id_per_frame=chunk_id_per_frame,
            dtype=inputs_embeds_q.dtype,
            device=inputs_embeds_q.device,
        )

        position_ids_q = question_only_inputs["position_ids"]
        if reset_chunk_position_ids:
            position_ids_q, _ = build_per_chunk_position_ids(
                position_ids=position_ids_q,
                frame_token_spans=question_only_inputs["frame_token_spans"],
                chunk_id_per_frame=chunk_id_per_frame,
                seq_len=int(inputs_embeds_q.shape[1]),
            )

        with chunked_causal_mask_override(chunked_mask_4d):
            prefill_output = qa._run_with_collector(
                collector,
                use_cache=False,
                input_ids=None,
                inputs_embeds=inputs_embeds_q,
                attention_mask=question_only_inputs["attention_mask"],
                position_ids=position_ids_q,
            )
        del prefill_output
        prefill_output = None

        attention_map_payload = collector.export_question_prefill_attention_maps(display_layer_indices)
        prefill_scores = collector.as_tensor()
        display_scores = prefill_scores[display_layer_indices].detach().cpu()

        return {
            "question_prefill_attention_maps": attention_map_payload,
            "question_prefill_attention_scores": display_scores,
            "question_prefill_display_layer_indices": [int(i) for i in display_layer_indices],
        }
    finally:
        if isinstance(question_only_inputs, dict):
            question_only_inputs.clear()
        del cached_embeds, cached_grid_thw, question_only_inputs, collector, prefill_output, prefill_scores
        release_cuda_cache()


def save_example_payload(
    *,
    examples_dir: Path,
    key: str,
    anno: dict[str, Any],
    split_name: str,
    video_path: str,
    decode_backend: str,
    recent_frame_indices: list[int],
    recent_chunk_ids: list[int],
    historical_candidate_indices: list[int],
    historical_candidate_scores: list[float],
    selected_historical_frame_indices_by_similarity: list[int],
    selected_historical_frame_indices: list[int],
    combined_frame_indices: list[int],
    similarity_text: str,
    chunk_metadata: dict[str, Any],
    attention_payload: dict[str, Any],
) -> Path:
    example_payload: dict[str, Any] = {
        "split": split_name,
        "task": anno["task"],
        "id": anno["id"],
        "question": anno.get("question"),
        "video_path": video_path,
        "decode_backend": decode_backend,
        "recent_frame_indices": [int(i) for i in recent_frame_indices],
        "recent_chunk_ids": [int(i) for i in recent_chunk_ids],
        "historical_candidate_frame_indices": [int(i) for i in historical_candidate_indices],
        "historical_candidate_frame_scores": [float(s) for s in historical_candidate_scores],
        "selected_historical_frame_indices_by_similarity": [int(i) for i in selected_historical_frame_indices_by_similarity],
        "selected_historical_frame_indices": [int(i) for i in selected_historical_frame_indices],
        "combined_frame_indices_for_inference": [int(i) for i in combined_frame_indices],
        "similarity_text": similarity_text,
        "chunked_attention": chunk_metadata,
    }
    example_payload.update(attention_payload)

    examples_dir.mkdir(parents=True, exist_ok=True)
    example_name = slugify(key) or f"example-{anno.get('id', 'unknown')}"
    example_path = examples_dir / f"{example_name}.pt"
    torch.save(example_payload, example_path)
    return example_path


def evaluate_vrag_top5_backward_realtime(
    *,
    anno: dict[str, Any],
    split_name: str,
    chunked_dir: str,
    qa: Qwen3Recent4FrameSaliencyAnalyzer,
    siglip_encoder: SiglipFrameEncoder,
    chunk_duration: float,
    fps: float,
    decode_max_frames: int,
    recent_frames_only: int,
    top_k_historical: int,
    chunk_size: int,
    reset_chunk_position_ids: bool = False,
    examples_dir: Path | None = None,
    save_example: bool = False,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    prompt = build_ovo_prompt(anno["task"], anno)
    similarity_text = str(anno.get("question", "")).strip() or prompt
    ground_truth = chr(65 + int(anno["gt"]))

    chunks, decode_backend = decode_full_video_to_chunks(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        decode_max_frames=decode_max_frames,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    frames, frame_rows, recent_indices, recent_chunk_ids = flatten_chunks(chunks, recent_frames_only)
    num_sampled_frames = len(frames)
    del chunks

    if num_sampled_frames == 0:
        raise ValueError(f"No frames decoded from video: {video_path}")

    recent_index_set = set(int(i) for i in recent_indices)
    historical_candidate_indices = [i for i in range(num_sampled_frames) if i not in recent_index_set]

    historical_candidate_scores: list[float] = []
    selected_historical_frame_indices_by_similarity: list[int] = []
    similarity_rank_by_frame_index: dict[int, int] = {}
    similarity_score_by_frame_index: dict[int, float] = {}

    if historical_candidate_indices:
        historical_frames = [frames[frame_index] for frame_index in historical_candidate_indices]
        siglip_features = siglip_encoder.encode_frames(historical_frames)
        siglip_question_feature = siglip_encoder.encode_text(similarity_text)
        historical_candidate_scores = cosine_scores_against_query(
            siglip_features, siglip_question_feature
        ).tolist()
        del historical_frames, siglip_features, siglip_question_feature

        top_k = min(int(top_k_historical), len(historical_candidate_scores))
        ranked_candidate_positions = sorted(
            range(len(historical_candidate_scores)),
            key=lambda position: -float(historical_candidate_scores[position]),
        )[:top_k]
        selected_historical_frame_indices_by_similarity = [
            int(historical_candidate_indices[position]) for position in ranked_candidate_positions
        ]
        similarity_rank_by_frame_index = {
            int(historical_candidate_indices[position]): rank
            for rank, position in enumerate(ranked_candidate_positions, start=1)
        }
        similarity_score_by_frame_index = {
            int(historical_candidate_indices[position]): float(historical_candidate_scores[position])
            for position in ranked_candidate_positions
        }

    selected_historical_frame_indices = sorted(selected_historical_frame_indices_by_similarity)
    combined_frame_indices = sorted(set(selected_historical_frame_indices) | recent_index_set)
    if not combined_frame_indices:
        raise ValueError(f"No frames selected for inference: {video_path}")

    combined_frames = [frames[frame_index] for frame_index in combined_frame_indices]
    del frames

    relative_position_denom = max(1, num_sampled_frames - 1)
    combined_frame_relative_positions = [
        float(frame_index) / float(relative_position_denom)
        for frame_index in combined_frame_indices
    ]
    combined_frame_mean_relative_position = (
        float(sum(combined_frame_relative_positions)) / len(combined_frame_relative_positions)
        if combined_frame_relative_positions
        else 0.0
    )

    t0 = time.perf_counter()
    response, chunk_metadata = generate_with_chunked_attention(
        qa=qa,
        combined_frames=combined_frames,
        combined_frame_indices=combined_frame_indices,
        recent_index_set=recent_index_set,
        chunk_size=chunk_size,
        prompt=prompt,
        reset_chunk_position_ids=reset_chunk_position_ids,
    )
    generate_time = time.perf_counter() - t0
    release_cuda_cache()

    example_path: Path | None = None
    example_save_error: str | None = None
    if save_example and examples_dir is not None:
        attention_payload = None
        try:
            attention_payload = capture_question_prefill_attention_maps(
                qa=qa,
                combined_frames=combined_frames,
                combined_frame_indices=combined_frame_indices,
                recent_index_set=recent_index_set,
                chunk_size=chunk_size,
                question_text=similarity_text,
                reset_chunk_position_ids=reset_chunk_position_ids,
            )
            if attention_payload is not None:
                key = make_ovo_key(anno)
                example_path = save_example_payload(
                    examples_dir=examples_dir,
                    key=key,
                    anno=anno,
                    split_name=split_name,
                    video_path=video_path,
                    decode_backend=decode_backend,
                    recent_frame_indices=recent_indices,
                    recent_chunk_ids=recent_chunk_ids,
                    historical_candidate_indices=historical_candidate_indices,
                    historical_candidate_scores=historical_candidate_scores,
                    selected_historical_frame_indices_by_similarity=selected_historical_frame_indices_by_similarity,
                    selected_historical_frame_indices=selected_historical_frame_indices,
                    combined_frame_indices=combined_frame_indices,
                    similarity_text=similarity_text,
                    chunk_metadata=chunk_metadata,
                    attention_payload=attention_payload,
                )
        except Exception as exc:
            example_save_error = f"{type(exc).__name__}: {exc}"
        finally:
            del attention_payload
            release_cuda_cache()

    selected_historical_frame_rows = [
        {
            "frame_index": int(frame_rows[frame_index]["frame_index"]),
            "chunk_index": int(frame_rows[frame_index]["chunk_index"]),
            "timestamp": float(frame_rows[frame_index]["timestamp"]),
            "is_recent": bool(frame_rows[frame_index]["is_recent"]),
            "similarity_score": similarity_score_by_frame_index[int(frame_index)],
            "similarity_rank": similarity_rank_by_frame_index[int(frame_index)],
        }
        for frame_index in selected_historical_frame_indices
    ]
    final_chunk_ids = sorted(
        {int(frame_rows[frame_index]["chunk_index"]) for frame_index in combined_frame_indices}
    )
    correct = int(score_ovo_br(response, ground_truth))

    del combined_frames

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
        "num_historical_candidate_frames": len(historical_candidate_indices),
        "historical_candidate_frame_indices": [int(i) for i in historical_candidate_indices],
        "historical_candidate_frame_scores": [float(s) for s in historical_candidate_scores],
        "analysis_sampling_strategy": format_analysis_sampling_strategy(decode_max_frames),
        "selected_historical_frame_indices_by_similarity": selected_historical_frame_indices_by_similarity,
        "selected_historical_frame_indices": [int(i) for i in selected_historical_frame_indices],
        "selected_historical_frames": selected_historical_frame_rows,
        "combined_frame_indices_for_inference": [int(i) for i in combined_frame_indices],
        "combined_frame_relative_positions": combined_frame_relative_positions,
        "combined_frame_mean_relative_position": combined_frame_mean_relative_position,
        "final_chunk_ids": final_chunk_ids,
        "chunk_size": int(chunk_metadata["chunk_size"]),
        "chunk_assignment_by_frame_index": {
            int(frame_index): int(chunk_metadata["chunk_id_per_frame"][position])
            for position, frame_index in enumerate(combined_frame_indices)
        },
        "retrieved_chunk_sizes": chunk_metadata["retrieved_chunk_sizes"],
        "recent_chunk_id": int(chunk_metadata["recent_chunk_id"]),
        "num_retrieved_chunks": int(chunk_metadata["num_retrieved_chunks"]),
        "num_chunks_total": int(chunk_metadata["num_chunks_total"]),
        "generate_time": generate_time,
        "ttft_seconds": float(getattr(qa, "_last_ttft_seconds", 0.0) or 0.0),
        "num_vision_tokens": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_vision_tokens_before": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_vision_tokens_after": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_frames": int(getattr(qa, "_last_num_vision_frames", len(combined_frame_indices)) or 0),
        "matrix_example_saved": example_path is not None,
        "example_path": str(example_path) if example_path is not None else None,
        "example_save_error": example_save_error,
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
        description="OVO-Bench Visual-RAG top-K with chunked attention for Qwen3-VL (Test 5)"
    )
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_vrag_chunked_qwen3vl")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument(
        "--max_analysis_frames",
        "--max_frames",
        dest="_deprecated_max_analysis_frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--decode_max_frames",
        type=int,
        default=DECODE_MAX_FRAMES,
        help=(
            "Maximum number of frames decoded for V-RAG retrieval. "
            "SigLIP similarity is computed over the decoded non-recent frames. Default: 768."
        ),
    )
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument("--siglip_model_name", default="google/siglip-so400m-patch14-384")
    parser.add_argument(
        "--siglip_device",
        default="auto",
        help="SigLIP device. 'auto' selects the visible CUDA GPU with the most free memory; use cuda:N or cpu to override.",
    )
    parser.add_argument("--analysis_scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--max_samples_per_subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top_k_historical",
        type=int,
        default=DEFAULT_TOP_K_HISTORICAL_FRAMES,
        help=(
            "Number of non-recent frames to retrieve by SigLIP cosine similarity and "
            "append to the recent-frame input before Qwen3-VL inference. Default: 5."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Chunk size for retrieved historical frames in the chunked attention mask. "
            "Retrieved frames are grouped in temporal order into chunks of this size; "
            "recent frames form a single separate chunk whose size always equals "
            "--recent_frames_only. Cross-chunk attention between frame tokens is blocked "
            "in every decoder layer. Default: 2."
        ),
    )
    parser.add_argument(
        "--reset_chunk_position_ids",
        action="store_true",
        help=(
            "PCW-style positional encoding: each frame chunk receives mRoPE coordinates "
            "as if it were the only video input. All chunks (retrieved and recent) share "
            "the same base position, and question/assistant tokens shift to start right "
            "after the longest chunk's max position. rope_deltas is recomputed so the "
            "decode step continues correctly. Default: off (standard whole-sequence mRoPE)."
        ),
    )
    parser.add_argument(
        "--save_example_matrices",
        type=int,
        default=0,
        help=(
            "Save up to N per-task Qwen3-VL question-prefill attention examples under "
            "<result_dir>/examples/ for later heatmap plotting. Requires "
            "--attn_implementation=eager. Set to 0 to disable. Default: 0."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        help=(
            "Attention backend for Qwen3-VL. Default: 'sdpa'. Chunked attention requires "
            "'sdpa' or 'eager' because 'flash_attention_2' does not support arbitrary 4D masks."
        ),
    )
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

    if args.max_samples_per_split is not None and args.max_samples_per_split < 1:
        raise ValueError("--max_samples_per_split must be >= 1 when provided.")
    if args.max_samples_per_subset is not None and args.max_samples_per_subset < 1:
        raise ValueError("--max_samples_per_subset must be >= 1 when provided.")
    if args.max_samples_per_split is not None and args.max_samples_per_subset is not None:
        raise ValueError("Use either --max_samples_per_split or --max_samples_per_subset, not both.")
    if args.decode_max_frames < 1:
        raise ValueError("--decode_max_frames must be >= 1.")
    if args.top_k_historical < 0:
        raise ValueError("--top_k_historical must be >= 0.")
    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be >= 1.")
    if args.attn_implementation == "flash_attention_2":
        raise ValueError(
            "--attn_implementation=flash_attention_2 is not supported: flash_attention_2 "
            "does not accept arbitrary 4D attention masks. Use 'sdpa' (default) or 'eager'."
        )
    if args.save_example_matrices > 0 and args.attn_implementation != "eager":
        raise ValueError(
            "--save_example_matrices requires --attn_implementation=eager because "
            "SDPA does not return attention weights for the saved heatmap payloads."
        )

    model_label = format_model_label(args.top_k_historical, args.chunk_size)

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
    accelerator.print(
        f"OVO-Bench V-RAG Top-{int(args.top_k_historical)} + Chunked Attention "
        f"(chunk_size={int(args.chunk_size)}) Evaluation ({model_label})"
    )
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}")
    if EXCLUDED_BACKWARD_TASKS:
        accelerator.print(f"Excluded backward tasks: {', '.join(sorted(EXCLUDED_BACKWARD_TASKS))}")
    accelerator.print(f"Excluded forward tasks: {', '.join(EXCLUDED_FORWARD_TASKS)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Window: recent_frames_only={args.recent_frames_only}, "
        f"decode_max_frames={args.decode_max_frames}, "
        f"candidate_frame_policy={CANDIDATE_FRAME_POLICY}, "
        f"top_k_historical={int(args.top_k_historical)}, "
        f"chunk_size={int(args.chunk_size)}, "
        f"reset_chunk_position_ids={bool(args.reset_chunk_position_ids)}, "
        f"text_attn_implementation={args.attn_implementation}, "
        f"vision_attn_implementation={VISION_ATTN_IMPLEMENTATION}, "
        f"chunk_duration={args.chunk_duration}, fps={args.fps}"
    )
    accelerator.print(f"Scope: {args.analysis_scope}")
    if args.max_samples_per_subset is not None:
        accelerator.print(f"Sampling: up to {args.max_samples_per_subset} per subset/task")
    elif split_sample_cap is not None:
        accelerator.print(f"Sampling: up to {split_sample_cap} per split")
    else:
        accelerator.print("Sampling: full split")
    accelerator.print(f"Backward subsets: {format_task_counts(EVAL_BACKWARD_TASKS, backward_selected_counts, backward_available_counts)}")
    accelerator.print(f"Realtime subsets: {format_task_counts(REAL_TIME_TASKS, realtime_selected_counts, realtime_available_counts)}")
    accelerator.print(f"{'=' * 60}\n")

    evaluator = Qwen3Recent4FrameSaliencyAnalyzer(
        model_name=args.model_path,
        device="auto" if args.model_device == "auto" else accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
        siglip_model_name=args.siglip_model_name,
        siglip_device=args.siglip_device,
    )
    siglip_encoder = evaluator.get_siglip_encoder()
    accelerator.print(f"SigLIP device: {format_siglip_device_for_log(siglip_encoder.device)}")

    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)

    checkpoint_path = get_checkpoint_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    done_path = get_done_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    if os.path.exists(done_path):
        os.remove(done_path)
    (
        backward_results,
        realtime_results,
        forward_results,
        done_keys,
        saved_examples_by_task,
    ) = load_checkpoint_state(checkpoint_path)

    examples_dir = Path(args.result_dir) / "examples"
    if args.save_example_matrices > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)
    save_example_cap = int(args.save_example_matrices)

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
                task_name = str(anno.get("task", ""))
                save_example = (
                    save_example_cap > 0
                    and saved_examples_by_task[task_name] < save_example_cap
                )
                try:
                    result = evaluate_vrag_top5_backward_realtime(
                        anno=anno,
                        split_name=split_name,
                        chunked_dir=args.chunked_dir,
                        qa=evaluator,
                        siglip_encoder=siglip_encoder,
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        decode_max_frames=args.decode_max_frames,
                        recent_frames_only=args.recent_frames_only,
                        top_k_historical=args.top_k_historical,
                        chunk_size=args.chunk_size,
                        reset_chunk_position_ids=args.reset_chunk_position_ids,
                        examples_dir=examples_dir,
                        save_example=save_example,
                    )
                    if result.get("matrix_example_saved"):
                        saved_examples_by_task[task_name] += 1
                except Exception as exc:
                    result = build_error_record(anno, split_name, args.chunked_dir, exc)
                    release_cuda_cache()

                result_sink.append(result)
                done_keys.add(key)
                append_checkpoint_row(checkpoint_file, result)

    write_done_marker(done_path)

    if accelerator.is_main_process:
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)

        summary_config = {
            "generated_at": datetime.now().isoformat(),
            "model_label": model_label,
            "model_path": args.model_path,
            "siglip_model_name": args.siglip_model_name,
            "siglip_device": args.siglip_device,
            "resolved_siglip_device": str(siglip_encoder.device),
            "anno_path": args.anno_path,
            "chunked_dir": args.chunked_dir,
            "result_dir": args.result_dir,
            "analysis_scope": args.analysis_scope,
            "recent_frames_only": args.recent_frames_only,
            "decode_max_frames": args.decode_max_frames,
            "candidate_frame_policy": CANDIDATE_FRAME_POLICY,
            "top_k_historical": int(args.top_k_historical),
            "chunk_size": int(args.chunk_size),
            "reset_chunk_position_ids": bool(args.reset_chunk_position_ids),
            "chunk_duration": args.chunk_duration,
            "fps": args.fps,
            "max_qa_tokens": args.max_qa_tokens,
            "save_example_matrices": save_example_cap,
            "attn_implementation": args.attn_implementation,
            "vision_attn_implementation": VISION_ATTN_IMPLEMENTATION,
            "model_device": args.model_device,
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
        print_eval_summary(model_label, summary)

        os.makedirs(args.result_dir, exist_ok=True)
        summary_path = os.path.join(args.result_dir, "summary.json")
        write_json(summary_path, summary)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.result_dir,
            f"qwen3vl_vrag_top{int(args.top_k_historical)}_chunk{int(args.chunk_size)}_results_{timestamp}.json",
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
