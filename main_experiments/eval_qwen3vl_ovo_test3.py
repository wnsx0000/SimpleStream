"""
OVO-Bench attention-guided top-4 frame evaluation for Qwen3-VL.

For each sample, this script prefills Qwen3-VL on a candidate frame pool
(recent frames + uniform context up to ``--max_analysis_frames``), captures
question-prefill attention at a single decoder layer specified by
``--layer_number``, scores each candidate frame by summing attention over its
vision-token span, keeps the top-4 frames, reorders them temporally, and then
generates the answer from those 4 frames.

This mirrors ``eval_qwen3vl_ovo_test2.py`` but replaces SigLIP similarity
scoring with Qwen3-VL self-attention scoring (same capture path as
``eval_qwen3vl_ovo_test1_2.py``). ``--attn_implementation eager`` is required.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
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
    LayerwiseFrameAttentionCollector,
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

MODEL_LABEL = "Qwen3-VL-Attn-Top4"
EXCLUDED_FORWARD_TASKS = ("REC", "SSR", "CRR")
EXCLUDED_BACKWARD_TASKS = frozenset({"HLD"})
EVAL_BACKWARD_TASKS = [task for task in BACKWARD_TASKS if task not in EXCLUDED_BACKWARD_TASKS]
EVAL_TASK_SET = frozenset([*EVAL_BACKWARD_TASKS, *REAL_TIME_TASKS])
TOP_K_FRAMES = 4


class AttentionTop4QA(Qwen3Recent4FrameSaliencyAnalyzer):
    """Qwen3-VL wrapper that ranks candidate frames by decoder attention at one layer."""

    @torch.inference_mode()
    def score_frames_by_layer_attention(
        self,
        frames: list[Image.Image],
        question: str,
        layer_number: int,
    ) -> list[float]:
        text_layers = self._get_text_layers()
        num_layers = len(text_layers)
        if not (0 <= int(layer_number) < num_layers):
            raise ValueError(
                f"--layer_number={layer_number} out of range [0, {num_layers})"
            )

        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        frame_token_counts = frame_token_counts_from_grid(cached_grid_thw, self.merge_size)
        prefill_inputs = self._build_cached_multimodal_inputs(
            cached_embeds=cached_embeds,
            cached_grid_thw=cached_grid_thw,
            frame_token_counts=frame_token_counts,
            question=question,
        )
        collector = LayerwiseFrameAttentionCollector(
            frame_token_spans=prefill_inputs["frame_token_spans"],
            query_positions=prefill_inputs["question_token_positions"],
            num_layers=num_layers,
        )
        try:
            self._run_with_collector(
                collector,
                use_cache=False,
                input_ids=None,
                inputs_embeds=prefill_inputs["inputs_embeds"],
                attention_mask=prefill_inputs["attention_mask"],
                position_ids=prefill_inputs["position_ids"],
            )
            scores = collector.as_tensor()[int(layer_number)].tolist()
        finally:
            del cached_embeds, cached_grid_thw, prefill_inputs, collector
            release_unused_cuda_memory()
        return [float(value) for value in scores]


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
    print(f"OVO-Bench Attention Top-4 Results ({model_label})")
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


def evaluate_attention_top4_backward_realtime(
    *,
    anno: dict[str, Any],
    split_name: str,
    chunked_dir: str,
    qa: AttentionTop4QA,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    max_analysis_frames: int,
    layer_number: int,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    prompt = build_ovo_prompt(anno["task"], anno)
    scoring_text = str(anno.get("question", "")).strip() or prompt
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
    t_score = time.perf_counter()
    analysis_frame_scores = qa.score_frames_by_layer_attention(
        analysis_frames, scoring_text, layer_number,
    )
    score_time = time.perf_counter() - t_score
    del analysis_frames

    top_k = min(TOP_K_FRAMES, len(analysis_frame_scores))
    ranked_candidate_positions = sorted(
        range(len(analysis_frame_scores)),
        key=lambda position: -float(analysis_frame_scores[position]),
    )[:top_k]
    selected_frame_indices_by_attention = [
        int(analysis_frame_indices[position]) for position in ranked_candidate_positions
    ]
    selected_frame_indices_for_inference = sorted(selected_frame_indices_by_attention)
    selected_frames = [frames[frame_index] for frame_index in selected_frame_indices_for_inference]

    t0 = time.perf_counter()
    response = qa.generate_from_frames(selected_frames, prompt)
    generate_time = time.perf_counter() - t0

    attention_rank_by_frame_index = {
        int(analysis_frame_indices[position]): rank
        for rank, position in enumerate(ranked_candidate_positions, start=1)
    }
    attention_score_by_frame_index = {
        int(analysis_frame_indices[position]): float(analysis_frame_scores[position])
        for position in ranked_candidate_positions
    }
    selected_frame_rows = [
        {
            "frame_index": int(frame_rows[frame_index]["frame_index"]),
            "chunk_index": int(frame_rows[frame_index]["chunk_index"]),
            "timestamp": float(frame_rows[frame_index]["timestamp"]),
            "is_recent": bool(frame_rows[frame_index]["is_recent"]),
            "attention_score": attention_score_by_frame_index[int(frame_index)],
            "attention_rank": attention_rank_by_frame_index[int(frame_index)],
        }
        for frame_index in selected_frame_indices_for_inference
    ]
    final_chunk_ids = sorted({int(frame_rows[frame_index]["chunk_index"]) for frame_index in selected_frame_indices_for_inference})
    correct = int(score_ovo_br(response, ground_truth))

    del selected_frames

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
        "analysis_frame_scores": [float(score) for score in analysis_frame_scores],
        "analysis_sampling_strategy": analysis_sampling_strategy,
        "scoring_layer": int(layer_number),
        "selected_frame_indices_by_attention": selected_frame_indices_by_attention,
        "selected_frame_indices_for_inference": [int(index) for index in selected_frame_indices_for_inference],
        "selected_frames": selected_frame_rows,
        "final_chunk_ids": final_chunk_ids,
        "score_time": score_time,
        "generate_time": generate_time,
        "ttft_seconds": float(getattr(qa, "_last_ttft_seconds", 0.0) or 0.0),
        "num_vision_tokens": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_vision_tokens_before": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_vision_tokens_after": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "num_frames": int(getattr(qa, "_last_num_vision_frames", len(selected_frame_indices_for_inference)) or 0),
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
    parser = argparse.ArgumentParser(description="OVO-Bench attention-guided top-4 frame evaluation for Qwen3-VL")
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_attn_top4_qwen3vl")
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
        "--layer_number",
        type=int,
        required=True,
        help="Qwen3-VL decoder layer index whose question-prefill attention is used to rank candidate frames.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        help="Must be 'eager' to capture attention weights at the scoring layer.",
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
    if args.layer_number < 0:
        raise ValueError("--layer_number must be >= 0")
    if args.max_samples_per_split is not None and args.max_samples_per_split < 1:
        raise ValueError("--max_samples_per_split must be >= 1 when provided.")
    if args.max_samples_per_subset is not None and args.max_samples_per_subset < 1:
        raise ValueError("--max_samples_per_subset must be >= 1 when provided.")
    if args.max_samples_per_split is not None and args.max_samples_per_subset is not None:
        raise ValueError("Use either --max_samples_per_split or --max_samples_per_subset, not both.")

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
    accelerator.print(f"OVO-Bench Attention Top-4 Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}")
    if EXCLUDED_BACKWARD_TASKS:
        accelerator.print(f"Excluded backward tasks: {', '.join(sorted(EXCLUDED_BACKWARD_TASKS))}")
    accelerator.print(f"Excluded forward tasks: {', '.join(EXCLUDED_FORWARD_TASKS)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Window: recent_frames_only={args.recent_frames_only}, "
        f"max_analysis_frames={args.max_analysis_frames}, "
        f"top_k={TOP_K_FRAMES}, layer_number={args.layer_number}, "
        f"attn_implementation={args.attn_implementation}, "
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

    evaluator = AttentionTop4QA(
        model_name=args.model_path,
        device="auto" if args.model_device == "auto" else accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
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
                    result = evaluate_attention_top4_backward_realtime(
                        anno=anno,
                        split_name=split_name,
                        chunked_dir=args.chunked_dir,
                        qa=evaluator,
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        recent_frames_only=args.recent_frames_only,
                        max_analysis_frames=args.max_analysis_frames,
                        layer_number=args.layer_number,
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
            "top_k": TOP_K_FRAMES,
            "layer_number": args.layer_number,
            "attn_implementation": args.attn_implementation,
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
        output_path = os.path.join(args.result_dir, f"qwen3vl_attn_top4_results_{timestamp}.json")
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
