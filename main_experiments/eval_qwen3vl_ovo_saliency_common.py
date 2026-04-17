from __future__ import annotations

import gc
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from lib.frame_saliency_qwen3 import build_experiment_summary, save_json, slugify
from lib.recent_window_eval import build_ovo_prompt, load_jsonl_results
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

EXCLUDED_BACKWARD_TASKS = frozenset({"HLD"})
EVAL_BACKWARD_TASKS = [task for task in BACKWARD_TASKS if task not in EXCLUDED_BACKWARD_TASKS]


@dataclass
class SaliencyExperimentConfig:
    run_label: str
    anno_path: str
    chunked_dir: str
    result_dir: str
    recent_frames_only: int = 4
    chunk_duration: float = 1.0
    fps: float = 1.0
    analysis_scope: str = "full"
    max_samples_per_split: int | None = None
    max_samples_per_subset: int | None = None
    similarity_backends: list[str] = field(default_factory=list)
    attention_modes: list[str] = field(default_factory=list)
    save_example_matrices: int = 5
    save_raw_attn_examples: int = 0
    max_analysis_frames: int = 768
    seed: int = 42
    extra_summary_config: dict[str, Any] = field(default_factory=dict)


def add_common_saliency_args(
    parser: Any,
    *,
    default_result_dir: str,
    include_save_raw_attn_examples: bool,
) -> None:
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default=default_result_dir)
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--analysis_scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--max_samples_per_subset", type=int, default=None)
    parser.add_argument("--save_example_matrices", type=int, default=5)
    if include_save_raw_attn_examples:
        parser.add_argument("--save_raw_attn_examples", type=int, default=0)
    parser.add_argument("--max_analysis_frames", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)


def make_key(task: str, sample_id: Any) -> str:
    return f"{task}:{sample_id}"


def smoke_cap_or_default(scope: str, max_samples_per_split: int | None) -> int | None:
    if max_samples_per_split is not None:
        return max_samples_per_split
    return 8 if scope == "smoke" else None


def select_split_annotations(
    annotations: list[dict[str, Any]],
    allowed_tasks: list[str],
    rng: random.Random,
    max_samples_per_split: int | None = None,
    max_samples_per_subset: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {task: [] for task in allowed_tasks}
    for anno in annotations:
        task = str(anno.get("task", ""))
        if task in grouped:
            grouped[task].append(anno)

    available_counts = {task: len(grouped[task]) for task in allowed_tasks}

    if max_samples_per_subset is not None:
        selected: list[dict[str, Any]] = []
        selected_counts: dict[str, int] = {}
        for task in allowed_tasks:
            task_annos = list(grouped[task])
            rng.shuffle(task_annos)
            chosen = task_annos[:max_samples_per_subset]
            selected_counts[task] = len(chosen)
            selected.extend(chosen)
        rng.shuffle(selected)
        return selected, available_counts, selected_counts

    pooled = [anno for task in allowed_tasks for anno in grouped[task]]
    rng.shuffle(pooled)
    if max_samples_per_split is not None:
        pooled = pooled[:max_samples_per_split]
    selected_counter = Counter(str(anno.get("task", "")) for anno in pooled)
    selected_counts = {task: int(selected_counter.get(task, 0)) for task in allowed_tasks}
    return pooled, available_counts, selected_counts


def format_task_counts(
    allowed_tasks: list[str],
    selected_counts: dict[str, int],
    available_counts: dict[str, int],
) -> str:
    return ", ".join(
        f"{task}={selected_counts.get(task, 0)}/{available_counts.get(task, 0)}"
        for task in allowed_tasks
    )


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def format_mean_std(mean_value: Any, std_value: Any) -> str:
    if mean_value is None:
        return "n/a"
    mean = float(mean_value)
    if std_value is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {float(std_value):.4f}"


def print_metric_summary(summary: dict[str, Any]) -> None:
    metric_labels = {
        "siglip_similarity": "SigLIP similarity",
        "question_prefill_attention": "Question prefill attention",
    }
    metrics = summary.get("metrics", {})
    if not metrics:
        return

    print("Aggregate metrics (pooled):")
    for metric_name, label in metric_labels.items():
        stats = metrics.get(metric_name)
        if not stats:
            continue
        mean_percentile = format_mean_std(
            stats.get("recent4_mean_percentile_mean"),
            stats.get("recent4_mean_percentile_std"),
        )
        print(
            f"  {label}: recent4_mean_percentile={mean_percentile} "
            f"(n={stats.get('count', 0)})"
        )


def run_saliency_experiment(
    analyzer: Any,
    config: SaliencyExperimentConfig,
) -> dict[str, Any]:
    if config.max_samples_per_split is not None and config.max_samples_per_split < 1:
        raise ValueError("--max_samples_per_split must be >= 1 when provided.")
    if config.max_samples_per_subset is not None and config.max_samples_per_subset < 1:
        raise ValueError("--max_samples_per_subset must be >= 1 when provided.")
    if config.max_samples_per_split is not None and config.max_samples_per_subset is not None:
        raise ValueError("Use either --max_samples_per_split or --max_samples_per_subset, not both.")
    if not config.similarity_backends and not config.attention_modes:
        raise ValueError("At least one similarity backend or attention mode must be enabled.")

    split_sample_cap = (
        None
        if config.max_samples_per_subset is not None
        else smoke_cap_or_default(config.analysis_scope, config.max_samples_per_split)
    )

    with open(config.anno_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    rng = random.Random(config.seed)
    backward_anno, backward_available_counts, backward_selected_counts = select_split_annotations(
        annotations,
        EVAL_BACKWARD_TASKS,
        rng,
        max_samples_per_split=split_sample_cap,
        max_samples_per_subset=config.max_samples_per_subset,
    )
    realtime_anno, realtime_available_counts, realtime_selected_counts = select_split_annotations(
        annotations,
        REAL_TIME_TASKS,
        rng,
        max_samples_per_split=split_sample_cap,
        max_samples_per_subset=config.max_samples_per_subset,
    )

    result_dir = Path(config.result_dir)
    records_path = result_dir / "records.jsonl"
    examples_dir = result_dir / "examples"
    result_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    existing_records, done_keys = load_jsonl_results(str(records_path))
    existing_records = [
        record for record in existing_records if str(record.get("task", "")) not in EXCLUDED_BACKWARD_TASKS
    ]
    saved_examples_by_task = Counter(
        str(record.get("task", ""))
        for record in existing_records
        if record.get("matrix_example_saved")
    )
    saved_raw_by_task = Counter(
        str(record.get("task", ""))
        for record in existing_records
        if record.get("raw_attention_saved")
    )

    print("\n" + "=" * 60)
    print(config.run_label)
    print("=" * 60)
    print(f"Backward: {len(backward_anno)}")
    print(f"Realtime: {len(realtime_anno)}")
    if EXCLUDED_BACKWARD_TASKS:
        print(f"Excluded backward tasks: {', '.join(sorted(EXCLUDED_BACKWARD_TASKS))}")
    print(f"Similarity: {config.similarity_backends or 'disabled'}")
    print(f"Attention: {config.attention_modes or 'disabled'}")
    print(f"Scope: {config.analysis_scope}")
    if config.max_samples_per_subset is not None:
        print(f"Sampling: up to {config.max_samples_per_subset} per subset/task")
    elif split_sample_cap is not None:
        print(f"Sampling: up to {split_sample_cap} per split")
    else:
        print("Sampling: full split")
    print(f"Backward subsets: {format_task_counts(EVAL_BACKWARD_TASKS, backward_selected_counts, backward_available_counts)}")
    print(f"Realtime subsets: {format_task_counts(REAL_TIME_TASKS, realtime_selected_counts, realtime_available_counts)}")
    print(f"Result Dir: {result_dir}")
    print("=" * 60 + "\n")

    all_records = [
        record for record in existing_records if str(record.get("task", "")) not in EXCLUDED_BACKWARD_TASKS
    ]

    with records_path.open("a", encoding="utf-8") as handle:
        for split_name, split_annos in (("backward", backward_anno), ("realtime", realtime_anno)):
            for anno in tqdm(split_annos, desc=split_name.capitalize()):
                key = make_key(anno["task"], anno["id"])
                if key in done_keys:
                    continue

                video_path = os.path.join(config.chunked_dir, f"{anno['id']}.mp4")
                task_name = str(anno["task"])
                save_example = saved_examples_by_task[task_name] < int(config.save_example_matrices)
                save_raw = bool(config.attention_modes) and (
                    saved_raw_by_task[task_name] < int(config.save_raw_attn_examples)
                )

                try:
                    record, example_payload = analyzer.analyze_sample(
                        video_path=video_path,
                        prompt=build_ovo_prompt(anno["task"], anno),
                        similarity_text=str(anno.get("question", "")).strip(),
                        chunk_duration=config.chunk_duration,
                        fps=config.fps,
                        recent_frames_only=config.recent_frames_only,
                        similarity_backends=config.similarity_backends,
                        attention_modes=config.attention_modes,
                        max_analysis_frames=config.max_analysis_frames,
                        save_example_matrices=save_example,
                        save_raw_attentions=save_raw,
                    )
                    output = {
                        "_key": key,
                        "split": split_name,
                        "id": anno["id"],
                        "task": anno["task"],
                        "video": anno.get("video"),
                        "question": anno.get("question"),
                        "ground_truth": chr(65 + int(anno["gt"])),
                        **record,
                    }
                    if example_payload is not None:
                        example_payload.update(
                            {
                                "split": split_name,
                                "task": anno["task"],
                                "id": anno["id"],
                                "question": anno.get("question"),
                            }
                        )
                        example_name = slugify(key) or f"example-{len(all_records)}"
                        example_path = examples_dir / f"{example_name}.pt"
                        torch.save(example_payload, example_path)
                        payload_has_raw_attention = any(
                            str(key_name).startswith("raw_")
                            for key_name in example_payload.keys()
                        )
                        output["example_saved"] = True
                        output["matrix_example_saved"] = bool(save_example)
                        output["raw_attention_saved"] = bool(save_raw and payload_has_raw_attention)
                        output["example_path"] = str(example_path)
                        if save_example:
                            saved_examples_by_task[task_name] += 1
                        if save_raw and payload_has_raw_attention:
                            saved_raw_by_task[task_name] += 1
                    else:
                        output["example_saved"] = False
                        output["matrix_example_saved"] = False
                        output["raw_attention_saved"] = False
                except Exception as exc:
                    output = {
                        "_key": key,
                        "split": split_name,
                        "id": anno["id"],
                        "task": anno["task"],
                        "video": anno.get("video"),
                        "question": anno.get("question"),
                        "ground_truth": chr(65 + int(anno["gt"])),
                        "video_path": video_path,
                        "error": str(exc),
                    }
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                append_record(handle, output)
                all_records.append(output)
                done_keys.add(key)

    summary_config = {
        "generated_at": datetime.now().isoformat(),
        "anno_path": config.anno_path,
        "chunked_dir": config.chunked_dir,
        "result_dir": config.result_dir,
        "recent_frames_only": config.recent_frames_only,
        "chunk_duration": config.chunk_duration,
        "fps": config.fps,
        "analysis_scope": config.analysis_scope,
        "max_samples_per_split": split_sample_cap,
        "max_samples_per_subset": config.max_samples_per_subset,
        "excluded_backward_tasks": sorted(EXCLUDED_BACKWARD_TASKS),
        "similarity_backends": list(config.similarity_backends),
        "attention_modes": list(config.attention_modes),
        "save_example_matrices": config.save_example_matrices,
        "save_raw_attn_examples": config.save_raw_attn_examples,
        "max_analysis_frames": config.max_analysis_frames,
        "seed": config.seed,
    }
    summary_config.update(config.extra_summary_config)

    summary = build_experiment_summary(all_records, config=summary_config)
    save_json(result_dir / "summary.json", summary)

    print("\n" + "=" * 60)
    print_metric_summary(summary)
    print(f"Records saved to: {records_path}")
    print(f"Summary saved to: {result_dir / 'summary.json'}")
    print("=" * 60 + "\n")

    return summary
