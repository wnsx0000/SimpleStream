"""
OVO-Bench recent4 saliency analysis for Qwen3-VL.

Compares the frames selected by SimpleStream's recent4 policy
on the backward and realtime OVO-Bench splits using:
- SigLIP-SO400M frame-question cosine similarity
- Qwen3 text self-attention aggregated layer-wise
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import (  # noqa: E402
    Qwen3Recent4FrameSaliencyAnalyzer,
    build_experiment_summary,
    parse_csv_options,
    save_json,
    slugify,
)
from lib.recent_window_eval import build_ovo_prompt, load_jsonl_results  # noqa: E402
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS  # noqa: E402


def make_key(task: str, sample_id: Any) -> str:
    return f"{task}:{sample_id}"


def smoke_cap_or_default(scope: str, max_samples_per_split: int | None) -> int | None:
    if max_samples_per_split is not None:
        return max_samples_per_split
    return 8 if scope == "smoke" else None


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def maybe_generate_plots(result_dir: str) -> str | None:
    try:
        from analysis.plot_recent_frame_saliency import generate_plots

        generate_plots(result_dir)
        return None
    except Exception as exc:
        return str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="OVO-Bench backward/realtime recent4 saliency analysis for Qwen3-VL")
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_saliency_qwen3vl")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--analysis_scope", choices=["smoke", "full"], default="full")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--similarity_backends", default="siglip")
    parser.add_argument("--siglip_model_name", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--attention_modes", default="first_token,question_prefill")

    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--save_example_matrices", type=int, default=8)
    parser.add_argument("--save_raw_attn_examples", type=int, default=0)
    parser.add_argument("--max_analysis_frames", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    similarity_backends = parse_csv_options(args.similarity_backends)
    attention_modes = parse_csv_options(args.attention_modes)
    supported_similarity = {"siglip"}
    supported_attention = {"first_token", "question_prefill"}
    unsupported_similarity = sorted(set(similarity_backends) - supported_similarity)
    unsupported_attention = sorted(set(attention_modes) - supported_attention)
    if unsupported_similarity:
        raise ValueError(f"Unsupported similarity backends: {unsupported_similarity}")
    if unsupported_attention:
        raise ValueError(f"Unsupported attention modes: {unsupported_attention}")
    if not similarity_backends and not attention_modes:
        raise ValueError("At least one similarity backend or attention mode must be enabled.")

    sample_cap = smoke_cap_or_default(args.analysis_scope, args.max_samples_per_split)

    with open(args.anno_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    backward_anno = [anno for anno in annotations if anno["task"] in BACKWARD_TASKS]
    realtime_anno = [anno for anno in annotations if anno["task"] in REAL_TIME_TASKS]

    random.seed(args.seed)
    random.shuffle(backward_anno)
    random.shuffle(realtime_anno)
    if sample_cap is not None:
        backward_anno = backward_anno[:sample_cap]
        realtime_anno = realtime_anno[:sample_cap]

    result_dir = Path(args.result_dir)
    records_path = result_dir / "records.jsonl"
    examples_dir = result_dir / "examples"
    result_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    existing_records, done_keys = load_jsonl_results(str(records_path))
    saved_examples = sum(1 for record in existing_records if record.get("matrix_example_saved"))
    saved_raw = sum(1 for record in existing_records if record.get("raw_attention_saved"))

    print("\n" + "=" * 60)
    print("OVO-Bench Recent4 Saliency Analysis (Qwen3-VL)")
    print("=" * 60)
    print(f"Backward: {len(backward_anno)}")
    print(f"Realtime: {len(realtime_anno)}")
    print(f"Similarity: {similarity_backends or 'disabled'}")
    print(f"Attention: {attention_modes or 'disabled'}")
    print(f"Scope: {args.analysis_scope}")
    print(f"Result Dir: {result_dir}")
    print("=" * 60 + "\n")

    analyzer = Qwen3Recent4FrameSaliencyAnalyzer(
        model_name=args.model_path,
        device="auto",
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        siglip_model_name=args.siglip_model_name,
    )

    all_records = list(existing_records)

    with records_path.open("a", encoding="utf-8") as handle:
        for split_name, split_annos in (
            ("backward", backward_anno),
            ("realtime", realtime_anno),
        ):
            for anno in tqdm(split_annos, desc=split_name.capitalize()):
                key = make_key(anno["task"], anno["id"])
                if key in done_keys:
                    continue

                video_path = os.path.join(args.chunked_dir, f"{anno['id']}.mp4")
                save_example = saved_examples < int(args.save_example_matrices)
                save_raw = saved_raw < int(args.save_raw_attn_examples)

                try:
                    record, example_payload = analyzer.analyze_sample(
                        video_path=video_path,
                        prompt=build_ovo_prompt(anno["task"], anno),
                        similarity_text=str(anno.get("question", "")).strip(),
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        recent_frames_only=args.recent_frames_only,
                        similarity_backends=similarity_backends,
                        attention_modes=attention_modes,
                        max_analysis_frames=args.max_analysis_frames,
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
                        example_name = slugify(key) or f"example-{len(all_records)}"
                        example_path = examples_dir / f"{example_name}.pt"
                        torch.save(example_payload, example_path)
                        output["example_saved"] = True
                        output["matrix_example_saved"] = bool(save_example)
                        output["raw_attention_saved"] = bool(save_raw)
                        output["example_path"] = str(example_path)
                        if save_example:
                            saved_examples += 1
                        if save_raw:
                            saved_raw += 1
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

                append_record(handle, output)
                all_records.append(output)
                done_keys.add(key)

    config = {
        "generated_at": datetime.now().isoformat(),
        "model_path": args.model_path,
        "anno_path": args.anno_path,
        "chunked_dir": args.chunked_dir,
        "recent_frames_only": args.recent_frames_only,
        "chunk_duration": args.chunk_duration,
        "fps": args.fps,
        "analysis_scope": args.analysis_scope,
        "max_samples_per_split": sample_cap,
        "similarity_backends": similarity_backends,
        "siglip_model_name": args.siglip_model_name,
        "attention_modes": attention_modes,
        "attn_implementation": args.attn_implementation,
        "save_example_matrices": args.save_example_matrices,
        "save_raw_attn_examples": args.save_raw_attn_examples,
        "max_analysis_frames": args.max_analysis_frames,
        "seed": args.seed,
    }
    summary = build_experiment_summary(all_records, config=config)
    plot_error = maybe_generate_plots(str(result_dir))
    if plot_error:
        summary["plot_generation_error"] = plot_error
    save_json(result_dir / "summary.json", summary)

    print("\n" + "=" * 60)
    print(f"Records saved to: {records_path}")
    print(f"Summary saved to: {result_dir / 'summary.json'}")
    if plot_error:
        print(f"Plot generation skipped: {plot_error}")
    else:
        print(f"Plots saved to: {result_dir / 'plots'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
