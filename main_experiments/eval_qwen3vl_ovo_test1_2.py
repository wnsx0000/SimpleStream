"""
OVO-Bench recent4 saliency analysis using Qwen3-VL attention only.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import (  # noqa: E402
    build_qwen3_attention_frame_saliency_analyzer,
    parse_csv_options,
)
from main_experiments.eval_qwen3vl_ovo_saliency_common import (  # noqa: E402
    SaliencyExperimentConfig,
    add_common_saliency_args,
    run_saliency_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OVO-Bench backward/realtime recent4 Qwen3-VL attention saliency analysis",
    )
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    add_common_saliency_args(
        parser,
        default_result_dir="results/ovo_saliency_qwen3vl_attention",
        include_save_raw_attn_examples=True,
    )
    parser.add_argument("--attention_modes", default="question_prefill")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    attention_modes = parse_csv_options(args.attention_modes)
    supported_attention = {"question_prefill"}
    unsupported_attention = sorted(set(attention_modes) - supported_attention)
    if unsupported_attention:
        raise ValueError(f"Unsupported attention modes: {unsupported_attention}")
    if not attention_modes:
        raise ValueError("At least one attention mode must be enabled.")

    analyzer = build_qwen3_attention_frame_saliency_analyzer(
        model_name=args.model_path,
        device="auto",
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    )
    config = SaliencyExperimentConfig(
        run_label="OVO-Bench Recent4 Saliency Analysis (Qwen3-VL Attention)",
        anno_path=args.anno_path,
        chunked_dir=args.chunked_dir,
        result_dir=args.result_dir,
        recent_frames_only=args.recent_frames_only,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        analysis_scope=args.analysis_scope,
        max_samples_per_split=args.max_samples_per_split,
        max_samples_per_subset=args.max_samples_per_subset,
        similarity_backends=[],
        attention_modes=attention_modes,
        save_example_matrices=args.save_example_matrices,
        save_raw_attn_examples=args.save_raw_attn_examples,
        max_analysis_frames=args.max_analysis_frames,
        seed=args.seed,
        extra_summary_config={
            "script": "eval_qwen3vl_ovo_test1_2.py",
            "mode": "qwen3_attention",
            "model_path": args.model_path,
            "attn_implementation": args.attn_implementation,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    run_saliency_experiment(analyzer=analyzer, config=config)


if __name__ == "__main__":
    main()
