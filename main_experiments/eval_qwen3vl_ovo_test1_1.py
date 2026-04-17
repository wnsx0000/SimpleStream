"""
OVO-Bench recent4 saliency analysis using SigLIP frame-question similarity only.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.frame_saliency_qwen3 import build_siglip_frame_saliency_analyzer  # noqa: E402
from main_experiments.eval_qwen3vl_ovo_saliency_common import (  # noqa: E402
    SaliencyExperimentConfig,
    add_common_saliency_args,
    run_saliency_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OVO-Bench backward/realtime recent4 SigLIP saliency analysis",
    )
    add_common_saliency_args(
        parser,
        default_result_dir="results/ovo_saliency_qwen3vl_siglip",
        include_save_raw_attn_examples=False,
    )
    parser.add_argument("--siglip_model_name", default="google/siglip-so400m-patch14-384")
    parser.add_argument(
        "--siglip_device",
        default="auto",
        help="SigLIP device. 'auto' selects the visible CUDA GPU with the most free memory; use cuda:N or cpu to override.",
    )
    args = parser.parse_args()

    analyzer = build_siglip_frame_saliency_analyzer(
        siglip_model_name=args.siglip_model_name,
        device=args.siglip_device,
    )
    config = SaliencyExperimentConfig(
        run_label="OVO-Bench Recent4 Saliency Analysis (SigLIP)",
        anno_path=args.anno_path,
        chunked_dir=args.chunked_dir,
        result_dir=args.result_dir,
        recent_frames_only=args.recent_frames_only,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        analysis_scope=args.analysis_scope,
        max_samples_per_split=args.max_samples_per_split,
        max_samples_per_subset=args.max_samples_per_subset,
        similarity_backends=["siglip"],
        attention_modes=[],
        save_example_matrices=args.save_example_matrices,
        save_raw_attn_examples=0,
        max_analysis_frames=args.max_analysis_frames,
        seed=args.seed,
        extra_summary_config={
            "script": "eval_qwen3vl_ovo_test1_1.py",
            "mode": "siglip_similarity",
            "siglip_model_name": args.siglip_model_name,
            "siglip_device": args.siglip_device,
        },
    )
    run_saliency_experiment(analyzer=analyzer, config=config)


if __name__ == "__main__":
    main()
