#!/bin/bash
# Sequential full-subset saliency runs for Qwen3-VL:
# 1) question_prefill attention
# 2) SigLIP similarity
# 3) first_token attention

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
SIGLIP_MODEL_NAME="${SIGLIP_MODEL_NAME:-google/siglip-so400m-patch14-384}"
ANNO_PATH="${OVO_ANNO_PATH:-data/ovo_bench/ovo_bench_new.json}"
CHUNKED_DIR="${OVO_CHUNKED_DIR:-data/ovo_bench/chunked_videos}"
RESULT_ROOT="${OVO_RESULT_ROOT:-main_experiments/results}"
RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY:-4}"
CHUNK_DURATION="${CHUNK_DURATION:-1.0}"
FPS="${FPS:-1.0}"
MAX_ANALYSIS_FRAMES="${MAX_ANALYSIS_FRAMES:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
ANALYSIS_SCOPE="${ANALYSIS_SCOPE:-full}"
MAX_SAMPLES_PER_SUBSET="${MAX_SAMPLES_PER_SUBSET:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"

run_case() {
    local case_name="$1"
    local script_path="$2"
    local attention_modes="${3:-}"
    local result_dir="${RESULT_ROOT}/ovo_qwen3vl_frame_saliency_full_${case_name}_${BATCH_TAG}"
    local cmd=(
        "${PYTHON_BIN}" "${script_path}"
        --anno_path "${ANNO_PATH}"
        --chunked_dir "${CHUNKED_DIR}"
        --result_dir "${result_dir}"
        --analysis_scope "${ANALYSIS_SCOPE}"
        --recent_frames_only "${RECENT_FRAMES_ONLY}"
        --chunk_duration "${CHUNK_DURATION}"
        --fps "${FPS}"
        --max_analysis_frames "${MAX_ANALYSIS_FRAMES}"
    )

    if [[ "${script_path}" == "main_experiments/eval_qwen3vl_ovo_test1_1.py" ]]; then
        cmd+=(--siglip_model_name "${SIGLIP_MODEL_NAME}")
    else
        cmd+=(
            --model_path "${MODEL_PATH}"
            --attention_modes "${attention_modes}"
            --attn_implementation "${ATTN_IMPLEMENTATION}"
            --max_new_tokens "${MAX_NEW_TOKENS}"
        )
    fi

    if [[ -n "${MAX_SAMPLES_PER_SUBSET}" ]]; then
        cmd+=(--max_samples_per_subset "${MAX_SAMPLES_PER_SUBSET}")
    fi

    echo
    echo "============================================================"
    echo "Starting ${case_name}"
    echo "Result dir: ${result_dir}"
    echo "Script: ${script_path}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "Analysis scope=${ANALYSIS_SCOPE}"
    if [[ -n "${MAX_SAMPLES_PER_SUBSET}" ]]; then
        echo "Max samples per subset=${MAX_SAMPLES_PER_SUBSET}"
    fi
    echo "============================================================"

    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
    "${cmd[@]}"
}

run_case "question_prefill" "main_experiments/eval_qwen3vl_ovo_test1_2.py" "question_prefill"
run_case "siglip" "main_experiments/eval_qwen3vl_ovo_test1_1.py"
run_case "first_token" "main_experiments/eval_qwen3vl_ovo_test1_2.py" "first_token"

echo
echo "Completed batch ${BATCH_TAG}"
