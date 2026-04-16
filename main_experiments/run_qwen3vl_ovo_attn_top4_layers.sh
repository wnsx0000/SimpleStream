#!/bin/bash
# Sequential layer sweep for Qwen3-VL OVO-Bench attention top-4 evaluation.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash main_experiments/run_qwen3vl_ovo_attn_top4_layers.sh 0,18,32,35

Runs eval_qwen3vl_ovo_test3.py once per layer in the given order.

Common overrides:
  CUDA_VISIBLE_DEVICES=6,7
  PYTHON_BIN=/path/to/python
  MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
  OVO_ANNO_PATH=data/ovo_bench/ovo_bench_new.json
  OVO_CHUNKED_DIR=data/ovo_bench/chunked_videos
  OVO_RESULT_ROOT=main_experiments/results
  BATCH_TAG=20260416_150000
  MODEL_DEVICE=auto
USAGE
}

if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    if [[ $# -ne 1 ]]; then
        exit 2
    fi
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
ANNO_PATH="${OVO_ANNO_PATH:-data/ovo_bench/ovo_bench_new.json}"
CHUNKED_DIR="${OVO_CHUNKED_DIR:-data/ovo_bench/chunked_videos}"
RESULT_ROOT="${OVO_RESULT_ROOT:-main_experiments/results}"
RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY:-4}"
MAX_ANALYSIS_FRAMES="${MAX_ANALYSIS_FRAMES:-12}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MODEL_DEVICE="${MODEL_DEVICE:-auto}"
CHUNK_DURATION="${CHUNK_DURATION:-1.0}"
FPS="${FPS:-1.0}"
MAX_QA_TOKENS="${MAX_QA_TOKENS:-256}"
ANALYSIS_SCOPE="${ANALYSIS_SCOPE:-full}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-}"
MAX_SAMPLES_PER_SUBSET="${MAX_SAMPLES_PER_SUBSET:-}"
SEED="${SEED:-42}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a LAYERS <<< "$1"
if [[ ${#LAYERS[@]} -eq 0 ]]; then
    echo "No layers were provided." >&2
    usage >&2
    exit 2
fi

for i in "${!LAYERS[@]}"; do
    layer="${LAYERS[$i]//[[:space:]]/}"
    if [[ -z "${layer}" ]]; then
        echo "Empty layer value in input: $1" >&2
        exit 2
    fi
    if [[ ! "${layer}" =~ ^[0-9]+$ ]]; then
        echo "Layer must be a non-negative integer, got: ${layer}" >&2
        exit 2
    fi
    LAYERS[$i]="${layer}"
done

cd "${REPO_ROOT}"
mkdir -p "${RESULT_ROOT}"

echo "Batch tag: ${BATCH_TAG}"
echo "Layers: ${LAYERS[*]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Result root: ${RESULT_ROOT}"

run_layer() {
    local layer="$1"
    local result_dir="${RESULT_ROOT}/ovo_qwen3vl_attn_top4_layer${layer}_${BATCH_TAG}"
    local log_path="${RESULT_ROOT}/nohup_ovo_qwen3vl_attn_top4_layer${layer}_${BATCH_TAG}.log"
    local cmd=(
        "${PYTHON_BIN}" "main_experiments/eval_qwen3vl_ovo_test3.py"
        --model_path "${MODEL_PATH}"
        --anno_path "${ANNO_PATH}"
        --chunked_dir "${CHUNKED_DIR}"
        --result_dir "${result_dir}"
        --recent_frames_only "${RECENT_FRAMES_ONLY}"
        --max_analysis_frames "${MAX_ANALYSIS_FRAMES}"
        --layer_number "${layer}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
        --model_device "${MODEL_DEVICE}"
        --chunk_duration "${CHUNK_DURATION}"
        --fps "${FPS}"
        --max_qa_tokens "${MAX_QA_TOKENS}"
        --analysis_scope "${ANALYSIS_SCOPE}"
        --seed "${SEED}"
    )

    if [[ -n "${MAX_SAMPLES_PER_SPLIT}" ]]; then
        cmd+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")
    fi
    if [[ -n "${MAX_SAMPLES_PER_SUBSET}" ]]; then
        cmd+=(--max_samples_per_subset "${MAX_SAMPLES_PER_SUBSET}")
    fi

    echo
    echo "============================================================"
    echo "Starting layer ${layer}"
    echo "Result dir: ${result_dir}"
    echo "Log: ${log_path}"
    echo "============================================================"

    if HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
        TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
        HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
        "${cmd[@]}" > "${log_path}" 2>&1; then
        echo "Completed layer ${layer}"
    else
        local exit_code=$?
        echo "Layer ${layer} failed with exit code ${exit_code}. See ${log_path}" >&2
        return "${exit_code}"
    fi
}

for layer in "${LAYERS[@]}"; do
    run_layer "${layer}"
done

echo
echo "Completed all layers for batch ${BATCH_TAG}"
