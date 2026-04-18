#!/bin/bash
# Run the README V-RAG full commands as two sequential GPU queues:
# - Test 4 on GPUs 4,5: TOP_K=4 then TOP_K=8
# - Test 5 on GPUs 6,7: CHUNK_SIZE=4 then CHUNK_SIZE=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p main_experiments/results

wait_for_run() {
    local pid="$1"
    local label="$2"
    local log_path="$3"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] started ${label}"
    echo "  pid=${pid}"
    echo "  log=${log_path}"

    if wait "${pid}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] completed ${label}"
    else
        local status=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] failed ${label} with status ${status}"
        echo "  see log: ${log_path}"
        return "${status}"
    fi
}

run_gpu45_queue() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 4,5 queue started"

    # README lines 535-554
    TOP_K=4
    DECODE_MAX_FRAMES=128
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
    LOG_PATH="./main_experiments/results/nohup_ovo_qwen3vl_vrag_top${TOP_K}_${RUN_TAG}_flash.log"
    CUDA_VISIBLE_DEVICES=4,5 nohup python main_experiments/eval_qwen3vl_ovo_test4.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --anno_path data/ovo_bench/ovo_bench_new.json \
        --chunked_dir data/ovo_bench/chunked_videos \
        --result_dir "main_experiments/results/ovo_qwen3vl_vrag_top${TOP_K}_${RUN_TAG}_flash" \
        --analysis_scope full \
        --recent_frames_only 4 \
        --decode_max_frames "${DECODE_MAX_FRAMES}" \
        --top_k_historical "${TOP_K}" \
        --chunk_duration 1.0 \
        --fps 1.0 \
        --siglip_model_name google/siglip-so400m-patch14-384 \
        --siglip_device auto \
        --save_example_matrices 0 \
        --attn_implementation flash_attention_2 \
        --model_device auto \
        > "${LOG_PATH}" 2>&1 &
    wait_for_run "$!" "test4 TOP_K=${TOP_K} on GPU 4,5" "${LOG_PATH}"

    # README lines 559-578
    TOP_K=8
    DECODE_MAX_FRAMES=128
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
    LOG_PATH="./main_experiments/results/nohup_ovo_qwen3vl_vrag_top${TOP_K}_${RUN_TAG}_flash.log"
    CUDA_VISIBLE_DEVICES=4,5 nohup python main_experiments/eval_qwen3vl_ovo_test4.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --anno_path data/ovo_bench/ovo_bench_new.json \
        --chunked_dir data/ovo_bench/chunked_videos \
        --result_dir "main_experiments/results/ovo_qwen3vl_vrag_top${TOP_K}_${RUN_TAG}_flash" \
        --analysis_scope full \
        --recent_frames_only 4 \
        --decode_max_frames "${DECODE_MAX_FRAMES}" \
        --top_k_historical "${TOP_K}" \
        --chunk_duration 1.0 \
        --fps 1.0 \
        --siglip_model_name google/siglip-so400m-patch14-384 \
        --siglip_device auto \
        --save_example_matrices 0 \
        --attn_implementation flash_attention_2 \
        --model_device auto \
        > "${LOG_PATH}" 2>&1 &
    wait_for_run "$!" "test4 TOP_K=${TOP_K} on GPU 4,5" "${LOG_PATH}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 4,5 queue completed"
}

run_gpu67_queue() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 6,7 queue started"

    # README lines 682-703
    TOP_K=4
    CHUNK_SIZE=4
    DECODE_MAX_FRAMES=128
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
    LOG_PATH="./main_experiments/results/nohup_ovo_qwen3vl_vrag_top${TOP_K}_chunk${CHUNK_SIZE}_${RUN_TAG}.log"
    CUDA_VISIBLE_DEVICES=6,7 nohup python main_experiments/eval_qwen3vl_ovo_test5.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --anno_path data/ovo_bench/ovo_bench_new.json \
        --chunked_dir data/ovo_bench/chunked_videos \
        --result_dir "main_experiments/results/ovo_qwen3vl_vrag_top${TOP_K}_chunk${CHUNK_SIZE}_${RUN_TAG}" \
        --analysis_scope full \
        --recent_frames_only 4 \
        --decode_max_frames "${DECODE_MAX_FRAMES}" \
        --top_k_historical "${TOP_K}" \
        --chunk_size "${CHUNK_SIZE}" \
        --chunk_duration 1.0 \
        --fps 1.0 \
        --siglip_model_name google/siglip-so400m-patch14-384 \
        --siglip_device auto \
        --save_example_matrices 0 \
        --attn_implementation sdpa \
        --model_device auto \
        > "${LOG_PATH}" 2>&1 &
    wait_for_run "$!" "test5 TOP_K=${TOP_K} CHUNK_SIZE=${CHUNK_SIZE} on GPU 6,7" "${LOG_PATH}"

    # README lines 708-729
    TOP_K=4
    CHUNK_SIZE=2
    DECODE_MAX_FRAMES=128
    RUN_TAG=$(date +%Y%m%d_%H%M%S)
    LOG_PATH="./main_experiments/results/nohup_ovo_qwen3vl_vrag_top${TOP_K}_chunk${CHUNK_SIZE}_${RUN_TAG}.log"
    CUDA_VISIBLE_DEVICES=6,7 nohup python main_experiments/eval_qwen3vl_ovo_test5.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --anno_path data/ovo_bench/ovo_bench_new.json \
        --chunked_dir data/ovo_bench/chunked_videos \
        --result_dir "main_experiments/results/ovo_qwen3vl_vrag_top${TOP_K}_chunk${CHUNK_SIZE}_${RUN_TAG}" \
        --analysis_scope full \
        --recent_frames_only 4 \
        --decode_max_frames "${DECODE_MAX_FRAMES}" \
        --top_k_historical "${TOP_K}" \
        --chunk_size "${CHUNK_SIZE}" \
        --chunk_duration 1.0 \
        --fps 1.0 \
        --siglip_model_name google/siglip-so400m-patch14-384 \
        --siglip_device auto \
        --save_example_matrices 0 \
        --attn_implementation sdpa \
        --model_device auto \
        > "${LOG_PATH}" 2>&1 &
    wait_for_run "$!" "test5 TOP_K=${TOP_K} CHUNK_SIZE=${CHUNK_SIZE} on GPU 6,7" "${LOG_PATH}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 6,7 queue completed"
}

run_gpu45_queue &
pid_gpu45=$!

run_gpu67_queue &
pid_gpu67=$!

status=0
wait "${pid_gpu45}" || status=$?
wait "${pid_gpu67}" || status=$?

if [[ "${status}" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] all queues completed"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] one or more queues failed"
fi

exit "${status}"
