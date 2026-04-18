#!/bin/bash
# Wait for GPUs 4,5,6,7 to become idle, run Test 3 first, then run Test 2
# and Test 5 queues. This script never terminates existing GPU processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
ANNO_PATH="${ANNO_PATH:-data/ovo_bench/ovo_bench_new.json}"
CHUNKED_DIR="${CHUNKED_DIR:-data/ovo_bench/chunked_videos}"
RESULT_ROOT="${RESULT_ROOT:-main_experiments/results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"

ANALYSIS_SCOPE="${ANALYSIS_SCOPE:-full}"
RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY:-4}"
DECODE_MAX_FRAMES="${DECODE_MAX_FRAMES:-128}"
CHUNK_DURATION="${CHUNK_DURATION:-1.0}"
FPS="${FPS:-1.0}"
MAX_QA_TOKENS="${MAX_QA_TOKENS:-256}"
SEED="${SEED:-42}"

SIGLIP_MODEL_NAME="${SIGLIP_MODEL_NAME:-google/siglip-so400m-patch14-384}"
SIGLIP_DEVICE="${SIGLIP_DEVICE:-auto}"

TEST3_LAYERS="${TEST3_LAYERS:-32,35,0}"
TEST3_MODEL_DEVICE="${TEST3_MODEL_DEVICE:-auto}"
TEST3_MAX_ANALYSIS_FRAMES="${TEST3_MAX_ANALYSIS_FRAMES:-12}"
TEST3_ATTN_IMPLEMENTATION="${TEST3_ATTN_IMPLEMENTATION:-eager}"

TEST2_ATTN_IMPLEMENTATION="${TEST2_ATTN_IMPLEMENTATION:-flash_attention_2}"
TEST2_MODEL_DEVICE="${TEST2_MODEL_DEVICE:-auto}"

TEST5_ATTN_IMPLEMENTATION="${TEST5_ATTN_IMPLEMENTATION:-sdpa}"
TEST5_MODEL_DEVICE="${TEST5_MODEL_DEVICE:-auto}"
TEST5_SAVE_EXAMPLE_MATRICES="${TEST5_SAVE_EXAMPLE_MATRICES:-0}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

format_duration() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
        "$((total_seconds / 3600))" \
        "$(((total_seconds % 3600) / 60))" \
        "$((total_seconds % 60))"
}

log() {
    echo "[$(timestamp)] $*"
}

csv_to_array() {
    local csv="$1"
    local -n out_array="$2"

    IFS=',' read -r -a out_array <<< "${csv}"
    for i in "${!out_array[@]}"; do
        out_array[$i]="${out_array[$i]//[[:space:]]/}"
    done
}

validate_test3_layers() {
    local layers=()
    local layer

    csv_to_array "${TEST3_LAYERS}" layers
    if [[ ${#layers[@]} -eq 0 ]]; then
        echo "TEST3_LAYERS must not be empty." >&2
        exit 2
    fi

    for layer in "${layers[@]}"; do
        if [[ -z "${layer}" ]]; then
            echo "TEST3_LAYERS contains an empty layer: ${TEST3_LAYERS}" >&2
            exit 2
        fi
        if [[ ! "${layer}" =~ ^[0-9]+$ ]]; then
            echo "TEST3_LAYERS must contain only non-negative integers, got: ${layer}" >&2
            exit 2
        fi
        if [[ "${layer}" == "18" ]]; then
            echo "Layer 18 is intentionally excluded. TEST3_LAYERS=${TEST3_LAYERS}" >&2
            exit 2
        fi
    done
}

list_busy_gpu_processes() {
    local gpu_csv="$1"
    local gpus=()
    local gpu
    local output
    local line

    csv_to_array "${gpu_csv}" gpus
    for gpu in "${gpus[@]}"; do
        if [[ -z "${gpu}" ]]; then
            continue
        fi

        if ! output="$(
            nvidia-smi -i "${gpu}" \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>&1
        )"; then
            echo "nvidia-smi query failed for GPU ${gpu}: ${output}" >&2
            return 1
        fi
        if [[ -z "${output}" ]]; then
            continue
        fi

        while IFS= read -r line; do
            if [[ -n "${line}" ]]; then
                printf 'GPU %s: %s\n' "${gpu}" "${line}"
            fi
        done <<< "${output}"
    done
}

wait_for_gpus_free() {
    local gpu_csv="$1"
    local label="$2"
    local start_ts
    local now_ts
    local elapsed
    local busy

    log "checking GPU availability for ${label}: GPUs ${gpu_csv}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        if ! busy="$(list_busy_gpu_processes "${gpu_csv}")"; then
            log "failed to query GPU processes for ${label}"
            return 1
        fi
        if [[ -n "${busy}" ]]; then
            log "DRY_RUN: these processes would be waited on before ${label}:"
            while IFS= read -r line; do
                echo "  ${line}"
            done <<< "${busy}"
        else
            log "DRY_RUN: GPUs ${gpu_csv} are currently free for ${label}"
        fi
        return 0
    fi

    start_ts="$(date +%s)"
    while true; do
        if ! busy="$(list_busy_gpu_processes "${gpu_csv}")"; then
            log "failed to query GPU processes for ${label}"
            return 1
        fi
        if [[ -z "${busy}" ]]; then
            elapsed="$(format_duration "$(("$(date +%s)" - start_ts))")"
            log "GPUs ${gpu_csv} are free for ${label}; waited ${elapsed}"
            return 0
        fi

        now_ts="$(date +%s)"
        elapsed="$(format_duration "$((now_ts - start_ts))")"
        log "waiting for GPUs ${gpu_csv} before ${label}; elapsed ${elapsed}; busy processes:"
        while IFS= read -r line; do
            echo "  ${line}"
        done <<< "${busy}"
        sleep "${POLL_SECONDS}"
    done
}

print_command() {
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
}

require_summary() {
    local label="$1"
    local result_dir="$2"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    if [[ ! -f "${result_dir}/summary.json" ]]; then
        log "failed ${label}: missing ${result_dir}/summary.json"
        return 1
    fi
}

run_logged_command() {
    local label="$1"
    local log_path="$2"
    local result_dir="$3"
    shift 3

    local start_ts
    local end_ts
    local elapsed
    local pid

    log "started ${label}"
    echo "  log=${log_path}"
    if [[ -n "${result_dir}" ]]; then
        echo "  result_dir=${result_dir}"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "$@"
        log "DRY_RUN: completed ${label}"
        return 0
    fi

    mkdir -p "$(dirname "${log_path}")"
    start_ts="$(date +%s)"
    "$@" > "${log_path}" 2>&1 &
    pid="$!"
    echo "  pid=${pid}"

    if wait "${pid}"; then
        end_ts="$(date +%s)"
        elapsed="$(format_duration "$((end_ts - start_ts))")"
        if ! require_summary "${label}" "${result_dir}"; then
            return 1
        fi
        log "completed ${label}; elapsed ${elapsed}"
    else
        local status=$?
        end_ts="$(date +%s)"
        elapsed="$(format_duration "$((end_ts - start_ts))")"
        log "failed ${label} with status ${status}; elapsed ${elapsed}"
        echo "  see log: ${log_path}"
        return "${status}"
    fi
}

verify_test3_summaries() {
    local layers=()
    local layer
    local result_dir

    csv_to_array "${TEST3_LAYERS}" layers
    for layer in "${layers[@]}"; do
        result_dir="${RESULT_ROOT}/ovo_qwen3vl_attn_top4_layer${layer}_${RUN_TAG}"
        if ! require_summary "test3 layer ${layer}" "${result_dir}"; then
            return 1
        fi
    done
}

run_test3() {
    local log_path="${RESULT_ROOT}/nohup_ovo_qwen3vl_attn_top4_layers_${RUN_TAG}.log"
    local start_ts
    local end_ts
    local elapsed

    wait_for_gpus_free "4,5,6,7" "test3 layers ${TEST3_LAYERS}"

    log "started test3 layers ${TEST3_LAYERS} on GPUs 4,5,6,7"
    echo "  log=${log_path}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command \
            env CUDA_VISIBLE_DEVICES=4,5,6,7 BATCH_TAG="${RUN_TAG}" \
            PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" \
            OVO_ANNO_PATH="${ANNO_PATH}" OVO_CHUNKED_DIR="${CHUNKED_DIR}" \
            OVO_RESULT_ROOT="${RESULT_ROOT}" RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY}" \
            MAX_ANALYSIS_FRAMES="${TEST3_MAX_ANALYSIS_FRAMES}" \
            ATTN_IMPLEMENTATION="${TEST3_ATTN_IMPLEMENTATION}" \
            MODEL_DEVICE="${TEST3_MODEL_DEVICE}" CHUNK_DURATION="${CHUNK_DURATION}" \
            FPS="${FPS}" MAX_QA_TOKENS="${MAX_QA_TOKENS}" \
            ANALYSIS_SCOPE="${ANALYSIS_SCOPE}" SEED="${SEED}" \
            bash main_experiments/run_qwen3vl_ovo_attn_top4_layers.sh "${TEST3_LAYERS}"
        log "DRY_RUN: completed test3 layers ${TEST3_LAYERS}"
        return 0
    fi

    start_ts="$(date +%s)"
    if env CUDA_VISIBLE_DEVICES=4,5,6,7 BATCH_TAG="${RUN_TAG}" \
        PYTHON_BIN="${PYTHON_BIN}" MODEL_PATH="${MODEL_PATH}" \
        OVO_ANNO_PATH="${ANNO_PATH}" OVO_CHUNKED_DIR="${CHUNKED_DIR}" \
        OVO_RESULT_ROOT="${RESULT_ROOT}" RECENT_FRAMES_ONLY="${RECENT_FRAMES_ONLY}" \
        MAX_ANALYSIS_FRAMES="${TEST3_MAX_ANALYSIS_FRAMES}" \
        ATTN_IMPLEMENTATION="${TEST3_ATTN_IMPLEMENTATION}" \
        MODEL_DEVICE="${TEST3_MODEL_DEVICE}" CHUNK_DURATION="${CHUNK_DURATION}" \
        FPS="${FPS}" MAX_QA_TOKENS="${MAX_QA_TOKENS}" \
        ANALYSIS_SCOPE="${ANALYSIS_SCOPE}" SEED="${SEED}" \
        bash main_experiments/run_qwen3vl_ovo_attn_top4_layers.sh "${TEST3_LAYERS}" \
        > "${log_path}" 2>&1; then
        verify_test3_summaries
        end_ts="$(date +%s)"
        elapsed="$(format_duration "$((end_ts - start_ts))")"
        log "completed test3 layers ${TEST3_LAYERS}; elapsed ${elapsed}"
    else
        local status=$?
        end_ts="$(date +%s)"
        elapsed="$(format_duration "$((end_ts - start_ts))")"
        log "failed test3 layers ${TEST3_LAYERS} with status ${status}; elapsed ${elapsed}"
        echo "  see log: ${log_path}"
        return "${status}"
    fi
}

run_test2_gpu45() {
    local result_dir="${RESULT_ROOT}/ovo_qwen3vl_siglip_top4_all_decoded_cap${DECODE_MAX_FRAMES}_${RUN_TAG}"
    local log_path="${RESULT_ROOT}/nohup_ovo_qwen3vl_siglip_top4_all_decoded_cap${DECODE_MAX_FRAMES}_${RUN_TAG}.log"
    local cmd=(
        env CUDA_VISIBLE_DEVICES=4,5
        "${PYTHON_BIN}" main_experiments/eval_qwen3vl_ovo_test2.py
        --model_path "${MODEL_PATH}"
        --anno_path "${ANNO_PATH}"
        --chunked_dir "${CHUNKED_DIR}"
        --result_dir "${result_dir}"
        --analysis_scope "${ANALYSIS_SCOPE}"
        --recent_frames_only "${RECENT_FRAMES_ONLY}"
        --decode_max_frames "${DECODE_MAX_FRAMES}"
        --chunk_duration "${CHUNK_DURATION}"
        --fps "${FPS}"
        --max_qa_tokens "${MAX_QA_TOKENS}"
        --siglip_model_name "${SIGLIP_MODEL_NAME}"
        --siglip_device "${SIGLIP_DEVICE}"
        --attn_implementation "${TEST2_ATTN_IMPLEMENTATION}"
        --seed "${SEED}"
    )

    if [[ -n "${TEST2_MODEL_DEVICE}" ]]; then
        cmd+=(--model_device "${TEST2_MODEL_DEVICE}")
    fi

    wait_for_gpus_free "4,5" "test2 on GPUs 4,5"
    run_logged_command "test2 SigLIP top4 cap${DECODE_MAX_FRAMES} on GPU 4,5" "${log_path}" "${result_dir}" "${cmd[@]}"
}

run_test5() {
    local gpu_csv="$1"
    local top_k="$2"
    local chunk_size="$3"
    local result_dir="${RESULT_ROOT}/ovo_qwen3vl_vrag_top${top_k}_chunk${chunk_size}_${RUN_TAG}"
    local log_path="${RESULT_ROOT}/nohup_ovo_qwen3vl_vrag_top${top_k}_chunk${chunk_size}_${RUN_TAG}.log"

    wait_for_gpus_free "${gpu_csv}" "test5 TOP_K=${top_k} CHUNK_SIZE=${chunk_size} on GPUs ${gpu_csv}"
    run_logged_command \
        "test5 TOP_K=${top_k} CHUNK_SIZE=${chunk_size} on GPU ${gpu_csv}" \
        "${log_path}" \
        "${result_dir}" \
        env CUDA_VISIBLE_DEVICES="${gpu_csv}" \
        "${PYTHON_BIN}" main_experiments/eval_qwen3vl_ovo_test5.py \
        --model_path "${MODEL_PATH}" \
        --anno_path "${ANNO_PATH}" \
        --chunked_dir "${CHUNKED_DIR}" \
        --result_dir "${result_dir}" \
        --analysis_scope "${ANALYSIS_SCOPE}" \
        --recent_frames_only "${RECENT_FRAMES_ONLY}" \
        --decode_max_frames "${DECODE_MAX_FRAMES}" \
        --top_k_historical "${top_k}" \
        --chunk_size "${chunk_size}" \
        --chunk_duration "${CHUNK_DURATION}" \
        --fps "${FPS}" \
        --max_qa_tokens "${MAX_QA_TOKENS}" \
        --siglip_model_name "${SIGLIP_MODEL_NAME}" \
        --siglip_device "${SIGLIP_DEVICE}" \
        --save_example_matrices "${TEST5_SAVE_EXAMPLE_MATRICES}" \
        --attn_implementation "${TEST5_ATTN_IMPLEMENTATION}" \
        --model_device "${TEST5_MODEL_DEVICE}" \
        --seed "${SEED}"
}

run_gpu45_queue() {
    log "GPU 4,5 queue started"
    run_test2_gpu45
    run_test5 "4,5" 4 1
    log "GPU 4,5 queue completed"
}

run_gpu67_queue() {
    log "GPU 6,7 queue started"
    run_test5 "6,7" 8 8
    run_test5 "6,7" 8 2
    log "GPU 6,7 queue completed"
}

run_test2_test5_queues() {
    local pid_gpu45
    local pid_gpu67
    local status=0

    wait_for_gpus_free "4,5,6,7" "test2/test5 parallel queues"

    run_gpu45_queue &
    pid_gpu45="$!"
    run_gpu67_queue &
    pid_gpu67="$!"

    if wait "${pid_gpu45}"; then
        log "GPU 4,5 queue exit status 0"
    else
        status="$?"
        log "GPU 4,5 queue failed with status ${status}"
    fi

    if wait "${pid_gpu67}"; then
        log "GPU 6,7 queue exit status 0"
    else
        local status_gpu67=$?
        log "GPU 6,7 queue failed with status ${status_gpu67}"
        if [[ "${status}" -eq 0 ]]; then
            status="${status_gpu67}"
        fi
    fi

    if [[ "${status}" -ne 0 ]]; then
        log "one or more test2/test5 queues failed"
        return "${status}"
    fi

    log "all test2/test5 queues completed"
}

main() {
    local script_start
    local script_end
    local elapsed

    validate_test3_layers

    cd "${REPO_ROOT}"
    mkdir -p "${RESULT_ROOT}"

    script_start="$(date +%s)"
    log "run tag: ${RUN_TAG}"
    log "result root: ${RESULT_ROOT}"
    log "dry run: ${DRY_RUN}"
    log "no existing GPU process will be terminated by this script"

    wait_for_gpus_free "4,5,6,7" "initial wait"
    run_test3
    wait_for_gpus_free "4,5,6,7" "post-test3 cleanup"
    run_test2_test5_queues

    script_end="$(date +%s)"
    elapsed="$(format_duration "$((script_end - script_start))")"
    log "all requested work completed; total elapsed ${elapsed}"
}

main "$@"
