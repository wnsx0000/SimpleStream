<div align="center">

<img src="assets/logo.svg" width="75%">

### A Simple Baseline for Streaming Video Understanding

<p>
<a href="https://arxiv.org/abs/2604.02317"><img src="https://img.shields.io/badge/arXiv-2604.02317-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/papers/2604.02317"><img src="https://img.shields.io/badge/🤗-Paper_Page-yellow" alt="HF Paper"></a>
<a href="https://simple-stream.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
</p>

<img src="assets/teaser.png" width="90%">

A sliding-window baseline that feeds only the most recent ***N*** frames to an off-the-shelf VLM **matches or surpasses** published streaming video understanding models. No memory bank, no retrieval, no compression.

</div>

## 🚀 News

📄 **[2026/04]** SimpleStream paper is released.

💻 **[2026/04]** Code and evaluation scripts are open-sourced.

## ✨ Highlights

- **Simple yet strong.** With only 4 recent frames, SimpleStream reaches **67.7%** on OVO-Bench and **80.59%** on StreamingBench, surpassing all published streaming methods.
- **Perception-memory trade-off.** Adding historical context improves recall but consistently degrades real-time perception, which dominates aggregate scores.
- **Training-free.** SimpleStream uses off-the-shelf VLMs (Qwen2.5-VL, Qwen3-VL) with zero fine-tuning.

## 📊 Main Results

<p align="center">
<img src="assets/main_results.png" width="90%">
</p>

## 🛠️ Getting Started

### Environment Setup

```bash
conda create -n SimpleStream python=3.10 -y
conda activate SimpleStream
pip install -r requirements.txt

# for my environment
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Optional: faster attention backend
pip install flash-attn --no-build-isolation
```

Qwen3-VL experiments require `transformers>=4.57.0` because the
`transformers.models.qwen3_vl` module is not available in older releases such
as `4.45.0`.

### Models

Downloaded automatically from HuggingFace on first run:
- `Qwen/Qwen3-VL-8B-Instruct` (primary)
- `Qwen/Qwen2.5-VL-7B-Instruct` (cross-validation)

### Data Preparation

- **OVO-Bench**: Download from [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench). Place annotations at `data/ovo_bench/ovo_bench_new.json` and chunked videos at `data/ovo_bench/chunked_videos/`.
- **StreamingBench**: Download from [StreamingBench](https://github.com/Infini-AI-Lab/StreamingBench). Place questions at `data/streamingbench/questions_real.json` and videos at `data/streamingbench/videos/`.

## 🧪 Experiments

<details>
<summary><b>Qwen3-VL on OVO-Bench</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --num_processes=2 \
    main_experiments/eval_qwen3vl_ovo.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_recent8 \
    --recent_frames_only 8 \
    --chunk_duration 1.0 \
    --fps 1.0
```

Or use the convenience launcher for 4-GPU:
```bash
bash main_experiments/run_qwen3vl_ovo_4gpu.sh
```
</details>

<details>
<summary><b>Qwen2.5-VL on OVO-Bench</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 \
    main_experiments/eval_qwen25vl_ovo.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen25vl_recent8 \
    --recent_frames_only 8 \
    --chunk_duration 1.0 \
    --fps 1.0
```
</details>

<details>
<summary><b>Qwen3-VL on OVO-Bench with no acceleration (auto device)</b></summary>

``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python main_experiments/eval_qwen3vl_ovo.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_recent8_auto \
    --recent_frames_only 8 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --model_device auto
```
</details>

<details>
<summary><b>StreamingBench</b></summary>

`--top-k 0` disables retrieval and keeps only the most recent chunks.

```bash
CUDA_VISIBLE_DEVICES=0 python main_experiments/eval_streamingbench.py \
    --anno-path data/streamingbench/questions_real.json \
    --video-dir data/streamingbench/videos \
    --top-k 0 \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --output-dir main_experiments/results/streamingbench_recent4
```
</details>

<details>
<summary><b>Efficiency Benchmark</b></summary>

Measures TTFT, throughput, and memory from a user-provided source video.

```bash
CUDA_VISIBLE_DEVICES=0 python efficiency/eval_efficiency.py \
    --source-video /path/to/your/source_video.mp4 \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --chunk-size 8 \
    --recent-frames 4
```
</details>

<details>
<summary><b>Scoring</b></summary>

```bash
python scoring/score_ovo_bench.py \
    --result_path main_experiments/results/ovo_qwen3vl_recent8/qwen3vl_results_*.json
```
</details>

## Additional Experiments


<details>
<summary><b>main results directory</b></summary>

- qwen3-vl-8B reproducing: ovo_qwen3vl_recent4
- test1-1 (SigLIP cosine similarity mean percentile, use only 12 frames including recent 4 frames): ovo_qwen3vl_siglip_subset20_20260415_151005
- test1-1 (SigLIP cosine similarity mean percentile, all decoded frames cap 768): ovo_qwen3vl_siglip_subset20_all_frames_20260417_114820
- test1-2 (layer-wise attention score mean percentile, attention heatmap): ovo_qwen3vl_attention_subset20_20260416_192329
- test2 (SigLIP top-4 frame inference from uniformly sampled 12 frames): ovo_qwen3vl_siglip_top4_all_20260415_205218_uniform
- test2 (SigLIP top-4 frame inference from uniformly sampled 12 frames including recent 4 frames): ovo_qwen3vl_siglip_top4_all_20260416_141554_always_recent4
- test2 (SigLIP top-4 frame inference from all decoded frames, qwen cap 768): 

</details>

<details>
<summary><b>Qwen3-VL on OVO-Bench</b></summary>

```bash
CUDA_VISIBLE_DEVICES=6,7 nohup python main_experiments/eval_qwen3vl_ovo.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_recent4 \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --model_device auto \
    --fps 1.0 \
    > ./main_experiments/results/nohup_ovo_qwen3vl_recent4_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
</details>

<details>
<summary><b>Test 1</b></summary>

Measures whether the `recent4` frames selected by SimpleStream are also salient 
among all sampled frames in the same window for the OVO-Bench backward and
realtime splits. The script computes:
- SigLIP-SO400M frame-question similarity
- Layer-wise Qwen3 attention scores for `question_prefill`

The HLD backward-tracing subset is excluded from Test 1 measurement, summaries,
averages, and plots.

Outputs are saved under `records.jsonl`, `summary.json`, and `examples/`.
The top-level fields in `summary.json` remain pooled across the analyzed records,
while split-specific summaries are stored under `summary["splits"]["backward"]`
and `summary["splits"]["realtime"]`. Each split section also includes
`task_mean_metrics`, which averages the task subset summaries within that split
with equal task weight. Metric summaries now include both `*_mean` and `*_std`
fields. Plot generation is run separately with
`python analysis/plot_recent_frame_saliency.py --result-dir <result_dir>`.
The pooled `plots/` directory contains the question-prefill layer-wise mean and
std line plots, a task/split/total mean-percentile bar plot, saved-example
heatmaps, and saved-example average attention maps.
Example exports use task/subset caps: `--save_example_matrices 3` means up to
3 saved examples per OVO task such as `ASI`, `EPM`, or `STU`, not 3 total.
When `question_prefill` example exports are enabled, each saved example also
produces `question_prefill_frame_frame_maps.png` and
`question_prefill_question_frame_maps.png` under `plots/examples/<example>/`.
The pooled `plots/` directory additionally includes
`question_prefill_frame_frame_maps_average.png` and
`question_prefill_question_frame_maps_average.png`, averaged over the saved
example subset only.
The SigLIP similarity test (test1-1) decodes the full video at `--fps 1.0`
with the same `qwen_vl_utils` sampling policy used by Test 2, capped at
`--max_analysis_frames` frames (default 768). Videos longer than this cap are
uniformly sampled by `qwen_vl_utils`, with the recent 4 frames always
included. SigLIP similarity is computed over all decoded frames.
The attention saliency test (test1-2) still uses `--max_analysis_frames` to
subsample frames via the `uniform_with_recent_anchor` policy.

The combined runner `main_experiments/eval_qwen3vl_ovo_test1.py` is deprecated.
Run SigLIP similarity and attention saliency with separate entrypoints.

<!-- CUDA_LAUNCH_BLOCKING=1  -->

Use `--max_samples_per_subset 50` to randomly sample up to 50 examples from
each OVO subset/task independently (for example `EPM`, `STU`, `OCR`) within the
backward/realtime splits. When `--max_samples_per_subset` is set, it overrides
the default smoke split cap.

question_prefill test.
For Qwen3-VL-8B, this run saves 12 decoder layers:
`0, 9, 18, 26, 28, 29, 30, 31, 32, 33, 34, 35`. These are the union of 5
uniform anchor layers (`0, 9, 18, 26, 35`) and the final 8-layer tail window
(`28` through `35`).
Saved example plots include both frame-to-frame and question-to-frame pooled
attention maps, plus averages over the saved example subset.

attention scoring test.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main_experiments/eval_qwen3vl_ovo_test1_2.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_attention_subset20_$(date +%Y%m%d_%H%M%S) \
    --analysis_scope full \
    --max_samples_per_subset 20 \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --max_analysis_frames 12 \
    --attention_modes question_prefill \
    --attn_implementation eager \
    > ./main_experiments/results/nohup_ovo_qwen3vl_attention_subset20_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Generate plots later from a saved result directory.

```bash
python analysis/plot_recent_frame_saliency.py \
    --result-dir main_experiments/results/ovo_qwen3vl_attention_subset20_20260416_192329
```

siglip similarity test.
Decodes all frames at fps 1.0 (capped at 768 by default) and computes SigLIP
cosine similarity against the question for every decoded frame.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main_experiments/eval_qwen3vl_ovo_test1_1.py \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_siglip_subset20_all_frames_$(date +%Y%m%d_%H%M%S) \
    --analysis_scope full \
    --max_samples_per_subset 20 \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --siglip_model_name google/siglip-so400m-patch14-384 \
    > ./main_experiments/results/nohup_ovo_qwen3vl_siglip_subset20_all_frames_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Generate plots from a saved SigLIP similarity result directory.

```bash
python analysis/plot_siglip_similarity.py \
    --result-dir main_experiments/results/ovo_qwen3vl_siglip_subset20_20260415_151005
```
</details>

<details>
<summary><b>Test 2</b></summary>

Runs OVO-Bench backward/realtime evaluation with SigLIP-guided frame selection.
The HLD backward-tracing subset is excluded.
For each sample, the script decodes the full video at `--fps 1.0` with the
same `qwen_vl_utils` sampling policy used by the Qwen-VL pipeline. Decoding is
explicitly capped at 768 frames; videos longer than this are uniformly sampled
across the full time range by `qwen_vl_utils`. The script computes SigLIP
cosine similarity between every decoded frame and the question, selects the
top-4 frames, then reorders those four frames temporally before Qwen3-VL
inference.

Outputs are saved under `results_incremental.jsonl`, `summary.json`, and
`qwen3vl_siglip_top4_results_*.json`. `summary.json` reports subset/task
accuracy, split-level official averages and pooled accuracy, plus both
`Official Total Avg.` and `Pooled Overall Acc.` for the full run.

Use `--max_samples_per_subset 50` to sample up to 50 examples independently
from each OVO subset/task within those splits. `--max_analysis_frames` and
`--max_frames` are accepted only as deprecated no-op arguments for backward
compatibility; Test 2 always scores all decoded frames after the qwen 768-frame
decode cap.

siglip top-4 full-candidate test.

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=6,7 nohup python main_experiments/eval_qwen3vl_ovo_test2.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir "main_experiments/results/ovo_qwen3vl_siglip_top4_all_decoded_cap768_${RUN_TAG}" \
    --analysis_scope full \
    --recent_frames_only 4 \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --siglip_model_name google/siglip-so400m-patch14-384 \
    > "./main_experiments/results/nohup_ovo_qwen3vl_siglip_top4_all_decoded_cap768_${RUN_TAG}.log" 2>&1 &
```
</details>

<details>
<summary><b>Test 3</b></summary>

Runs OVO-Bench backward/realtime evaluation with attention-guided top-4 frame
selection. For each sample, the script builds a candidate frame pool from the
full decoded video using the same `uniform_with_recent_anchor` policy as
Test 1/2 (recent frames are always included; the remaining budget up to
`--max_analysis_frames` is filled uniformly), prefills Qwen3-VL on those
frames plus the question, captures question-prefill self-attention at the
decoder layer specified by `--layer_number`, sums attention over each frame's
vision-token span to produce a per-frame score, selects the top-4 frames, and
then reorders them temporally before running Qwen3-VL inference on just those
four frames. The HLD backward-tracing subset is excluded.

`--layer_number` is required. `--attn_implementation eager` is required
because the scoring path needs attention weights (flash-attention does not
return them). With `--max_analysis_frames 12 --recent_frames_only 4`, the
candidate pool is exactly "recent 4 + uniform 8".

Outputs are saved under `results_incremental.jsonl`, `summary.json`, and
`qwen3vl_attn_top4_results_*.json`. Each record includes
`analysis_frame_indices`, `analysis_frame_scores`, `scoring_layer`,
`selected_frame_indices_by_attention`, and
`selected_frame_indices_for_inference`. `summary.json` reports subset/task
accuracy, split-level official averages and pooled accuracy, plus
`Official Total Avg.` and `Pooled Overall Acc.` for the full run.

attention top-4 test. candidate layers -> 0, 18, 32, 35

Run one layer manually:

```bash
LAYER=0
RUN_TAG=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=5,7 nohup python main_experiments/eval_qwen3vl_ovo_test3.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir "main_experiments/results/ovo_qwen3vl_attn_top4_layer${LAYER}_${RUN_TAG}" \
    --recent_frames_only 4 \
    --max_analysis_frames 12 \
    --layer_number "${LAYER}" \
    --attn_implementation eager \
    --model_device auto \
    --chunk_duration 1.0 \
    --fps 1.0 \
    > "./main_experiments/results/nohup_ovo_qwen3vl_attn_top4_layer${LAYER}_${RUN_TAG}.log" 2>&1 &
```

Run a minimal smoke job (1 backward sample + 1 realtime sample):

```bash
LAYER=0
RUN_TAG=$(date +%Y%m%d_%H%M%S)_smoke
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python main_experiments/eval_qwen3vl_ovo_test3.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir "main_experiments/results/ovo_qwen3vl_attn_top4_layer${LAYER}_${RUN_TAG}" \
    --recent_frames_only 4 \
    --max_analysis_frames 12 \
    --layer_number "${LAYER}" \
    --attn_implementation eager \
    --model_device auto \
    --chunk_duration 1.0 \
    --fps 1.0 \
    --analysis_scope smoke \
    --max_samples_per_split 1 \
    > "./main_experiments/results/nohup_ovo_qwen3vl_attn_top4_layer${LAYER}_${RUN_TAG}.log" 2>&1 &
```

Run multiple layer candidates sequentially by passing a comma-separated layer
list. The script executes layers in the given order, reusing one batch tag for
all result directories and per-layer logs.

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=4,5,6,7 BATCH_TAG="${RUN_TAG}" nohup bash main_experiments/run_qwen3vl_ovo_attn_top4_layers.sh 0,18,32,35 \
    > "./main_experiments/results/nohup_ovo_qwen3vl_attn_top4_layers_${RUN_TAG}.log" 2>&1 &
```

Generate plots from a saved attention top-4 result directory. The script reads
`results_incremental.jsonl`, computes the per-record mean relative position of
the 4 selected frames within the sampled video (using the
`selected_frame_relative_positions` / `selected_frame_mean_relative_position`
fields written by the eval; falls back to recomputing from
`selected_frame_indices_for_inference` and `num_sampled_frames` for older runs),
and writes `plots/attn_top4_selected_position.png` with the same per-subset /
split-avg / total-avg layout as the SigLIP test1 plot. The HLD backward
subset is excluded.

```bash
python analysis/plot_attn_top4_selection.py \
    --result-dir main_experiments/results/ovo_qwen3vl_attn_top4_layer0_20260416_214018
```
</details>

## 📢 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{simplestream2026,
  title={A Simple Baseline for Streaming Video Understanding},
  author={Shen, Yujiao and Tian, Shulin and Yang, Jingkang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2604.02317},
  year={2026}
}
```

## 🙏 Acknowledgement

- [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL): the VLM backbone used in our experiments.
- [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) and [StreamingBench](https://github.com/Infini-AI-Lab/StreamingBench): the evaluation benchmarks.
