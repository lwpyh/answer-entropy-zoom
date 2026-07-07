# Answer Entropy Gate for Selective Video Zoom

**Answer Token Entropy (ATE) gating for efficient tool use in long video QA.**

Uses the entropy of the model's answer token distribution to decide whether to invoke a video zoom/frame-inspect tool. If the model is already confident, skip the tool; if uncertain, trigger it.

Built on top of [VideoZoomer](https://arxiv.org/abs/2512.22315) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

---

## Method

```
R1 (no-tool greedy decode)
  → compute H over answer token logits
  → -H > threshold  →  confident  →  return answer
  → -H ≤ threshold  →  uncertain  →  invoke zoom/frame-inspect  →  R2
```

Three variants are implemented:

- **ATE single-tool** — one zoom tool, one extra round if uncertain
- **ATE multi-tool** — model chooses between `video_zoom` and `frame_inspect`; re-gates after each tool call (up to 5 rounds)
- **Open-loop multi-tool** — no gate; model freely calls tools until it outputs a final answer

---

## Install

```bash
git clone https://github.com/lwpyh/answer-entropy-zoom
cd answer-entropy-zoom
conda create -n VideoZoomer python=3.11 -y
conda activate VideoZoomer
pip install -r requirements.txt
pip install -e .
```

Set your tokens before running:
```bash
export HF_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_key"   # only needed for GPT-based reward scoring
export HF_HOME="/path/to/hf_cache"
```

---

## Data Preparation

Download benchmark data and videos:

```bash
# LongVideoReason
huggingface-cli download LongVideo-Reason/longvideo-reason \
    --repo-type dataset --local-dir longvideo-reason/data

# MLVU / LVB / LVBench / VideoMME — follow each benchmark's official instructions
```

Edit the relevant YAML in `longvideo-reason/` to set `json_path` and `video_path` to your local paths.

---

## Running Experiments

All scripts are in `scripts/`. Edit the `MODEL_PATH` and data paths at the top of each script before submitting.

### ATE single-tool zoom (per benchmark)

```bash
sbatch scripts/eval_mlvu_ate_zoom.sh
sbatch scripts/eval_lvb_ate_zoom.sh
sbatch scripts/eval_videomme_ate_zoom.sh
sbatch scripts/eval_answer_token_entropy_zoom.sh   # LongVideoReason
```

### ATE multi-tool (zoom + frame inspect)

Requires [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with the task configs in `lmms_eval/tasks/`:

```bash
sbatch scripts/eval_mlvu_mt.sh
sbatch scripts/eval_lvb_mt.sh
sbatch scripts/eval_lvbench_mt.sh
sbatch scripts/eval_vmme_mt.sh
sbatch scripts/eval_lvr_mt.sh
```

### Open-loop multi-tool

```bash
sbatch scripts/eval_mlvu_mt_open.sh
sbatch scripts/eval_lvb_mt_open.sh
sbatch scripts/eval_lvbench_mt_open.sh
sbatch scripts/eval_vmme_mt_open.sh
sbatch scripts/eval_lvr_mt_open.sh
```

### Error detection / uncertainty signals

```bash
sbatch scripts/eval_error_detection.sh       # baseline greedy + U_vote, U_traj
sbatch scripts/eval_beam_uncertainty.sh      # beam-path entropy signals
```

---

## Repository Structure

```
main_infer_hmm_zoom.py            ATE zoom gate (primary single-tool)
main_infer_error_detection.py     Baseline for first-error detection
main_infer_beam_uncertainty.py    Beam-path uncertainty signals
main_infer_tool_uncertainty.py    Multi-round tool uncertainty
main_infer_tool.py                Always-zoom baseline
main_infer_greedy.py              No-tool greedy baseline
scripts/
  eval_*_ate_zoom.sh              ATE single-tool eval (per benchmark)
  eval_*_mt.sh                    ATE multi-tool eval
  eval_*_mt_open.sh               Open-loop multi-tool eval
  eval_beam_uncertainty.sh        Beam uncertainty signals
longvideo-reason/
  LongVideoReason_test_fixed.json Test set (fixed answer format)
  eval_longvideoreason.yaml       Dataset config
```
