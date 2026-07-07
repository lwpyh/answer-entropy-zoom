# Answer Entropy Gate for Selective Video Zoom

**Answer Token Entropy (ATE) gating for efficient tool use in long video QA.**

Uses the entropy of the model's answer token distribution to decide whether to invoke a video zoom tool. If the model is already confident, skip the zoom; if uncertain, trigger it.

Built on top of [VideoZoomer](https://arxiv.org/abs/2512.22315).

---

## Method

```
R1 (no-tool greedy decode)
  → compute H over answer token logits
  → -H > threshold  →  confident  →  return answer
  → -H ≤ threshold  →  uncertain  →  invoke zoom  →  R2 with dense frames
```

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

```bash
export HF_TOKEN="your_huggingface_token"
export HF_HOME="/path/to/hf_cache"
```

---

## Data Preparation

```bash
huggingface-cli download LongVideo-Reason/longvideo-reason \
    --repo-type dataset --local-dir longvideo-reason/data

huggingface-cli download LongVideo-Reason/longvideo_eval_videos \
    --repo-type dataset --local-dir longvideo-reason/videos
```

Edit `longvideo-reason/eval_longvideoreason.yaml` — set `json_path` to your local `LongVideoReason_test_fixed.json` and `video_path` to your videos directory.

---

## Run

```bash
sbatch scripts/eval_answer_token_entropy_zoom.sh
```

Edit `MODEL_PATH` and data paths at the top of the script before submitting.

---

## Repository Structure

```
main_infer_hmm_zoom.py            Primary inference script (ATE zoom gate)
main_infer_greedy.py              No-tool greedy baseline
main_infer_tool.py                Always-zoom baseline
scripts/
  eval_answer_token_entropy_zoom.sh   SLURM eval script
longvideo-reason/
  LongVideoReason_test_fixed.json     Test set
  eval_longvideoreason.yaml           Dataset config (edit paths before use)
```
