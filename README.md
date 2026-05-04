# Answer Entropy Gate for Selective Video Zoom

**Answer-distribution entropy gating for efficient tool use in long video QA.**

Instead of always invoking a temporal zoom tool, we ask: *do k independent reasoning chains already agree on the answer?* If yes, skip the zoom. If not, execute it.

Built on top of [VideoZoomer](https://arxiv.org/abs/2512.22315).

---

## Method

Before invoking a zoom tool, draw *k* temperature samples from the model and measure how much the extracted MC answers (A/B/C/D) agree:

```
R1  →  sample k responses at temperature T
     →  extract answer letter from each sample
     →  H_answer = -Σ p_i · log(p_i)  over {A,B,C,D}
     →  score = -H_answer

score > threshold  →  confident  →  SKIP zoom, use majority-vote answer
score ≤ threshold  →  uncertain  →  EXECUTE zoom  →  R2 with dense frames
```

- **H = 0** — all k samples agree → skip zoom
- **H = log(4)** — uniform distribution → execute zoom
- Threshold `−0.30` skips ~10–20% of samples (those where ≥4/5 chains agree)
- No logprobs, no keyword lists, no calibrated constants — only one free parameter (threshold)

---

## Install

```bash
git clone https://github.com/lwpyh/answer-entropy-zoom
cd answer-entropy-zoom
conda create -n answer-entropy-zoom python=3.11 -y
conda activate answer-entropy-zoom
pip install -r requirements.txt
pip install -e .
pip install httpx==0.23.3
```

---

## Data Preparation

### LongVideoReason-eval (1k test samples)

1. Download the dataset and videos from HuggingFace:
   ```bash
   huggingface-cli download LongVideo-Reason/longvideo-reason --repo-type dataset --local-dir longvideo-reason/data
   huggingface-cli download LongVideo-Reason/longvideo_eval_videos --repo-type dataset --local-dir longvideo-reason/videos
   ```

2. Edit `longvideo-reason/eval_longvideoreason.yaml` — update the `json_path` field to point to your local `LongVideoReason_test_fixed.json`.

### VideoMME

Download videos and annotations from the [VideoMME](https://video-mme.github.io) official site.

---

## Usage

```bash
python main_infer_hmm_zoom.py \
    --data_path           longvideo-reason/eval_longvideoreason.yaml \
    --model_path          <path-to-videozoomer-checkpoint> \
    --video_root          /path/to/videos \
    --output_dir          ./results/answer_entropy \
    --score_mode          answer_entropy \
    --entropy_threshold   -0.30 \
    --answer_k            5 \
    --answer_temperature  0.7 \
    --fps                 0.5 \
    --frames_upbound      64 \
    --max_rounds          5 \
    --batch_size          32
```

Or via the SLURM script (edit paths inside first):
```bash
sbatch scripts/eval_answer_entropy_zoom.sh
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--score_mode` | `hmm` | Set to `answer_entropy` to enable the entropy gate |
| `--entropy_threshold` | `-0.30` | Skip zoom when `score = -H_answer > threshold` |
| `--answer_k` | `5` | Number of temperature samples |
| `--answer_temperature` | `0.7` | Sampling temperature |
| `--fps` | `0.5` | Frame rate for R1 video load |
| `--frames_upbound` | `64` | Max frames per round |
| `--max_rounds` | `5` | Maximum tool call rounds |
| `--batch_size` | `32` | vLLM batch size |

---

## Repository Structure

```
main_infer_hmm_zoom.py          Primary inference: VideoZoomer + answer entropy gate
main_infer_adaptive_zoom.py     Base module: data loading, video I/O, prompt builders
scripts/
  eval_answer_entropy_zoom.sh   SLURM evaluation script
longvideo-reason/
  eval_longvideoreason.yaml     Dataset config (edit json_path before use)
```
