#!/usr/bin/env python3
"""
main_infer_beam_uncertainty.py  ── Beam-path Uncertainty Signals

Goal: extract uncertainty signals from B parallel diverse/beam paths in a
      SINGLE vLLM inference pass (n=B, logprobs=K), and evaluate their
      power for first-error detection.

══════════════════════════════════════════════════════════════════════════════
Signal design  (all from one pass: n=B=5, temperature=T, logprobs=20)
══════════════════════════════════════════════════════════════════════════════

[Text-level]
  U_beam_vote    : 1 − max_count/B          answer disagreement
  U_beam_ent     : H(empirical answer dist)  soft answer entropy

[Logprob-level — from per-path mean log-likelihoods]
  U_beam_spread  : std(mean_LL per path)     logprob dispersion  ↑=uncertain
  U_beam_gap     : −(LL_best − LL_2nd)       negated conf. gap   ↑=uncertain

[Sequence-level — logprob vectors as hidden-state proxy]
  U_lp_traj_cos  : 1 − mean_pairwise cos of resampled chosen-logprob sequences
                   Captures: do paths have the same "confidence rhythm"?

  U_lp_shape_cos : 1 − mean_pairwise cos of mean-pooled sorted-top-K vectors
                   At each position, vLLM returns top-K logprob values.
                   Sort them (ignore which token) → K-dim distribution shape.
                   Mean-pool over sequence → path's global "spread pattern".
                   Captures: do paths show the same confidence distribution shape?

══════════════════════════════════════════════════════════════════════════════
Analysis
══════════════════════════════════════════════════════════════════════════════
  Labels (acc_notool) are taken from --ref_jsonl (existing error_detection
  JSONL), joined by problem_id.  Baseline signals printed for comparison.

══════════════════════════════════════════════════════════════════════════════
Usage
══════════════════════════════════════════════════════════════════════════════
  # Full run:
  python main_infer_beam_uncertainty.py \\
      --data_path  .../eval_deltaS_v2.yaml \\
      --output_dir ./infer_results/beam_uncertainty \\
      --ref_jsonl  ./infer_results/error_detection_full/results_error_detection.jsonl

  # Analysis only (re-analyse saved JSONL, no GPU needed):
  python main_infer_beam_uncertainty.py --analyze_only \\
      --output_dir ./infer_results/beam_uncertainty \\
      --ref_jsonl  ./infer_results/error_detection_full/results_error_detection.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   default=None,
                   help="YAML dataset config (required unless --analyze_only)")
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/beam_uncertainty")

    # Reference JSONL from existing error_detection run (for labels + baselines)
    p.add_argument("--ref_jsonl",   default=None,
                   help="Existing error_detection JSONL with acc_notool + baseline signals")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)

    # Video
    p.add_argument("--fps",            type=float, default=0.2)
    p.add_argument("--min_pixels",     type=int,   default=12544)
    p.add_argument("--frames_upbound", type=int,   default=120)
    p.add_argument("--max_tokens",     type=int,   default=2048)

    # Beam / diverse sampling
    p.add_argument("--n_paths",      type=int,   default=5,
                   help="Number of parallel paths B")
    p.add_argument("--sample_temp",  type=float, default=0.4,
                   help="Sampling temperature (lower → more beam-like)")
    p.add_argument("--logprobs_k",   type=int,   default=20,
                   help="Top-K vocab logprobs returned per token (for shape vector)")
    p.add_argument("--traj_len",     type=int,   default=64,
                   help="Resampled sequence length for U_lp_traj_cos")

    # Misc
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--system_prompt", default="You are a helpful assistant.")
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip inference; load existing JSONL and re-analyse")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    samples = []
    for ds in cfg.get("datasets", []):
        with open(ds["json_path"]) as f:
            samples.extend(json.load(f))
    return samples


def resolve_video_path(video_rel: str, video_root: str) -> str:
    p = Path(video_rel)
    if p.is_absolute():
        return str(p)
    parts = p.parts
    rel = Path(*parts[1:]) if parts and parts[0] in (".", "..") else p
    return str(Path(video_root) / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# Video loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_frames(video_path, fps, max_pixels, min_pixels, frames_upbound):
    from decord import VideoReader, cpu
    from PIL import Image

    vr      = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total   = len(vr)
    avg_fps = vr.get_avg_fps()

    stride  = max(1, int(avg_fps / fps)) if fps > 0 else 1
    indices = list(range(0, total, stride))
    if len(indices) > frames_upbound:
        sel     = np.linspace(0, len(indices) - 1, frames_upbound, dtype=int)
        indices = [indices[s] for s in sel]

    frame_times = [idx / avg_fps for idx in indices]
    raw = vr.get_batch(indices).asnumpy()

    frames = []
    for arr in raw:
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        h, w = img.height, img.width
        px = h * w
        if px > max_pixels:
            s = (max_pixels / px) ** 0.5
            img = img.resize((max(2, int(w*s/2)*2), max(2, int(h*s/2)*2)), Image.LANCZOS)
        elif px < min_pixels:
            s = (min_pixels / px) ** 0.5
            img = img.resize((max(2, int(w*s/2)*2), max(2, int(h*s/2)*2)), Image.LANCZOS)
        frames.append(img)
    return frame_times, frames


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

def build_vllm_input(question, frame_times, frames, processor, system_prompt):
    vision_str = "".join(
        f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
        for i, t in enumerate(frame_times)
    )
    user_content = (question.replace("<image>", vision_str, 1)
                    if "<image>" in question else vision_str + "\n" + question)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return {
        "prompt": processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
        "multi_modal_data": {"image": frames},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Answer extraction & scoring
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mc_answer(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    fragment = m.group(1).strip() if m else text
    matches  = re.findall(r"\b([A-D])\b", fragment)
    if matches:
        return matches[-1]
    if fragment and fragment[0] in "ABCD":
        return fragment[0]
    matches = re.findall(r"([A-D])\.", text)
    return matches[-1] if matches else None


def score_majority(answers: list, gt: str) -> float:
    """Score the majority-vote answer against ground truth."""
    gt_m  = re.search(r"<answer>(.*?)</answer>", gt, re.DOTALL)
    gt_c  = gt_m.group(1).strip() if gt_m else gt.strip()
    gt_l  = extract_mc_answer(gt_c) or gt_c.strip()
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    majority = Counter(valid).most_common(1)[0][0]
    return 1.0 if majority == gt_l else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level logprob helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _path_mean_ll(comp_output) -> float:
    """Mean log-likelihood of chosen tokens for one CompletionOutput."""
    lps = []
    for tok_id, step in zip(comp_output.token_ids, comp_output.logprobs or []):
        if step and tok_id in step:
            lps.append(step[tok_id].logprob)
    return float(np.mean(lps)) if lps else -100.0


def _chosen_lp_seq(comp_output) -> np.ndarray:
    """
    Extract the sequence of chosen-token logprobs for one path.
    If the chosen token is missing from the top-K dict, fall back to
    the minimum logprob in the returned set.
    """
    lps = []
    for tok_id, step in zip(comp_output.token_ids, comp_output.logprobs or []):
        if not step:
            continue
        if tok_id in step:
            lps.append(step[tok_id].logprob)
        else:
            lps.append(min(lp.logprob for lp in step.values()))
    return np.array(lps, dtype=float) if lps else np.array([-100.0])


def _shape_vec(comp_output, K: int) -> np.ndarray:
    """
    For each token position: take the top-K logprob VALUES (sorted descending),
    ignoring WHICH tokens they correspond to.  This gives a K-dim vector that
    describes the "spread" of the output distribution at that step.
    Mean-pool over the whole sequence → one K-dim vector per path.

    Why sort and ignore token identity?
    - Different paths produce different token sequences, so token IDs are
      not aligned across paths.
    - The distribution shape (how concentrated vs spread the probability mass
      is) is comparable regardless of which tokens are in top-K.
    - This is a proxy for the hidden state: similar h_t → similar logit
      distributions → similar shape vectors.
    """
    vecs = []
    for step in (comp_output.logprobs or []):
        if not step:
            continue
        # Sort logprob values descending
        vals = sorted((lp.logprob for lp in step.values()), reverse=True)
        # Pad to exactly K (repeat last value) or truncate
        if len(vals) >= K:
            vals = vals[:K]
        else:
            vals = vals + [vals[-1]] * (K - len(vals))
        vecs.append(vals)
    if not vecs:
        return np.zeros(K)
    return np.mean(vecs, axis=0)  # (K,)


def _resample(seq: np.ndarray, L: int) -> np.ndarray:
    """
    Linearly interpolate a 1-D sequence to exactly L evenly-spaced points.
    Used to make variable-length logprob trajectories comparable.
    """
    if len(seq) == 0:
        return np.zeros(L)
    if len(seq) == L:
        return seq.copy()
    old_x = np.linspace(0.0, 1.0, len(seq))
    new_x = np.linspace(0.0, 1.0, L)
    return np.interp(new_x, old_x, seq)


def _pairwise_cos_dist(mat: np.ndarray) -> float:
    """
    Given a (B, D) matrix, compute 1 − mean pairwise cosine similarity.
    Returns NaN if fewer than 2 valid (non-zero) rows.
    """
    B = mat.shape[0]
    if B < 2:
        return np.nan
    norms = np.linalg.norm(mat, axis=1)          # (B,)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return np.nan
    mat_n = mat / np.maximum(norms[:, None], 1e-12)
    sims  = []
    for i in range(B):
        for j in range(i + 1, B):
            if valid[i] and valid[j]:
                sims.append(float(mat_n[i] @ mat_n[j]))
    return float(1.0 - np.mean(sims)) if sims else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Main signal computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_beam_signals(comp_outputs, traj_len: int = 64) -> dict:
    """
    Compute all 6 beam-path uncertainty signals from B CompletionOutputs.

    Requires: comp_output.logprobs populated (logprobs=K in SamplingParams).

    Returns
    -------
    dict with keys:
        U_beam_vote, U_beam_ent,         # text-level
        U_beam_spread, U_beam_gap,       # logprob-level
        U_lp_traj_cos, U_lp_shape_cos,  # sequence-level proxy
        _answers                          # raw answers (debugging)
    """
    B        = len(comp_outputs)
    answers  = [extract_mc_answer(o.text) for o in comp_outputs]
    mean_lls = np.array([_path_mean_ll(o) for o in comp_outputs])

    # ── Text-level ──────────────────────────────────────────────────────── #

    valid_ans = [a for a in answers if a is not None]

    if valid_ans:
        max_cnt    = max(Counter(valid_ans).values())
        U_beam_vote = float(1.0 - max_cnt / B)
        cnt   = Counter(valid_ans)
        total = sum(cnt.values())
        probs = np.array([v / total for v in cnt.values()])
        U_beam_ent  = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        U_beam_vote = 1.0
        U_beam_ent  = float(np.log(4))   # maximum entropy (4 choices)

    # ── Logprob-level ────────────────────────────────────────────────────── #

    U_beam_spread = float(np.std(mean_lls)) if B >= 2 else 0.0

    sorted_ll = np.sort(mean_lls)[::-1]
    U_beam_gap = float(-(sorted_ll[0] - sorted_ll[1])) if B >= 2 else 0.0

    # ── Sequence-level: logprob trajectory (U_lp_traj_cos) ──────────────── #
    # Each path: sequence of chosen-token logprobs, resampled to traj_len points
    lp_seqs   = [_chosen_lp_seq(o) for o in comp_outputs]
    resampled = np.array([_resample(s, traj_len) for s in lp_seqs])  # (B, L)
    U_lp_traj_cos = _pairwise_cos_dist(resampled)

    # ── Sequence-level: distribution shape (U_lp_shape_cos) ──────────────── #
    # Infer K from the first non-empty logprob step
    K = None
    for o in comp_outputs:
        for step in (o.logprobs or []):
            if step:
                K = len(step)
                break
        if K is not None:
            break

    if K and K > 0:
        shape_vecs    = np.array([_shape_vec(o, K) for o in comp_outputs])  # (B, K)
        U_lp_shape_cos = _pairwise_cos_dist(shape_vecs)
    else:
        U_lp_shape_cos = np.nan

    return {
        "U_beam_vote":    float(U_beam_vote),
        "U_beam_ent":     float(U_beam_ent),
        "U_beam_spread":  float(U_beam_spread),
        "U_beam_gap":     float(U_beam_gap),
        "U_lp_traj_cos":  float(U_lp_traj_cos) if not np.isnan(U_lp_traj_cos) else None,
        "U_lp_shape_cos": float(U_lp_shape_cos) if not np.isnan(U_lp_shape_cos) else None,
        "_answers":       answers,   # kept for debugging, stripped before saving
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

NEW_SIGNALS = [
    "U_beam_vote", "U_beam_ent",
    "U_beam_spread", "U_beam_gap",
    "U_lp_traj_cos", "U_lp_shape_cos",
]
BASELINE_SIGNALS = ["U_vote", "U_traj", "U_combo_v2"]
DERIVED_SIGNALS  = ["U_combo_beam", "U_combo_best"]

SIGNAL_META = {
    "U_beam_vote":    ("U_beam_vote    answer disagree  1−max/B           [text]",  "+"),
    "U_beam_ent":     ("U_beam_ent     answer entropy   H(p_ans)          [text]",  "+"),
    "U_beam_spread":  ("U_beam_spread  std(mean_LL per path)         [logprob]",    "+"),
    "U_beam_gap":     ("U_beam_gap     −(LL_best−LL_2nd) ↑=uncertain [logprob]",   "+"),
    "U_lp_traj_cos":  ("U_lp_traj_cos  1−cos(chosen_lp_seqs resampled)    [seq]",  "+"),
    "U_lp_shape_cos": ("U_lp_shape_cos 1−cos(sorted_topK mean-pool)       [seq]",  "+"),
    # Baselines
    "U_vote":         ("U_vote         [BASELINE diverse K=5 T=0.7]",              "+"),
    "U_traj":         ("U_traj         [BASELINE TF-IDF CoT distance]",             "+"),
    "U_combo_v2":     ("U_combo_v2     [BASELINE minmax mean(vote+traj)]",          "+"),
    # Hardcoded combos (rank-pct within batch, fixed AUC-proportional weights)
    "U_combo_beam":   ("U_combo_beam   [beam-only 3-sig: ent+vote+traj_cos]",       "+"),
    "U_combo_best":   ("U_combo_best   [Cx_best+  5-sig: +U_vote+U_traj]",          "+"),
}

# ── Fixed combo weights (calibrated on eval_deltaS_v2, n=1601) ──────────── #
# AUC-proportional: w_i = (AUC_i − 0.5) / Σ(AUC_j − 0.5)
# Cx_best+  (5-sig, AUC=0.6869): ent=0.6593 vote=0.6532 traj_cos=0.5665
#                                 U_vote=0.6474 U_traj=0.6156
COMBO_BEST_WEIGHTS: dict = {
    "U_beam_ent":    0.2691,
    "U_beam_vote":   0.1743,
    "U_lp_traj_cos": 0.1123,
    "U_vote":        0.2490,
    "U_traj":        0.1953,
}
# Beam-only fallback (3-sig, AUC≈0.67) — usable without a ref_jsonl
COMBO_BEAM_WEIGHTS: dict = {
    "U_beam_ent":    0.4842,
    "U_beam_vote":   0.3137,
    "U_lp_traj_cos": 0.2021,
}


def _auc_roc(labels, scores):
    pairs   = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tp, fp, auc, prev_fp = 0, 0, 0.0, 0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


def _pearson(x, y):
    x, y = np.array(x, float), np.array(y, float)
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = np.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum())
    return float(num / den) if den > 1e-12 else 0.0


def _spearman(x, y):
    def rank(v):
        o = np.argsort(v)
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(len(v)) + 1
        return r
    return _pearson(rank(np.array(x, float)), rank(np.array(y, float)))


def _rankpct(arr: np.ndarray) -> np.ndarray:
    out  = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    n    = mask.sum()
    if n < 2:
        return out
    order = np.argsort(arr[mask])
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    out[mask] = ranks / (n - 1)
    return out


def _compute_combo(records: list, weights: dict) -> np.ndarray:
    """
    Compute weighted rank-pct combo score for each record.
    Only uses signals present in the records; re-normalises weights for missing ones.
    Returns float array of length len(records) (nan where insufficient data).
    """
    present = {
        k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in records])
        for k in weights
        if any(r.get(k) is not None for r in records)
    }
    if len(present) < 2:
        return np.full(len(records), np.nan)
    total_w = sum(weights[k] for k in present)
    combo   = np.zeros(len(records))
    for k, arr in present.items():
        combo += (weights[k] / total_w) * _rankpct(arr)
    return combo


def _eval_signal(sig, valid, errors):
    """Compute Pearson, Spearman, AUC for one signal. Returns (pr, sr, sp, auc, n_ok)."""
    vals = np.array([np.nan if r.get(sig) is None else float(r[sig]) for r in valid])
    ok   = np.isfinite(vals) & np.isfinite(errors)
    n_ok = int(ok.sum())
    if n_ok < 10:
        return None, None, None, None, n_ok
    v, e = vals[ok], errors[ok]
    try:
        from scipy import stats as ss
        pr, _  = ss.pearsonr(v, e)
        sr, sp = ss.spearmanr(v, e)
    except ImportError:
        pr, sr, sp = _pearson(v, e), _spearman(v, e), np.nan
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(e.astype(int), v))
    except Exception:
        auc = _auc_roc(e.astype(int).tolist(), v.tolist())
    return float(pr), float(sr), float(sp), float(auc), n_ok


def analyze(results: list, output_dir: str, ref_jsonl: str = None):
    """
    Main analysis function.

    1. Join new beam results with ref JSONL (by problem_id) for:
       - ground truth labels (acc_notool)
       - baseline signals (U_vote, U_traj, U_combo_v2)
    2. Compute AUC-ROC + Spearman ρ for all signals.
    3. Build AUC-proportional combos from the 6 new signals.
    4. Print binned error-rate table and outcome-group means.
    """

    # ── Load reference JSONL ─────────────────────────────────────────────── #
    ref_lookup = {}
    if ref_jsonl and os.path.exists(ref_jsonl):
        print(f"[analysis] Loading reference from {ref_jsonl}")
        with open(ref_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("problem_id") is not None:
                        ref_lookup[r["problem_id"]] = r

    # ── Merge: pull acc_notool + baseline signals from ref ───────────────── #
    merged = []
    n_from_ref = 0
    for r in results:
        pid = r.get("problem_id")
        rec = dict(r)
        if pid in ref_lookup:
            ref = ref_lookup[pid]
            # Use ref acc_notool as canonical label (greedy, consistent)
            rec["acc_notool"] = ref.get("acc_notool", rec.get("acc_notool"))
            for k in BASELINE_SIGNALS:
                rec.setdefault(k, ref.get(k))
            n_from_ref += 1
        merged.append(rec)

    # ── Compute & inject hardcoded combo scores ──────────────────────────── #
    beam_combo = _compute_combo(merged, COMBO_BEAM_WEIGHTS)
    best_combo = _compute_combo(merged, COMBO_BEST_WEIGHTS)
    for i, rec in enumerate(merged):
        if np.isfinite(beam_combo[i]):
            rec["U_combo_beam"] = float(beam_combo[i])
        if np.isfinite(best_combo[i]):
            rec["U_combo_best"] = float(best_combo[i])

    # Rewrite JSONL with combo fields (beam signals + combo scores)
    out_jsonl = os.path.join(output_dir, "results_beam_uncertainty.jsonl")
    with open(out_jsonl, "w") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_beam = int(np.isfinite(beam_combo).sum())
    n_best = int(np.isfinite(best_combo).sum())
    print(f"[combo]  U_combo_beam written for {n_beam}/{len(merged)} records")
    print(f"[combo]  U_combo_best written for {n_best}/{len(merged)} records  → {out_jsonl}")

    valid = [r for r in merged if r.get("acc_notool") is not None]
    if not valid:
        print("[analysis] No valid records with acc_notool — cannot evaluate.")
        return

    errors   = np.array([1.0 - float(r["acc_notool"]) for r in valid])
    n        = len(valid)
    err_rate = errors.mean()
    has_ref  = n_from_ref > 0

    print(f"\n{'='*92}")
    print(f"BEAM-PATH UNCERTAINTY  (n={n},  error_rate={err_rate:.3f},  "
          f"ref_joined={n_from_ref})")
    print(f"{'='*92}")

    # ── Per-signal table ─────────────────────────────────────────────────── #
    all_signals = (NEW_SIGNALS
                   + (BASELINE_SIGNALS if has_ref else [])
                   + DERIVED_SIGNALS)

    hdr = (f"{'Signal':<62} {'Pearson':>8}  {'Spearman':>9}  "
           f"{'p-val':>7}  {'AUC-ROC':>7}")
    print(hdr)
    print("-" * 92)

    rows   = []
    aucs   = {}   # signal → AUC (for combo weighting)

    sep_printed   = False
    derived_sep   = False
    for sig in all_signals:
        if sig in BASELINE_SIGNALS and not sep_printed:
            print(f"{'─'*92}")
            sep_printed = True
        if sig in DERIVED_SIGNALS and not derived_sep:
            print(f"{'─'*92}")
            derived_sep = True

        if sig not in SIGNAL_META:
            continue
        label, exp = SIGNAL_META[sig]

        pr, sr, sp, auc, n_ok = _eval_signal(sig, valid, errors)
        if pr is None:
            print(f"{label:<62} {'—':>8}  {'—':>9}  {'—':>7}  {'—':>7}  (n={n_ok})")
            continue

        sig_mark = "*" if (not np.isnan(sp) and sp < 0.05) else " "
        ok_sign  = "✓" if (exp == "+" and sr > 0) or (exp == "−" and sr < 0) else "✗"
        auc_str  = f"{auc:.4f}" if not np.isnan(auc) else "   n/a"
        sp_str   = f"{sp:.4f}" if not np.isnan(sp)  else "   n/a"

        print(f"{label:<62} {pr:>+8.4f}  {sr:>+8.4f}{sig_mark}  "
              f"{sp_str}  {auc_str}  {ok_sign}")
        rows.append(dict(signal=sig, n=n_ok, pearson=pr, spearman=sr,
                         spearman_p=sp if not np.isnan(sp) else None,
                         auc=auc if not np.isnan(auc) else None))
        if not np.isnan(auc):
            aucs[sig] = auc

    print(f"{'='*92}")
    print("* Spearman p<0.05   ✓/✗ = expected direction")

    # ── AUC-proportional combo signals ──────────────────────────────────── #
    print(f"\n{'─'*92}")
    print("COMBO SIGNALS  (rank-pct normalised, AUC-proportional weights)")
    print(f"{'─'*92}")
    _analyze_combos(valid, errors, aucs, rows)

    # ── Binned analysis ──────────────────────────────────────────────────── #
    print(f"\n{'─'*92}")
    print("BINNED ERROR RATE  (signal quintiles → mean error rate)")
    print(f"{'─'*92}")
    print(f"{'Signal':<20}  Q1(low)   Q2        Q3        Q4        Q5(high)  Δ(Q5−Q1)")
    print("-" * 92)
    for sig in all_signals:
        vals = np.array([np.nan if r.get(sig) is None else float(r[sig]) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(errors)
        if ok.sum() < 20:
            continue
        v, e = vals[ok], errors[ok]
        q    = np.percentile(v, [0, 20, 40, 60, 80, 100])
        bins = []
        for lo, hi in zip(q[:-1], q[1:]):
            mask = (v >= lo) & (v <= hi)
            bins.append(e[mask].mean() if mask.sum() > 0 else np.nan)
        delta  = bins[-1] - bins[0]
        b_str  = "  ".join(f"{x:.4f}" for x in bins)
        print(f"{sig:<20}  {b_str}  {delta:>+.4f}")

    # ── Outcome group means ──────────────────────────────────────────────── #
    print(f"\n{'─'*92}")
    print("SIGNAL MEAN BY OUTCOME  (correct vs wrong)")
    print(f"{'─'*92}")
    correct = [r for r in valid if float(r.get("acc_notool", 0)) >= 1.0]
    wrong   = [r for r in valid if float(r.get("acc_notool", 1)) < 1.0]
    print(f"{'Signal':<20}  "
          f"{'Correct (n='+str(len(correct))+')':>20}  "
          f"{'Wrong (n='+str(len(wrong))+')':>18}  "
          f"{'Δ(wrong−corr)':>14}")
    print("-" * 76)
    for sig in all_signals:
        c_v = [float(r[sig]) for r in correct if r.get(sig) is not None
               and np.isfinite(float(r[sig]))]
        w_v = [float(r[sig]) for r in wrong   if r.get(sig) is not None
               and np.isfinite(float(r[sig]))]
        if not c_v or not w_v:
            continue
        cm, wm = np.mean(c_v), np.mean(w_v)
        print(f"{sig:<20}  {cm:>20.4f}  {wm:>18.4f}  {wm - cm:>+14.4f}")

    # ── Save ─────────────────────────────────────────────────────────────── #
    out_path = os.path.join(output_dir, "beam_uncertainty_analysis.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_path}")


def _analyze_combos(valid, errors, aucs, rows):
    """
    Build combo signals from the 6 new beam signals using:
      - rank-pct normalisation (robust to distribution shape)
      - AUC-proportional weighting where individual AUCs are available

    Combos evaluated:
      Cx_text   : vote + ent              (text only)
      Cx_lp     : spread + gap            (logprob only)
      Cx_seq    : traj_cos + shape_cos    (sequence proxy only)
      Cx_all    : all 6 signals equally   (full ensemble)
      Cx_best   : AUC-weighted combo of whichever new signals beat 0.5
    """
    def _auc_eval(vals):
        ok = np.isfinite(vals) & np.isfinite(errors)
        if ok.sum() < 10:
            return np.nan
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(errors[ok].astype(int), vals[ok]))
        except Exception:
            return _auc_roc(errors[ok].astype(int).tolist(), vals[ok].tolist())

    # Build rank-pct arrays for new signals + baselines (needed for cross combos)
    raw  = {}
    rank = {}
    for k in NEW_SIGNALS + BASELINE_SIGNALS:
        raw[k]  = np.array([np.nan if r.get(k) is None else float(r[k])
                            for r in valid])
        rank[k] = _rankpct(raw[k])

    # Helper: nanmean of selected rank arrays per sample
    def combo(keys):
        available = [k for k in keys if np.isfinite(rank[k]).sum() >= 10]
        if not available:
            return np.full(len(valid), np.nan)
        mat = np.array([rank[k] for k in available])  # (|keys|, n)
        return np.nanmean(mat, axis=0)                 # (n,)

    # AUC-weighted combo of new signals only that beat random (AUC > 0.5)
    useful = {k: v for k, v in aucs.items() if k in NEW_SIGNALS and v > 0.5}
    if useful:
        total_excess = sum(v - 0.5 for v in useful.values())
        weights      = {k: (v - 0.5) / total_excess for k, v in useful.items()}
        mat          = np.array([rank[k] * weights[k] for k in useful])
        Cx_best      = np.nansum(mat, axis=0)
    else:
        Cx_best = combo(NEW_SIGNALS)

    # AUC-weighted combo of ALL signals (new + baseline) that beat random
    all_useful = {k: v for k, v in aucs.items()
                  if k in NEW_SIGNALS + BASELINE_SIGNALS
                  and np.isfinite(v) and v > 0.5}
    if all_useful:
        total_excess_all = sum(v - 0.5 for v in all_useful.values())
        weights_all      = {k: (v - 0.5) / total_excess_all for k, v in all_useful.items()}
        mat_all          = np.array([rank[k] * weights_all[k] for k in all_useful])
        Cx_best_plus     = np.nansum(mat_all, axis=0)
        n_all_useful     = len(all_useful)
    else:
        Cx_best_plus = combo(NEW_SIGNALS + BASELINE_SIGNALS)
        n_all_useful = 0

    combos = {
        "Cx_text    (vote+ent)":            combo(["U_beam_vote",   "U_beam_ent"]),
        "Cx_lp      (spread+gap)":          combo(["U_beam_spread", "U_beam_gap"]),
        "Cx_seq     (traj+shape)":          combo(["U_lp_traj_cos", "U_lp_shape_cos"]),
        "Cx_all     (all 6 equal)":         combo(NEW_SIGNALS),
        "Cx_best    (AUC-wtd new only)":    Cx_best,
        "── cross combos ──":               None,
        "Cx_ent+traj  (beam_ent+U_traj)":  combo(["U_beam_ent",  "U_traj"]),
        "Cx_vote+traj (beam_vote+U_traj)":  combo(["U_beam_vote", "U_traj"]),
        "Cx_ev+traj   (ent+vote+U_traj)":   combo(["U_beam_ent",  "U_beam_vote", "U_traj"]),
        "Cx_best+   (AUC-wtd new+base)":    Cx_best_plus,
    }

    print(f"{'Combo':<46}  {'AUC-ROC':>8}  {'Spearman':>9}  {'n_weights':>10}")
    print("-" * 78)
    for name, vals in combos.items():
        if vals is None:           # separator row
            print(f"  {name}")
            continue
        auc = _auc_eval(vals)
        ok  = np.isfinite(vals) & np.isfinite(errors)
        sr  = _spearman(vals[ok], errors[ok]) if ok.sum() >= 2 else np.nan
        if "best+" in name:
            tag = f"({n_all_useful} sigs)"
        elif "best" in name:
            tag = f"({len(useful)} sigs)"
        else:
            tag = ""
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "   n/a"
        sr_str  = f"{sr:+.4f}" if not np.isnan(sr) else "   n/a"
        print(f"{name:<46}  {auc_str}  {sr_str}  {tag:>10}")
        rows.append(dict(signal=name.strip(),
                         auc=float(auc) if not np.isnan(auc) else None,
                         spearman=float(sr) if not np.isnan(sr) else None))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_beam_uncertainty.jsonl")

    # ── Analysis-only mode ──────────────────────────────────────────────── #
    if args.analyze_only:
        print(f"[analysis] Loading {output_jsonl}")
        with open(output_jsonl) as f:
            results = [json.loads(l) for l in f if l.strip()]
        print(f"[analysis] {len(results)} records")
        analyze(results, args.output_dir, ref_jsonl=args.ref_jsonl)
        return

    if not args.data_path:
        raise ValueError("--data_path required unless --analyze_only is set")

    # ── Load data ───────────────────────────────────────────────────────── #
    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    # ── Processor ───────────────────────────────────────────────────────── #
    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ── vLLM ────────────────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    print(f"[vLLM]  Initialising (tp={args.tensor_parallel_size}, "
          f"util={args.gpu_memory_utilization})")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        mm_processor_kwargs={
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        },
        limit_mm_per_prompt={"image": args.frames_upbound},
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    # Single sampling params: n=B paths, logprobs=K for shape vector
    sp = SamplingParams(
        n=args.n_paths,
        max_tokens=args.max_tokens,
        temperature=args.sample_temp,
        logprobs=args.logprobs_k,    # top-K logprobs per token, per path
        detokenize=True,
    )
    print(f"[vLLM]  Sampling: n={args.n_paths}, T={args.sample_temp}, "
          f"logprobs={args.logprobs_k}, max_tokens={args.max_tokens}")

    # ── Inference loop ──────────────────────────────────────────────────── #
    results = []
    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            vllm_inputs   = []
            valid_samples = []
            for sample in batch:
                try:
                    vp = resolve_video_path(sample["videos"][0], args.video_root)
                    ft, fr = load_video_frames(
                        vp, args.fps, args.max_pixels,
                        args.min_pixels, args.frames_upbound)
                    inp = build_vllm_input(
                        sample["problem"], ft, fr, processor, args.system_prompt)
                    vllm_inputs.append(inp)
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id','?')}: {e}")

            if not valid_samples:
                continue

            # Single inference: all B paths per sample
            beam_outs = llm.generate(vllm_inputs, sp)

            for sample, b_out in zip(valid_samples, beam_outs):
                gt      = sample.get("solution", "")
                signals = compute_beam_signals(b_out.outputs, traj_len=args.traj_len)

                answers = signals.pop("_answers")
                acc     = score_majority(answers, gt)

                record = {
                    "problem_id":  sample.get("problem_id"),
                    "data_source": sample.get("data_source"),
                    "delta_s":     sample.get("delta_s"),
                    "acc_notool":  float(acc),    # majority-vote accuracy
                    **signals,
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nResults JSONL: {output_jsonl}  ({len(results)} samples)")
    analyze(results, args.output_dir, ref_jsonl=args.ref_jsonl)


if __name__ == "__main__":
    main()
