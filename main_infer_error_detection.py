#!/usr/bin/env python3
"""
main_infer_error_detection.py  ── v2 First-Error Detection

Goal: predict (WITHOUT ground truth) whether the no-tool answer is wrong.
All signals are GLOBAL-level (one scalar per sample), not token-level entropy.

══════════════════════════════════════════════════════════════════════════════
Inference passes
══════════════════════════════════════════════════════════════════════════════
  Phase 1a  greedy (logprobs=20, T=0)
              → E_bar   : −mean(logprob)            [v1 baseline]
              → U_rough : mean|Δlogprob|             [v1 baseline]

  Phase 1b  diverse (K=5, T=0.7, logprobs=1)
              → U_vote  : answer disagreement        [v1 improved]
              → U_ens   : sample weight entropy      [v1]
              → U_gap   : LL_best − LL_2nd (↑=cert) [v1]
              → U_traj  : mean pairwise CoT distance [NEW — text TF-IDF/Jaccard]

  Phase 1c  alignment check  (text-only, T=0, max_tokens=64)
              Input : question + greedy reasoning (no video)
              Prompt: "Based ONLY on the above reasoning, what is the answer?"
              → U_align : 1 if model's answer ≠ greedy answer, else 0   [NEW]

  Phase 1d  self-verification  (text-only, T=0, max_tokens=64)
              Input : question → original response → "double-check" turn
              → U_flip  : 1 if answer changes after self-check, else 0  [NEW]

══════════════════════════════════════════════════════════════════════════════
Analysis
══════════════════════════════════════════════════════════════════════════════
  (A) Correlation table : each signal × error (1 − acc_notool)
  (B) AUC-ROC           : per signal (higher = better predictor)
  (C) Binned analysis   : signal quintiles → mean error rate
  (D) Combined signal   : U_vote + U_traj + U_align + U_flip ensemble

══════════════════════════════════════════════════════════════════════════════
Usage
══════════════════════════════════════════════════════════════════════════════
  # Full run:
  python main_infer_error_detection.py \\
      --data_path  .../eval_deltaS_v2.yaml \\
      --output_dir ./infer_results/error_detection

  # Analysis only (re-analyse existing JSONL):
  python main_infer_error_detection.py --analyze_only \\
      --output_dir ./infer_results/error_detection

  # Skip expensive passes:
  python main_infer_error_detection.py ... --skip_align --skip_flip
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
# System prompts
# ═══════════════════════════════════════════════════════════════════════════════

NOTOOL_SYSTEM_PROMPT = "You are a helpful assistant."

ALIGN_SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "Read the question and the provided reasoning chain. "
    "Then output the answer that the reasoning supports."
)

VERIFY_SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "Re-read the question and the previous reasoning. "
    "Double-check every step and output your final answer."
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   default=None)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/error_detection")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)

    # Phase 1 video params (aligned with main_infer_tool_uncertainty.py defaults)
    p.add_argument("--fps",            type=float, default=0.2)
    p.add_argument("--min_pixels",     type=int,   default=12544)
    p.add_argument("--frames_upbound", type=int,   default=120)
    p.add_argument("--max_tokens",     type=int,   default=2048)

    # Diverse sampling
    p.add_argument("--n_samples",   type=int,   default=5,
                   help="K for diverse sampling (Phase 1b)")
    p.add_argument("--sample_temp", type=float, default=0.7)
    p.add_argument("--logprobs_k",  type=int,   default=20)

    # Misc
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip inference; load existing JSONL and re-analyse")
    p.add_argument("--skip_align",   action="store_true",
                   help="Skip Phase 1c (U_align) — saves ~1 LLM pass")
    p.add_argument("--skip_flip",    action="store_true",
                   help="Skip Phase 1d (U_flip)  — saves ~1 LLM pass")
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

def _frame_vision_tokens(frame_times):
    return "".join(
        f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
        for i, t in enumerate(frame_times)
    )


def build_notool_input(question, frame_times, frames, processor):
    """Phase 1a/1b: standard video-QA prompt (same as v1)."""
    vision_str   = _frame_vision_tokens(frame_times)
    user_content = (question.replace("<image>", vision_str, 1)
                    if "<image>" in question else vision_str + "\n" + question)
    msgs = [
        {"role": "system", "content": NOTOOL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return {
        "prompt": processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
        "multi_modal_data": {"image": frames},
    }


def _strip_question_visuals(question: str) -> str:
    """Remove <image> and frame tokens from question text for text-only prompts."""
    q = re.sub(r"<image>", "", question)
    q = re.sub(r"<frame\d+_time[\d.]+s>.*?<\|vision_end\|>", "", q)
    q = re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "", q, flags=re.DOTALL)
    return q.strip()


def _extract_reasoning(response_text: str, max_chars: int = 3000) -> str:
    """Extract reasoning content from a model response, stripping answer tags."""
    # Try to extract <think> block first
    m = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    reasoning = m.group(1).strip() if m else response_text

    # Remove any embedded <answer> tags
    reasoning = re.sub(r"<answer>.*?</answer>", "", reasoning, flags=re.DOTALL).strip()

    # Truncate to avoid context overflow
    if len(reasoning) > max_chars:
        reasoning = reasoning[:max_chars] + "\n...[truncated]"
    return reasoning


def build_align_prompt(greedy_response: str, question: str, processor) -> str:
    """
    Phase 1c — Alignment Check (text-only).

    Feed the greedy reasoning back and ask the model:
    "Based ONLY on this reasoning, what is the answer?"
    If the model gives a different answer, U_align = 1 (misaligned).
    """
    reasoning = _extract_reasoning(greedy_response)
    q_text    = _strip_question_visuals(question)

    user_content = (
        f"Question:\n{q_text}\n\n"
        f"A student wrote the following reasoning:\n\n"
        f"{reasoning}\n\n"
        f"Based ONLY on the reasoning above (do NOT re-solve the question yourself), "
        f"what answer does the reasoning reach? "
        f"Output your answer inside <answer> and </answer> tags, e.g. <answer>A</answer>."
    )
    msgs = [
        {"role": "system", "content": ALIGN_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_flip_prompt(greedy_response: str, question: str, processor) -> str:
    """
    Phase 1d — Self-Verification (text-only).

    Show the model its own answer and ask it to double-check.
    If the answer changes, U_flip = 1.
    """
    q_text = _strip_question_visuals(question)

    # Keep the full original response (including answer) but truncate if huge
    original = greedy_response
    if len(original) > 3000:
        original = original[:3000] + "\n...[truncated]"

    msgs = [
        {"role": "system",    "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user",      "content": f"Question:\n{q_text}"},
        {"role": "assistant", "content": original},
        {"role": "user",      "content": (
            "Please carefully re-read the question and your reasoning above. "
            "Double-check every logical step. "
            "Is your final answer correct? "
            "Output your final (possibly revised) answer inside "
            "<answer> and </answer> tags, e.g. <answer>A</answer>."
        )},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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


def score_text(text: str, gt: str) -> float:
    gt_m = re.search(r"<answer>(.*?)</answer>", gt, re.DOTALL)
    gt_c = gt_m.group(1).strip() if gt_m else gt.strip()
    try:
        from verl.utils.reward_score.openr1 import judge_multi_choice
        return float(judge_multi_choice(text, gt_c))
    except Exception:
        pred = extract_mc_answer(text)
        gt_l = extract_mc_answer(gt_c) or gt_c.strip()
        return 1.0 if pred == gt_l else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_greedy_signals(logprobs_list, token_ids) -> dict:
    """Phase 1a → E_bar, U_rough (global-level, from v1)."""
    T = len(token_ids)
    if T == 0 or not logprobs_list:
        return {"E_bar": np.nan, "U_rough": np.nan}

    chosen_lps = np.full(T, np.nan)
    for t, (tok_id, step) in enumerate(zip(token_ids, logprobs_list)):
        if step is None:
            continue
        if tok_id in step:
            chosen_lps[t] = step[tok_id].logprob
        else:
            chosen_lps[t] = min(lp.logprob for lp in step.values())

    valid      = np.isfinite(chosen_lps)
    chosen_lps = np.where(valid, chosen_lps, np.nanmean(chosen_lps))

    E_bar   = float(-np.mean(chosen_lps))
    U_rough = float(np.mean(np.abs(np.diff(chosen_lps)))) if T >= 2 else 0.0
    return {"E_bar": E_bar, "U_rough": U_rough}


def _sample_mean_ll(comp_output) -> float:
    lps = []
    for tok_id, step in zip(comp_output.token_ids, comp_output.logprobs or []):
        if step and tok_id in step:
            lps.append(step[tok_id].logprob)
    return float(np.mean(lps)) if lps else -100.0


def _traj_variance_tfidf(texts: list) -> float:
    """
    U_traj (TF-IDF cosine distance variant).
    Compute mean pairwise cosine DISTANCE among CoT texts.
    Falls back to Jaccard distance if sklearn is unavailable.
    """
    if len(texts) < 2:
        return 0.0

    # Clean: strip <answer> and <think> wrappers, keep reasoning content
    cleaned = []
    for t in texts:
        c = re.sub(r"<answer>.*?</answer>", "", t, flags=re.DOTALL)
        c = re.sub(r"</?think>", "", c)
        cleaned.append(c.strip() or t)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=1)
        X   = vec.fit_transform(cleaned).toarray()
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_n = X / np.maximum(norms, 1e-12)
        # Cosine similarity matrix
        sim_matrix = X_n @ X_n.T
        n = len(texts)
        sims = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
        return float(1.0 - np.mean(sims))  # distance = 1 − similarity
    except ImportError:
        pass

    # Fallback: Jaccard on word sets
    def jaccard_dist(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa and not sb:
            return 0.0
        return 1.0 - len(sa & sb) / len(sa | sb)

    n = len(cleaned)
    dists = [jaccard_dist(cleaned[i], cleaned[j])
             for i in range(n) for j in range(i+1, n)]
    return float(np.mean(dists))


def compute_diverse_signals(comp_outputs) -> dict:
    """
    Phase 1b → U_vote, U_ens, U_gap, U_traj (all global-level).
    """
    k         = len(comp_outputs)
    mean_lls  = np.array([_sample_mean_ll(o) for o in comp_outputs])
    answers   = [extract_mc_answer(o.text) for o in comp_outputs]
    texts     = [o.text for o in comp_outputs]

    # U_vote: fraction of samples that disagree with majority answer
    valid_ans = [a for a in answers if a is not None]
    U_vote    = 1.0 - max(Counter(valid_ans).values()) / k if valid_ans else 1.0

    # U_ens: sample weight entropy (weight ∝ exp(mean_LL))
    w     = np.exp(mean_lls - mean_lls.max())
    w    /= w.sum()
    U_ens = float(-np.sum(w * np.log(w + 1e-12)))

    # U_gap: per-token LL gap between best and second-best sample (↑ = more certain)
    order = np.argsort(mean_lls)[::-1]
    U_gap = float(mean_lls[order[0]] - mean_lls[order[1]]) if k >= 2 else 0.0

    # U_traj: reasoning-path diversity (NEW)
    U_traj = _traj_variance_tfidf(texts)

    return {"U_vote": U_vote, "U_ens": U_ens, "U_gap": U_gap, "U_traj": U_traj}


def compute_u_align(greedy_answer, align_response_text) -> float:
    """
    Phase 1c → U_align.
    1 if alignment check gives a different answer, 0 if consistent.
    """
    if align_response_text is None:
        return np.nan
    align_ans = extract_mc_answer(align_response_text)
    if align_ans is None or greedy_answer is None:
        return np.nan
    return 0.0 if align_ans == greedy_answer else 1.0


def compute_u_flip(greedy_answer, flip_response_text) -> float:
    """
    Phase 1d → U_flip.
    1 if the self-verification produces a different answer, 0 if stable.
    """
    if flip_response_text is None:
        return np.nan
    flip_ans = extract_mc_answer(flip_response_text)
    if flip_ans is None or greedy_answer is None:
        return np.nan
    return 0.0 if flip_ans == greedy_answer else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

SIGNAL_META = {
    # key: (display label, expected direction for error, is_higher_more_uncertain)
    "E_bar":   ("E_bar   −mean(logprob)                   [v1 baseline]", "+"),
    "U_rough": ("U_rough mean|Δlogprob|                   [v1 baseline]", "+"),
    "U_vote":  ("U_vote  answer disagreement  (K diverse) [v1 improved]", "+"),
    "U_ens":   ("U_ens   sample weight entropy             [v1]",          "+"),
    "U_gap":   ("U_gap   LL_best−LL_2nd  ↑=certain        [v1]",          "−"),
    "U_traj":  ("U_traj  CoT pairwise dist (TF-IDF)       [NEW v2]",      "+"),
    "U_align": ("U_align reasoning-answer mismatch         [NEW v2]",      "+"),
    "U_flip":  ("U_flip  self-verify flip rate             [NEW v2]",      "+"),
    "U_combo":    ("U_combo    minmax(vote+traj+align+flip)      [combo v1]", "+"),
    "U_combo_v2": ("U_combo_v2 minmax(vote+traj)               [combo v2]", "+"),
    "U_combo_v3": ("U_combo_v3 rank-pct w.avg(vote+traj)       [combo v3]", "+"),
    "U_combo_v4": ("U_combo_v4 mm(vote)+rank(traj) w.avg       [combo v4]", "+"),
}
SIGNAL_ORDER = ["E_bar", "U_rough", "U_vote", "U_ens", "U_gap",
                "U_traj", "U_align", "U_flip",
                "U_combo", "U_combo_v2", "U_combo_v3", "U_combo_v4"]


def _auc_roc_manual(labels, scores):
    """Compute AUC-ROC without sklearn using the trapezoidal rule."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


def _pearson_manual(x, y):
    x, y = np.array(x, float), np.array(y, float)
    mx, my = x.mean(), y.mean()
    num  = ((x - mx) * (y - my)).sum()
    den  = np.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum())
    return float(num / den) if den > 1e-12 else 0.0


def _spearman_manual(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v)) + 1
        return r
    return _pearson_manual(rank(np.array(x, float)), rank(np.array(y, float)))


def _rankpct(arr):
    """
    Percentile-rank normalisation, NaN-safe.  Output is in [0, 1].
    More robust than min-max: invariant to outliers and monotone transforms.
    Each signal becomes uniformly distributed, making them directly comparable.
    """
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    n = mask.sum()
    if n < 2:
        return out
    order = np.argsort(arr[mask])
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    out[mask] = ranks / (n - 1)   # [0, 1]
    return out


def _add_combo_signal(results):
    """
    Add three combo signals:
      U_combo    – minmax mean of (vote + traj + align + flip)  [original]
      U_combo_v2 – minmax mean of (vote + traj)                 [drop degenerate signals]
      U_combo_v3 – rank-pct weighted sum of (vote + traj)       [improved normalisation]
                   weights proportional to individual (AUC − 0.5):
                     w_vote ≈ 0.56  (AUC 0.647 → excess 0.147)
                     w_traj ≈ 0.44  (AUC 0.616 → excess 0.116)
    """
    def minmax(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        return (arr - lo) / (hi - lo + 1e-12)

    # ── raw arrays ──────────────────────────────────────────────────────── #
    raw = {}
    for k in ["U_vote", "U_traj", "U_align", "U_flip"]:
        raw[k] = np.array([np.nan if r.get(k) is None else float(r[k]) for r in results])

    # ── normalised arrays ────────────────────────────────────────────────── #
    mm   = {k: minmax(v)   for k, v in raw.items()}   # min-max
    rank = {k: _rankpct(v) for k, v in raw.items()}   # percentile rank

    # AUC-proportional weights for vote and traj
    # Excess AUC: vote=0.1474, traj=0.1156  → total=0.2630
    W_VOTE, W_TRAJ = 0.1474 / 0.2630, 0.1156 / 0.2630   # ≈ 0.561, 0.439

    for i, r in enumerate(results):
        # U_combo: original (minmax, equal weights, all 4 signals)
        parts = [mm[k][i] for k in ["U_vote", "U_traj", "U_align", "U_flip"]
                 if np.isfinite(mm[k][i])]
        r["U_combo"] = float(np.mean(parts)) if parts else np.nan

        # U_combo_v2: minmax, equal weights, vote+traj only
        parts_v2 = [mm[k][i] for k in ["U_vote", "U_traj"] if np.isfinite(mm[k][i])]
        r["U_combo_v2"] = float(np.mean(parts_v2)) if parts_v2 else np.nan

        # U_combo_v3: rank-pct both, AUC-proportional weights, vote+traj only
        rv, rt = rank["U_vote"][i], rank["U_traj"][i]
        if np.isfinite(rv) and np.isfinite(rt):
            r["U_combo_v3"] = float(W_VOTE * rv + W_TRAJ * rt)
        elif np.isfinite(rv):
            r["U_combo_v3"] = float(rv)
        elif np.isfinite(rt):
            r["U_combo_v3"] = float(rt)
        else:
            r["U_combo_v3"] = np.nan

        # U_combo_v4: minmax(U_vote) + rank-pct(U_traj), AUC-proportional weights
        # U_vote is discrete → preserve its natural step structure via minmax
        # U_traj is continuous → rank-pct removes its right-skew
        mv, rt4 = mm["U_vote"][i], rank["U_traj"][i]
        if np.isfinite(mv) and np.isfinite(rt4):
            r["U_combo_v4"] = float(W_VOTE * mv + W_TRAJ * rt4)
        elif np.isfinite(mv):
            r["U_combo_v4"] = float(mv)
        elif np.isfinite(rt4):
            r["U_combo_v4"] = float(rt4)
        else:
            r["U_combo_v4"] = np.nan

    return results


def analyze(results: list, output_dir: str):
    results = _add_combo_signal(results)

    # error = 1 if notool wrong (binary per sample)
    valid  = [r for r in results if r.get("acc_notool") is not None]
    errors = np.array([1.0 - float(r["acc_notool"]) for r in valid])
    n      = len(valid)
    err_rate = errors.mean()

    try:
        from scipy import stats as ss
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    try:
        from sklearn.metrics import roc_auc_score as _auc_fn
        def auc_fn(labels, scores): return _auc_fn(labels, scores)
        HAS_SKLEARN = True
    except ImportError:
        def auc_fn(labels, scores): return _auc_roc_manual(labels, scores)
        HAS_SKLEARN = False

    print(f"\n{'='*82}")
    print(f"FIRST-ERROR DETECTION  (n={n},  error_rate={err_rate:.3f})")
    print(f"{'='*82}")
    hdr = (f"{'Signal':<52} {'Exp':>3}  {'Pearson':>8}  "
           f"{'Spearman':>9}  {'p-val':>7}  {'AUC-ROC':>7}")
    print(hdr)
    print("-" * 82)

    rows = []
    for sig in SIGNAL_ORDER:
        if sig not in SIGNAL_META:
            continue
        label, exp = SIGNAL_META[sig]
        vals = np.array([np.nan if r.get(sig) is None else float(r[sig]) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(errors)
        n_ok = ok.sum()
        if n_ok < 5:
            print(f"{label:<52} {exp:>3}  {'—':>8}  {'—':>9}  {'—':>7}  {'—':>7}  (n={n_ok})")
            continue

        v, e = vals[ok], errors[ok]

        if HAS_SCIPY:
            pr, pp = ss.pearsonr(v, e)
            sr, sp = ss.spearmanr(v, e)
        else:
            pr, pp = _pearson_manual(v, e), np.nan
            sr, sp = _spearman_manual(v, e), np.nan

        # AUC: higher signal → predicts error=1; flip if exp="−"
        score_for_auc = -v if exp == "−" else v
        try:
            auc = auc_fn(e.astype(int), score_for_auc)
        except Exception:
            auc = np.nan

        sig_mark = "*" if (not np.isnan(sp) and sp < 0.05) else " "
        ok_sign  = "✓" if (exp == "+" and sr > 0) or (exp == "−" and sr < 0) else "✗"
        auc_str  = f"{auc:.4f}" if not np.isnan(auc) else "   n/a"
        p_str    = f"{sp:.4f}" if not np.isnan(sp) else "   n/a"

        print(f"{label:<52} {exp:>3}  {pr:>+8.4f}  {sr:>+8.4f}{sig_mark}  {p_str}  {auc_str}  {ok_sign}")
        rows.append(dict(
            signal=sig, n=int(n_ok),
            pearson=float(pr),
            pearson_p=float(pp) if not np.isnan(pp) else None,
            spearman=float(sr),
            spearman_p=float(sp) if not np.isnan(sp) else None,
            auc=float(auc) if not np.isnan(auc) else None,
            expected=exp,
        ))

    print(f"{'='*82}")
    print("* Spearman p<0.05   ✓/✗ = expected direction matched/missed")

    # ── Binned analysis ────────────────────────────────────────────────────── #
    print(f"\n{'─'*82}")
    print("BINNED ANALYSIS  (signal quintiles → mean error rate)")
    print(f"{'─'*82}")
    print(f"{'Signal':<15}  Q1(low)  Q2       Q3       Q4       Q5(high)  trend  Δ(Q5−Q1)")
    print("-" * 82)
    for sig in SIGNAL_ORDER:
        vals = np.array([np.nan if r.get(sig) is None else float(r[sig]) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(errors)
        if ok.sum() < 20:
            continue
        v, e  = vals[ok], errors[ok]
        q     = np.percentile(v, [0, 20, 40, 60, 80, 100])
        bms   = []
        for lo, hi in zip(q[:-1], q[1:]):
            mask = (v >= lo) & (v <= hi)
            bms.append(e[mask].mean() if mask.sum() > 0 else np.nan)
        trend    = "↑" if bms[-1] > bms[0] else "↓"
        delta    = bms[-1] - bms[0]
        bms_str  = "  ".join(f"{x:.4f}" for x in bms)
        print(f"{sig:<15}  {bms_str}  {trend}  {delta:+.4f}")

    # ── Outcome group analysis ─────────────────────────────────────────────── #
    print(f"\n{'─'*82}")
    print("SIGNAL MEAN BY OUTCOME GROUP")
    print(f"{'─'*82}")
    correct = [r for r in valid if float(r.get("acc_notool", 0)) >= 1.0]
    wrong   = [r for r in valid if float(r.get("acc_notool", 1)) < 1.0]
    print(f"{'Signal':<15}  {'Correct (n='+str(len(correct))+')':>18}  "
          f"{'Wrong (n='+str(len(wrong))+')':>16}  {'Δ(wrong−correct)':>18}")
    print("-" * 70)
    for sig in SIGNAL_ORDER:
        c_vals = [float(r.get(sig, np.nan)) for r in correct if np.isfinite(r.get(sig, np.nan) or np.nan)]
        w_vals = [float(r.get(sig, np.nan)) for r in wrong   if np.isfinite(r.get(sig, np.nan) or np.nan)]
        if not c_vals or not w_vals:
            continue
        c_mean, w_mean = np.mean(c_vals), np.mean(w_vals)
        delta = w_mean - c_mean
        print(f"{sig:<15}  {c_mean:>18.4f}  {w_mean:>16.4f}  {delta:>+18.4f}")

    # ── Save ──────────────────────────────────────────────────────────────── #
    out_path = os.path.join(output_dir, "error_detection_analysis.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved analysis: {out_path}")

    # Also save results with U_combo back to JSONL for downstream use
    combo_path = os.path.join(output_dir, "results_error_detection_with_combo.jsonl")
    with open(combo_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved results : {combo_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_error_detection.jsonl")

    # ── Analysis-only mode ──────────────────────────────────────────────── #
    if args.analyze_only:
        print(f"[analysis] Loading {output_jsonl}")
        with open(output_jsonl) as f:
            results = [json.loads(l) for l in f if l.strip()]
        print(f"[analysis] {len(results)} records")
        analyze(results, args.output_dir)
        return

    if not args.data_path:
        raise ValueError("--data_path required unless --analyze_only is set")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    print(f"[vllm]  Initialising (tp={args.tensor_parallel_size}, "
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

    # Sampling params
    greedy_sp  = SamplingParams(n=1, max_tokens=args.max_tokens,
                                temperature=0.0, logprobs=args.logprobs_k,
                                detokenize=True)
    diverse_sp = SamplingParams(n=args.n_samples, max_tokens=args.max_tokens,
                                temperature=args.sample_temp, logprobs=1,
                                detokenize=True)
    # Phases 1c & 1d: short text-only check, just need the answer letter
    check_sp   = SamplingParams(n=1, max_tokens=64, temperature=0.0,
                                detokenize=True)

    results = []
    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            # ── Prepare video inputs ────────────────────────────────────── #
            notool_inputs = []
            valid_samples = []
            for sample in batch:
                try:
                    vp = resolve_video_path(sample["videos"][0], args.video_root)
                    ft, fr = load_video_frames(
                        vp, args.fps, args.max_pixels,
                        args.min_pixels, args.frames_upbound)
                    notool_inputs.append(
                        build_notool_input(sample["problem"], ft, fr, processor))
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id','?')}: {e}")

            if not valid_samples:
                continue

            # ── Phase 1a: greedy (logprobs) ──────────────────────────────── #
            greedy_outs = llm.generate(notool_inputs, greedy_sp)

            # ── Phase 1b: diverse (K samples) ────────────────────────────── #
            diverse_outs = llm.generate(notool_inputs, diverse_sp)

            # ── Phase 1c: alignment check (text-only) ───────────────────── #
            align_outs = None
            if not args.skip_align:
                align_prompts = [
                    build_align_prompt(g.outputs[0].text, s["problem"], processor)
                    for s, g in zip(valid_samples, greedy_outs)
                ]
                # Plain strings → no multi_modal_data needed
                align_outs = llm.generate(align_prompts, check_sp)

            # ── Phase 1d: self-verification (text-only) ──────────────────── #
            flip_outs = None
            if not args.skip_flip:
                flip_prompts = [
                    build_flip_prompt(g.outputs[0].text, s["problem"], processor)
                    for s, g in zip(valid_samples, greedy_outs)
                ]
                flip_outs = llm.generate(flip_prompts, check_sp)

            # ── Combine results ───────────────────────────────────────────── #
            for i, (sample, g_out, d_out) in enumerate(
                    zip(valid_samples, greedy_outs, diverse_outs)):
                g  = g_out.outputs[0]
                gt = sample.get("solution", "")

                greedy_answer = extract_mc_answer(g.text)
                acc_notool    = score_text(g.text, gt)

                # Phase 1a signals
                sig_greedy = compute_greedy_signals(
                    g.logprobs, list(g.token_ids))

                # Phase 1b signals
                sig_diverse = compute_diverse_signals(d_out.outputs)

                # Phase 1c
                u_align = compute_u_align(
                    greedy_answer,
                    align_outs[i].outputs[0].text if align_outs else None)

                # Phase 1d
                u_flip = compute_u_flip(
                    greedy_answer,
                    flip_outs[i].outputs[0].text if flip_outs else None)

                record = {
                    "problem_id":     sample.get("problem_id"),
                    "data_source":    sample.get("data_source"),
                    "acc_notool":     float(acc_notool),
                    "greedy_answer":  greedy_answer,
                    "delta_s":        sample.get("delta_s"),
                    # greedy text (for debugging)
                    "response_notool": g.text,
                    # Phase 1c/1d responses (for debugging)
                    "response_align": (align_outs[i].outputs[0].text
                                       if align_outs else None),
                    "response_flip":  (flip_outs[i].outputs[0].text
                                       if flip_outs else None),
                    # ── Signals ──
                    **sig_greedy,
                    **sig_diverse,
                    "U_align": float(u_align) if not np.isnan(u_align) else None,
                    "U_flip":  float(u_flip)  if not np.isnan(u_flip)  else None,
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nResults JSONL: {output_jsonl}  ({len(results)} samples)")
    analyze(results, args.output_dir)


if __name__ == "__main__":
    main()
