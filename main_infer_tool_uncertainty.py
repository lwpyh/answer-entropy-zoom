#!/usr/bin/env python3
"""
main_infer_tool_uncertainty.py

For each sample this script runs THREE inference passes:

  Phase 1a  no-tool greedy  (logprobs=k)  → acc_notool  +  Cat-1 / Cat-3 metrics
  Phase 1b  no-tool diverse (n=4, T=0.7)  →                 Cat-2 / Cat-3 metrics
  Phase 2   tool multi-turn               → acc_tool_final, n_tool_calls, n_rounds

Analysis produced:
  (A) Global accuracy table        acc_notool | acc_tool_final | Δacc
  (B) Breakdown by n_tool_calls    accuracy per zoom-call group
  (C) Corr: 9 metrics × Δacc      does uncertainty predict tool benefit?
  (D) Corr: 9 metrics × delta_s   comparison with ground-truth Δs
  (E) Metric dist by n_tool_calls  do uncertain samples zoom more?
  (F) Δacc–delta_s agreement       is our measured Δacc consistent with labelled Δs?
  (G) Binned Δacc analysis         uncertainty quintiles → mean Δacc

───────────────────────────────────────────────────────────────────────────────
Usage
-----
  # Full run:
  python main_infer_tool_uncertainty.py \\
      --data_path  .../eval_deltaS.yaml \\
      --video_root /data/DERI-Gong/jh015/VideoZoomer \\
      --output_dir ./infer_results/tool_uncertainty

  # Analysis only on existing JSONL:
  python main_infer_tool_uncertainty.py --analyze_only \\
      --output_dir ./infer_results/tool_uncertainty
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════════════════

NOTOOL_SYSTEM_PROMPT = "You are a helpful assistant."

TOOL_SYSTEM_PROMPT = (
    'You are a helpful assistant. You will receive a low-frame-rate video and '
    'related questions. You can analyze the video content to answer the question '
    'and trigger high-frame-rate inspections when finer temporal resolution is '
    'needed. When you detect ambiguous motion/objects that require closer '
    'inspection, wrap your request in <video_zoom></video_zoom> tags and provide '
    'the exact time segment and target frame rate in JSON format: '
    '<video_zoom> {"segment": [start_sec, end_sec], "fps": n} </video_zoom>, '
    'it will return the video clip at the target fps to help you better answer '
    'the question. Note that the total frames num of the request clip cannot '
    'exceed 16 (e.g., (end_sec - start_sec) * fps \u2264 16) and DO NOT include '
    '<answer> tags in this round. \n'
    ' Example usage: <video_zoom> {"segment": [4.0, 6.0], "fps": 2} </video_zoom>. '
    'If the initial tool response does not provide sufficient information to '
    'answer the question, you may continue to request additional video zoom '
    'inspections as needed, until you either (1) gather enough information to '
    'form a complete answer, or (2) are explicitly instructed to stop using the '
    'tool. Output the thinking process within <think> </think> tags, once you '
    'confirm your final answer, place the final answer in \\boxed{} inside '
    '<answer> and </answer>.'
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   default=None)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/tool_uncertainty")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)

    # Phase 1 (no-tool) video – aligned with eval_deltaS_infer.sh
    p.add_argument("--notool_fps",            type=float, default=0.2)
    p.add_argument("--notool_min_pixels",     type=int,   default=12544)
    p.add_argument("--notool_frames_upbound", type=int,   default=120)
    p.add_argument("--notool_max_tokens",     type=int,   default=2048)

    # Phase 2 (tool) video – aligned with eval_deltaS_infer_tool.sh
    p.add_argument("--tool_fps",              type=float, default=0.5)
    p.add_argument("--tool_min_pixels",       type=int,   default=25088)
    p.add_argument("--tool_frames_upbound",   type=int,   default=64)
    p.add_argument("--tool_max_tokens",       type=int,   default=4096)
    p.add_argument("--tool_limit_mm",         type=int,   default=128,
                   help="Max images per prompt across all tool-use turns")
    p.add_argument("--tool_max_rounds",       type=int,   default=5)
    p.add_argument("--tool_max_frames_per_call", type=int, default=16)
    p.add_argument("--tool_workers",          type=int,   default=8)

    # Uncertainty sampling
    p.add_argument("--n_samples",   type=int,   default=4)
    p.add_argument("--sample_temp", type=float, default=0.7)
    p.add_argument("--logprobs_k",  type=int,   default=20)

    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip inference; load existing JSONL and run analysis only")
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
    raw         = vr.get_batch(indices).asnumpy()

    frames = []
    for arr in raw:
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        h, w = img.height, img.width
        px   = h * w
        if px > max_pixels:
            s   = (max_pixels / px) ** 0.5
            img = img.resize((max(2, int(w*s/2)*2), max(2, int(h*s/2)*2)), Image.LANCZOS)
        elif px < min_pixels:
            s   = (min_pixels / px) ** 0.5
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
    vision_str   = _frame_vision_tokens(frame_times)
    user_content = (question.replace("<image>", vision_str, 1)
                    if "<image>" in question else vision_str + "\n" + question)
    msgs = [
        {"role": "system", "content": NOTOOL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return {
        "prompt":           processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
        "multi_modal_data": {"image": frames},
    }


def build_tool_initial_prompt(question, frame_times, processor):
    vision_str   = _frame_vision_tokens(frame_times)
    user_content = (question.replace("<image>", vision_str, 1)
                    if "<image>" in question else vision_str + "\n" + question)
    msgs = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_tool_response_turn(frame_times, is_last_round: bool) -> str:
    pad = _frame_vision_tokens(frame_times)
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        "<tool_response>\nThe frames of the video clip are shown below:\n"
        + pad + "\n</tool_response>\n"
        "continue your reasoning process inside <think> and </think> "
        "and then write your final answer inside <answer> and </answer>"
    )
    if is_last_round:
        msg += ("Do not call <video_zoom> in this round, "
                "give a final answer based on information above.")
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


def build_tool_error_turn(error_str: str, is_last_round: bool) -> str:
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        + error_str
        + "\nPlease analyze the error information obtained from the function "
        "tool and adjust your response. "
        "Continue your reasoning process inside <think> and </think>."
    )
    if is_last_round:
        msg += (" Do not call <video_zoom> in this round, "
                "give a final answer based on information above.")
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


_ZOOM_RE = re.compile(r"<video_zoom>(.*?)</video_zoom>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_tool_call(text: str):
    m = _ZOOM_RE.search(text)
    if not m:
        return None
    jsons = _JSON_RE.findall(m.group(1))
    if not jsons:
        return None
    try:
        obj = json.loads(jsons[0])
        return float(obj["segment"][0]), float(obj["segment"][1]), float(obj["fps"])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mc_answer(text: str):
    m        = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    fragment = m.group(1).strip() if m else text
    matches  = re.findall(r"([A-Z])\.", fragment)
    if matches:
        return matches[-1]
    if fragment and fragment[0] in "ABCD":
        return fragment[0]
    matches = re.findall(r"([A-Z])\.", text)
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


def score_responses(all_responses, gt: str) -> float:
    return score_text(" ".join(all_responses), gt)


# ═══════════════════════════════════════════════════════════════════════════════
# Uncertainty metrics  (same as main_infer_uncertainty.py)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory_metrics(logprobs_list, token_ids, response_text: str) -> dict:
    T = len(token_ids)
    if T == 0 or not logprobs_list:
        return {"U_rough": np.nan, "U_dH": np.nan, "U_ans_stab": np.nan, "E_bar": np.nan}

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

    entropies = np.zeros(T)
    for t, step in enumerate(logprobs_list):
        if step is None:
            continue
        lps   = np.array([lp.logprob for lp in step.values()])
        probs = np.exp(lps - lps.max())
        probs /= probs.sum()
        entropies[t] = -float(np.sum(probs * np.log(probs + 1e-12)))

    U_rough = float(np.mean(np.abs(np.diff(chosen_lps)))) if T >= 2 else 0.0
    U_dH    = float(np.mean(np.abs(np.diff(entropies))))  if T >= 2 else 0.0

    ans_s = response_text.find("<answer>")
    ans_e = response_text.find("</answer>")
    if ans_s != -1 and ans_e != -1:
        tc  = max(len(response_text), 1)
        ts  = max(0, int(ans_s / tc * T))
        te  = min(T, int(ans_e / tc * T) + 1)
        alp = chosen_lps[ts:te]
    else:
        alp = chosen_lps[max(0, int(0.8 * T)):]
    U_ans_stab = float(np.mean(np.abs(np.diff(alp)))) if len(alp) >= 2 else U_rough

    E_bar = float(-np.mean(chosen_lps))
    return {"U_rough": U_rough, "U_dH": U_dH, "U_ans_stab": U_ans_stab, "E_bar": E_bar}


def _sample_mean_ll(comp_output) -> float:
    lps = []
    for tok_id, step in zip(comp_output.token_ids, comp_output.logprobs or []):
        if step and tok_id in step:
            lps.append(step[tok_id].logprob)
    return float(np.mean(lps)) if lps else -100.0


def compute_diversity_metrics(comp_outputs) -> dict:
    k        = len(comp_outputs)
    mean_lls = np.array([_sample_mean_ll(o) for o in comp_outputs])
    answers  = [extract_mc_answer(o.text) for o in comp_outputs]

    order    = np.argsort(mean_lls)[::-1]
    sorted_l = mean_lls[order]
    U_gap    = float(sorted_l[0] - sorted_l[1]) if k >= 2 else 0.0

    w     = np.exp(mean_lls - mean_lls.max())
    w    /= w.sum()
    U_ens = float(-np.sum(w * np.log(w + 1e-12)))

    valid_ans = [a for a in answers if a is not None]
    U_vote    = 1.0 - max(Counter(valid_ans).values()) / k if valid_ans else 1.0
    dE        = -U_gap
    N_eff     = float(np.exp(U_ens))
    return {"U_gap": U_gap, "U_ens": U_ens, "U_vote": U_vote, "dE": dE, "N_eff": N_eff}


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_INFO = {
    "U_rough":    ("1. Logprob roughness  mean|Δℓ|",                   "+"),
    "U_dH":       ("2. Entropy roughness  mean|ΔH|",                   "+"),
    "U_ans_stab": ("3. Answer-region logprob roughness",                "+"),
    "E_bar":      ("7. Seq-level energy  −mean(ℓ)         [Cat 3]",    "+"),
    "U_gap":      ("4. Best-vs-second gap  L1−L2 (↑=cert) [Cat 2]",   "−"),
    "U_ens":      ("5. Sample weight entropy  H(w)         [Cat 2]",   "+"),
    "U_vote":     ("6. Answer vote disagreement            [Cat 2]",   "+"),
    "dE":         ("8. Energy gap  E2−E1  (↓=cert)        [Cat 3]",   "−"),
    "N_eff":      ("9. Effective candidates  exp(H(w))     [Cat 3]",   "+"),
}
METRIC_ORDER = ["U_rough", "U_dH", "U_ans_stab", "E_bar",
                "U_gap", "U_ens", "U_vote", "dE", "N_eff"]


def _corr_table(results, target_key, target_label, output_dir, fname):
    """Correlation table: 9 metrics × target_key."""
    valid = [r for r in results if r.get(target_key) is not None]
    if not valid:
        print(f"[analysis] No '{target_key}' values — skipping.")
        return []

    try:
        from scipy import stats as ss
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    ds = np.array([float(r[target_key]) for r in valid])
    print(f"\n{'='*78}")
    print(f"CORRELATION WITH {target_label}   "
          f"(n={len(valid)}, mean={ds.mean():.3f}, std={ds.std():.3f})")
    print(f"{'='*78}")
    print(f"{'Metric':<50} {'Exp':>3}  {'Pearson r':>9}  {'Spearman ρ':>10}  {'p-val':>7}")
    print("-" * 78)

    rows = []
    for m in METRIC_ORDER:
        label, exp = METRIC_INFO[m]
        vals = np.array([float(r.get(m, np.nan)) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(ds)
        if ok.sum() < 5:
            continue
        v, d = vals[ok], ds[ok]
        if HAS_SCIPY:
            pr, pp = ss.pearsonr(v, d)
            sr, sp = ss.spearmanr(v, d)
        else:
            pr = float(np.corrcoef(v, d)[0, 1]); pp = np.nan
            rv = np.argsort(np.argsort(v)).astype(float)
            rd = np.argsort(np.argsort(d)).astype(float)
            sr = float(np.corrcoef(rv, rd)[0, 1]); sp = np.nan

        sig     = "*" if (not np.isnan(sp) and sp < 0.05) else " "
        ok_sign = "✓" if (exp == "+" and sr > 0) or (exp == "−" and sr < 0) else "✗"
        print(f"{label:<50} {exp:>3}  {pr:>+9.4f}  {sr:>+9.4f}{sig}   {sp:>7.4f}  {ok_sign}")
        rows.append(dict(metric=m, target=target_key,
                         pearson=float(pr), pearson_p=float(pp) if not np.isnan(pp) else None,
                         spearman=float(sr), spearman_p=float(sp) if not np.isnan(sp) else None,
                         expected=exp))

    print(f"{'='*78}")
    print("* Spearman p<0.05   ✓/✗ = expected direction matched/missed")

    if fname:
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved: {path}")
    return rows


def analyze(results: list, output_dir: str):
    # ── (A) Global accuracy ──────────────────────────────────────────────── #
    notool_accs = [r["acc_notool"]    for r in results if "acc_notool"    in r]
    tool_accs   = [r["acc_tool_final"] for r in results if "acc_tool_final" in r]
    delta_accs  = [r["delta_acc"]     for r in results if "delta_acc"     in r]

    print(f"\n{'='*60}")
    print("PHASE ACCURACY COMPARISON")
    print(f"{'='*60}")
    print(f"  No-tool  (Phase 1)  : {np.mean(notool_accs):.4f}  (n={len(notool_accs)})")
    print(f"  Tool final (Phase 2): {np.mean(tool_accs):.4f}  (n={len(tool_accs)})")
    print(f"  Δacc (tool − notool): {np.mean(delta_accs):+.4f}")
    print(f"{'='*60}")

    # Distribution of delta_acc
    n_improved  = sum(1 for d in delta_accs if d > 0)
    n_same      = sum(1 for d in delta_accs if d == 0)
    n_worsened  = sum(1 for d in delta_accs if d < 0)
    total       = len(delta_accs)
    print(f"\n  Tool helped  (Δacc>0): {n_improved:>3} / {total}  ({100*n_improved/total:.1f}%)")
    print(f"  No change   (Δacc=0): {n_same:>3} / {total}  ({100*n_same/total:.1f}%)")
    print(f"  Tool hurt   (Δacc<0): {n_worsened:>3} / {total}  ({100*n_worsened/total:.1f}%)")

    # ── (A') Did <video_zoom> actually help? binary zoom vs no-zoom ─────── #
    zoomed     = [r for r in results if r.get("n_tool_calls", 0) >= 1
                  and "acc_notool" in r and "acc_tool_final" in r]
    not_zoomed = [r for r in results if r.get("n_tool_calls", 0) == 0
                  and "acc_notool" in r and "acc_tool_final" in r]

    print(f"\n{'='*65}")
    print("DID <video_zoom> HELP?  (binary zoomed vs not-zoomed)")
    print(f"{'='*65}")
    print(f"  {'Group':<20}  {'n':>5}  {'acc_notool':>10}  {'acc_tool':>8}  {'Δacc':>7}")
    print(f"  {'-'*55}")
    for label, grp in [("zoomed (≥1 call)", zoomed), ("not zoomed (0 calls)", not_zoomed)]:
        if not grp:
            continue
        an = np.mean([r["acc_notool"]     for r in grp])
        at = np.mean([r["acc_tool_final"] for r in grp])
        da = np.mean([r["delta_acc"]      for r in grp])
        pct_improved = 100 * sum(1 for r in grp if r["delta_acc"] > 0) / len(grp)
        pct_hurt     = 100 * sum(1 for r in grp if r["delta_acc"] < 0) / len(grp)
        print(f"  {label:<20}  {len(grp):>5}  {an:>10.4f}  {at:>8.4f}  {da:>+7.4f}")
        print(f"  {'':20}  {'':5}  helped={pct_improved:.1f}%  hurt={pct_hurt:.1f}%")
    zoom_rate = 100 * len(zoomed) / (len(zoomed) + len(not_zoomed)) if (zoomed or not_zoomed) else 0
    print(f"\n  Zoom rate: {zoom_rate:.1f}%  ({len(zoomed)} / {len(zoomed)+len(not_zoomed)} samples called tool)")
    print(f"{'='*65}")

    # ── (B) Breakdown by n_tool_calls ───────────────────────────────────── #
    by_calls: dict = defaultdict(list)
    for r in results:
        if "n_tool_calls" in r and "acc_notool" in r and "acc_tool_final" in r:
            k = min(r["n_tool_calls"], 4)   # group 4+ together
            by_calls[k].append(r)

    print(f"\n{'='*65}")
    print("BREAKDOWN BY N_TOOL_CALLS  (each group vs. no-tool baseline)")
    print(f"{'='*65}")
    print(f"{'n_calls':>8}  {'n':>5}  {'acc_notool':>10}  {'acc_tool':>8}  "
          f"{'Δacc':>7}  {'%improve':>9}")
    print("-" * 55)
    for k in sorted(by_calls.keys()):
        grp    = by_calls[k]
        an     = np.mean([r["acc_notool"]    for r in grp])
        at     = np.mean([r["acc_tool_final"] for r in grp])
        da     = np.mean([r["delta_acc"]     for r in grp])
        pct    = 100 * sum(1 for r in grp if r["delta_acc"] > 0) / len(grp)
        label  = f"{k}+" if k == 4 else str(k)
        print(f"{label:>8}  {len(grp):>5}  {an:>10.4f}  {at:>8.4f}  "
              f"{da:>+7.4f}  {pct:>8.1f}%")
    print(f"{'='*65}")

    # ── (C) Correlation: 9 metrics × delta_acc ──────────────────────────── #
    _corr_table(results, "delta_acc", "Δacc (tool−notool)",
                output_dir, "corr_metrics_vs_delta_acc.json")

    # Binned: uncertainty quintiles → mean Δacc
    valid_da  = [r for r in results if r.get("delta_acc") is not None]
    da_arr    = np.array([float(r["delta_acc"]) for r in valid_da])
    print(f"\n{'─'*78}")
    print("BINNED ANALYSIS  (uncertainty quintiles → mean Δacc)")
    print(f"{'─'*78}")
    print(f"{'Metric':<25}  Q1(low)   Q2        Q3        Q4        Q5(high)  trend")
    print("-" * 78)
    for m in METRIC_ORDER:
        vals = np.array([float(r.get(m, np.nan)) for r in valid_da])
        ok   = np.isfinite(vals) & np.isfinite(da_arr)
        if ok.sum() < 20:
            continue
        v, d     = vals[ok], da_arr[ok]
        q        = np.percentile(v, [0, 20, 40, 60, 80, 100])
        bms      = []
        for lo, hi in zip(q[:-1], q[1:]):
            mask = (v >= lo) & (v <= hi)
            bms.append(d[mask].mean() if mask.sum() > 0 else np.nan)
        trend    = "↑" if bms[-1] > bms[0] else "↓"
        vals_str = "  ".join(f"{x:+.3f}" for x in bms)
        print(f"{m:<25}  {vals_str}  {trend}")

    # ── (D) Correlation: 9 metrics × delta_s ────────────────────────────── #
    _corr_table(results, "delta_s", "delta_s (ground truth)",
                output_dir, "corr_metrics_vs_delta_s.json")

    # ── (E) Metric distributions by n_tool_calls ────────────────────────── #
    group_keys = sorted(by_calls.keys())
    print(f"\n{'─'*78}")
    print("UNCERTAINTY METRICS BY N_TOOL_CALLS  (mean per group)")
    print(f"{'─'*78}")
    header = f"{'Metric':<15}" + "".join(f"  {'≥4' if k==4 else str(k)+' calls':>10}" for k in group_keys)
    print(header)
    print("-" * (15 + 12 * len(group_keys)))
    for m in METRIC_ORDER:
        row = f"{m:<15}"
        for k in group_keys:
            vals = [float(r[m]) for r in by_calls[k] if r.get(m) is not None and np.isfinite(r.get(m, np.nan))]
            row += f"  {np.mean(vals):>10.4f}" if vals else f"  {'n/a':>10}"
        print(row)

    # ── (H) Multi-round accuracy flow ───────────────────────────────────── #
    multi = [r for r in results
             if r.get("per_round_scores") and r.get("acc_notool") is not None]
    if multi:
        print(f"\n{'='*78}")
        print("MULTI-ROUND ACCURACY FLOW  (notool baseline → each round → final)")
        print(f"{'='*78}")

        # Table: one row per distinct n_rounds value
        max_rounds = max(len(r["per_round_scores"]) for r in multi)
        hdr = f"  {'n_rounds':>8}  {'n':>5}  {'notool':>7}"
        for i in range(max_rounds):
            hdr += f"  {'r'+str(i+1):>7}"
        hdr += f"  {'final':>7}  {'Δacc':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        by_nr = defaultdict(list)
        for r in multi:
            by_nr[len(r["per_round_scores"])].append(r)

        for nr in sorted(by_nr):
            grp = by_nr[nr]
            nt  = np.mean([r["acc_notool"]    for r in grp])
            fi  = np.mean([r["acc_tool_final"] for r in grp])
            da  = fi - nt
            row = f"  {nr:>8}  {len(grp):>5}  {nt:>7.4f}"
            for i in range(max_rounds):
                scores = [r["per_round_scores"][i] for r in grp
                          if i < len(r["per_round_scores"])]
                row += f"  {np.mean(scores):>7.4f}" if scores else f"  {'—':>7}"
            row += f"  {fi:>7.4f}  {da:>+7.4f}"
            print(row)

        # State-flow: how samples move between correct(1) and wrong(0)
        print(f"\n  STATE FLOW  notool → final  (by n_rounds)")
        print(f"  {'n_rounds':>8}  {'0→0':>6}  {'0→1 ✓fix':>9}  {'1→0 ✗brk':>9}  {'1→1':>6}")
        print("  " + "-" * 46)
        for nr in sorted(by_nr):
            grp = by_nr[nr]
            c   = Counter((int(r["acc_notool"]), int(r["acc_tool_final"])) for r in grp)
            n   = len(grp)
            print(f"  {nr:>8}  "
                  f"{c[(0,0)]:>4} ({100*c[(0,0)]/n:4.1f}%)  "
                  f"{c[(0,1)]:>4} ({100*c[(0,1)]/n:4.1f}%)  "
                  f"{c[(1,0)]:>4} ({100*c[(1,0)]/n:4.1f}%)  "
                  f"{c[(1,1)]:>4} ({100*c[(1,1)]/n:4.1f}%)")

        # For samples with ≥2 tool calls: did needing more rounds correlate with
        # harder questions (lower notool) or with more benefit?
        multi_tool = [r for r in multi if len(r["per_round_scores"]) >= 3]
        single_tool = [r for r in multi if len(r["per_round_scores"]) == 2]
        if multi_tool and single_tool:
            print(f"\n  SINGLE vs MULTI tool-call comparison")
            print(f"  {'group':<22}  {'n':>5}  {'notool':>7}  {'final':>7}  {'Δacc':>7}  {'fix%':>6}  {'brk%':>6}")
            print("  " + "-" * 65)
            for label, grp in [("1 tool call (2 rounds)", single_tool),
                                ("2+ tool calls (3+ rounds)", multi_tool)]:
                nt  = np.mean([r["acc_notool"]    for r in grp])
                fi  = np.mean([r["acc_tool_final"] for r in grp])
                fix = 100 * sum(1 for r in grp if r["acc_notool"] == 0 and r["acc_tool_final"] == 1) / len(grp)
                brk = 100 * sum(1 for r in grp if r["acc_notool"] == 1 and r["acc_tool_final"] == 0) / len(grp)
                print(f"  {label:<22}  {len(grp):>5}  {nt:>7.4f}  {fi:>7.4f}  {fi-nt:>+7.4f}  {fix:>5.1f}%  {brk:>5.1f}%")

        print(f"{'='*78}")

    # ── (F) Δacc – delta_s agreement ────────────────────────────────────── #
    paired = [(r["delta_acc"], float(r["delta_s"]))
              for r in results
              if r.get("delta_acc") is not None and r.get("delta_s") is not None]
    if paired:
        da_arr2, ds_arr = zip(*paired)
        da_arr2, ds_arr = np.array(da_arr2), np.array(ds_arr)
        try:
            from scipy import stats as ss
            pr, pp = ss.pearsonr(da_arr2, ds_arr)
            sr, sp = ss.spearmanr(da_arr2, ds_arr)
            print(f"\n{'─'*78}")
            print(f"Δacc ↔ delta_s agreement  (n={len(paired)}):")
            print(f"  Pearson r  = {pr:+.4f}  (p={pp:.4f})")
            print(f"  Spearman ρ = {sr:+.4f}  (p={sp:.4f})")
            # Confusion-style table
            agree = sum(1 for d, s in paired if (d > 0 and s > 0) or (d <= 0 and s <= 0))
            print(f"  Sign agree = {agree}/{len(paired)} = {100*agree/len(paired):.1f}%  "
                  f"(Δacc and Δs both positive or both non-positive)")
        except Exception:
            pass

    # ── Save combined results ────────────────────────────────────────────── #
    corr_path = os.path.join(output_dir, "analysis_summary.json")
    summary   = {
        "n":             len(results),
        "acc_notool":    float(np.mean(notool_accs)) if notool_accs else None,
        "acc_tool_final": float(np.mean(tool_accs))  if tool_accs   else None,
        "delta_acc_mean": float(np.mean(delta_accs)) if delta_accs  else None,
        "n_improved":    n_improved,
        "n_same":        n_same,
        "n_worsened":    n_worsened,
        "by_n_tool_calls": {
            str(k): {
                "n":           len(grp),
                "acc_notool":  float(np.mean([r["acc_notool"]    for r in grp])),
                "acc_tool":    float(np.mean([r["acc_tool_final"] for r in grp])),
                "delta_acc":   float(np.mean([r["delta_acc"]     for r in grp])),
            }
            for k, grp in by_calls.items()
        },
    }
    with open(corr_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAnalysis summary saved to {corr_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_tool_uncertainty.jsonl")

    # ── Analysis-only mode ──────────────────────────────────────────────── #
    if args.analyze_only:
        print(f"[analysis] Loading {output_jsonl}")
        with open(output_jsonl) as f:
            results = [json.loads(l) for l in f if l.strip()]
        analyze(results, args.output_dir)
        return

    if not args.data_path:
        raise ValueError("--data_path is required unless --analyze_only is set")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    # limit_mm_per_prompt must cover both phases: max(notool_frames, tool_limit_mm)
    max_mm = max(args.notool_frames_upbound, args.tool_limit_mm)
    print(f"[vllm]  Initialising (tp={args.tensor_parallel_size}, "
          f"util={args.gpu_memory_utilization}, limit_mm={max_mm})")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        mm_processor_kwargs={
            "max_pixels": args.max_pixels,
            "min_pixels": args.tool_min_pixels,   # use larger (tool) min so vLLM handles both
        },
        limit_mm_per_prompt={"image": max_mm},
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    # Sampling params
    greedy_sp = SamplingParams(n=1, max_tokens=args.notool_max_tokens,
                               temperature=0.0, logprobs=args.logprobs_k, detokenize=True)
    diverse_sp = SamplingParams(n=args.n_samples, max_tokens=args.notool_max_tokens,
                                temperature=args.sample_temp, logprobs=1, detokenize=True)
    tool_sp    = SamplingParams(n=1, max_tokens=args.tool_max_tokens, temperature=0.0,
                                stop=["</video_zoom>"], include_stop_str_in_output=True,
                                detokenize=True)
    final_sp   = SamplingParams(n=1, max_tokens=args.tool_max_tokens, temperature=0.0,
                                detokenize=True)

    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    results = []
    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            # ── Preprocess both no-tool and tool inputs ────────────────── #
            notool_inputs  = []
            tool_prompts   = []
            tool_images    = []
            valid_samples  = []

            for sample in batch:
                try:
                    vp = resolve_video_path(sample["videos"][0], args.video_root)

                    # Phase 1 video (low fps, more frames)
                    ft1, fr1 = load_video_frames(
                        vp, args.notool_fps, args.max_pixels,
                        args.notool_min_pixels, args.notool_frames_upbound)
                    notool_inputs.append(
                        build_notool_input(sample["problem"], ft1, fr1, processor))

                    # Phase 2 video (tool fps, fewer initial frames)
                    ft2, fr2 = load_video_frames(
                        vp, args.tool_fps, args.max_pixels,
                        args.tool_min_pixels, args.tool_frames_upbound)
                    tool_prompts.append(
                        build_tool_initial_prompt(sample["problem"], ft2, processor))
                    tool_images.append(list(fr2))

                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id','?')}: {e}")

            if not valid_samples:
                continue

            # ── Phase 1a: no-tool greedy (logprobs for uncertainty) ────── #
            greedy_outs  = llm.generate(notool_inputs, greedy_sp)

            # ── Phase 1b: no-tool diverse (neighbourhood metrics) ─────── #
            diverse_outs = llm.generate(notool_inputs, diverse_sp)

            # ── Phase 2: tool-use multi-turn ──────────────────────────── #
            N             = len(valid_samples)
            accum_prompts = list(tool_prompts)
            accum_images  = [list(imgs) for imgs in tool_images]
            all_responses = [[] for _ in range(N)]
            active        = list(range(N))

            for round_idx in range(args.tool_max_rounds):
                if not active:
                    break
                is_last_round = (round_idx == args.tool_max_rounds - 1)
                sp = final_sp if is_last_round else tool_sp

                vllm_inputs = [
                    {"prompt": accum_prompts[i], "multi_modal_data": {"image": accum_images[i]}}
                    for i in active
                ]
                outputs = llm.generate(vllm_inputs, sp)

                tool_queue  = {}
                next_active = []

                for local_j, (i, out) in enumerate(zip(active, outputs)):
                    resp = out.outputs[0].text
                    all_responses[i].append(resp)
                    tc   = None if is_last_round else parse_tool_call(resp)
                    if tc:
                        tool_queue[local_j] = (i, tc)
                        next_active.append(i)
                        accum_prompts[i] += resp

                if tool_queue and not is_last_round:
                    is_next_last = (round_idx == args.tool_max_rounds - 2)
                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {}
                        for local_j, (i, (s_t, e_t, fps_z)) in tool_queue.items():
                            video_path = resolve_video_path(
                                valid_samples[i]["videos"][0], args.video_root)
                            f = ex.submit(
                                extract_video_clip,
                                video_path=video_path,
                                start_time=s_t, end_time=e_t, fps=fps_z,
                                max_pixels=args.max_pixels,
                                min_pixels=args.tool_min_pixels,
                                max_frames=args.tool_max_frames_per_call,
                                storage_system="local",
                            )
                            futures[f] = (local_j, i)

                        for f in as_completed(futures):
                            local_j, i = futures[f]
                            result = f.result()
                            if isinstance(result, dict):
                                accum_prompts[i] += build_tool_response_turn(
                                    result["frame_time"], is_next_last)
                                accum_images[i].extend(result["frames"])
                            else:
                                accum_prompts[i] += build_tool_error_turn(
                                    str(result), is_next_last)

                active = next_active

            # ── Combine and save ───────────────────────────────────────── #
            for i, (sample, g_out, d_out) in enumerate(
                    zip(valid_samples, greedy_outs, diverse_outs)):
                g  = g_out.outputs[0]
                gt = sample.get("solution", "")

                traj  = compute_trajectory_metrics(g.logprobs, list(g.token_ids), g.text)
                divers = compute_diversity_metrics(d_out.outputs)

                acc_notool    = score_text(g.text, gt)
                acc_tool_final = score_responses(all_responses[i], gt)
                n_tool_calls  = sum(1 for r in all_responses[i] if parse_tool_call(r) is not None)

                # Per-round scores: score the concatenation of responses up to each round
                per_round_scores = []
                for r_idx in range(1, len(all_responses[i]) + 1):
                    s = score_responses(all_responses[i][:r_idx], gt)
                    per_round_scores.append(float(s))

                record = {
                    "problem_id":       sample.get("problem_id"),
                    "data_source":      sample.get("data_source"),
                    "problem":          sample["problem"],
                    "solution":         gt,
                    # Phase 1
                    "response_notool":  g.text,
                    "acc_notool":       float(acc_notool),
                    "diverse_answers":  [extract_mc_answer(o.text) for o in d_out.outputs],
                    # Phase 2
                    "responses_tool":   all_responses[i],
                    "per_round_scores": per_round_scores,
                    "acc_tool_final":   float(acc_tool_final),
                    "n_tool_calls":     n_tool_calls,
                    "n_rounds":         len(all_responses[i]),
                    # Derived
                    "delta_acc":        float(acc_tool_final - acc_notool),
                    "delta_s":          sample.get("delta_s"),
                    # 9 uncertainty metrics
                    **traj,
                    **divers,
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nResults JSONL: {output_jsonl}  ({len(results)} samples)")
    analyze(results, args.output_dir)


if __name__ == "__main__":
    main()
