#!/usr/bin/env python3
"""
Uncertainty estimation for VideoZoomer no-tool inference.

Extends main_infer.py with 9 uncertainty metrics across 3 categories,
then correlates each metric against delta_s (S_tool - S_no_tool).

───────────────────────────────────────────────────────────────────────────────
9 METRICS

Category 1 – Token Trajectory Instability  (single greedy run + per-token logprobs)
  1. U_rough    : mean |ℓ_t − ℓ_{t-1}|              (logprob roughness)
  2. U_dH       : mean |H_t − H_{t-1}|              (entropy roughness, approx top-k)
  3. U_ans_stab : U_rough restricted to <answer> region

Category 2 – Top-k Neighbourhood Disagreement  (n_samples diverse runs)
  4. U_gap      : L(1) − L(2)  best-vs-second mean log-likelihood  (↑ = certain)
  5. U_ens      : H(softmax(mean_logprobs))  weight entropy across samples
  6. U_vote     : 1 − max_answer_count / k   answer disagreement fraction

Category 3 – Energy Landscape  (derived from same runs)
  7. E_bar      : −mean(ℓ_greedy)  sequence-level energy
  8. dE         : E(2) − E(1) = −U_gap  energy gap to local alternative
  9. N_eff      : exp(H(w))  effective number of candidate solutions

Theory: high uncertainty → far from high-likelihood + high-confidence region
        → tool call more likely to help → positive correlation with delta_s
        Exception: U_gap, dE are "confidence" metrics → negative expected correlation.

───────────────────────────────────────────────────────────────────────────────
Usage
-----
  # Full run (inference + analysis):
  python main_infer_uncertainty.py \\
      --data_path  .../eval_deltaS.yaml \\
      --video_root /data/DERI-Gong/jh015/VideoZoomer \\
      --output_dir ./infer_results/uncertainty

  # Analysis only on existing JSONL:
  python main_infer_uncertainty.py --analyze_only \\
      --output_dir ./infer_results/uncertainty
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
    p.add_argument("--output_dir",  default="./infer_results/uncertainty")

    # vLLM (aligned with eval_deltaS_infer.sh)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_tokens",             type=int,   default=2048)

    # Video (aligned with eval_deltaS_infer.sh notool settings)
    p.add_argument("--max_pixels",     type=int,   default=100352)
    p.add_argument("--min_pixels",     type=int,   default=12544)
    p.add_argument("--fps",            type=float, default=0.2)
    p.add_argument("--frames_upbound", type=int,   default=120)
    p.add_argument("--system_prompt",  default="You are a helpful assistant.")

    # Uncertainty estimation
    p.add_argument("--n_samples",    type=int,   default=4,
                   help="Diverse samples per question for metrics 4–9")
    p.add_argument("--sample_temp",  type=float, default=0.7,
                   help="Sampling temperature for diverse samples")
    p.add_argument("--logprobs_k",   type=int,   default=20,
                   help="Top-k vocab logprobs returned per token (metric 2 entropy approx)")

    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip inference; load existing JSONL and run analysis only")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading  (identical to main_infer.py)
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
    if parts and parts[0] in (".", ".."):
        rel = Path(*parts[1:])
    else:
        rel = p
    return str(Path(video_root) / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# Video preprocessing  (identical to main_infer.py)
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
        img    = Image.fromarray(arr.astype("uint8"), "RGB")
        h, w   = img.height, img.width
        pixels = h * w
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            img   = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        elif pixels < min_pixels:
            scale = (min_pixels / pixels) ** 0.5
            img   = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        frames.append(img)
    return frame_times, frames


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building  (identical to main_infer.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_vllm_input(question, frame_times, frames, processor, system_prompt):
    if "<image>" in question:
        frame_tokens = "".join(
            f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
            for i, t in enumerate(frame_times)
        )
        user_content = question.replace("<image>", frame_tokens, 1)
    else:
        user_content = "".join(
            f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
            for i, t in enumerate(frame_times)
        ) + "\n" + question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return {
        "prompt":           processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "multi_modal_data": {"image": frames},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Answer extraction helper
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mc_answer(text: str):
    """Extract final multiple-choice letter (A/B/C/D) from model response."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    fragment = m.group(1).strip() if m else text
    # Look for "X." pattern (e.g. "C.")
    matches = re.findall(r"([A-Z])\.", fragment)
    if matches:
        return matches[-1]
    # Bare letter at start
    if fragment and fragment[0] in "ABCD":
        return fragment[0]
    # Fallback: scan whole text backwards
    matches = re.findall(r"([A-Z])\.", text)
    return matches[-1] if matches else None


def score_response(text: str, ground_truth: str) -> float:
    gt_m = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    gt   = gt_m.group(1).strip() if gt_m else ground_truth.strip()
    try:
        from verl.utils.reward_score.openr1 import judge_multi_choice
        return float(judge_multi_choice(text, gt))
    except Exception:
        pred = extract_mc_answer(text)
        gt_l = extract_mc_answer(gt) or gt.strip()
        return 1.0 if pred == gt_l else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: Token Trajectory Instability
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory_metrics(logprobs_list, token_ids, response_text: str) -> dict:
    """
    logprobs_list : List[Dict[int, Logprob]]  — from output.logprobs
    token_ids     : List[int]                 — from output.token_ids
    response_text : str                       — from output.text

    Returns metrics 1 (U_rough), 2 (U_dH), 3 (U_ans_stab), 7 (E_bar).
    """
    T = len(token_ids)
    if T == 0 or not logprobs_list:
        return {"U_rough": np.nan, "U_dH": np.nan, "U_ans_stab": np.nan, "E_bar": np.nan}

    # ── Per-token logprob of the chosen token ─────────────────────────────── #
    chosen_lps = np.full(T, np.nan)
    for t, (tok_id, step) in enumerate(zip(token_ids, logprobs_list)):
        if step is None:
            continue
        if tok_id in step:
            chosen_lps[t] = step[tok_id].logprob
        else:
            # Fallback: lowest logprob in the returned set
            chosen_lps[t] = min(lp.logprob for lp in step.values())
    valid = np.isfinite(chosen_lps)
    chosen_lps = np.where(valid, chosen_lps, np.nanmean(chosen_lps))

    # ── Per-token entropy (approx from top-k vocab logprobs) ──────────────── #
    # Note: top-k under-estimates true entropy; used only for relative comparison.
    entropies = np.zeros(T)
    for t, step in enumerate(logprobs_list):
        if step is None:
            continue
        lps   = np.array([lp.logprob for lp in step.values()])
        probs = np.exp(lps - lps.max())
        probs /= probs.sum()
        entropies[t] = -float(np.sum(probs * np.log(probs + 1e-12)))

    # ── Metric 1: U_rough ─────────────────────────────────────────────────── #
    U_rough = float(np.mean(np.abs(np.diff(chosen_lps)))) if T >= 2 else 0.0

    # ── Metric 2: U_dH ────────────────────────────────────────────────────── #
    U_dH = float(np.mean(np.abs(np.diff(entropies)))) if T >= 2 else 0.0

    # ── Metric 3: U_ans_stab  (roughness in answer region) ───────────────── #
    # Approximate answer-token range by character-position ratio.
    ans_start_c = response_text.find("<answer>")
    ans_end_c   = response_text.find("</answer>")
    if ans_start_c != -1 and ans_end_c != -1:
        total_c  = max(len(response_text), 1)
        t_start  = max(0,  int(ans_start_c / total_c * T))
        t_end    = min(T,  int(ans_end_c   / total_c * T) + 1)
        ans_lps  = chosen_lps[t_start:t_end]
    else:
        # No answer tag: use last 20% of tokens as proxy
        ans_lps = chosen_lps[max(0, int(0.8 * T)):]
    U_ans_stab = float(np.mean(np.abs(np.diff(ans_lps)))) if len(ans_lps) >= 2 else U_rough

    # ── Metric 7: E_bar  (sequence-level energy) ──────────────────────────── #
    E_bar = float(-np.mean(chosen_lps))

    return {"U_rough": U_rough, "U_dH": U_dH, "U_ans_stab": U_ans_stab, "E_bar": E_bar}


# ═══════════════════════════════════════════════════════════════════════════════
# Categories 2 & 3: Top-k Diversity / Energy Landscape
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_mean_ll(comp_output) -> float:
    """Mean log-likelihood of chosen tokens for one CompletionOutput."""
    lps = []
    for tok_id, step in zip(comp_output.token_ids, comp_output.logprobs or []):
        if step and tok_id in step:
            lps.append(step[tok_id].logprob)
    return float(np.mean(lps)) if lps else -100.0


def compute_diversity_metrics(comp_outputs) -> dict:
    """
    comp_outputs : List[CompletionOutput]  — output.outputs from a sampling run.

    Returns metrics 4 (U_gap), 5 (U_ens), 6 (U_vote), 8 (dE), 9 (N_eff).
    """
    k        = len(comp_outputs)
    mean_lls = np.array([_sample_mean_ll(o) for o in comp_outputs])
    answers  = [extract_mc_answer(o.text) for o in comp_outputs]

    # Sort descending: best sample first
    order    = np.argsort(mean_lls)[::-1]
    sorted_l = mean_lls[order]

    # Metric 4: U_gap = L(1) - L(2)  (higher gap → more confident)
    U_gap = float(sorted_l[0] - sorted_l[1]) if k >= 2 else 0.0

    # Metric 5: U_ens = H(w) where w_i ∝ exp(L_i)
    w     = np.exp(mean_lls - mean_lls.max())
    w    /= w.sum()
    U_ens = float(-np.sum(w * np.log(w + 1e-12)))

    # Metric 6: U_vote = 1 - max_count / k  (0 = unanimous, 1 = fully split)
    valid_ans = [a for a in answers if a is not None]
    if valid_ans:
        max_cnt = max(Counter(valid_ans).values())
        U_vote  = 1.0 - max_cnt / k
    else:
        U_vote = 1.0

    # Metric 8: dE = E(2) - E(1) = -(L(1) - L(2)) = -U_gap
    dE = -U_gap

    # Metric 9: N_eff = exp(H(w))
    N_eff = float(np.exp(U_ens))

    return {"U_gap": U_gap, "U_ens": U_ens, "U_vote": U_vote, "dE": dE, "N_eff": N_eff}


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════════

# Expected direction: positive = "higher metric → higher delta_s (tool helps more)"
METRIC_INFO = {
    "U_rough":    ("1. Logprob roughness  mean|Δℓ|",                  "+"),
    "U_dH":       ("2. Entropy roughness  mean|ΔH|",                  "+"),
    "U_ans_stab": ("3. Answer-region logprob roughness",               "+"),
    "E_bar":      ("7. Seq-level energy  −mean(ℓ)            [Cat 3]", "+"),
    "U_gap":      ("4. Best-vs-second gap  L1−L2  (↑=certain) [Cat 2]", "−"),
    "U_ens":      ("5. Sample weight entropy  H(w)             [Cat 2]", "+"),
    "U_vote":     ("6. Answer vote disagreement  1−max/k       [Cat 2]", "+"),
    "dE":         ("8. Energy gap  E2−E1  (↓=certain)         [Cat 3]", "−"),
    "N_eff":      ("9. Effective candidates  exp(H(w))         [Cat 3]", "+"),
}
METRIC_ORDER = ["U_rough", "U_dH", "U_ans_stab", "E_bar",
                "U_gap", "U_ens", "U_vote", "dE", "N_eff"]


def analyze_correlations(results: list, output_dir: str):
    """Print and save correlation table between all 9 metrics and delta_s."""
    valid = [r for r in results if r.get("delta_s") is not None]
    if not valid:
        print("[analysis] No delta_s values found.")
        return

    ds = np.array([float(r["delta_s"]) for r in valid])
    print(f"\n{'='*78}")
    print(f"CORRELATION WITH delta_s   (n={len(valid)}, "
          f"mean={ds.mean():.3f}, std={ds.std():.3f})")
    print(f"{'='*78}")

    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("[warning] scipy not available — Spearman/Pearson computed manually")

    header = (f"{'Metric':<50} {'Exp':>3}  "
              f"{'Pearson r':>9}  {'Spearman ρ':>10}  {'p-val':>7}")
    print(header)
    print("-" * 78)

    rows = []
    for m in METRIC_ORDER:
        label, exp_sign = METRIC_INFO[m]
        vals = np.array([float(r.get(m, np.nan)) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(ds)
        if ok.sum() < 5:
            continue
        v, d = vals[ok], ds[ok]

        if HAS_SCIPY:
            pr, pp = scipy_stats.pearsonr(v, d)
            sr, sp = scipy_stats.spearmanr(v, d)
        else:
            pr = float(np.corrcoef(v, d)[0, 1])
            pp = np.nan
            # Spearman via rank
            rv  = np.argsort(np.argsort(v)).astype(float)
            rd  = np.argsort(np.argsort(d)).astype(float)
            sr  = float(np.corrcoef(rv, rd)[0, 1])
            sp  = np.nan

        sig = "*" if (not np.isnan(sp) and sp < 0.05) else " "
        ok_sign = "✓" if (exp_sign == "+" and sr > 0) or (exp_sign == "−" and sr < 0) else "✗"
        print(f"{label:<50} {exp_sign:>3}  "
              f"{pr:>+9.4f}  {sr:>+9.4f}{sig}   {sp:>7.4f}  {ok_sign}")
        rows.append(dict(metric=m, label=label, expected=exp_sign,
                         pearson=pr, pearson_p=pp, spearman=sr, spearman_p=sp))

    print(f"{'='*78}")
    print("* Spearman p < 0.05   ✓/✗ = expected direction matched/missed")

    # ── Binned analysis ────────────────────────────────────────────────────── #
    print(f"\n{'─'*78}")
    print("BINNED ANALYSIS  (uncertainty quintiles → mean delta_s)")
    print(f"{'─'*78}")
    print(f"{'Metric':<25}  Q1(low)   Q2        Q3        Q4        Q5(high)  trend")
    print("-" * 78)
    for m in METRIC_ORDER:
        vals = np.array([float(r.get(m, np.nan)) for r in valid])
        ok   = np.isfinite(vals) & np.isfinite(ds)
        if ok.sum() < 20:
            continue
        v, d     = vals[ok], ds[ok]
        q_edges  = np.percentile(v, [0, 20, 40, 60, 80, 100])
        bin_means = []
        for lo, hi in zip(q_edges[:-1], q_edges[1:]):
            mask = (v >= lo) & (v <= hi)
            bin_means.append(d[mask].mean() if mask.sum() > 0 else np.nan)
        trend = "↑" if bin_means[-1] > bin_means[0] else "↓"
        vals_str = "  ".join(f"{x:+.3f}" for x in bin_means)
        print(f"{m:<25}  {vals_str}  {trend}")

    # ── Save correlation table ─────────────────────────────────────────────── #
    corr_path = os.path.join(output_dir, "correlation_report.json")
    with open(corr_path, "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: None if np.isnan(x) else float(x))
    print(f"\nCorrelation table saved to {corr_path}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_uncertainty.jsonl")

    # ── Analysis-only mode ────────────────────────────────────────────────── #
    if args.analyze_only:
        print(f"[analysis] Loading {output_jsonl}")
        with open(output_jsonl) as f:
            results = [json.loads(l) for l in f if l.strip()]
        analyze_correlations(results, args.output_dir)
        return

    # ── Normal mode: inference + uncertainty + analysis ───────────────────── #
    if not args.data_path:
        raise ValueError("--data_path is required unless --analyze_only is set")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    print(f"[vllm]  Initialising {args.model_path}  "
          f"(tp={args.tensor_parallel_size}, util={args.gpu_memory_utilization})")
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

    # ── Greedy run: per-token logprobs for trajectory metrics ─────────────── #
    # logprobs=k returns top-k vocab distribution at each step for entropy approx.
    # temperature=0 → argmax; vLLM still computes logprobs from raw softmax(logits).
    greedy_sp = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=0.0,
        logprobs=args.logprobs_k,
        detokenize=True,
    )

    # ── Diverse run: n_samples outputs for neighbourhood metrics ──────────── #
    # logprobs=1 is enough to get chosen-token logprob for mean-LL computation.
    diverse_sp = SamplingParams(
        n=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.sample_temp,
        logprobs=1,
        detokenize=True,
    )

    results       = []
    total_correct = 0.0
    total_scored  = 0

    print(f"\n[run]  greedy (logprobs={args.logprobs_k}) + "
          f"diverse (n={args.n_samples}, T={args.sample_temp}) per batch")

    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            # Build vLLM inputs
            vllm_inputs   = []
            valid_samples = []
            for sample in batch:
                try:
                    vp = resolve_video_path(sample["videos"][0], args.video_root)
                    ft, frames = load_video_frames(
                        vp, args.fps, args.max_pixels, args.min_pixels, args.frames_upbound
                    )
                    inp = build_vllm_input(
                        sample["problem"], ft, frames, processor, args.system_prompt
                    )
                    vllm_inputs.append(inp)
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id','?')}: {e}")

            if not vllm_inputs:
                continue

            # ── Two inference calls per batch ─────────────────────────────── #
            greedy_outs  = llm.generate(vllm_inputs, greedy_sp)
            diverse_outs = llm.generate(vllm_inputs, diverse_sp)

            for sample, g_out, d_out in zip(valid_samples, greedy_outs, diverse_outs):
                g = g_out.outputs[0]

                # Category 1 + 7: trajectory metrics from greedy run
                traj  = compute_trajectory_metrics(
                    g.logprobs, list(g.token_ids), g.text
                )
                # Categories 2 + 3: diversity metrics from sampling run
                divers = compute_diversity_metrics(d_out.outputs)

                # Primary reward: greedy response vs ground truth
                gt     = sample.get("solution", "")
                reward = score_response(g.text, gt)
                total_correct += reward
                total_scored  += 1

                record = {
                    "problem_id":  sample.get("problem_id"),
                    "data_source": sample.get("data_source"),
                    "problem":     sample["problem"],
                    "solution":    gt,
                    "response":    g.text,
                    "reward":      reward,
                    "delta_s":     sample.get("delta_s"),
                    **traj,
                    **divers,
                    # Snapshot of diverse answers for inspection
                    "diverse_answers": [extract_mc_answer(o.text) for o in d_out.outputs],
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    # ── Accuracy summary ──────────────────────────────────────────────────── #
    acc = total_correct / total_scored if total_scored else 0.0
    print(f"\n{'='*60}")
    print(f"Accuracy (greedy): {acc:.4f}  ({total_correct:.0f} / {total_scored})")
    print(f"Results JSONL:     {output_jsonl}")

    # ── Correlation analysis ───────────────────────────────────────────────── #
    analyze_correlations(results, args.output_dir)


if __name__ == "__main__":
    main()
