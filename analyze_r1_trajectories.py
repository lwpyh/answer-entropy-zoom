#!/usr/bin/env python3
"""
analyze_r1_trajectories.py — Offline HMM/feature analysis for zoom trigger

Goal: Learn which features of R1 reasoning (before zoom decision) predict
      whether zoom would improve the final answer.

Data design:
  - temp_vote no-zoom samples (551): have R1 NOTOOL_SYS text + acc_notool
  - greedy_baseline same samples: have acc_zoom (after zoom, via TOOL_SYS)
  - Label: zoom_benefit = greedy_acc - notool_acc > 0 → zoom helped → should zoom

Features extracted from R1 text (without requiring <think> tags):
  1. S/A/V/F state classification of full text
  2. Transition matrix features (when trajectory length > 1)
  3. Uncertainty keyword density
  4. Think chain length (chars / words)
  5. Conclusion confidence markers

Outputs:
  - Discriminative feature weights for online inference
  - Calibrated threshold
  - Saved: analysis_hmm_transitions.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Step-level classifier (S/A/V/F)
# ─────────────────────────────────────────────────────────────────────────────

FINAL_PATS = [
    r'therefore', r'thus', r'so the answer', r'final answer', r'answer is',
    r'correct answer', r'the answer', r'in conclusion', r'to summarize',
    r'</answer>', r'\\boxed', r'i would (?:choose|select|say)',
    r'option [A-D] is correct', r'my answer',
]
VERIFY_PATS = [
    r'\bwait\b', r'let me (?:check|verify|reconsider|re-examine|look again)',
    r'double.?check', r'\bconfirm\b', r'\bhmm\b', r'but wait',
    r'however', r'on the other hand',
    r"i.m not (?:sure|certain)", r"it.s (?:unclear|ambiguous|hard to tell)",
    r'need to (?:check|verify|reconsider)', r'let me re',
    r'but (?:looking|considering|thinking)',
]
ANALYSIS_PATS = [
    r'calculat', r'comput', r'\d+\s*[+\-×÷*/]\s*\d+',
    r'because', r'since\s+\w', r'given that',
    r'this (?:means|suggests|indicates|implies|shows)',
    r'compar', r'contrast', r'analyz', r'reason', r'deduc',
    r'so (?:it|this|that|the)', r'which (?:means|suggests|indicates)',
    r'based on', r'from (?:this|the|these)',
    r'percentage', r'ratio', r'proportion',
]
SETUP_PATS = [
    r'the video (?:shows|depicts|begins|starts|features)',
    r'i can (?:see|observe|notice|identify)',
    r'in the (?:clip|video|frame|footage)',
    r'at \d+\.?\d*\s*s', r'at \d+:\d+',
    r'the question (?:asks|is about)',
    r'option [A-D][:\.]', r'^[A-D][:\.]',
]
UNCERTAINTY_PATS = [
    r'\bnot sure\b', r'\buncertain\b', r'\bhard to tell\b', r'\bunclear\b',
    r'\bdifficult to\b', r'\bcould be\b', r'\bmight be\b', r'\bpossibly\b',
    r'\bperhaps\b', r'\bi think\b', r'\bseems like\b', r'\bappears to\b',
    r'\bneed (?:more|to see|to zoom|to look closer)\b',
    r'\bnot (?:visible|clear|shown)\b', r'\bcannot (?:see|tell|determine)\b',
]

STATES = ['S', 'A', 'V', 'F']
STATE_IDX = {s: i for i, s in enumerate(STATES)}


def classify_text(text: str) -> str:
    """Classify a segment of text as S/A/V/F."""
    t = text.lower()
    for p in FINAL_PATS:
        if re.search(p, t):
            return 'F'
    for p in VERIFY_PATS:
        if re.search(p, t):
            return 'V'
    for p in ANALYSIS_PATS:
        if re.search(p, t):
            return 'A'
    return 'S'


def segment_text(text: str) -> list:
    """Split reasoning text into step-level segments."""
    lines = text.strip().split('\n')
    segments = []
    buf = []
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                segments.append(' '.join(buf))
                buf = []
            continue
        if line.startswith(('-', '•', '*', '·')):
            buf.append(line.lstrip('-•*· '))
        else:
            if buf:
                segments.append(' '.join(buf))
                buf = []
            for sent in re.split(r'(?<=[.!?])\s+', line):
                if len(sent.strip()) > 5:
                    buf.append(sent.strip())
    if buf:
        segments.append(' '.join(buf))
    return [s for s in segments if len(s.strip()) > 5]


def get_trajectory(text: str) -> list:
    """Get SAVF category sequence from raw reasoning text."""
    # Extract from <think> block if present
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_text = m.group(1).strip() if m else text.strip()

    # Remove answer/tool tags
    think_text = re.sub(r'<(?:answer|video_zoom)[^>]*>.*', '', think_text, flags=re.DOTALL)
    think_text = think_text.strip()
    if not think_text:
        return []

    steps = segment_text(think_text)
    cats = [classify_text(s) for s in steps]
    return [c for c in cats if c]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from raw text
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(raw_output: str) -> dict:
    """
    Extract features from R1 raw output for zoom-trigger prediction.
    Works with or without <think> tags.
    """
    traj = get_trajectory(raw_output)

    # Clean text for analysis
    m = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    think_text = m.group(1).strip() if m else raw_output.strip()
    think_text = re.sub(r'<(?:answer|video_zoom)[^>]*>.*', '', think_text, flags=re.DOTALL).strip()
    t = think_text.lower()

    # ── Trajectory features ─────────────────────────────────────────────── #
    n_steps = len(traj)
    state_counts = Counter(traj)
    s_ratio  = state_counts.get('S', 0) / max(n_steps, 1)
    a_ratio  = state_counts.get('A', 0) / max(n_steps, 1)
    v_ratio  = state_counts.get('V', 0) / max(n_steps, 1)
    f_ratio  = state_counts.get('F', 0) / max(n_steps, 1)
    av_ratio = a_ratio + v_ratio    # analytical depth
    ends_f   = int(traj[-1] == 'F') if traj else 0

    # Transition matrix features (meaningful only when n_steps > 1)
    trans_counts = Counter()
    for a, b in zip(traj[:-1], traj[1:]):
        trans_counts[(a, b)] += 1
    n_trans = sum(trans_counts.values()) + 1e-8
    t_vf = trans_counts.get(('V', 'F'), 0) / n_trans  # verify→final: good
    t_af = trans_counts.get(('A', 'F'), 0) / n_trans  # analysis→final: good
    t_sa = trans_counts.get(('S', 'A'), 0) / n_trans  # setup→analysis: good
    t_vv = trans_counts.get(('V', 'V'), 0) / n_trans  # verify loop: bad (EIRL)
    t_aa = trans_counts.get(('A', 'A'), 0) / n_trans  # analysis loop: bad (EIRL)
    t_ss = trans_counts.get(('S', 'S'), 0) / n_trans  # setup loop: bad (EIRL)

    # ── Text surface features ────────────────────────────────────────────── #
    n_words = len(think_text.split())
    n_chars = len(think_text)

    # Uncertainty keyword density
    n_uncertain = sum(1 for p in UNCERTAINTY_PATS if re.search(p, t))
    uncertain_density = n_uncertain / max(n_words / 50, 1)  # per 50 words

    # Confidence markers
    confidence_pats = [
        r'clearly', r'obviously', r'definitely', r'certainly', r'confirmed',
        r'i can (?:now |clearly )?see', r'this confirms', r'it is (?:clear|evident)',
        r'therefore the (?:correct )?answer', r'the answer is (?:clearly|definitely)',
    ]
    n_confident = sum(1 for p in confidence_pats if re.search(p, t))

    # Full-text state classification
    full_state = classify_text(think_text)

    return {
        'n_steps':          n_steps,
        's_ratio':          s_ratio,
        'a_ratio':          a_ratio,
        'v_ratio':          v_ratio,
        'f_ratio':          f_ratio,
        'av_ratio':         av_ratio,
        'ends_f':           ends_f,
        't_vf':             t_vf,
        't_af':             t_af,
        't_sa':             t_sa,
        't_vv':             t_vv,
        't_aa':             t_aa,
        't_ss':             t_ss,
        'n_words':          n_words,
        'n_chars':          n_chars,
        'n_uncertain_kw':   n_uncertain,
        'uncertain_density':uncertain_density,
        'n_confident_kw':   n_confident,
        'full_state':       full_state,
        'full_state_S':     int(full_state == 'S'),
        'full_state_A':     int(full_state == 'A'),
        'full_state_V':     int(full_state == 'V'),
        'full_state_F':     int(full_state == 'F'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrix estimation (EIRL-style)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_T(samples, label: int):
    """Estimate row-normalised 4×4 transition matrix for given label (0/1)."""
    counts = np.ones((4, 4)) * 0.5   # Laplace smoothing
    for s in samples:
        if s['zoom_benefit_label'] != label:
            continue
        traj = s['trajectory']
        for a, b in zip(traj[:-1], traj[1:]):
            if a in STATE_IDX and b in STATE_IDX:
                counts[STATE_IDX[a], STATE_IDX[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# Load and pair data
# ─────────────────────────────────────────────────────────────────────────────

def load_paired_data():
    """
    Cross-reference temp_vote no-zoom R1 outputs with greedy baseline accuracy.

    Label:
      zoom_benefit = greedy_acc - notool_acc > 0  →  1 (zoom helped, SHOULD zoom)
      zoom_benefit ≤ 0                            →  0 (zoom didn't help, skip)
    """
    # Load greedy baseline (has acc_zoom for all samples)
    greedy_by_pid = {}
    gp = BASE / "infer_results" / "greedy_baseline" / "results_greedy_merged.jsonl"
    with open(gp) as f:
        for line in f:
            s = json.loads(line)
            greedy_by_pid[s['problem_id']] = float(s['acc_final'])

    # Load temp_vote (no-zoom samples have R1 text + notool acc)
    tv_path = BASE / "infer_results" / "temp_vote" / "results_tvote.jsonl"
    samples = []
    n_matched = 0
    with open(tv_path) as f:
        for line in f:
            s = json.loads(line)
            if s.get('zoom_triggered'):
                continue  # R1 text overwritten by R2+

            acc_notool = float(s['acc_final'])
            acc_zoom   = greedy_by_pid.get(s['problem_id'], None)
            if acc_zoom is None:
                continue

            zoom_benefit = acc_zoom - acc_notool   # >0 means zoom helped
            traj = get_trajectory(s['raw_output'])
            feats = extract_features(s['raw_output'])

            samples.append({
                'problem_id':       s['problem_id'],
                'acc_notool':       acc_notool,
                'acc_zoom':         acc_zoom,
                'zoom_benefit':     zoom_benefit,
                'zoom_benefit_label': 1 if zoom_benefit > 0 else 0,
                'r1_agreement':     s.get('r1_agreement', None),
                'trajectory':       traj,
                **feats,
            })
            n_matched += 1

    print(f"[paired data] {n_matched} samples matched (temp_vote no-zoom ∩ greedy)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance analysis
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'n_steps', 's_ratio', 'a_ratio', 'v_ratio', 'f_ratio', 'av_ratio', 'ends_f',
    't_vf', 't_af', 't_sa', 't_vv', 't_aa', 't_ss',
    'n_words', 'n_uncertain_kw', 'uncertain_density', 'n_confident_kw',
    'full_state_S', 'full_state_A', 'full_state_V', 'full_state_F',
]


def point_biserial_correlation(samples, feature):
    """Compute point-biserial correlation between feature and zoom_benefit_label."""
    vals  = np.array([s[feature] for s in samples], dtype=float)
    labels = np.array([s['zoom_benefit_label'] for s in samples], dtype=float)
    # Exclude NaN
    mask = ~np.isnan(vals)
    vals, labels = vals[mask], labels[mask]
    if vals.std() < 1e-9:
        return 0.0
    corr = np.corrcoef(vals, labels)[0, 1]
    return float(corr)


def mean_by_label(samples, feature, label):
    vals = [s[feature] for s in samples if s['zoom_benefit_label'] == label]
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Composite score and threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_score(samples, corr_weights: dict):
    """
    Build a linear composite score:
       score = sum_i(w_i * feature_i)
    Positive score → model reasoning is confident → SKIP zoom
    Negative score → model uncertain → EXECUTE zoom
    """
    scored = []
    for s in samples:
        score = 0.0
        for feat, w in corr_weights.items():
            v = s.get(feat, 0.0)
            if v is not None and not np.isnan(v):
                score += w * v
        scored.append({'score': score, **s})
    return scored


def calibrate_threshold(scored_samples):
    """Scan thresholds; below threshold → zoom triggered."""
    scored_samples = sorted(scored_samples, key=lambda x: x['score'])
    scores  = np.array([s['score'] for s in scored_samples])
    benefit = np.array([s['zoom_benefit'] for s in scored_samples])

    # Expected accuracy under "zoom if score < thr" strategy
    # Base: always zoom → acc = acc_zoom
    # Proposed: zoom only for score < thr → acc = mix of acc_notool (skip) + acc_zoom (zoom)
    acc_always_zoom   = np.mean([s['acc_zoom']   for s in scored_samples])
    acc_never_zoom    = np.mean([s['acc_notool'] for s in scored_samples])

    thresholds = np.percentile(scores, np.arange(5, 95, 5))
    results = []
    for thr in np.unique(thresholds):
        zoom_mask = scores < thr
        n_zoom = zoom_mask.sum()
        n_skip = (~zoom_mask).sum()
        if n_zoom == 0 or n_skip == 0:
            continue
        # Accuracy under this policy:
        # - zoomed samples → use acc_zoom (greedy baseline with zoom)
        # - skipped samples → use acc_notool (R1 NOTOOL_SYS answer)
        acc_zoom_subset  = np.mean([s['acc_zoom']   for s, z in zip(scored_samples, zoom_mask) if z])
        acc_skip_subset  = np.mean([s['acc_notool'] for s, z in zip(scored_samples, zoom_mask) if not z])
        policy_acc = (n_zoom * acc_zoom_subset + n_skip * acc_skip_subset) / (n_zoom + n_skip)

        results.append({
            'threshold':      float(thr),
            'n_zoom':         int(n_zoom),
            'zoom_rate':      float(n_zoom / len(scored_samples)),
            'acc_zoom_sub':   float(acc_zoom_subset),
            'acc_skip_sub':   float(acc_skip_subset),
            'policy_acc':     float(policy_acc),
            'vs_always_zoom': float(policy_acc - acc_always_zoom),
            'vs_never_zoom':  float(policy_acc - acc_never_zoom),
        })

    return results, acc_always_zoom, acc_never_zoom


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HMM Trajectory Analysis — Paired Zoom-Benefit Labels")
    print("=" * 60)

    samples = load_paired_data()
    n_should_zoom  = sum(1 for s in samples if s['zoom_benefit_label'] == 1)
    n_skip_zoom    = sum(1 for s in samples if s['zoom_benefit_label'] == 0)
    zoom_benefit   = np.mean([s['zoom_benefit'] for s in samples])
    print(f"\n[label distribution]")
    print(f"  zoom helped (label=1): {n_should_zoom} ({n_should_zoom/len(samples)*100:.1f}%)")
    print(f"  zoom hurt   (label=0): {n_skip_zoom}  ({n_skip_zoom/len(samples)*100:.1f}%)")
    print(f"  mean zoom_benefit (greedy - notool): {zoom_benefit:+.4f}")
    print(f"  acc_always_zoom = {np.mean([s['acc_zoom'] for s in samples]):.4f}")
    print(f"  acc_never_zoom  = {np.mean([s['acc_notool'] for s in samples]):.4f}")

    # ── Trajectory length analysis ──────────────────────────────────────── #
    lengths = [len(s['trajectory']) for s in samples]
    has_trans = sum(1 for l in lengths if l > 1)
    print(f"\n[trajectory lengths]")
    print(f"  mean={np.mean(lengths):.1f}  median={np.median(lengths):.0f}  max={max(lengths)}")
    print(f"  with transitions (len>1): {has_trans}/{len(lengths)} ({has_trans/len(lengths)*100:.1f}%)")

    # ── Transition matrix analysis (EIRL-style, on full 1548-sample dataset) #
    # Load all records for T matrix estimation
    all_records = []
    gp = BASE / "infer_results" / "greedy_baseline" / "results_greedy_merged.jsonl"
    with open(gp) as f:
        for line in f:
            s = json.loads(line)
            traj = get_trajectory(s['raw_output'])
            if traj:
                all_records.append({'trajectory': traj, 'acc': float(s['acc_final'])})

    tv_path = BASE / "infer_results" / "temp_vote" / "results_tvote.jsonl"
    with open(tv_path) as f:
        for line in f:
            s = json.loads(line)
            if not s.get('zoom_triggered'):
                traj = get_trajectory(s['raw_output'])
                if traj:
                    all_records.append({'trajectory': traj, 'acc': float(s['acc_final'])})

    def estimate_T(records, label_val):
        counts = np.ones((4, 4)) * 0.5
        for r in records:
            if r['acc'] != label_val:
                continue
            for a, b in zip(r['trajectory'][:-1], r['trajectory'][1:]):
                if a in STATE_IDX and b in STATE_IDX:
                    counts[STATE_IDX[a], STATE_IDX[b]] += 1
        return counts / counts.sum(axis=1, keepdims=True)

    T_correct   = estimate_T(all_records, 1.0)
    T_incorrect = estimate_T(all_records, 0.0)
    delta_T     = T_correct - T_incorrect

    print(f"\n[Transition matrices from {len(all_records)} records]")
    print(f"  ΔT = T_correct − T_incorrect  (S A V F):")
    for i, s in enumerate(STATES):
        print(f"    {s}: {delta_T[i].round(4)}")

    eirl_deltas = {
        ('V','F'): +0.065, ('A','F'): +0.045, ('S','A'): +0.042,
        ('V','V'): -0.043, ('A','A'): -0.020, ('S','S'): -0.028,
    }
    print(f"\n  EIRL Table 18 vs our ΔT:")
    for (a, b), ev in eirl_deltas.items():
        ov = delta_T[STATE_IDX[a], STATE_IDX[b]]
        match = '✓' if (ev > 0) == (ov > 0) else '✗'
        print(f"    {a}→{b}: EIRL={ev:+.4f}  ours={ov:+.4f}  {match}")

    # ── Feature-level analysis on paired data ────────────────────────────── #
    print(f"\n[Feature correlation with zoom_benefit_label]")
    print(f"  {'Feature':25s}  {'PBcorr':>8s}  {'mean(label=1)':>13s}  {'mean(label=0)':>13s}  {'Δ':>8s}")
    corrs = {}
    for feat in FEATURE_NAMES:
        corr = point_biserial_correlation(samples, feat)
        m1 = mean_by_label(samples, feat, 1)
        m0 = mean_by_label(samples, feat, 0)
        corrs[feat] = corr
        print(f"  {feat:25s}  {corr:>+8.4f}  {m1:>13.4f}  {m0:>13.4f}  {m1-m0:>+8.4f}")

    # ── Build composite score ────────────────────────────────────────────── #
    # Use point-biserial as weights; negate so score > 0 = should zoom
    # Features positively correlated with label=1 (should zoom) get positive weight
    # We keep only features with |corr| > 0.02 to avoid noise
    WEIGHT_FEATURES = {
        # Positive = zoom more likely to help (model is uncertain/setup-heavy)
        'full_state_S':     +1.0,   # entire R1 text is just description → zoom
        'uncertain_density':+1.5,   # uncertainty keywords → zoom
        'n_uncertain_kw':   +0.5,
        's_ratio':          +1.0,   # lots of setup steps → zoom
        # Negative = zoom less likely to help (model already reasoning/concluding)
        'full_state_A':     -1.0,   # analysis text → skip zoom
        'full_state_V':     -1.0,   # verification text → skip zoom
        'full_state_F':     -1.5,   # already has conclusion → definitely skip
        'av_ratio':         -2.0,   # high A+V fraction → skip zoom
        'ends_f':           -1.5,   # trajectory ends in final state → skip
        'n_confident_kw':   -1.0,   # confident language → skip
        # Transition-based (EIRL-derived, for samples with transitions)
        't_vf':             -1.0,   # V→F: success signal, skip zoom
        't_af':             -0.8,   # A→F: success signal, skip zoom
        't_vv':             +0.5,   # V→V loop: uncertainty, zoom
        't_ss':             +0.5,   # S→S loop: setup only, zoom
        # Length: longer reasoning = more confident (model worked through it)
        'n_words':          -0.02,  # small penalty per word
    }

    # Scale weights by point-biserial correlation (direction alignment)
    final_weights = {}
    for feat, base_w in WEIGHT_FEATURES.items():
        corr = corrs.get(feat, 0.0)
        # Keep sign from base_w (domain knowledge), scale by |corr|
        final_weights[feat] = base_w * (1 + 2 * abs(corr))  # amplify high-corr features

    print(f"\n[Composite score weights]")
    print(f"  (score > threshold → skip zoom, score < threshold → execute zoom)")
    for feat, w in sorted(final_weights.items(), key=lambda x: -abs(x[1])):
        direction = 'zoom' if w > 0 else 'skip'
        print(f"  {feat:25s}: {w:>+8.4f}  [{direction}]")

    scored = build_composite_score(samples, final_weights)

    # ── Threshold calibration ────────────────────────────────────────────── #
    cal_results, acc_always_zoom, acc_never_zoom = calibrate_threshold(scored)

    print(f"\n[Baseline accuracies on 551 paired samples]")
    print(f"  Always zoom (greedy): {acc_always_zoom:.4f}")
    print(f"  Never zoom (notool):  {acc_never_zoom:.4f}")
    print(f"\n[Policy accuracy vs zoom rate]")
    print(f"  {'Threshold':>10s}  {'ZoomRate':>8s}  {'PolicyAcc':>9s}  "
          f"{'vs_zoom':>8s}  {'vs_notool':>9s}")
    for c in cal_results:
        print(f"  {c['threshold']:>+10.4f}  {c['zoom_rate']:>8.3f}  "
              f"{c['policy_acc']:>9.4f}  {c['vs_always_zoom']:>+8.4f}  "
              f"{c['vs_never_zoom']:>+9.4f}")

    # Best: maximize policy_acc
    best = max(cal_results, key=lambda x: x['policy_acc']) if cal_results else None
    if best:
        print(f"\n  Best threshold: {best['threshold']:+.4f}  "
              f"acc={best['policy_acc']:.4f}  zoom_rate={best['zoom_rate']:.2f}")

    # Also find threshold that best approximates desired ~30% zoom rate
    target_rate_results = sorted(
        [c for c in cal_results if 0.25 <= c['zoom_rate'] <= 0.45],
        key=lambda x: -x['policy_acc']
    )
    if target_rate_results:
        target = target_rate_results[0]
        print(f"  Best at 25-45% zoom rate: threshold={target['threshold']:+.4f}  "
              f"acc={target['policy_acc']:.4f}  zoom_rate={target['zoom_rate']:.2f}")

    # ── HMM score distribution by label ─────────────────────────────────── #
    scores_1 = [s['score'] for s in scored if s['zoom_benefit_label'] == 1]
    scores_0 = [s['score'] for s in scored if s['zoom_benefit_label'] == 0]
    print(f"\n[Composite score distribution]")
    print(f"  Should zoom (label=1): mean={np.mean(scores_1):+.4f}  std={np.std(scores_1):.4f}")
    print(f"  Skip zoom   (label=0): mean={np.mean(scores_0):+.4f}  std={np.std(scores_0):.4f}")
    print(f"  Separation: {np.mean(scores_0) - np.mean(scores_1):+.4f}")

    # ── Save results ─────────────────────────────────────────────────────── #
    out = {
        "n_paired_samples":    len(samples),
        "n_should_zoom":       n_should_zoom,
        "n_skip_zoom":         n_skip_zoom,
        "acc_always_zoom":     float(acc_always_zoom),
        "acc_never_zoom":      float(acc_never_zoom),

        # EIRL-style transition matrices (estimated from full 1548 sample set)
        "T_correct":           T_correct.tolist(),
        "T_incorrect":         T_incorrect.tolist(),
        "delta_T":             delta_T.tolist(),
        "states":              STATES,

        "eirl_comparison": {
            f"{a}->{b}": {
                "eirl_delta": ev,
                "our_delta":  float(delta_T[STATE_IDX[a], STATE_IDX[b]]),
                "sign_match": (ev > 0) == (float(delta_T[STATE_IDX[a], STATE_IDX[b]]) > 0),
            }
            for (a, b), ev in eirl_deltas.items()
        },

        # Feature weights for online composite score
        "composite_weights":   {k: float(v) for k, v in final_weights.items()},

        # Point-biserial correlations
        "feature_correlations": {k: float(v) for k, v in corrs.items()},

        # Calibration results
        "threshold_calibration": cal_results,
        "recommended_threshold": float(best['threshold']) if best else 0.0,
        "recommended_zoom_rate": float(best['zoom_rate']) if best else 0.3,
        "best_policy_acc":       float(best['policy_acc']) if best else 0.0,
        "target_rate_threshold": float(target['threshold']) if target_rate_results else 0.0,

        "score_dist": {
            "label_1_mean": float(np.mean(scores_1)) if scores_1 else 0,
            "label_1_std":  float(np.std(scores_1))  if scores_1 else 0,
            "label_0_mean": float(np.mean(scores_0)) if scores_0 else 0,
            "label_0_std":  float(np.std(scores_0))  if scores_0 else 0,
        },
    }

    out_path = BASE / "analysis_hmm_transitions.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved]  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
