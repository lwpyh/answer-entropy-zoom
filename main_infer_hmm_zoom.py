#!/usr/bin/env python3
"""
main_infer_hmm_zoom.py  ─  Single-pass HMM-based zoom trigger

Design (replaces temp_vote, avoids NOTOOL_SYS degradation):
  Round 1  [TOOL_SYS, greedy, T=0]
    - Model generates <think>...</think><video_zoom>...</video_zoom>  (zoom call)
      OR <think>...</think><answer>...</answer>  (direct answer)
    - If model gave answer → done (no zoom needed)
    - If model called zoom → extract R1 <think> → compute HMM score

  HMM score (composite metric from EIRL explicit-stage framework):
    score = weighted sum of features from R1 reasoning trajectory
    + state-frequency features (S/A/V/F proportions)
    + uncertainty keyword density
    + transition matrix features (our ΔT, calibrated from 1548 greedy+tv samples)
    score > hmm_threshold  →  reasoning is confident → SKIP zoom (R1 answer)
    score ≤ hmm_threshold  →  reasoning is uncertain → EXECUTE zoom → Round 2

  Round 2+  [TOOL_SYS, greedy, same as greedy_baseline]
    - Normal zoom execution and continuation

Advantages over temp_vote:
  ✓ No NOTOOL_SYS degradation (R1 always uses TOOL_SYS)
  ✓ Single forward pass (no 5× temperature sampling)
  ✓ Interpretable: zoom decision has linguistic justification
  ✓ Zoom rate tunable via --hmm_threshold

Offline calibration:
  Run analyze_r1_trajectories.py first to produce analysis_hmm_transitions.json
  Default weights are embedded (from our LVR dataset calibration).
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    TOOL_SYS,
    load_dataset,
    resolve_video_path,
    load_video_frames,
    build_tool_initial_prompt,
    build_notool_prompt,
    build_tool_response_turn,
    extract_mc_answer,
    parse_zoom_call,
    score_answer,
    _frame_tokens,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HMM / Feature extraction (mirrors analyze_r1_trajectories.py)
# ═══════════════════════════════════════════════════════════════════════════════

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
    r'based on', r'from (?:this|the|these)',
    r'percentage', r'ratio', r'proportion',
]
SETUP_PATS = [
    r'the video (?:shows|depicts|begins|starts|features)',
    r'i can (?:see|observe|notice|identify)',
    r'in the (?:clip|video|frame|footage)',
    r'at \d+\.?\d*\s*s',
    r'the question (?:asks|is about)',
]
UNCERTAINTY_PATS = [
    r'\bnot sure\b', r'\buncertain\b', r'\bhard to tell\b', r'\bunclear\b',
    r'\bdifficult to\b', r'\bcould be\b', r'\bmight be\b', r'\bpossibly\b',
    r'\bperhaps\b', r'\bi think\b', r'\bseems like\b', r'\bappears to\b',
    r'\bneed (?:more|to see|to zoom|to look closer)\b',
    r'\bnot (?:visible|clear|shown)\b', r'\bcannot (?:see|tell|determine)\b',
]
CONFIDENCE_PATS = [
    r'clearly', r'obviously', r'definitely', r'certainly', r'confirmed',
    r'i can (?:now |clearly )?see', r'this confirms', r'it is (?:clear|evident)',
    r'therefore the (?:correct )?answer', r'the answer is (?:clearly|definitely)',
]

STATES = ['S', 'A', 'V', 'F']
STATE_IDX = {s: i for i, s in enumerate(STATES)}


def _classify(text: str) -> str:
    t = text.lower()
    for p in FINAL_PATS:
        if re.search(p, t): return 'F'
    for p in VERIFY_PATS:
        if re.search(p, t): return 'V'
    for p in ANALYSIS_PATS:
        if re.search(p, t): return 'A'
    return 'S'


def _segment(text: str) -> list:
    lines = text.strip().split('\n')
    segs, buf = [], []
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                segs.append(' '.join(buf)); buf = []
            continue
        if line.startswith(('-', '•', '*', '·')):
            buf.append(line.lstrip('-•*· '))
        else:
            if buf:
                segs.append(' '.join(buf)); buf = []
            for sent in re.split(r'(?<=[.!?])\s+', line):
                if len(sent.strip()) > 5: buf.append(sent.strip())
    if buf: segs.append(' '.join(buf))
    return [s for s in segs if len(s.strip()) > 5]


def extract_think_text(raw_r1: str) -> str:
    """
    Extract thinking content from R1 output.
    Handles both explicit <think>...</think> and raw text before zoom/answer tag.
    """
    m = re.search(r'<think>(.*?)</think>', raw_r1, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # No think tags: take everything before the tool call
        text = re.sub(r'<(?:video_zoom|answer)[^>]*>.*', '', raw_r1, flags=re.DOTALL).strip()
    return text


def compute_hmm_score(think_text: str, weights: dict) -> dict:
    """
    Compute composite HMM score from R1 think chain.

    Higher score → model reasoning is confident/analytical → SKIP zoom
    Lower score  → model is uncertain/setup-heavy → EXECUTE zoom

    Returns dict with score + diagnostic sub-features.
    """
    t = think_text.lower()

    # ── Trajectory features ─────────────────────────────────────────────── #
    steps = _segment(think_text)
    traj  = [_classify(s) for s in steps]
    n     = max(len(traj), 1)

    from collections import Counter
    state_cnt = Counter(traj)
    s_ratio  = state_cnt.get('S', 0) / n
    a_ratio  = state_cnt.get('A', 0) / n
    v_ratio  = state_cnt.get('V', 0) / n
    f_ratio  = state_cnt.get('F', 0) / n
    av_ratio = a_ratio + v_ratio
    ends_f   = int(traj[-1] == 'F') if traj else 0

    # Transition features
    from collections import Counter as Counter2
    trans = Counter2(zip(traj[:-1], traj[1:]))
    n_trans = sum(trans.values()) + 1e-8
    t_vf = trans.get(('V', 'F'), 0) / n_trans
    t_af = trans.get(('A', 'F'), 0) / n_trans
    t_sa = trans.get(('S', 'A'), 0) / n_trans
    t_vv = trans.get(('V', 'V'), 0) / n_trans
    t_aa = trans.get(('A', 'A'), 0) / n_trans
    t_ss = trans.get(('S', 'S'), 0) / n_trans

    # ── Text surface features ────────────────────────────────────────────── #
    n_words     = len(think_text.split())
    n_uncertain = sum(1 for p in UNCERTAINTY_PATS if re.search(p, t))
    uncertain_d = n_uncertain / max(n_words / 50, 1)
    n_confident = sum(1 for p in CONFIDENCE_PATS if re.search(p, t))

    full_state  = _classify(think_text)

    feats = {
        'n_steps':           len(traj),
        's_ratio':           s_ratio,
        'a_ratio':           a_ratio,
        'v_ratio':           v_ratio,
        'f_ratio':           f_ratio,
        'av_ratio':          av_ratio,
        'ends_f':            ends_f,
        't_vf':              t_vf,
        't_af':              t_af,
        't_sa':              t_sa,
        't_vv':              t_vv,
        't_aa':              t_aa,
        't_ss':              t_ss,
        'n_words':           n_words,
        'n_uncertain_kw':    n_uncertain,
        'uncertain_density': uncertain_d,
        'n_confident_kw':    n_confident,
        'full_state_S':      int(full_state == 'S'),
        'full_state_A':      int(full_state == 'A'),
        'full_state_V':      int(full_state == 'V'),
        'full_state_F':      int(full_state == 'F'),
        'trajectory':        ''.join(traj),
        'full_state':        full_state,
    }

    # ── Composite score ──────────────────────────────────────────────────── #
    score = sum(
        weights.get(k, 0.0) * v
        for k, v in feats.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    feats['hmm_score'] = score
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy-based scoring (zero-shot, no GT needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _token_entropy(lp_dict: dict) -> float:
    """
    Compute entropy of a token's distribution from top-K vLLM logprobs dict.
    Uses the top-K probs directly + residual mass for the rest of the vocab.

    H = -sum_{top-K} p_i*log(p_i)  -  p_residual*log(p_residual)

    Returns entropy in nats.
    """
    lps   = np.array([v.logprob for v in lp_dict.values()], dtype=np.float64)
    probs = np.exp(lps)
    probs = np.clip(probs, 0.0, 1.0)
    top_sum = probs.sum()
    residual = max(0.0, 1.0 - top_sum)
    H = -np.sum(probs * np.log(probs + 1e-12))
    if residual > 1e-9:
        H -= residual * np.log(residual)
    return float(H)


def _think_token_range(text: str, token_logprobs: list, tokenizer):
    """
    Return (think_text, start_idx, end_idx) — indices into token_logprobs
    that correspond to the <think>...</think> content.
    Falls back to the full sequence if no <think> tags found.
    """
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        think_text  = m.group(1)
        prefix_ntok = len(tokenizer.encode(text[:m.start(1)], add_special_tokens=False))
        think_ntok  = len(tokenizer.encode(think_text,        add_special_tokens=False))
        s = min(prefix_ntok, len(token_logprobs))
        e = min(prefix_ntok + think_ntok, len(token_logprobs))
        if e > s:
            return think_text, s, e
    return text, 0, len(token_logprobs)


def _split_sentences(text: str) -> list:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 8]


def _sent_entropies(think_text: str, token_ents: list) -> list:
    """Map token entropies to sentence-level means via char/token ratio."""
    sents = _split_sentences(think_text)
    if not sents or not token_ents:
        return []
    T = len(token_ents)
    chars_per_tok = max(len(think_text), 1) / T
    sent_ents, offset = [], 0
    for s in sents:
        n = max(1, round(len(s) / chars_per_tok))
        seg = token_ents[offset: offset + n]
        if seg:
            sent_ents.append(float(np.mean(seg)))
        offset += n
        if offset >= T:
            break
    return sent_ents


def compute_early_entropy_score(token_ents: list, think_text: str,
                                early_frac: float = 0.75) -> dict:
    """
    Sentence-level entropy using only the FIRST `early_frac` of think-chain tokens.
    Avoids contamination from the zoom-request boilerplate at the end:
      "I will request a higher frame rate clip..." → always low entropy → misleads last_H.
    By cutting at 75%, we score only the reasoning part.
    Higher score → confident reasoning → SKIP zoom.
    """
    T = len(token_ents)
    if T < 4:
        return compute_sent_entropy_score(token_ents, think_text)

    cutoff = max(int(T * early_frac), 1)
    early_ents = token_ents[:cutoff]
    early_text = think_text[:int(len(think_text) * early_frac)] if think_text else ''

    sent_ents = _sent_entropies(early_text, early_ents)
    if not sent_ents:
        q      = max(len(early_ents) // 4, 1)
        last_H = float(np.mean(early_ents[-q:]))
        trend  = last_H - float(np.mean(early_ents[:q]))
    else:
        q      = max(len(sent_ents) // 4, 1)
        last_H = float(np.mean(sent_ents[-q:]))
        trend  = last_H - float(np.mean(sent_ents[:q]))

    score = -1.5 * last_H - 0.6 * trend
    return {
        'early_entropy_score': score,
        'last_H':              last_H,
        'trend':               trend,
        'n_sents':             len(sent_ents) if sent_ents else 0,
        'early_cutoff':        cutoff,
    }


# Keywords indicating the question requires fine-grained temporal/spatial detail → need zoom
_ZOOM_NEED_KW = frozenset([
    'fast', 'quickly', 'rapid', 'speed', 'motion', 'moving', 'action',
    'detail', 'close', 'precise', 'specific', 'exact', 'identify', 'recognize',
    'read', 'text', 'sign', 'number', 'count', 'small', 'brief', 'moment',
    'second', 'frame', 'transition', 'flash', 'blink', 'appear', 'disappear',
])
# Keywords indicating a broad/summary question → zoom less needed
_NO_ZOOM_KW = frozenset([
    'overall', 'general', 'main', 'primary', 'purpose', 'intention', 'theme',
    'topic', 'category', 'type', 'kind', 'describe', 'summary', 'summarize',
    'throughout', 'entire', 'whole', 'structure',
])


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds (0.0 on failure)."""
    try:
        import av as _av
        container = _av.open(video_path)
        dur = float(container.duration) / 1e6  # av uses microseconds
        container.close()
        return max(dur, 0.0)
    except Exception:
        return 0.0


def compute_question_prior_score(question: str, video_path: str,
                                 n_think_tokens: int = 0) -> dict:
    """
    Prior-based zoom decision — no logprobs needed.
    Uses:
      (1) question text keywords (detail/motion → need zoom; summary → skip)
      (2) video duration (longer video → harder to find specific moments → zoom)
      (3) think chain length (short think chain → model confident → skip)
    Higher score → skip zoom.
    """
    q_lower = question.lower()
    words   = set(re.findall(r'\w+', q_lower))

    n_need  = len(words & _ZOOM_NEED_KW)
    n_skip  = len(words & _NO_ZOOM_KW)
    kw_score = float(n_skip - n_need)          # positive → skip

    duration = _get_video_duration(video_path)
    if   duration <= 0:    dur_score =  0.0
    elif duration <  60:   dur_score =  0.5    # short → less need for zoom
    elif duration < 120:   dur_score =  0.0
    elif duration < 300:   dur_score = -0.5
    else:                  dur_score = -1.0    # long → more need for zoom

    # More think tokens → more uncertain → zoom
    think_score = max(0.0, (150.0 - n_think_tokens) / 150.0)  # 1.0 at 0 tokens, 0 at 150+

    score = 0.5 * kw_score + 0.8 * dur_score + 0.4 * think_score
    return {
        'question_prior_score': score,
        'n_zoom_need_kw':       n_need,
        'n_skip_kw':            n_skip,
        'duration_secs':        duration,
        'dur_score':            dur_score,
        'kw_score':             kw_score,
        'n_think_tokens':       n_think_tokens,
    }


def compute_combined_score(think_text: str, token_ents: list,
                           question: str, video_path: str,
                           kw_weights: dict) -> dict:
    """
    Direction 1: Combined zoom-trigger score.
    Normalizes and combines three complementary signals:
      - keyword score  (hmm_zoom_v2 rules, best single method 0.774)
      - early_entropy  (reasoning-part entropy, skip_acc=0.698 best)
      - question_prior (video duration + question keywords, skip_acc=0.786)

    Each signal is z-normalized using distribution stats from 1000-sample calibration:
      keyword:        mean=-2.268, std=2.861
      early_entropy:  mean=-1.192, std=0.683
      question_prior: mean=-0.271, std=0.648

    Weights: keyword=0.5, early_entropy=0.3, question_prior=0.2
    Combined threshold ≈ 0 (z-normalized). Higher → skip zoom.
    """
    # ── keyword score ────────────────────────────────────────────────── #
    kw_feats  = compute_hmm_score(think_text, kw_weights)
    kw_raw    = kw_feats['hmm_score']
    kw_norm   = (kw_raw - (-2.268)) / 2.861

    # ── early entropy (first 75% of think tokens) ─────────────────────── #
    ee_feats  = compute_early_entropy_score(token_ents, think_text)
    ee_raw    = ee_feats['early_entropy_score']
    ee_norm   = (ee_raw - (-1.192)) / 0.683

    # ── question prior (video duration + question keywords) ──────────── #
    n_think   = len(token_ents)
    qp_feats  = compute_question_prior_score(question, video_path, n_think)
    qp_raw    = qp_feats['question_prior_score']
    qp_norm   = (qp_raw - (-0.271)) / 0.648

    score = 0.5 * kw_norm + 0.3 * ee_norm + 0.2 * qp_norm
    return {
        'combined_score':   score,
        'kw_norm':          kw_norm,
        'ee_norm':          ee_norm,
        'qp_norm':          qp_norm,
        'kw_raw':           kw_raw,
        'ee_raw':           ee_raw,
        'qp_raw':           qp_raw,
        'n_think_tokens':   n_think,
        **{f'kw_{k}': v for k, v in kw_feats.items() if k != 'hmm_score'},
        **{f'ee_{k}': v for k, v in ee_feats.items() if k != 'early_entropy_score'},
        **{f'qp_{k}': v for k, v in qp_feats.items() if k != 'question_prior_score'},
    }


def compute_sent_entropy_score(token_ents: list, think_text: str) -> dict:
    """
    Sentence-level entropy composite score (no HMM — HMM degenerates on this data).
    Score = -1.5*last_H - 0.6*trend   (sentence-level)
    Cohen's d: last_H=-1.24, trend=-1.05 vs token-level last_H=-0.85
    Higher score → more confident → SKIP zoom.
    """
    sent_ents = _sent_entropies(think_text, token_ents)
    if not sent_ents:
        # fallback to token-level
        T = len(token_ents) if token_ents else 1
        q = max(T // 4, 1)
        last_H = float(np.mean(token_ents[-q:])) if token_ents else 1.0
        trend  = 0.0
    else:
        q      = max(len(sent_ents) // 4, 1)
        last_H = float(np.mean(sent_ents[-q:]))
        trend  = last_H - float(np.mean(sent_ents[:q]))

    score = -1.5 * last_H - 0.6 * trend
    return {
        'sent_entropy_score': score,
        'last_H':             last_H,
        'trend':              trend,
        'n_sents':            len(sent_ents) if sent_ents else 0,
    }


def compute_entropy_hmm_score(entropies: list, hmm_model, C_state: int, U_state: int) -> dict:
    """
    Decode entropy sequence with a pre-fitted GaussianHMM(K=2).
    Returns composite score based on Viterbi state trajectory.
    Higher score → confident think chain → SKIP zoom.
    """
    T = len(entropies)
    if T < 2:
        return {'hmm_score': 0.0, 'C_ratio': 0.5, 'ends_in_C': 0,
                'last_q_C': 0.5, 'last_H': entropies[0] if entropies else 1.0,
                'trend': 0.0}

    q = max(T // 4, 1)
    H = np.clip(np.array(entropies, dtype=np.float32), 0.05, None).reshape(-1, 1)
    states, _ = hmm_model.decode(H, algorithm='viterbi')

    C_ratio   = float((states == C_state).mean())
    ends_in_C = int(states[-1] == C_state)
    last_q_C  = float((states[-q:] == C_state).mean())
    n_CU      = sum(1 for a, b in zip(states[:-1], states[1:])
                    if a == C_state and b == U_state)

    last_H = float(np.mean(entropies[-q:]))
    trend  = last_H - float(np.mean(entropies[:q]))

    score = (
        -1.5 * last_H
        -0.6 * trend
        +0.8 * C_ratio
        +0.4 * ends_in_C
        +0.5 * last_q_C
        -0.3 * (n_CU / max(T - 1, 1))
    )
    return {
        'hmm_score':  score,
        'C_ratio':    C_ratio,
        'ends_in_C':  ends_in_C,
        'last_q_C':   last_q_C,
        'last_H':     last_H,
        'trend':      trend,
    }


def compute_entropy_score(text: str, token_logprobs: list, tokenizer) -> dict:
    """
    Compute mean token entropy of think chain from vLLM logprobs (top-20).

    Higher entropy → model is uncertain about its own tokens → EXECUTE zoom
    Lower  entropy → model generates fluently/confidently  → SKIP zoom

    Score returned = -mean_entropy  (higher score → more confident → skip zoom,
    consistent with the rest of the codebase where score > threshold → skip).

    No GT or keyword lists needed.
    """
    if not token_logprobs:
        return {'entropy_score': -float('inf'), 'mean_entropy': float('inf'),
                'n_think_tokens': 0, 'segments': []}

    think_text, s, e = _think_token_range(text, token_logprobs, tokenizer)
    think_lp_dicts   = token_logprobs[s:e] or token_logprobs

    entropies = [_token_entropy(d) for d in think_lp_dicts]
    mean_H    = float(np.mean(entropies))

    # Per-segment breakdown
    segs = _segment(think_text)
    seg_results, offset = [], 0
    for seg in segs:
        n = len(tokenizer.encode(seg, add_special_tokens=False))
        seg_ents = entropies[offset : offset + n]
        if seg_ents:
            seg_results.append({
                'text':         seg[:60],
                'mean_entropy': float(np.mean(seg_ents)),
                'n_tokens':     n,
            })
        offset += n

    T = len(entropies)
    q = max(T // 4, 1)
    entropy_first = float(np.mean(entropies[:q]))
    entropy_last  = float(np.mean(entropies[-q:]))
    entropy_trend = entropy_last - entropy_first   # positive = getting more uncertain
    entropy_max   = float(np.max(entropies))
    entropy_std   = float(np.std(entropies))

    return {
        'entropy_score':   -mean_H,        # higher → confident → skip zoom
        'mean_entropy':     mean_H,
        'entropy_first':    entropy_first,
        'entropy_last':     entropy_last,
        'entropy_trend':    entropy_trend,  # >0 = increasingly uncertain
        'entropy_max':      entropy_max,
        'entropy_std':      entropy_std,
        'n_think_tokens':   T,
        'n_total_tokens':   len(token_logprobs),
        'think_text':       think_text,     # R1 think chain for keyword cross-analysis
        'token_entropies':  entropies,      # full per-token list for offline analysis
        'segments':         seg_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Answer distribution entropy  (NeurIPS Direction: answer-level uncertainty)
# Directly measures "will the model give different answers?" instead of proxy
# token-level uncertainty. No hand-crafted keywords or logprobs needed.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_answer_dist_entropy(texts: list) -> dict:
    """
    Extract MC answer from k temperature samples → compute answer distribution entropy.

    Key insight: directly measures answer-level uncertainty rather than token-level
    language uncertainty (which is confounded by zoom-request boilerplate).

    H = 0:         all k samples agree → perfectly confident → SKIP zoom
    H = log(4)≈1.39: uniform distribution → maximally uncertain → EXECUTE zoom

    score = -H_answer  (higher = more confident = skip zoom, consistent convention)
    Returns majority_answer for direct use when zoom is skipped (no 2nd LLM pass).
    """
    from collections import Counter
    answers = []
    for t in texts:
        ans = extract_mc_answer(t)
        if ans and ans in 'ABCD':
            answers.append(ans)

    if not answers:
        # All k samples called zoom or gave no answer → treat as maximally uncertain
        return {
            'answer_entropy_score': -np.log(4),
            'H_answer':             np.log(4),
            'answer_dist':          {c: 0.25 for c in 'ABCD'},
            'majority_answer':      None,
            'n_valid':              0,
        }

    counts   = Counter(answers)
    total    = len(answers)
    dist     = {c: counts.get(c, 0) / total for c in 'ABCD'}
    H        = -sum(p * np.log(p + 1e-12) for p in dist.values() if p > 0)
    majority = counts.most_common(1)[0][0]

    return {
        'answer_entropy_score': -H,   # higher = more confident = skip zoom
        'H_answer':             H,
        'answer_dist':          dist,
        'majority_answer':      majority,
        'n_valid':              total,
    }


def compute_answer_jsd(dist1: dict, dist2: dict) -> float:
    """
    Jensen-Shannon divergence between two answer distributions.
    JSD = 0:      distributions identical  → answer has converged → STOP zooming
    JSD = log(2): maximally different      → answer still changing → CONTINUE
    Used by the Wald-style optimal-stopping iter controller (--answer_wald).
    """
    choices = 'ABCD'
    p = np.array([dist1.get(c, 0.0) for c in choices], dtype=np.float64) + 1e-12
    q = np.array([dist2.get(c, 0.0) for c in choices], dtype=np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m      = 0.5 * (p + q)
    kl_pm  = float(np.sum(p * np.log(p / m)))
    kl_qm  = float(np.sum(q * np.log(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def fit_answer_gmm_batch(H_values: list) -> dict:
    """
    Fit a 2-component Gaussian Mixture Model on the batch of H_answer values.

    Completely unsupervised (Baum-Welch / EM) — no labels, no pre-calibration.
    Learns two hidden states from the current batch distribution:
      C (Confident): low-entropy cluster
      U (Uncertain): high-entropy cluster

    The decision boundary adapts to the batch rather than using a fixed threshold.
    This is the 'POMDP belief state' layer: for each sample we compute
      b = P(Confident | H_obs) = posterior probability of the confident component.
    """
    from sklearn.mixture import GaussianMixture
    X   = np.clip(np.array(H_values, dtype=np.float64), 0.0, None).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components    = 2,
        covariance_type = 'full',
        n_init          = 5,
        random_state    = 42,
    )
    gmm.fit(X)
    means   = gmm.means_.flatten()
    C_state = int(np.argmin(means))   # lower mean = Confident
    U_state = 1 - C_state
    return {'model': gmm, 'C_state': C_state, 'U_state': U_state,
            'means': means.tolist()}


def compute_answer_hmm_belief(H_value: float, gmm_artifact: dict) -> dict:
    """
    Compute belief state P(Confident | H_obs) using the fitted GMM.
    score = P(Confident)  (higher = more confident = skip zoom)
    """
    model   = gmm_artifact['model']
    C_state = gmm_artifact['C_state']
    X          = np.array([[max(H_value, 0.0)]])
    posteriors = model.predict_proba(X)        # shape (1, 2)
    p_conf     = float(posteriors[0, C_state])
    return {
        'answer_hmm_score': p_conf,
        'p_confident':      p_conf,
        'H_answer':         H_value,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Default weights (from analyze_r1_trajectories.py on LVR dataset)
# Sign convention: weight > 0 → feature favors ZOOM, weight < 0 → favors SKIP
# At test time: zoom if score > threshold
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    # Negative = skip zoom (model already reasoning well)
    'av_ratio':          -2.1189,
    'full_state_F':      -1.6637,
    'ends_f':            -1.6114,
    'n_confident_kw':    -1.1079,
    't_vf':              -1.1058,
    'full_state_V':      -1.0864,
    'full_state_A':      -1.0174,
    't_af':              -0.8500,
    'n_words':           -0.0217,
    # Positive = zoom (model uncertain / setup-heavy)
    'uncertain_density': +1.8130,
    'full_state_S':      +1.0797,
    's_ratio':           +1.0663,
    'n_uncertain_kw':    +0.6006,
    't_ss':              +0.5769,
    't_vv':              +0.5000,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, prompt, images, question=''):
        self.pid           = pid
        self.gt            = gt
        self.video_path    = video_path
        self.prompt        = prompt
        self.images        = images
        self.question      = question
        self.n_tool_calls  = 0
        self.n_rounds      = 0
        self.final_answer  = None
        self.acc_final     = None
        self.raw_output    = ""
        self.zoom_skipped   = False    # HMM decided to skip zoom
        self.zoom_triggered = False    # HMM decided to execute zoom
        self.hmm_score      = None
        self.hmm_features   = None
        self.round_entropies  = []      # mean entropy per round (for iter_entropy control)
        self.prev_answer_dist = None    # answer distribution from previous round (for answer_wald)
        self.prev_h_answer    = None    # H_answer from previous round (for recursive_ae)
        self.iter_ae_prev_h   = None    # H_answer from previous force-answer check (for iter_ae_delta)
        self.notool_prompt    = None    # NOTOOL_SYS prompt for ae_notool force-answer sampling


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/hmm_zoom_v2")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)
    p.add_argument("--min_pixels",             type=int,   default=25088)

    # Video & tool
    p.add_argument("--fps",                      type=float, default=0.5)
    p.add_argument("--frames_upbound",           type=int,   default=64)
    p.add_argument("--max_tokens",               type=int,   default=4096)
    p.add_argument("--tool_limit_mm",            type=int,   default=128)
    p.add_argument("--tool_max_frames_per_call", type=int,   default=16)
    p.add_argument("--tool_workers",             type=int,   default=8)
    p.add_argument("--max_rounds",               type=int,   default=5)

    # HMM zoom trigger
    p.add_argument(
        "--hmm_threshold", type=float, default=1.7,
        help=(
            "Composite HMM score threshold. "
            "score > threshold → skip zoom (confident R1). "
            "score ≤ threshold → execute zoom (uncertain R1). "
            "Calibrated values from analyze_r1_trajectories.py: "
            "  -6.0 ≈ 40%% zoom rate (acc < greedy), "
            "  +1.7 ≈ 85%% zoom rate (acc slightly > greedy), "
            "  -inf → always zoom (≡ greedy baseline). "
            "Default +1.7: skip zoom for top ~15%% most confident samples."
        ),
    )
    p.add_argument(
        "--hmm_weights_path", type=str, default=None,
        help="Path to analysis_hmm_transitions.json for calibrated weights. "
             "If None, uses embedded DEFAULT_WEIGHTS.",
    )
    p.add_argument(
        "--force_zoom_on_no_think", action="store_true", default=True,
        help="If R1 has no <think> block and model calls zoom, always execute zoom.",
    )

    # Scoring mode
    p.add_argument(
        "--score_mode", choices=["keyword", "ppl", "entropy", "entropy_hmm",
                                  "sent_entropy", "early_entropy", "question_prior",
                                  "combined",
                                  "answer_entropy", "answer_entropy_hmm"],
        default="keyword",
        help=(
            "keyword:            composite keyword-based HMM score (original). "
            "ppl:                mean logprob of think chain tokens (deprecated). "
            "entropy:            mean token entropy from top-20 logprobs. "
            "entropy_hmm:        GaussianHMM(K=2) on token entropy trajectory. "
            "sent_entropy:       sentence-level entropy last_H+trend composite. "
            "early_entropy:      sentence-level entropy on FIRST 75%% of think tokens. "
            "question_prior:     prior score from question keywords + video duration. "
            "combined:           z-normalized combination of keyword + early_entropy + question_prior. "
            "answer_entropy:     k temperature samples → H(P(A/B/C/D)); score=-H. "
            "                    No logprobs, no keywords. Direct answer uncertainty. "
            "answer_entropy_hmm: same + batch GMM (Baum-Welch) for data-adaptive belief state; "
            "                    score = P(Confident | H). No pre-calibrated threshold. "
            "score > threshold → skip zoom in all modes."
        ),
    )
    p.add_argument(
        "--entropy_hmm_model", type=str, default=None,
        help="Path to entropy_hmm_artifacts/hmm_k2.pkl (from fit_entropy_hmm.py). "
             "Required for --score_mode entropy_hmm.",
    )
    p.add_argument(
        "--ppl_threshold", type=float, default=-1.8,
        help="Threshold for ppl score mode (deprecated).",
    )
    p.add_argument(
        "--entropy_threshold", type=float, default=-2.5,
        help=(
            "Threshold for entropy score mode (= -mean_entropy of think chain). "
            "Typical range: -1.5 (few skips) to -3.5 (many skips). "
            "Default -2.5 ≈ skip top ~15%% most confident samples."
        ),
    )

    # Post-zoom iteration entropy control (Direction 2)
    p.add_argument(
        "--iter_entropy", action="store_true", default=False,
        help=(
            "Enable post-zoom iteration control via entropy trajectory. "
            "After each zoom round (R2+), compute mean token entropy. "
            "If delta_H = H_t - H_{t-1} >= iter_threshold → zoom not converging → "
            "force finalize with current answer instead of continuing. "
            "Works alongside any --score_mode."
        ),
    )
    p.add_argument(
        "--iter_threshold", type=float, default=0.0,
        help=(
            "Delta-entropy threshold for post-zoom stopping. "
            "If H_t - H_{t-1} >= iter_threshold → entropy not decreasing → stop. "
            "0.0 = stop if entropy doesn't decrease at all (strict). "
            "0.05 = allow slight increase (lenient). Default=0.0."
        ),
    )

    # Answer distribution entropy (NeurIPS Direction 2 & 3)
    p.add_argument(
        "--answer_k", type=int, default=5,
        help="Number of temperature samples for answer_entropy / answer_entropy_hmm modes. "
             "Higher k → better entropy estimate, more compute.",
    )
    p.add_argument(
        "--answer_temperature", type=float, default=0.7,
        help="Sampling temperature for answer_entropy modes.",
    )

    p.add_argument(
        "--ae_notool", action="store_true", default=False,
        help=(
            "Use NOTOOL_SYS prompt for answer_entropy force-answer k sampling. "
            "Eliminates zoom-call truncation in k chains: model sees original video+question "
            "with no tool definition, so all k chains produce valid answers. "
            "Gives cleaner H estimate vs default TOOL_SYS+_FORCE_ANS_TURN approach."
        ),
    )

    # Answer Wald iter control (NeurIPS Direction 4 — optimal stopping)
    p.add_argument(
        "--answer_wald", action="store_true", default=False,
        help=(
            "Enable Wald-style optimal stopping for R2+ iterations. "
            "After each zoom round, draw --wald_k answer samples (T=0.7) and "
            "compare with the previous round via Jensen-Shannon divergence. "
            "If JSD < wald_threshold → answer distribution converged → force stop. "
            "Fixes iter_entropy failure: compares same-type distributions (answers), "
            "not different-length text strings."
        ),
    )
    p.add_argument(
        "--wald_k", type=int, default=3,
        help="Number of temperature samples per round for answer_wald stopping test.",
    )
    p.add_argument(
        "--wald_threshold", type=float, default=0.05,
        help=(
            "JSD threshold for answer_wald convergence. "
            "JSD < threshold → answer stable → stop zooming. "
            "JSD = 0: identical distributions. JSD = log(2)≈0.693: maximally different. "
            "0.05 = stop if distributions are very similar (~95%% of mass on same answer). "
            "0.02 = stricter (require near-perfect agreement). Default=0.05."
        ),
    )

    # Iterative Answer Entropy (R2+ zoom gating — same mechanism as R1)
    p.add_argument(
        "--iter_ae", action="store_true", default=False,
        help=(
            "Apply the same answer_entropy check before EVERY R2+ zoom execution. "
            "When the model calls zoom in R2+, first run --answer_k force-answer samples "
            "(T=answer_temperature); if H < |entropy_threshold| → already confident → "
            "skip this zoom and return majority answer. "
            "Extends the R1 answer_entropy gate consistently to all rounds."
        ),
    )

    p.add_argument(
        "--iter_ae_delta", action="store_true", default=False,
        help=(
            "Add ΔH stopping to R2+ zoom gating: skip zoom if H_curr >= H_prev "
            "(entropy did not decrease after last zoom → zoom not helping). "
            "Uses the same force-answer sampling as --iter_ae. "
            "Can be used alone or combined with --iter_ae."
        ),
    )

    # Recursive Answer Entropy iter control (R2+ stopping)
    p.add_argument(
        "--recursive_ae", action="store_true", default=False,
        help=(
            "Enable Recursive Answer Entropy stopping for R2+ iterations. "
            "After each zoom round, draw --rae_k answer samples (T=answer_temperature) "
            "and compute H_t = H(P(A/B/C/D)). Stop if: "
            "(1) H_t < rae_low_threshold  → confident enough → stop, OR "
            "(2) H_t >= H_{t-1}           → zoom not reducing uncertainty → stop. "
            "More principled than Wald: stops on absolute confidence, not just consistency."
        ),
    )
    p.add_argument(
        "--rae_k", type=int, default=3,
        help="Number of temperature samples per R2+ round for recursive_ae stopping.",
    )
    p.add_argument(
        "--rae_low_threshold", type=float, default=0.15,
        help=(
            "H_answer threshold for recursive_ae confident-stop condition. "
            "H_t < rae_low_threshold → answer has converged → stop zooming. "
            "0.15 ≈ ~4/5 samples agree on same answer. Default=0.15."
        ),
    )

    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_hmm_zoom_v2.jsonl")

    # ── Score mode setup ─────────────────────────────────────────────────── #
    if args.score_mode == "keyword":
        if args.hmm_weights_path and os.path.exists(args.hmm_weights_path):
            with open(args.hmm_weights_path) as f:
                analysis = json.load(f)
            weights = analysis.get("composite_weights", DEFAULT_WEIGHTS)
            print(f"[score] keyword mode — weights from {args.hmm_weights_path}")
        else:
            weights = DEFAULT_WEIGHTS
            print(f"[score] keyword mode — using DEFAULT_WEIGHTS")
        threshold = args.hmm_threshold
    elif args.score_mode == "sent_entropy":
        weights   = None
        threshold = args.entropy_threshold
        print(f"[score] sent_entropy mode — sentence-level last_H+trend, threshold={threshold}")
    elif args.score_mode == "early_entropy":
        weights   = None
        threshold = args.entropy_threshold
        print(f"[score] early_entropy mode — first 75%% of think tokens (reasoning part only), threshold={threshold}")
    elif args.score_mode == "question_prior":
        weights   = None
        threshold = args.entropy_threshold
        print(f"[score] question_prior mode — question keywords + video duration (no logprobs), threshold={threshold}")
    elif args.score_mode == "combined":
        if args.hmm_weights_path and os.path.exists(args.hmm_weights_path):
            with open(args.hmm_weights_path) as f:
                analysis = json.load(f)
            weights = analysis.get("composite_weights", DEFAULT_WEIGHTS)
        else:
            weights = DEFAULT_WEIGHTS
        threshold = args.entropy_threshold
        print(f"[score] combined mode — z-norm(keyword)*0.5 + z-norm(early_entropy)*0.3 + "
              f"z-norm(question_prior)*0.2, threshold={threshold}")
    elif args.score_mode == "entropy_hmm":
        import pickle
        assert args.entropy_hmm_model, "--entropy_hmm_model required for entropy_hmm mode"
        with open(args.entropy_hmm_model, 'rb') as f:
            hmm_artifact = pickle.load(f)
        hmm_model = hmm_artifact['hmm']
        C_state   = hmm_artifact['C_state']
        U_state   = hmm_artifact['U_state']
        weights   = None
        threshold = args.entropy_threshold
        print(f"[score] entropy_hmm mode — GaussianHMM(K=2) Viterbi scoring, threshold={threshold}")
        print(f"[score] C_state={C_state}  U_state={U_state}")
    elif args.score_mode == "entropy":
        weights   = None
        threshold = args.entropy_threshold
        print(f"[score] entropy mode — mean token entropy (top-20 logprobs), threshold={threshold}")
        print(f"[score] entropy_score = -mean_entropy  (higher = more confident = skip zoom)")
    elif args.score_mode == "answer_entropy":
        weights   = None
        threshold = args.entropy_threshold   # use -H threshold (default -0.3 in script)
        print(f"[score] answer_entropy mode — k={args.answer_k} T={args.answer_temperature} samples "
              f"→ H(P(A/B/C/D)), score=-H, threshold={threshold}")
        print(f"[score] No logprobs, no keywords — direct answer uncertainty measurement.")
    elif args.score_mode == "answer_entropy_hmm":
        if args.hmm_weights_path and os.path.exists(args.hmm_weights_path):
            with open(args.hmm_weights_path) as f:
                analysis = json.load(f)
            weights = analysis.get("composite_weights", DEFAULT_WEIGHTS)
        else:
            weights = DEFAULT_WEIGHTS
        threshold = args.entropy_threshold   # P(Confident) threshold (default 0.6 in script)
        print(f"[score] answer_entropy_hmm mode — batch GMM belief state, "
              f"k={args.answer_k} T={args.answer_temperature}, threshold={threshold}")
        print(f"[score] score = P(Confident | H_answer) — data-adaptive, no pre-calibration.")
    else:
        weights   = None
        threshold = args.ppl_threshold
        print(f"[score] ppl mode — mean think-chain logprob, threshold={threshold}")

    print(f"[score] score > {threshold} → skip zoom, score ≤ {threshold} → execute zoom")

    # ── Data ─────────────────────────────────────────────────────────────── #
    print(f"[data] Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data] {len(samples)} samples")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    max_mm = max(args.frames_upbound, args.tool_limit_mm)
    print(f"[vLLM] Initialising (tp={args.tensor_parallel_size})")
    llm = LLM(
        model                  = args.model_path,
        tensor_parallel_size   = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len          = args.max_model_len,
        dtype                  = "bfloat16",
        trust_remote_code      = True,
        mm_processor_kwargs    = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels},
        limit_mm_per_prompt    = {"image": max_mm},
        enforce_eager          = False,
        enable_prefix_caching  = False,
    )

    greedy_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=(20 if args.score_mode in ("entropy", "entropy_hmm", "sent_entropy",
                                             "early_entropy", "combined")
                  else (1 if args.score_mode == "ppl" else None)),
    )
    # Used for zoom_skipped fallback: force model to answer without zoom
    force_ans_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )
    # Sampling params for R2+ rounds — includes logprobs when iter_entropy is active
    iter_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=(20 if args.iter_entropy else None),
    )

    # answer_entropy / answer_entropy_hmm: k temperature samples for R1 answer distribution
    answer_sp = SamplingParams(
        n=args.answer_k,
        temperature=args.answer_temperature,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=None,   # no logprobs needed — direct answer extraction
    ) if args.score_mode in ("answer_entropy", "answer_entropy_hmm") else None

    # answer_wald: k temperature samples per R2+ round for Wald stopping test
    wald_iter_sp = SamplingParams(
        n=args.wald_k,
        temperature=args.answer_temperature,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=None,
    ) if args.answer_wald else None

    # iter_ae: same force-answer sampling as R1, applied before every R2+ zoom
    iter_ae_sp = SamplingParams(
        n=args.answer_k,
        temperature=args.answer_temperature,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=None,
    ) if (args.iter_ae or args.iter_ae_delta) else None

    # recursive_ae: k temperature samples per R2+ round for H_answer stopping
    rae_iter_sp = SamplingParams(
        n=args.rae_k,
        temperature=args.answer_temperature,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=None,
    ) if args.recursive_ae else None

    if args.iter_entropy:
        print(f"[iter_entropy] enabled — post-zoom convergence check, "
              f"delta_H threshold={args.iter_threshold} "
              f"(stop if H_t - H_{{t-1}} >= threshold)")
    if args.answer_wald:
        print(f"[answer_wald] enabled — Wald JSD stopping, k={args.wald_k}, "
              f"T={args.answer_temperature}, JSD threshold={args.wald_threshold}")
    if args.iter_ae or args.iter_ae_delta:
        print(f"[iter_ae{'_delta' if args.iter_ae_delta else ''}] enabled — "
              f"k={args.answer_k}, T={args.answer_temperature}"
              + (f", abs_threshold={args.entropy_threshold}" if args.iter_ae else "")
              + (f", delta_H: stop if H_curr>=H_prev" if args.iter_ae_delta else ""))
    if args.recursive_ae:
        print(f"[recursive_ae] enabled — H_t<{args.rae_low_threshold} OR H_t>=H_{{t-1}} → stop, "
              f"k={args.rae_k}, T={args.answer_temperature}")

    print(f"[run]  TOOL_SYS R1 + HMM zoom gate, "
          f"fps={args.fps}, frames≤{args.frames_upbound}, max_rounds={args.max_rounds}")

    # Force-answer turn template (used in R1 answer_entropy and R2+ iter_ae)
    _FORCE_ANS_TURN = (
        "\n<|im_end|>\n<|im_start|>user\n"
        "<tool_response>\n"
        "The zoom tool is unavailable. Based on the video frames "
        "already shown, give your best final answer now.\n"
        "</tool_response>\n"
        "Do not call <video_zoom>. "
        "Write your final answer inside <answer> and </answer>."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # ── Resume ───────────────────────────────────────────────────────────── #
    done_pids, file_mode = set(), "w"
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try: done_pids.add(json.loads(line)["problem_id"])
                except Exception: pass
        if done_pids:
            print(f"[resume] {len(done_pids)} samples done, skipping")
            file_mode = "a"

    pending = [
        s for s in samples
        if str(s.get("problem_id", "")) not in done_pids
        and str(s.get("extra_info", {}).get("problem_id", "")) not in done_pids
    ]

    all_records = []
    n_correct = n_zoom_triggered = n_zoom_skipped = n_direct_answer = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="Batches",
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess ────────────────────────────────────────────── #
            active = []
            for sample in batch:
                pid      = (sample.get("problem_id") or
                            sample.get("extra_info", {}).get("problem_id", "?"))
                gt       = sample.get("solution", "")
                question = sample.get("problem", "")
                video_rel = (sample.get("videos") or [""])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound)
                    prompt = build_tool_initial_prompt(question, frame_times, processor)
                    state  = SampleState(pid, gt, video_path, prompt, list(frames), question=question)
                    if args.ae_notool and args.score_mode in ("answer_entropy", "answer_entropy_hmm"):
                        state.notool_prompt = build_notool_prompt(question, frame_times, processor,
                                                                  use_tool_sys=False)
                    active.append(state)
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ══════════════════════════════════════════════════════════════
            # Round 1: TOOL_SYS greedy — intercept zoom for HMM gate
            # ══════════════════════════════════════════════════════════════
            inputs = [
                {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                for s in active
            ]

            zoom_queue        = {}    # idx → (state, zoom_call_tuple)
            force_answer_queue = {}   # idx → state  (HMM skip → force-answer pass)
            next_active       = []    # continue to R2+

            # All modes: R1 greedy T=0 for canonical reasoning chain
            outputs = llm.generate(inputs, greedy_sp)

            # Deferred zoom-calling samples for answer_entropy batch scoring
            # (populated in the greedy loop, processed below with force-answer sampling)
            ae_deferred = []    # list of (idx, s, out, zoom_call)

            for idx, (s, out) in enumerate(zip(active, outputs)):
                text = out.outputs[0].text
                s.n_rounds   = 1
                s.raw_output = text

                zoom_call = parse_zoom_call(text)

                if zoom_call is None:
                    # Model gave a direct answer (no zoom) — done
                    s.final_answer = extract_mc_answer(text)
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)
                else:
                    # Model called zoom — apply score gate
                    think_text = extract_think_text(text)

                    # answer_entropy modes: defer to batch force-answer sampling below
                    if args.score_mode in ("answer_entropy", "answer_entropy_hmm"):
                        ae_deferred.append((idx, s, out, zoom_call))
                        continue   # skip inline scoring; handled in batch after this loop

                    if think_text:
                        if args.score_mode == "sent_entropy":
                            tok_lps   = out.outputs[0].logprobs or []
                            e_feats   = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            ents      = e_feats.get('token_entropies', [])
                            think_txt = e_feats.get('think_text', think_text)
                            feats     = compute_sent_entropy_score(ents, think_txt)
                            feats['mean_entropy']   = e_feats.get('mean_entropy')
                            feats['n_think_tokens'] = e_feats.get('n_think_tokens')
                            score     = feats['sent_entropy_score']
                        elif args.score_mode == "early_entropy":
                            tok_lps   = out.outputs[0].logprobs or []
                            e_feats   = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            ents      = e_feats.get('token_entropies', [])
                            think_txt = e_feats.get('think_text', think_text)
                            feats     = compute_early_entropy_score(ents, think_txt)
                            feats['mean_entropy']   = e_feats.get('mean_entropy')
                            feats['n_think_tokens'] = e_feats.get('n_think_tokens')
                            score     = feats['early_entropy_score']
                        elif args.score_mode == "question_prior":
                            tok_lps   = out.outputs[0].logprobs or []
                            e_feats   = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            n_think   = e_feats.get('n_think_tokens', 0)
                            feats     = compute_question_prior_score(s.question, s.video_path, n_think)
                            feats['mean_entropy']   = e_feats.get('mean_entropy')
                            feats['n_think_tokens'] = n_think
                            score     = feats['question_prior_score']
                        elif args.score_mode == "combined":
                            tok_lps   = out.outputs[0].logprobs or []
                            e_feats   = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            ents      = e_feats.get('token_entropies', [])
                            think_txt = e_feats.get('think_text', think_text)
                            feats     = compute_combined_score(
                                think_txt, ents, s.question, s.video_path, weights)
                            feats['mean_entropy']   = e_feats.get('mean_entropy')
                            score     = feats['combined_score']
                        elif args.score_mode == "entropy_hmm":
                            tok_lps  = out.outputs[0].logprobs or []
                            e_feats  = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            ents     = e_feats.get('token_entropies', [])
                            feats    = compute_entropy_hmm_score(ents, hmm_model, C_state, U_state)
                            feats['mean_entropy']    = e_feats.get('mean_entropy')
                            feats['n_think_tokens']  = e_feats.get('n_think_tokens')
                            score    = feats['hmm_score']
                        elif args.score_mode == "entropy":
                            tok_lps = out.outputs[0].logprobs or []
                            feats   = compute_entropy_score(text, tok_lps, processor.tokenizer)
                            score   = feats['entropy_score']
                        elif args.score_mode == "ppl":
                            tok_lps = out.outputs[0].logprobs or []
                            feats   = compute_ppl_score(text, tok_lps, processor.tokenizer)
                            score   = feats['ppl_score']
                        else:
                            feats = compute_hmm_score(think_text, weights)
                            score = feats['hmm_score']
                        s.hmm_score    = score
                        s.hmm_features = feats
                        # Record R1 entropy for iter_entropy baseline (ΔH in R2+)
                        if args.iter_entropy:
                            r1_H = feats.get('mean_entropy') or 0.0
                            s.round_entropies = [r1_H]
                    else:
                        score = -float('inf')   # no think → always zoom
                        s.hmm_score    = score
                        s.hmm_features = {}

                    if score > threshold:
                        # ── Confident: skip zoom → force-answer pass ──── #
                        # Append R1 output (think + zoom call) to conversation
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        # Add a "zoom unavailable" user turn asking for direct answer
                        s.prompt += (
                            "\n<|im_end|>\n<|im_start|>user\n"
                            "<tool_response>\n"
                            "The zoom tool is unavailable. Based on the video frames "
                            "already shown, give your best final answer now.\n"
                            "</tool_response>\n"
                            "Do not call <video_zoom>. "
                            "Write your final answer inside <answer> and </answer>."
                            "<|im_end|>\n<|im_start|>assistant\n"
                        )
                        s.zoom_skipped = True
                        force_answer_queue[idx] = s
                    else:
                        # ── Uncertain: execute zoom, continue to R2 ────── #
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        s.zoom_triggered = True
                        zoom_queue[idx] = (s, zoom_call)

            # ── answer_entropy: batch force-answer sampling for H_answer ── #
            # R1 was greedy (T=0). Now measure answer confidence by running
            # k temperature samples on "zoom unavailable → answer now" prompts.
            # This separates reasoning quality (T=0) from confidence measurement (T>0).
            if ae_deferred:
                # Build force-answer prompts for all deferred zoom-calling samples
                ae_fa_inputs = []
                for (idx, s, out, zoom_call) in ae_deferred:
                    if args.ae_notool and s.notool_prompt is not None:
                        # Clean NOTOOL_SYS prompt: model sees original frames+question only,
                        # no zoom tool definition → all k chains produce valid answers
                        fa_prompt = s.notool_prompt
                    else:
                        # Original: append R1 zoom call + "zoom unavailable" turn
                        text    = out.outputs[0].text
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        fa_prompt = s.prompt + text[:end_pos] + _FORCE_ANS_TURN
                    ae_fa_inputs.append({
                        "prompt": fa_prompt,
                        "multi_modal_data": {"image": list(s.images)},
                    })

                # k temperature samples per sample → answer distribution
                ae_fa_outputs = llm.generate(ae_fa_inputs, answer_sp)

                # Phase 1: H_answer for each sample
                ae_batch_h    = []
                ae_batch_feats = []
                for fa_out in ae_fa_outputs:
                    texts_k = [o.text for o in fa_out.outputs]
                    feats   = compute_answer_dist_entropy(texts_k)
                    ae_batch_feats.append(feats)
                    ae_batch_h.append(feats['H_answer'])

                # Phase 2: fit batch GMM (for answer_entropy_hmm)
                gmm_batch = None
                if args.score_mode == "answer_entropy_hmm" and len(ae_batch_h) >= 4:
                    try:
                        gmm_batch = fit_answer_gmm_batch(ae_batch_h)
                        print(f"[R1 GMM] means={[f'{m:.3f}' for m in gmm_batch['means']]}  "
                              f"C_state={gmm_batch['C_state']}")
                    except Exception as e:
                        print(f"[warn] GMM fitting failed ({e}); falling back to answer_entropy")

                # Phase 3: per-sample zoom trigger decisions
                for (idx, s, out, zoom_call), feats in zip(ae_deferred, ae_batch_feats):
                    text = out.outputs[0].text   # T=0 greedy canonical

                    if args.score_mode == "answer_entropy_hmm" and gmm_batch:
                        hmm_feats = compute_answer_hmm_belief(feats['H_answer'], gmm_batch)
                        score = hmm_feats['answer_hmm_score']
                        feats = {**feats, **hmm_feats}
                    else:
                        score = feats['answer_entropy_score']

                    s.hmm_score    = score
                    s.hmm_features = feats

                    if score > threshold:
                        # Skip zoom — use majority answer from force-answer samples
                        s.zoom_skipped = True
                        majority = feats['majority_answer']
                        if majority:
                            s.final_answer = majority
                            s.acc_final    = score_answer(majority, s.gt)
                            done.append(s)
                        else:
                            # All k samples gave no answer — fallback to greedy force-answer
                            end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                            s.prompt += text[:end_pos] + _FORCE_ANS_TURN
                            force_answer_queue[idx] = s
                    else:
                        # Execute zoom — use T=0 greedy canonical for R2+ context
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        s.zoom_triggered = True
                        # Seed iter_ae_delta with H_R1 so R2 can compare against it
                        if args.iter_ae_delta:
                            s.iter_ae_prev_h = feats['H_answer']
                        zoom_queue[idx] = (s, zoom_call)

            # ── Force-answer pass for HMM-skipped samples ─────────────── #
            if force_answer_queue:
                fa_inputs = [
                    {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                    for s in force_answer_queue.values()
                ]
                fa_outputs = llm.generate(fa_inputs, force_ans_sp)
                for s, out in zip(force_answer_queue.values(), fa_outputs):
                    fa_text = out.outputs[0].text
                    s.raw_output   = fa_text
                    s.n_rounds     = 2
                    s.final_answer = extract_mc_answer(fa_text)
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)

            # ── Execute R1 zoom clips (parallel) ──────────────────────── #
            if zoom_queue:
                with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                    futures = {
                        ex.submit(
                            extract_video_clip,
                            video_path     = s.video_path,
                            start_time     = s_t,
                            end_time       = e_t,
                            fps            = fps_z,
                            max_pixels     = args.max_pixels,
                            min_pixels     = args.min_pixels,
                            max_frames     = args.tool_max_frames_per_call,
                            storage_system = "local",
                        ): idx
                        for idx, (s, (s_t, e_t, fps_z)) in zoom_queue.items()
                    }
                    zoom_results = {futures[f]: f.result() for f in as_completed(futures)}

                for idx, (s, _) in zoom_queue.items():
                    result = zoom_results.get(idx)
                    if isinstance(result, dict):
                        times, frames = result["frame_time"], result["frames"]
                        s.prompt  += build_tool_response_turn(times, is_last=False)
                        s.images  += list(frames)
                        s.n_tool_calls += 1
                        next_active.append(s)
                    else:
                        # Zoom failed — finalize with R1
                        s.final_answer = extract_mc_answer(s.raw_output)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

            active = next_active

            # ══════════════════════════════════════════════════════════════
            # Round 2+: standard greedy continuation + optional iter_entropy gate
            # ══════════════════════════════════════════════════════════════
            for round_idx in range(2, args.max_rounds + 1):
                is_last = (round_idx == args.max_rounds)
                if not active:
                    break

                inputs  = [
                    {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                    for s in active
                ]
                # Select iter SamplingParams based on active stopping method
                if args.recursive_ae:
                    current_iter_sp = rae_iter_sp
                elif args.answer_wald:
                    current_iter_sp = wald_iter_sp
                else:
                    current_iter_sp = iter_sp
                outputs = llm.generate(inputs, current_iter_sp)

                zoom_queue2  = {}
                next_active2 = []

                for idx, (s, out) in enumerate(zip(active, outputs)):
                    text = out.outputs[0].text
                    s.n_rounds   = round_idx
                    s.raw_output = text
                    zoom_call = parse_zoom_call(text)

                    # ── recursive_ae: H_answer-based stopping ──────────── #
                    rae_force_stop   = False
                    rae_majority_ans = None
                    if args.recursive_ae and not is_last:
                        rae_texts = [o.text for o in out.outputs]
                        rae_feats = compute_answer_dist_entropy(rae_texts)
                        H_cur     = rae_feats['H_answer']
                        # Condition 1: confident enough
                        if H_cur < args.rae_low_threshold:
                            rae_force_stop   = True
                            rae_majority_ans = rae_feats['majority_answer']
                        # Condition 2: zoom not reducing uncertainty
                        elif s.prev_h_answer is not None and H_cur >= s.prev_h_answer:
                            rae_force_stop   = True
                            rae_majority_ans = rae_feats['majority_answer']
                        s.prev_h_answer = H_cur

                    # ── answer_wald: Wald JSD stopping ─────────────────── #
                    wald_force_stop  = False
                    wald_majority_ans = None
                    if args.answer_wald and not is_last:
                        wald_texts = [o.text for o in out.outputs]
                        wald_feats = compute_answer_dist_entropy(wald_texts)
                        cur_dist   = wald_feats['answer_dist']
                        if s.prev_answer_dist is not None:
                            jsd = compute_answer_jsd(s.prev_answer_dist, cur_dist)
                            if jsd < args.wald_threshold:
                                # Answer distribution converged → stop zooming
                                wald_force_stop   = True
                                wald_majority_ans = wald_feats['majority_answer']
                        s.prev_answer_dist = cur_dist

                    # ── iter_entropy convergence check ─────────────────── #
                    iter_force_stop = False
                    if args.iter_entropy and zoom_call is not None and not is_last:
                        tok_lps  = out.outputs[0].logprobs or []
                        i_feats  = compute_entropy_score(text, tok_lps, processor.tokenizer)
                        H_cur    = i_feats.get('mean_entropy', 0.0) or 0.0
                        s.round_entropies.append(H_cur)
                        if len(s.round_entropies) >= 2:
                            delta_H = s.round_entropies[-1] - s.round_entropies[-2]
                            if delta_H >= args.iter_threshold:
                                iter_force_stop = True

                    if (zoom_call is not None and not is_last
                            and not iter_force_stop
                            and not wald_force_stop
                            and not rae_force_stop):
                        end_pos  = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue2[idx] = (s, zoom_call)
                    else:
                        if rae_force_stop and rae_majority_ans:
                            s.final_answer = rae_majority_ans
                        elif wald_force_stop and wald_majority_ans:
                            s.final_answer = wald_majority_ans
                        else:
                            s.final_answer = extract_mc_answer(text)
                        s.acc_final = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                # ── iter_ae / iter_ae_delta: gate before R2+ zoom execution ─ #
                # Force-answer sampling → H_curr, then check:
                #   iter_ae:       H_curr < |threshold|        → already confident
                #   iter_ae_delta: H_curr >= H_prev            → zoom not helping
                # Either condition → skip this zoom.
                if (args.iter_ae or args.iter_ae_delta) and zoom_queue2:
                    iae_fa_inputs = []
                    iae_keys = list(zoom_queue2.keys())
                    for iae_idx in iae_keys:
                        s, _ = zoom_queue2[iae_idx]
                        fa_prompt = s.prompt + _FORCE_ANS_TURN
                        iae_fa_inputs.append({
                            "prompt": fa_prompt,
                            "multi_modal_data": {"image": list(s.images)},
                        })
                    iae_outputs = llm.generate(iae_fa_inputs, iter_ae_sp)
                    for iae_idx, fa_out in zip(iae_keys, iae_outputs):
                        s, _ = zoom_queue2[iae_idx]
                        texts_k = [o.text for o in fa_out.outputs]
                        feats   = compute_answer_dist_entropy(texts_k)
                        H_cur   = feats['H_answer']
                        skip = False
                        # Condition 1: absolute confidence (iter_ae)
                        if args.iter_ae and H_cur < abs(args.entropy_threshold):
                            skip = True
                        # Condition 2: zoom not reducing entropy (iter_ae_delta)
                        if args.iter_ae_delta and s.iter_ae_prev_h is not None and H_cur >= s.iter_ae_prev_h:
                            skip = True
                        # Update H_prev for next round regardless
                        s.iter_ae_prev_h = H_cur
                        if skip:
                            majority = feats['majority_answer']
                            s.final_answer = majority if majority else extract_mc_answer(s.raw_output)
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)
                            del zoom_queue2[iae_idx]

                if zoom_queue2:
                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {
                            ex.submit(
                                extract_video_clip,
                                video_path     = s.video_path,
                                start_time     = s_t,
                                end_time       = e_t,
                                fps            = fps_z,
                                max_pixels     = args.max_pixels,
                                min_pixels     = args.min_pixels,
                                max_frames     = args.tool_max_frames_per_call,
                                storage_system = "local",
                            ): idx
                            for idx, (s, (s_t, e_t, fps_z)) in zoom_queue2.items()
                        }
                        zoom_results2 = {futures[f]: f.result() for f in as_completed(futures)}

                    for idx, (s, _) in zoom_queue2.items():
                        result = zoom_results2.get(idx)
                        if isinstance(result, dict):
                            times, frames = result["frame_time"], result["frames"]
                            s.prompt  += build_tool_response_turn(times, is_last=False)
                            s.images  += list(frames)
                            s.n_tool_calls += 1
                            next_active2.append(s)
                        else:
                            s.final_answer = extract_mc_answer(s.raw_output)
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)

                active = next_active2

            # Force-finalize
            for s in active:
                s.final_answer = extract_mc_answer(s.raw_output)
                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                done.append(s)

            # ── Save ──────────────────────────────────────────────────── #
            for s in done:
                n_correct += int(s.acc_final or 0)
                if s.zoom_triggered:  n_zoom_triggered += 1
                if s.zoom_skipped:    n_zoom_skipped   += 1
                if not s.zoom_triggered and not s.zoom_skipped:
                    n_direct_answer += 1

                rec = {
                    "problem_id":      s.pid,
                    "gt":              s.gt,
                    "acc_final":       s.acc_final,
                    "final_answer":    s.final_answer,
                    "n_rounds":        s.n_rounds,
                    "n_tool_calls":    s.n_tool_calls,
                    "raw_output":      s.raw_output,
                    "zoom_triggered":  s.zoom_triggered,
                    "zoom_skipped":    s.zoom_skipped,
                    "hmm_score":       s.hmm_score,
                    "hmm_features":    s.hmm_features,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Summary ─────────────────────────────────────────────────────────── #
    n_this = len(all_records)
    print(f"\n{'='*60}")
    print(f"  HMM-Zoom inference ({n_this} new samples)")
    if n_this:
        print(f"  Accuracy (this run):   {n_correct}/{n_this} = {n_correct/n_this:.4f}")
        print(f"  Direct answer (no zoom call): {n_direct_answer}/{n_this} "
              f"({n_direct_answer/n_this*100:.1f}%)")
        print(f"  HMM skipped zoom:             {n_zoom_skipped}/{n_this} "
              f"({n_zoom_skipped/n_this*100:.1f}%)")
        print(f"  Zoom executed:                {n_zoom_triggered}/{n_this} "
              f"({n_zoom_triggered/n_this*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]

    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        n_zoom  = sum(1 for r in all_recs if r.get("zoom_triggered"))
        n_skip  = sum(1 for r in all_recs if r.get("zoom_skipped"))
        n_direct= sum(1 for r in all_recs if not r.get("zoom_triggered") and not r.get("zoom_skipped"))
        zoom_acc  = np.mean([r["acc_final"] for r in all_recs if r.get("zoom_triggered")]) if n_zoom else float('nan')
        skip_acc  = np.mean([r["acc_final"] for r in all_recs if r.get("zoom_skipped")])   if n_skip else float('nan')
        dir_acc   = np.mean([r["acc_final"] for r in all_recs if not r.get("zoom_triggered") and not r.get("zoom_skipped")]) if n_direct else float('nan')

        print(f"\n  Full file ({len(accs)} samples):  acc={np.mean(accs):.4f}")
        print(f"  ├─ Direct answer  ({n_direct:4d}):  acc={dir_acc:.4f}")
        print(f"  ├─ HMM skip zoom  ({n_skip:4d}):  acc={skip_acc:.4f}  (R1 answer used)")
        print(f"  └─ Zoom executed  ({n_zoom:4d}):  acc={zoom_acc:.4f}  (R2 answer used)")

        avg_rounds = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_tools  = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        print(f"\n  Avg rounds: {avg_rounds:.2f}  Avg tool calls: {avg_tools:.2f}")
        print(f"  Compute saving vs greedy: "
              f"~{(1 - avg_tools / 0.943) * 100:.0f}%% fewer zoom executions")

    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    summary = {
        "n_samples":      len(accs) if accs else 0,
        "accuracy":       float(np.mean(accs)) if accs else None,
        "mode":           "hmm_zoom_trigger",
        "hmm_threshold":  args.hmm_threshold,
        "zoom_triggered": n_zoom if accs else 0,
        "zoom_skipped":   n_skip if accs else 0,
        "direct_answer":  n_direct if accs else 0,
        "fps":            args.fps,
        "model":          args.model_path,
    }
    with open(os.path.join(args.output_dir, "hmm_zoom_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {args.output_dir}/hmm_zoom_summary.json")


if __name__ == "__main__":
    main()
