#!/usr/bin/env python3
"""
analyze_sentence_entropy_hmm.py

Use existing entropy_zoom_v2 token_entropies + think_text to build
sentence-level entropy sequences, then fit GaussianHMM(K=2).

No model reload needed — works entirely on stored results.
"""

import json, pickle, os, re
import numpy as np
from pathlib import Path

RESULTS = "infer_results/entropy_zoom_v2/results_hmm_zoom_v2.jsonl"
GREEDY  = "infer_results/greedy_baseline/results_greedy.jsonl"
HMM_RES = "infer_results/hmm_zoom_v2/results_hmm_zoom_v2.jsonl"
OUT_DIR = "entropy_hmm_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Sentence splitter ─────────────────────────────────────────────────────── #

def split_sentences(text: str) -> list:
    """Split think chain into sentences. Returns non-empty sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 8]
    return sents


def map_sentences_to_entropies(think_text: str, token_ents: list, sents: list) -> list:
    """
    Approximate token→sentence mapping using char/token ratio.
    Returns list of per-sentence mean entropy.
    """
    T = len(token_ents)
    if T == 0 or not sents:
        return []
    total_chars = max(len(think_text), 1)
    chars_per_tok = total_chars / T

    sent_entropies = []
    offset_tok = 0
    for s in sents:
        n_tok = max(1, round(len(s) / chars_per_tok))
        end   = min(offset_tok + n_tok, T)
        seg   = token_ents[offset_tok:end]
        if seg:
            sent_entropies.append(float(np.mean(seg)))
        offset_tok = end
        if offset_tok >= T:
            break
    return sent_entropies


# ── Load data ─────────────────────────────────────────────────────────────── #
print("[data] Loading ...")
ent_recs  = {r['problem_id']: r for r in (json.loads(l) for l in open(RESULTS))}
grdy_recs = {r['problem_id']: r for r in (json.loads(l) for l in open(GREEDY))}
hmm_recs  = {r['problem_id']: r for r in (json.loads(l) for l in open(HMM_RES))}

# ── Build sentence-level entropy sequences ────────────────────────────────── #
seqs, pids, sent_counts = [], [], []
for pid, r in ent_recs.items():
    feats      = r.get('hmm_features', {})
    token_ents = feats.get('token_entropies', [])
    think_text = feats.get('think_text', '')
    if not token_ents or not think_text:
        continue
    sents     = split_sentences(think_text)
    sent_ents = map_sentences_to_entropies(think_text, token_ents, sents)
    if len(sent_ents) < 2:
        continue
    seqs.append(np.array(sent_ents, dtype=np.float32))
    pids.append(pid)
    sent_counts.append(len(sent_ents))

print(f"[data] {len(seqs)} samples, sentence count: "
      f"mean={np.mean(sent_counts):.1f}  min={min(sent_counts)}  max={max(sent_counts)}")

# ── Fit GaussianHMM(K=2) on sentence-level entropy ───────────────────────── #
from hmmlearn.hmm import GaussianHMM

X_flat  = np.concatenate([s.reshape(-1, 1) for s in seqs])
lengths = [len(s) for s in seqs]

print(f"[hmm] Fitting K=2 on {len(X_flat)} sentences ...")
print(f"[hmm] Sentence entropy: mean={X_flat.mean():.3f}  std={X_flat.std():.3f}  "
      f"min={X_flat.min():.3f}  max={X_flat.max():.3f}")

hmm = GaussianHMM(n_components=2, covariance_type='diag',
                  n_iter=300, tol=1e-4, random_state=42)
hmm.fit(X_flat, lengths)

means = hmm.means_.flatten()
C_state = int(np.argmin(means))
U_state = int(np.argmax(means))
print(f"[hmm] State means: C={means[C_state]:.3f}  U={means[U_state]:.3f}")
print(f"[hmm] Transition matrix:\n{hmm.transmat_.round(3)}")

# ── Extract features per sample ───────────────────────────────────────────── #
def extract_features(sent_ents, hmm, C_state, U_state):
    T  = len(sent_ents)
    q  = max(T // 4, 1)
    H  = np.array(sent_ents, dtype=np.float32)

    states = np.atleast_1d(hmm.decode(H.reshape(-1, 1), algorithm='viterbi')[0])

    C_ratio   = float((states == C_state).mean())
    ends_in_C = int(states[-1] == C_state)
    last_q_C  = float((states[-q:] == C_state).mean())
    n_UC = sum(1 for a, b in zip(states[:-1], states[1:]) if a==U_state and b==C_state)
    n_CU = sum(1 for a, b in zip(states[:-1], states[1:]) if a==C_state and b==U_state)
    n_tr = max(T-1, 1)

    last_H  = float(np.mean(H[-q:]))
    first_H = float(np.mean(H[:q]))
    trend   = last_H - first_H
    mean_H  = float(np.mean(H))
    std_H   = float(np.std(H))

    # Composite score
    score = (
        -1.5 * last_H
        -0.6 * trend
        +0.8 * C_ratio
        +0.4 * ends_in_C
        +0.5 * last_q_C
        -0.3 * (n_CU / n_tr)
    )
    return {
        'sent_hmm_score': score,
        'C_ratio':        C_ratio,
        'ends_in_C':      ends_in_C,
        'last_q_C':       last_q_C,
        'n_UC':           n_UC,
        'n_CU':           n_CU,
        'last_H':         last_H,
        'trend':          trend,
        'mean_H':         mean_H,
        'std_H':          std_H,
        'state_seq':      states.tolist(),
        'T_sents':        T,
    }

rows = []
for pid, seq in zip(pids, seqs):
    if pid not in grdy_recs:
        continue
    feats = extract_features(seq, hmm, C_state, U_state)

    g_acc  = float(grdy_recs[pid]['acc_final'])
    hr     = hmm_recs.get(pid)
    r1_acc = float(hr['acc_final']) if (hr and hr.get('zoom_skipped')) else None

    feats['pid']         = pid
    feats['g_acc']       = g_acc
    feats['r1_acc']      = r1_acc
    feats['skip_should'] = int(g_acc <= r1_acc) if r1_acc is not None else None
    # Also store token-level last_H for comparison
    er_feats = ent_recs[pid].get('hmm_features', {})
    tok_ents = er_feats.get('token_entropies', [])
    if tok_ents:
        q_tok = max(len(tok_ents)//4, 1)
        feats['token_last_H'] = float(np.mean(tok_ents[-q_tok:]))
    else:
        feats['token_last_H'] = None
    rows.append(feats)

# ── Discriminability analysis ─────────────────────────────────────────────── #
labeled = [r for r in rows if r['skip_should'] is not None]
skip1   = [r for r in labeled if r['skip_should'] == 1]
skip0   = [r for r in labeled if r['skip_should'] == 0]
print(f"\n[analysis] labeled={len(labeled)}  skip_ok={len(skip1)}  skip_wrong={len(skip0)}")

feat_names = ['sent_hmm_score','C_ratio','ends_in_C','last_q_C',
              'last_H','trend','mean_H','std_H','token_last_H']
print(f"\n{'Feature':<18}  {'skip=1':>10}  {'skip=0':>10}  {'diff':>8}  {'Cohen_d':>8}")
print('-' * 65)
for f in feat_names:
    v1 = np.array([r[f] for r in skip1 if r.get(f) is not None])
    v0 = np.array([r[f] for r in skip0 if r.get(f) is not None])
    if len(v0) == 0: continue
    diff   = np.mean(v1) - np.mean(v0)
    pooled = np.sqrt((np.std(v1)**2 + np.std(v0)**2) / 2 + 1e-12)
    d = diff / pooled
    print(f"{f:<18}  {np.mean(v1):>10.4f}  {np.mean(v0):>10.4f}  {diff:>+8.4f}  {d:>+8.3f}")

# ── Oracle threshold simulation ───────────────────────────────────────────── #
all_scores  = np.array([r['sent_hmm_score'] for r in rows])
all_g_accs  = np.array([r['g_acc'] for r in rows])
all_r1_accs = np.array([
    r['r1_acc'] if r['r1_acc'] is not None else r['g_acc']
    for r in rows
])

print(f"\n[simulate] sent_hmm_score: mean={np.mean(all_scores):.3f}  "
      f"std={np.std(all_scores):.3f}  min={np.min(all_scores):.3f}  max={np.max(all_scores):.3f}")
print(f"\n{'skip%':>7}  {'threshold':>10}  {'sim_acc':>9}  {'n_skip':>7}  {'skip_wrong':>11}")
print('-' * 52)
print(f"  greedy baseline:  {np.mean(all_g_accs):.4f}")
for pct in [0.05, 0.10, 0.13, 0.15, 0.20, 0.25]:
    thr    = float(np.percentile(all_scores, 100*(1-pct)))
    skip_m = all_scores >= thr
    sim    = np.where(skip_m, all_r1_accs, all_g_accs)
    wrong  = int(np.sum(skip_m & (all_g_accs > all_r1_accs)))
    print(f"  {pct:>5.0%}  {thr:>10.4f}  {np.mean(sim):>9.4f}  {skip_m.sum():>7}  {wrong:>11}")

# ── Save ─────────────────────────────────────────────────────────────────── #
pkl_path = os.path.join(OUT_DIR, "sent_hmm_k2.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump({'hmm': hmm, 'C_state': C_state, 'U_state': U_state}, f)
print(f"\n[save] Saved to {pkl_path}")
