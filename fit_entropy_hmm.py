#!/usr/bin/env python3
"""
fit_entropy_hmm.py

Fit an unsupervised GaussianHMM(K=2) on token-entropy sequences from
entropy_zoom_v2 results.  No GT labels needed.

States discovered:
  C (confident) = low-entropy state
  U (uncertain) = high-entropy state

Saves:
  entropy_hmm_artifacts/hmm_k2.pkl      — fitted GaussianHMM
  entropy_hmm_artifacts/hmm_analysis.json — feature stats + analysis
"""

import json, pickle, os, re
import numpy as np
from pathlib import Path

RESULTS  = "infer_results/entropy_zoom_v2/results_hmm_zoom_v2.jsonl"
GREEDY   = "infer_results/greedy_baseline/results_greedy.jsonl"
HMM_RES  = "infer_results/hmm_zoom_v2/results_hmm_zoom_v2.jsonl"
OUT_DIR  = "entropy_hmm_artifacts"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────── #
print("[data] Loading results ...")
ent_recs  = {r['problem_id']: r for r in (json.loads(l) for l in open(RESULTS))}
grdy_recs = {r['problem_id']: r for r in (json.loads(l) for l in open(GREEDY))}
hmm_recs  = {r['problem_id']: r for r in (json.loads(l) for l in open(HMM_RES))}

# ── Extract entropy sequences ─────────────────────────────────────────────── #
seqs, pids = [], []
for pid, r in ent_recs.items():
    H = r.get('hmm_features', {}).get('token_entropies', [])
    if len(H) < 4:
        continue
    seqs.append(np.array(H, dtype=np.float32))
    pids.append(pid)

print(f"[data] {len(seqs)} sequences, lengths: "
      f"mean={np.mean([len(s) for s in seqs]):.0f}  "
      f"min={min(len(s) for s in seqs)}  max={max(len(s) for s in seqs)}")

# ── Fit GaussianHMM(K=2) ─────────────────────────────────────────────────── #
from hmmlearn.hmm import GaussianHMM

# Clip near-zero entropy tokens (punctuation/articles) so HMM splits on
# content-word uncertainty rather than punctuation vs content.
# Keep only tokens with entropy > 0.05 (essentially removes near-deterministic tokens).
seqs_clipped = [np.clip(s, 0.05, None) for s in seqs]

X_flat   = np.concatenate([s.reshape(-1, 1) for s in seqs_clipped])
lengths  = [len(s) for s in seqs_clipped]

print(f"[hmm] Fitting GaussianHMM K=2 on {len(X_flat)} tokens (clipped entropy ≥ 0.05) ...")
hmm = GaussianHMM(
    n_components=2,
    covariance_type='diag',
    n_iter=200,
    tol=1e-4,
    random_state=42,
)
hmm.fit(X_flat, lengths)

# Identify which state is C (confident=low entropy) vs U (uncertain=high entropy)
means = hmm.means_.flatten()
C_state = int(np.argmin(means))
U_state = int(np.argmax(means))
print(f"[hmm] State means: {means}  → C={C_state} (μ={means[C_state]:.3f}), U={U_state} (μ={means[U_state]:.3f})")
print(f"[hmm] Transition matrix:\n{hmm.transmat_}")

# ── Decode + extract features ─────────────────────────────────────────────── #
def hmm_features(H_seq, hmm, C_state, U_state):
    H_raw = np.array(H_seq, dtype=np.float32)
    H = np.clip(H_raw, 0.05, None).reshape(-1, 1)
    T = len(H)
    q = max(T // 4, 1)

    states = np.atleast_1d(hmm.decode(H, algorithm='viterbi')[0])

    C_ratio     = float((states == C_state).mean())
    ends_in_C   = int(states[-1] == C_state)
    last_q_C    = float((states[-q:] == C_state).mean())

    # U→C transitions (convergence events)
    n_UC = sum(1 for a, b in zip(states[:-1], states[1:])
               if a == U_state and b == C_state)
    n_CU = sum(1 for a, b in zip(states[:-1], states[1:])
               if a == C_state and b == U_state)
    n_trans = max(T - 1, 1)

    # Entropy statistics
    mean_H  = float(np.mean(H_seq))
    last_H  = float(np.mean(H_seq[-q:]))
    first_H = float(np.mean(H_seq[:q]))
    trend   = last_H - first_H
    std_H   = float(np.std(H_seq))

    # Composite HMM score: higher = more confident = skip zoom
    # Weights tuned from Cohen's d analysis:
    #   last_H:  d=-0.846  → weight on -last_H
    #   trend:   d=-0.501  → weight on -trend
    #   C_ratio: correlated with both → +C_ratio
    #   ends_in_C: binary strong signal → +ends_in_C
    score = (
        -1.5  * last_H
        -0.6  * trend
        +0.8  * C_ratio
        +0.4  * ends_in_C
        +0.5  * last_q_C
        -0.3  * (n_CU / n_trans)   # C→U divergence is bad
    )

    return {
        'hmm_score':   score,
        'C_ratio':     C_ratio,
        'ends_in_C':   ends_in_C,
        'last_q_C':    last_q_C,
        'n_UC':        n_UC,
        'n_CU':        n_CU,
        'mean_H':      mean_H,
        'last_H':      last_H,
        'trend':       trend,
        'std_H':       std_H,
        'state_seq':   states.tolist(),
    }

rows = []
for pid, seq in zip(pids, seqs):
    if pid not in grdy_recs:
        continue
    feats = hmm_features(seq, hmm, C_state, U_state)

    # Ground-truth label: skip_should = 1 if zoom doesn't help
    g_acc  = float(grdy_recs[pid]['acc_final'])
    hr = hmm_recs.get(pid)
    r1_acc = float(hr['acc_final']) if (hr and hr.get('zoom_skipped')) else None

    feats['pid']       = pid
    feats['g_acc']     = g_acc
    feats['r1_acc']    = r1_acc
    feats['skip_should'] = int(g_acc <= r1_acc) if r1_acc is not None else None
    rows.append(feats)

# ── Feature discrimination analysis ──────────────────────────────────────── #
labeled = [r for r in rows if r['skip_should'] is not None]
skip1   = [r for r in labeled if r['skip_should'] == 1]
skip0   = [r for r in labeled if r['skip_should'] == 0]
print(f"\n[analysis] labeled={len(labeled)}  skip_ok={len(skip1)}  skip_wrong={len(skip0)}")

feat_names = ['hmm_score','C_ratio','ends_in_C','last_q_C','last_H','trend','mean_H','std_H']
print(f"\n{'Feature':<14}  {'skip=1':>10}  {'skip=0':>10}  {'diff':>8}  {'Cohen_d':>8}")
print('-' * 60)
for f in feat_names:
    v1 = np.array([r[f] for r in skip1])
    v0 = np.array([r[f] for r in skip0])
    if len(v0) == 0:
        continue
    diff = np.mean(v1) - np.mean(v0)
    pooled = np.sqrt((np.std(v1)**2 + np.std(v0)**2) / 2 + 1e-12)
    d = diff / pooled
    print(f"{f:<14}  {np.mean(v1):>10.4f}  {np.mean(v0):>10.4f}  {diff:>+8.4f}  {d:>+8.3f}")

# ── Threshold simulation (oracle) ────────────────────────────────────────── #
all_scores  = np.array([r['hmm_score'] for r in rows])
all_g_accs  = np.array([r['g_acc']     for r in rows])
all_r1_accs = np.array([
    r['r1_acc'] if r['r1_acc'] is not None else r['g_acc']
    for r in rows
])

print(f"\n[simulate] hmm_score: mean={np.mean(all_scores):.3f}  std={np.std(all_scores):.3f}  "
      f"min={np.min(all_scores):.3f}  max={np.max(all_scores):.3f}")
print(f"\n{'skip%':>7}  {'threshold':>10}  {'sim_acc':>9}  {'n_skip':>7}  {'skip_wrong':>11}")
print('-' * 52)

greedy_acc = np.mean(all_g_accs)
print(f"  greedy baseline:  {greedy_acc:.4f}")

for pct in [0.05, 0.10, 0.13, 0.15, 0.20, 0.25]:
    thr      = float(np.percentile(all_scores, 100 * (1 - pct)))
    skip_m   = all_scores >= thr
    sim_accs = np.where(skip_m, all_r1_accs, all_g_accs)
    # wrong skip: skipped but g_acc > r1_acc
    wrong    = int(np.sum(skip_m & (all_g_accs > all_r1_accs)))
    print(f"  {pct:>5.0%}  {thr:>10.4f}  {np.mean(sim_accs):>9.4f}  "
          f"{skip_m.sum():>7}  {wrong:>11}")

# ── Save ─────────────────────────────────────────────────────────────────── #
hmm_path = os.path.join(OUT_DIR, "hmm_k2.pkl")
with open(hmm_path, 'wb') as f:
    pickle.dump({'hmm': hmm, 'C_state': C_state, 'U_state': U_state}, f)
print(f"\n[save] HMM saved to {hmm_path}")

analysis = {
    'C_state': C_state,
    'U_state': U_state,
    'C_mean':  float(means[C_state]),
    'U_mean':  float(means[U_state]),
    'transmat': hmm.transmat_.tolist(),
    'feature_stats': {
        f: {
            'skip1_mean': float(np.mean([r[f] for r in skip1])),
            'skip0_mean': float(np.mean([r[f] for r in skip0])) if skip0 else None,
        }
        for f in feat_names
    },
    'n_seqs': len(seqs),
    'n_labeled': len(labeled),
}
with open(os.path.join(OUT_DIR, "hmm_analysis.json"), 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"[save] Analysis saved to {OUT_DIR}/hmm_analysis.json")
