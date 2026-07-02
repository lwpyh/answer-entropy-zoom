#!/usr/bin/env python3
"""
analyze_tool_gating.py

Reframes the routing problem as binary classification:
  Positive : Δacc > 0  (tool HELPED  — should zoom)        n=27
  Negative : Δacc ≤ 0  (tool no-help or hurt — skip zoom)  n=270

For each of the 9 Phase-1 uncertainty metrics computes:
  - AUC-ROC  (rank-based, direction-aware)
  - Average Precision  (better for imbalanced classes)
  - Mann-Whitney U  p-value (non-parametric group comparison)
  - Mean in improved vs. not-improved group
  - Best threshold by F1 (precision/recall trade-off)

Also examines:
  - HURT sub-analysis: Δacc < 0 vs Δacc ≥ 0  (can we avoid harm?)
  - Feature combination: logistic regression AUC with all metrics
"""

import json, sys
import numpy as np
from pathlib import Path
from scipy import stats

JSONL = "/data/DERI-Gong/jh015/VideoZoomer/infer_results/tool_uncertainty/results_tool_uncertainty.jsonl"

METRICS = [
    ("U_rough",    "Logprob roughness",              "+"),
    ("U_dH",       "Entropy roughness",              "+"),
    ("U_ans_stab", "Answer-region roughness",        "+"),
    ("E_bar",      "Seq-level energy",               "+"),
    ("U_gap",      "Best-vs-second gap (↑=cert)",    "-"),
    ("U_ens",      "Sample weight entropy",          "+"),
    ("U_vote",     "Answer vote disagreement",       "+"),
    ("dE",         "Energy gap (↓=cert)",            "-"),
    ("N_eff",      "Effective candidates",           "+"),
]

# ── Load data ────────────────────────────────────────────────────────────── #
records = []
with open(JSONL) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

improved   = [r for r in records if r.get("delta_acc", 0) > 0]   # n=27
same       = [r for r in records if r.get("delta_acc", 0) == 0]  # n=261
hurt       = [r for r in records if r.get("delta_acc", 0) < 0]   # n=9
not_imp    = [r for r in records if r.get("delta_acc", 0) <= 0]  # n=270

print(f"Samples: {len(records)}")
print(f"  Improved  (Δacc>0) : {len(improved):>4}  ({100*len(improved)/len(records):.1f}%)")
print(f"  Same      (Δacc=0) : {len(same):>4}  ({100*len(same)/len(records):.1f}%)")
print(f"  Hurt      (Δacc<0) : {len(hurt):>4}  ({100*len(hurt)/len(records):.1f}%)")

# ── Helper: rank-based AUC ───────────────────────────────────────────────── #
def auc_roc(y_true, scores):
    """Compute AUC-ROC from scratch (no sklearn needed)."""
    pairs = sorted(zip(scores, y_true), key=lambda x: -x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp, auc_val = 0, 0, 0.0
    prev_fp = 0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
            auc_val += tp  # area under step
    return auc_val / (n_pos * n_neg)

def avg_precision(y_true, scores):
    """Average Precision (area under precision-recall curve)."""
    order = np.argsort(-np.array(scores))
    y_sorted = np.array(y_true)[order]
    precisions, recalls = [], []
    tp = 0
    for i, y in enumerate(y_sorted):
        if y == 1:
            tp += 1
        prec = tp / (i + 1)
        rec  = tp / max(sum(y_true), 1)
        precisions.append(prec)
        recalls.append(rec)
    # integrate: AP = sum of P(k)*ΔR(k)
    ap = 0.0
    prev_rec = 0.0
    for p, r in sorted(zip(precisions, recalls), key=lambda x: x[1]):
        ap += p * (r - prev_rec)
        prev_rec = r
    return ap

def best_f1_threshold(y_true, scores, direction="+"):
    """Find threshold maximising F1; return (threshold, precision, recall, f1)."""
    y_arr = np.array(y_true)
    s_arr = np.array(scores)
    if direction == "-":
        s_arr = -s_arr  # flip so larger = more positive
    thresholds = np.unique(s_arr)
    best = (0, 0, 0, 0, 0)  # thr, prec, rec, f1, acc
    for t in thresholds:
        pred = (s_arr >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y_arr == 1)))
        fp = int(np.sum((pred == 1) & (y_arr == 0)))
        fn = int(np.sum((pred == 0) & (y_arr == 1)))
        tn = int(np.sum((pred == 0) & (y_arr == 0)))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        acc  = (tp + tn) / len(y_arr)
        if f1 > best[3]:
            best = (t, prec, rec, f1, acc)
    return best

# ── Binary classification: improved vs. not-improved ────────────────────── #
print(f"\n{'='*90}")
print("BINARY GATING: can Phase-1 uncertainty predict Δacc > 0?")
print(f"  Positive (zoom → improved): n={len(improved)}")
print(f"  Negative (zoom → no-gain) : n={len(not_imp)}")
print(f"{'='*90}")
print(f"{'Metric':<40}  {'AUC':>5}  {'AP':>5}  {'MW p':>8}  {'mean+':>8}  {'mean-':>8}  {'BestF1':>7}  {'Prec':>6}  {'Rec':>6}")
print("-" * 90)

all_records_bin  = improved + not_imp
y_true_bin       = [1]*len(improved) + [0]*len(not_imp)

gating_results = []
for key, name, direction in METRICS:
    scores = [r.get(key, np.nan) for r in all_records_bin]
    valid  = [(y, s) for y, s in zip(y_true_bin, scores) if not np.isnan(s)]
    y_v, s_v = zip(*valid) if valid else ([], [])
    if not y_v:
        continue

    # AUC — if direction='-', higher score means LESS uncertain, flip
    if direction == "-":
        auc = auc_roc(list(y_v), [-s for s in s_v])
    else:
        auc = auc_roc(list(y_v), list(s_v))

    # AP
    if direction == "-":
        ap = avg_precision(list(y_v), [-s for s in s_v])
    else:
        ap = avg_precision(list(y_v), list(s_v))

    # Mann-Whitney
    pos_s = [s for y, s in zip(y_v, s_v) if y == 1]
    neg_s = [s for y, s in zip(y_v, s_v) if y == 0]
    _, mw_p = stats.mannwhitneyu(pos_s, neg_s, alternative='two-sided')

    mean_pos = np.mean(pos_s)
    mean_neg = np.mean(neg_s)

    # Best F1 threshold
    thr, prec, rec, f1, acc = best_f1_threshold(list(y_v), list(s_v), direction)
    star = "**" if mw_p < 0.05 else ("  " if mw_p < 0.10 else "  ")

    print(f"{name:<40}  {auc:.3f}  {ap:.3f}  {mw_p:>8.4f}{star}  {mean_pos:>8.4f}  {mean_neg:>8.4f}  {f1:>7.3f}  {prec:>6.3f}  {rec:>6.3f}")
    gating_results.append((key, name, direction, auc, ap, mw_p, mean_pos, mean_neg, f1, prec, rec))

# ── HURT sub-analysis ────────────────────────────────────────────────────── #
print(f"\n{'='*90}")
print("HURT PREVENTION: can Phase-1 uncertainty predict Δacc < 0?")
print(f"  Positive (zoom → hurt)  : n={len(hurt)}")
print(f"  Negative (zoom → ok)    : n={len(improved)+len(same)}")
print(f"{'='*90}")
print(f"{'Metric':<40}  {'AUC':>5}  {'MW p':>8}  {'mean hurt':>10}  {'mean ok':>10}")
print("-" * 70)

not_hurt = improved + same
all_records_hurt = hurt + not_hurt
y_true_hurt = [1]*len(hurt) + [0]*len(not_hurt)
for key, name, direction in METRICS:
    scores = [r.get(key, np.nan) for r in all_records_hurt]
    valid  = [(y, s) for y, s in zip(y_true_hurt, scores) if not np.isnan(s)]
    y_v, s_v = zip(*valid) if valid else ([], [])
    if not y_v:
        continue
    if direction == "-":
        auc = auc_roc(list(y_v), [-s for s in s_v])
    else:
        auc = auc_roc(list(y_v), list(s_v))
    pos_s = [s for y, s in zip(y_v, s_v) if y == 1]
    neg_s = [s for y, s in zip(y_v, s_v) if y == 0]
    _, mw_p = stats.mannwhitneyu(pos_s, neg_s, alternative='two-sided')
    mean_pos = np.mean(pos_s)
    mean_neg = np.mean(neg_s)
    star = " *" if mw_p < 0.05 else ("~" if mw_p < 0.10 else "  ")
    print(f"{name:<40}  {auc:.3f}  {mw_p:>8.4f}{star}  {mean_pos:>10.4f}  {mean_neg:>10.4f}")

# ── Combined feature: logistic regression AUC ───────────────────────────── #
print(f"\n{'='*90}")
print("COMBINED FEATURE IMPORTANCE (logistic regression, leave-one-out AUC)")
print(f"{'='*90}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X_all, y_all = [], []
    for r, y in zip(all_records_bin, y_true_bin):
        row = [r.get(key, np.nan) for key, _, _ in METRICS]
        if not any(np.isnan(row)):
            X_all.append(row)
            y_all.append(y)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Full-set AUC with standardised LR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train, test in cv.split(X_scaled, y_all):
        lr.fit(X_scaled[train], y_all[train])
        prob = lr.predict_proba(X_scaled[test])[:, 1]
        try:
            aucs.append(roc_auc_score(y_all[test], prob))
        except Exception:
            pass

    lr.fit(X_scaled, y_all)
    print(f"\n  5-fold CV AUC (all 9 features combined): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"\n  Logistic regression coefficients (standardised):")
    for (key, name, _), coef in zip(METRICS, lr.coef_[0]):
        print(f"    {name:<40}: {coef:>+.4f}")
except ImportError:
    print("  (sklearn not available — skipping combined analysis)")

# ── Best metric recommendation ───────────────────────────────────────────── #
print(f"\n{'='*90}")
print("RANKING BY AUC (improved vs. not-improved)")
print(f"{'='*90}")
ranked = sorted(gating_results, key=lambda x: -x[3])
for i, (key, name, direction, auc, ap, mw_p, mp, mn, f1, prec, rec) in enumerate(ranked, 1):
    sig = "**" if mw_p < 0.05 else ("~" if mw_p < 0.10 else "  ")
    print(f"  {i}. {name:<40}  AUC={auc:.3f}  AP={ap:.3f}  p={mw_p:.4f}{sig}")
print()
best = ranked[0]
print(f"  => Best single predictor: {best[1]}")
print(f"     AUC={best[3]:.3f}, AP={best[4]:.3f}, MW p={best[5]:.4f}")
print(f"     Mean (improved)={best[6]:.4f}  Mean (not)={best[7]:.4f}")
print(f"     Best-F1={best[8]:.3f}  Prec={best[9]:.3f}  Rec={best[10]:.3f}")
