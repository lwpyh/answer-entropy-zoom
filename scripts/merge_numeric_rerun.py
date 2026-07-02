"""
Merge re-run numeric results with existing YouTube results.

Usage:
    python scripts/merge_numeric_rerun.py greedy
    python scripts/merge_numeric_rerun.py branch
"""
import json
import sys
import os

BASE = '/data/DERI-Gong/jh015/VideoZoomer/infer_results'

CONFIGS = {
    'greedy': {
        'original': f'{BASE}/greedy_baseline/results_greedy.jsonl',
        'rerun':    f'{BASE}/greedy_numeric_rerun/results_greedy.jsonl',
        'merged':   f'{BASE}/greedy_baseline/results_greedy_merged.jsonl',
    },
    'branch': {
        'original': f'{BASE}/branch_baseline/results_branch.jsonl',
        'rerun':    f'{BASE}/branch_numeric_rerun/results_branch.jsonl',
        'merged':   f'{BASE}/branch_baseline/results_branch_merged.jsonl',
    },
}

def is_numeric(pid):
    parts = pid.split('_')
    return len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main(mode):
    cfg = CONFIGS[mode]
    original = load_jsonl(cfg['original'])
    rerun    = load_jsonl(cfg['rerun'])

    # Index rerun by problem_id
    rerun_by_pid = {r['problem_id']: r for r in rerun}

    merged = []
    replaced = 0
    for rec in original:
        pid = rec['problem_id']
        if is_numeric(pid) and pid in rerun_by_pid:
            merged.append(rerun_by_pid[pid])
            replaced += 1
        else:
            merged.append(rec)

    # Add any rerun records not in original (shouldn't happen, but just in case)
    orig_pids = {r['problem_id'] for r in original}
    for pid, rec in rerun_by_pid.items():
        if pid not in orig_pids:
            merged.append(rec)
            print(f'  [extra] {pid}')

    # Compute accuracy
    accs = [float(r['acc_final']) for r in merged if r.get('acc_final') is not None]
    numeric_new = [r for r in merged if is_numeric(r['problem_id'])]
    numeric_acc = sum(float(r['acc_final']) for r in numeric_new) / len(numeric_new) if numeric_new else 0
    youtube = [r for r in merged if not is_numeric(r['problem_id'])]
    youtube_acc = sum(float(r['acc_final']) for r in youtube) / len(youtube) if youtube else 0

    with open(cfg['merged'], 'w') as f:
        for rec in merged:
            f.write(json.dumps(rec) + '\n')

    print(f'[{mode}] merged {len(merged)} records -> {cfg["merged"]}')
    print(f'  replaced numeric: {replaced}/{len(rerun)}')
    print(f'  overall acc:  {sum(accs)/len(accs):.4f}  (n={len(accs)})')
    print(f'  numeric acc:  {numeric_acc:.4f}  (n={len(numeric_new)})')
    print(f'  youtube acc:  {youtube_acc:.4f}  (n={len(youtube)})')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'greedy'
    assert mode in CONFIGS, f'mode must be greedy or branch, got {mode}'
    main(mode)
