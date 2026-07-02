#!/usr/bin/env python3
"""Convert LongVideoBench val JSON to VideoZoomer format."""
import json
from collections import Counter

SOURCE  = "/data/DERI-Gong/jh015/VideoZoomer/lvb_val.json"
VIDEO_ROOT = "/gpfs/scratch/acw652/hf_home_longvb/datasets/longvideobench/videos"
OUTPUT  = "/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/lvb_val_fixed.json"

LETTERS = "ABCDE"

with open(SOURCE, encoding="utf-8") as f:
    raw = json.load(f)

print(f"Loaded {len(raw)} records")
n_cands = Counter(len(r.get("candidates", [])) for r in raw)
print(f"Candidates distribution: {dict(sorted(n_cands.items()))}")

records = []
for i, r in enumerate(raw):
    candidates  = r["candidates"]
    correct_idx = int(r["correct_choice"])
    solution    = LETTERS[correct_idx]
    question    = r["question"].strip()

    # Build option lines: (A) text\n(B) text\n...
    opts = "\n".join(f"({LETTERS[j]}) {c.strip()}" for j, c in enumerate(candidates))
    problem = f"<image>{question}\n{opts}\n"

    video_name = r["video_path"]  # bare filename
    record = {
        "videos": [f"{VIDEO_ROOT}/{video_name}"],
        "problem": problem,
        "solution": solution,
        "data_source": "lvb",
        "extra_info": {
            "question_id":      r.get("id", f"{r['video_id']}_{i}"),
            "video_id":         r["video_id"],
            "question_category": r.get("question_category", ""),
            "level":            r.get("level", ""),
            "topic_category":   r.get("topic_category", ""),
            "duration":         float(r["duration"]) if r.get("duration") is not None else None,
            "duration_group":   r.get("duration_group"),
            "type":             r.get("type", ""),
        },
        "problem_id": f"lvb_{r.get('id', i)}_{i}",
    }
    records.append(record)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} records → {OUTPUT}")
print("\nSample[0]:")
print(json.dumps(records[0], ensure_ascii=False, indent=2)[:800])
print("\nSample[100]:")
print(json.dumps(records[100], ensure_ascii=False, indent=2)[:600])
