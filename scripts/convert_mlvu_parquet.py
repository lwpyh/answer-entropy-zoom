#!/usr/bin/env python3
"""Convert MLVU test parquet to VideoZoomer JSON format."""
import json
import sys
import pyarrow.parquet as pq

PARQUET = "/data/DERI-Gong/jh015/VideoZoomer/mlvu_test.parquet"
VIDEO_ROOT = "/gpfs/scratch/acw652/mlvu_test"
OUTPUT = "/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/mlvu_test_fixed.json"

table = pq.read_table(PARQUET)
df = table.to_pydict()

keys = list(df.keys())
n = len(df[keys[0]])
print(f"Loaded {n} rows, columns: {keys}")

# Count answer distribution
from collections import Counter
ans_dist = Counter(df["answer"])
print(f"Answer distribution: {dict(sorted(ans_dist.items()))}")
print(f"Task types: {dict(Counter(df['task_type']))}")

records = []
for i in range(n):
    video_name = df["video_name"][i]
    question   = df["question"][i]
    answer     = df["answer"][i]
    task_type  = df["task_type"][i]
    question_id = df["question_id"][i]
    duration   = df["duration"][i]

    # Build problem: add <image> prefix (VideoZoomer convention)
    problem = f"<image>{question.strip()}\n"

    record = {
        "videos": [f"{VIDEO_ROOT}/{video_name}"],
        "problem": problem,
        "solution": answer,
        "data_source": "mlvu",
        "extra_info": {
            "task_type": task_type,
            "question_id": question_id,
            "duration": float(duration) if duration is not None else None,
        },
        "problem_id": f"mlvu_{question_id}_{i}",
    }
    records.append(record)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} records → {OUTPUT}")
print("Sample[0]:")
print(json.dumps(records[0], ensure_ascii=False, indent=2)[:600])
print("Sample[-1]:")
print(json.dumps(records[-1], ensure_ascii=False, indent=2)[:400])
