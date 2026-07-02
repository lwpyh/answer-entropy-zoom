#!/usr/bin/env python3
"""Convert MLVU-Dev HF arrow cache to VideoZoomer JSON format."""
import json
import os
from collections import Counter

import pyarrow as pa

ARROW  = "/data/home/acw652/.cache/huggingface/datasets/sy1998___mlvu_dev/default/0.0.0/96207eb9aa7101e2a495dd147684a7e618c79e12/mlvu_dev-test.arrow"
VIDEO_ROOT = "/data/home/acw652/.cache/huggingface/mlvu"
OUTPUT = "/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/mlvu_dev_fixed.json"

t = pa.ipc.open_stream(ARROW).read_all()
print(f"Loaded {t.num_rows} rows, columns: {t.schema.names}")

answer_dist = Counter(t["answer"][i].as_py() for i in range(t.num_rows))
print(f"Answer distribution: {dict(sorted(answer_dist.items()))}")
task_dist = Counter(t["task_type"][i].as_py() for i in range(t.num_rows))
print(f"Task types: {dict(sorted(task_dist.items()))}")

records = []
for i in range(t.num_rows):
    video_name  = t["video_name"][i].as_py()
    question    = t["question"][i].as_py().strip()
    answer      = t["answer"][i].as_py().strip()
    task_type   = t["task_type"][i].as_py()
    question_id = t["question_id"][i].as_py()
    duration    = t["duration"][i].as_py()

    video_path = os.path.join(VIDEO_ROOT, video_name)

    problem = f"<image>{question}\n"

    record = {
        "videos": [video_path],
        "problem": problem,
        "solution": answer,
        "data_source": "mlvu_dev",
        "extra_info": {
            "task_type":   task_type,
            "question_id": question_id,
            "duration":    float(duration) if duration is not None else None,
        },
        "problem_id": f"mlvu_dev_{question_id}_{i}",
    }
    records.append(record)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(records)} records → {OUTPUT}")
print("\nSample[0]:")
print(json.dumps(records[0], ensure_ascii=False, indent=2)[:600])
print("\nSample[-1]:")
print(json.dumps(records[-1], ensure_ascii=False, indent=2)[:400])

# Spot-check video existence
missing = sum(1 for r in records[:20] if not os.path.exists(r["videos"][0]))
print(f"\nMissing videos (first 20): {missing}")
