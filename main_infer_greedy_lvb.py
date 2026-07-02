#!/usr/bin/env python3
"""
Wrapper: greedy baseline for LongVideoBench val (4–5 option A-E).
Patches extract_mc_answer to handle A-E before running main_infer_greedy.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import main_infer_adaptive_zoom as _az


def _extract_mc_answer_ae(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    frag = m.group(1) if m else text[-200:]
    hits = re.findall(r"\b([A-E])\b", frag)
    if hits:
        return hits[-1]
    hits = re.findall(r"([A-E])\.", text)
    return hits[-1] if hits else None


_az.extract_mc_answer = _extract_mc_answer_ae

import runpy
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "main_infer_greedy.py"),
    run_name="__main__",
)
