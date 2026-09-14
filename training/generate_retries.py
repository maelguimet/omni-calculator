#!/usr/bin/env python3
"""Prepare source-preserving retry examples matching the Rust prompt exactly.

These are a supplemental candidate dataset. They are not part of the v1 run.
"""

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent
examples = [json.loads(line) for line in (ROOT / "train.jsonl").read_text().splitlines()]
rng = random.Random(812715)
rng.shuffle(examples)
errors = [
    "There is no complete literal expression to evaluate. Use function, convert, or algebra to interpret the requested operation.",
    "The tool changed or omitted a number. Copy every input quantity exactly, without calculating it.",
    "Algebra must preserve the expression exactly as written, including parentheses.",
]
result = []
for example in examples:
    calls = example["answers"]
    if not calls or calls[0]["name"] == "evaluate":
        continue
    error = rng.choice(errors if calls[0]["name"] == "algebra" else errors[:2])
    query = (f"Calculation request: {example['query']}\n"
             f"Previous tool call was rejected: {error}\n"
             "Correct the tool call, preserving all quantities and the requested operation.")
    result.append(dict(example, query=query, _family="retry_" + example["_family"]))
    if len(result) == 1536:
        break
(ROOT / "retry-supplement.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in result) + "\n")
print(f"Wrote {len(result)} supplemental retry examples; v1 train.jsonl unchanged.")
