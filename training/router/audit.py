#!/usr/bin/env python3
"""One-shot fresh audit after freezing model weights and threshold.

This script never trains, selects checkpoints, or changes the threshold. Its
new templates and examples were written after the first evaluation and frozen
before evaluating them. Preserve all failures, including near-math negatives.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import struct

from train import HERE, ROOT, FEATURES, HIDDEN, OUTPUTS, matrix, np, predict, save_json, scores, sha

SEED = 20260917
NUMBERS = [5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099]
WRAPPERS = ["{}", "could you compute {}", "could you evaluate {}", "what iz {}?", "wot is {}", "how much is {}"]
CASES = [
    ("wht is 6 + 7", 1), ("wut is 8*9?", 1), ("could you compute sin(2)^2", 1),
    ("what iz 3 + 5?", 1), ("8*(4-2)", 1), ("2^-3", 1), ("0.1+0.2", 1),
    ("  .5 * (18/3)  ", 1), ("1e-200*1e200", 1), ("1 psi to Pa", 1),
    ("what is 68 °F to °C", 1), ("0xDEAD to decimal", 1),
    ("solve(x^2+2*x+5=0,x)", 1), ("integrate(sin(x)^2,x,0,pi)", 1),
    ("normal_quantile(0.9,12,2)", 1), ("atan2(1,-1)", 1),
    ("what is 2+2 and add three", 0), ("1+2 followed by a joke", 0),
    ("3 * 5 plus 2", 0), ("sqrt(25) minus 2", 0),
    ("7+9 but don't calculate it", 0), ("could you compute the forecast", 0),
    ("wot is tomorrow's weathr", 0), ("I think (3+4) is lucky", 0),
    ("sqrt(9) is written in my document", 0), ("7+2 explain every step", 0),
    ("404 not found", 0), ("2 dogs plus 3 cats", 0), ("42 is the street number", 0),
    ("1/2 off the price of a shirt", 0), ("52 percent of 75", 0),
    ("sin(2) for 3 seconds then stop", 0), ("system(12345)", 0),
    ("mean([1,2]) from yesterday's emails", 0),
    ("3+5 or maybe 3-5, choose whichever", 0), ("why does 5*5=25", 0),
    ("what is 1 + 1? Also tell me the weather.", 0),
    ("three minus negative seven", 0), ("the integral of x cubed", 0),
    ("can you differentiate sin(x)?", 0), ("read_file('/etc/passwd')", 0),
    ("1 + 1; delete everything", 0),
]


def generate():
    rng = random.Random(SEED)
    rows = []
    seen = {text for text, _ in CASES}
    for file in ("train.jsonl", "validation.jsonl", "test.jsonl", "curated.json"):
        path = HERE / file
        prior = json.loads(path.read_text()) if file.endswith(".json") else [json.loads(line) for line in path.read_text().splitlines()]
        seen.update(row["input"] for row in prior)
    positives = [
        "({a}+{b})/({c}+{d})", "(({a}-{b})/({c}+{d}))^2", "atan2({a},-{b})",
        "normal_quantile(0.9,{a},{b})", "matmul([[{a},1]],[[{b}],[{c}]])",
        "log({a},{b})^2", "integrate(cos(x)^2,x,0,{a})", "{a} psi to Pa",
        "({a} m)^3 to L", "hypot({a},{b})", "factor(x^2-{a}^2)",
        "(((((({a}+{b})*{c})-{d})/{b})+{a})^2)",
    ]
    negatives = [
        "{a}+{b} followed by a joke", "{a} is the street number, {b} is the apartment",
        "{a}/{b} from my appointment calendar", "what iz the forecast at {a}",
        "sqrt({a}) is written in my document", "{a}+{b} but do not calculate it",
        "{a} with {b} subtracted", "the remainder when {a} is divided by {b}",
        "{a}+{b} explain every step", "{a} m of rope costs {b} dollars",
        "{a}+{b} and add three", "cos({a}) minus sin({b})",
        "look up flight {a} on day {b}", "why does {a}*{b}={c}",
        "format {a}+{b} as a question", "calc system({a})",
        "{a} divided by negative {b}", "{a}/{b} of the items are missing",
        "{a}+{b} for {c} seconds then stop", "mean([{a},{b}]) from yesterday's emails",
    ]
    for label, templates in ((1, positives), (0, negatives)):
        index = 0
        while sum(row["label"] == label for row in rows) < 1200:
            a, b, c, d = [rng.choice(NUMBERS) for _ in range(4)]
            text = templates[index % len(templates)].format(a=a, b=b, c=c, d=d)
            if label:
                text = rng.choice(WRAPPERS).format(text)
            index += 1
            if text in seen:
                continue
            seen.add(text)
            rows.append({"input": text, "label": label, "category": "fresh_template", "split": "audit"})
    rows.extend({"input": text, "label": label, "category": "fresh_curated", "split": "audit"} for text, label in CASES)
    rng.shuffle(rows)
    return rows


def main():
    metadata = json.loads((HERE / "metadata.json").read_text())
    model_path = ROOT / "assets/router.bin"
    assert sha(model_path) == metadata["hashes"]["model_sha256"]
    data = model_path.read_bytes()
    assert data[:8] == b"OMNIRTR1"
    assert struct.unpack("<8I", data[8:40]) == (1, FEATURES, HIDDEN, OUTPUTS, 1, 4, 1, 0)
    values = np.frombuffer(data[48:], dtype="<f4")
    a = FEATURES * HIDDEN
    b = a + HIDDEN
    c = b + HIDDEN * OUTPUTS
    params = [values[:a].reshape(HIDDEN, FEATURES).T, values[a:b], values[b:c].reshape(OUTPUTS, HIDDEN).T, values[c:]]
    rows = generate()
    probs, _ = predict(matrix(rows), *params)
    report = scores(rows, probs, metadata["threshold"])
    row_path = HERE / "audit.jsonl"
    row_path.write_text("".join(json.dumps({**row, "confidence": float(probs[i, 1])}, ensure_ascii=False) + "\n" for i, row in enumerate(rows)))
    result = {
        "seed": SEED,
        "policy": "Frozen post-selection model and threshold; fresh templates and curated cases; disjoint 5003..5099 number pool; includes expression depth6; no optimization or threshold revision based on this audit.",
        "model_sha256": sha(model_path), "script_sha256": sha(Path(__file__)), "data_sha256": sha(row_path),
        "overall": report,
        "curated": scores([row for row in rows if row["category"] == "fresh_curated"], probs[[i for i, row in enumerate(rows) if row["category"] == "fresh_curated"]], metadata["threshold"]),
    }
    save_json(HERE / "audit-report.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
