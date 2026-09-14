#!/usr/bin/env python3
"""Whitespace-robust reference classifier, with frozen v1 preserved separately."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import struct
import importlib.util
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "4")
source = Path(__file__).resolve().parents[1] / "router/train.py"
spec = importlib.util.spec_from_file_location("reference_v1_training", source)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20260915
original.HIDDEN = 48
FEATURES, HIDDEN = original.FEATURES, original.HIDDEN
LEX = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_µ°][A-Za-z_0-9µ°]*|[^\s]")
NUMERAL = re.compile(r"(?<![A-Za-z0-9_])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
original_feature_indices = original.feature_indices


def feature_indices(text):
    # Tool-reference decisions need not depend on the quantities. Arithmetic
    # still receives the untouched input, and signs remain model features.
    return original_feature_indices(NUMERAL.sub("0", text))


original.feature_indices = feature_indices
WRAPPERS = {
    "train": ["{}", "what is {}", "wat iz {}", "what's {}", "calculate {}", "compute {}", "evaluate {}", "please calculate {}", "{}?", "what is {}?"],
    "validation": ["please compute {}", "can you calculate {}", "{} please", "could you evaluate {}?", "wht is {}"],
    "test": ["WHAT IS {}?", "wat is {}?", "could you calculate {}", "please evaluate {}", "can you compute {}?"],
}

SCIENTIFIC_UNARY = "sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh sqrt cbrt exp ln log log2 log10 abs floor ceil trunc sign sind cosd tand asind acosd atand sec csc cot gamma lgamma erf erfc factorial conj conjugate real imag arg".split()
SCIENTIFIC_BINARY = "log round mod gcd lcm ncr npr atan2 hypot beta".split()
SCIENTIFIC_LIST = "sum product mean median variance stdev sample_variance sample_stdev min max rms geomean harmonic_mean norm normalize".split()
SCIENTIFIC_MATRIX = "det inverse transpose trace eigenvalues singular_values rank condition".split()


def scientific(rng, split):
    a, b, c = (number(rng, split) for _ in range(3))
    group = rng.randrange(8)
    if group == 0:
        return f"{rng.choice(SCIENTIFIC_UNARY)}({a})"
    if group == 1:
        return f"{rng.choice(SCIENTIFIC_BINARY)}({a},{b})"
    if group == 2:
        return f"{rng.choice(SCIENTIFIC_LIST)}([{a},{b},{c}])"
    if group == 3:
        return f"{rng.choice(SCIENTIFIC_MATRIX)}([[{a},{b}],[{c},{a}]])"
    if group == 4:
        name = rng.choice("normal student_t chi2 exponential uniform beta gamma weibull f".split())
        params = {"normal": f"{a},2", "student_t": "7", "chi2": "4", "exponential": "2", "uniform": "0,1", "beta": "2,3", "gamma": "3,2", "weibull": "2,3", "f": "5,7"}[name]
        operation = rng.choice(["pdf", "cdf", "quantile"])
        x = "0.975" if operation == "quantile" else a
        return f"{name}_{operation}({x},{params})"
    if group == 5:
        return rng.choice([f"dot([{a},{b}],[{b},{c}])", f"cross([{a},0,1],[0,{b},2])", f"quantile([{a},{b},{c}],0.75)", f"percentile([{a},{b},{c}],75)", f"covariance([{a},{b}],[{b},{c}])", f"correlation([{a},{b}],[{b},{c}])", f"linreg([{a},{b}],[{b},{c}])"])
    if group == 6:
        return rng.choice([f"matmul([[{a},1],[2,{b}]],[[1,0],[0,1]])", f"matrix_power([[{a},1],[0,{b}]],3)", f"least_squares([[1,{a}],[1,{b}],[1,{c}]],[2,3,4])", f"substitute(x^3+{a}*x,x,{b})", f"product_range(k,k,1,{a})"])
    return rng.choice([f"{a} degrees Celsius to Fahrenheit", f"{a} degree Fahrenheit to Celsius", f"{a} Kelvin to Celsius", f"{a} psi to Pa", f"({a} m)^3 to L"])


def number(rng, split):
    bounds = {"train": (0, 999), "validation": (2000, 8999), "test": (10001, 999999)}
    a, b = bounds[split]
    value = rng.randint(a, b)
    style = rng.randrange(6)
    if style == 0:
        value = f"{value}.{rng.randrange(1000):03d}"
    elif style == 1:
        value = f"{value}e{rng.choice(['+', '-'])}{rng.randrange(1,9)}"
    elif style == 2:
        value = f".{value}"
    return str(value)


def whitespace(text, rng):
    tokens = LEX.findall(text)
    out = []
    style = rng.randrange(5)
    for index, token in enumerate(tokens):
        if index:
            previous = tokens[index - 1]
            adjacent_words = (previous[-1].isalnum() or previous[-1] in "_.°") and (token[0].isalnum() or token[0] in "_.°")
            # Preserve word/number token boundaries; only vary whitespace at boundaries.
            if adjacent_words:
                separator = " "
            elif style == 0:
                separator = ""
            elif style == 1:
                separator = " "
            else:
                separator = rng.choice(["", " ", "  ", "\t"])
            out.append(separator)
        out.append(token)
    return "".join(out)


def positive(rng, split, index):
    # Reuse v1 expression families, but control wrappers and vary every token boundary.
    saved = original.WRAPPERS[split]
    original.WRAPPERS[split] = ["{}"]
    expression, category = original.positive(rng, split, index)
    original.WRAPPERS[split] = saved
    if index % 5 == 0:
        operands = [number(rng, split) for _ in range(rng.randrange(2, 6))]
        expression = operands[0]
        for operand in operands[1:]:
            expression += rng.choice(["+", "-", "*", "/", "^"]) + operand
        category = "unparenthesized_numeric_chain"
    elif index % 3 == 0:
        expression, category = scientific(rng, split), "scientific_vocabulary"
    if rng.random() < 0.8:
        expression = whitespace(expression, rng)
    return rng.choice(WRAPPERS[split]).format(expression), category


def negative(rng, split, index):
    a, b = number(rng, split), number(rng, split)
    if index % 4 == 0:
        expr = whitespace(f"{a}+{b}", rng)
        forms = {
            "train": [f"do not calculate {expr}", f"don't evaluate {expr}", f"explain why {expr}", f"write a poem about {expr}", f"{expr} then delete the file", f"what is the weather at {a}"],
            "validation": [f"{expr} but do not calculate it", f"never compute {expr}", f"describe the notation in {expr}", f"send {expr} by email", f"is it raining at {a}"],
            "test": [f"please do not evaluate {expr}", f"dont calculate {expr}", f"tell a story involving {expr}", f"translate {expr} into French", f"what will the forecast be at {a}"],
        }
        return rng.choice(forms[split]), "negation_discussion_or_action"
    text, category = original.negative(rng, split, index)
    # Include repeated spaces in language too: spacing cannot become the label.
    if rng.random() < 0.35:
        text = text.replace(" ", rng.choice(["  ", "\t"]))
    return text, category


def corpus(split, count):
    rng = random.Random(SEED + {"train": 0, "validation": 1, "test": 2}[split])
    rows, seen = [], {text for text, _ in original.CURATED}
    seen.update(["what is 1 + 1", "1 + 1", "0.7 - 0.4 - 0.3"])
    for index in range(count):
        label = index % 2
        for attempt in range(100):
            text, category = (positive if label else negative)(rng, split, index // 2 + attempt)
            if text not in seen:
                break
        else:
            continue
        seen.add(text)
        rows.append({"input": text, "label": label, "category": category, "split": split})
    rng.shuffle(rows)
    return rows


def main():
    start = time.monotonic()
    original.number = number
    datasets = {name: corpus(name, count) for name, count in [("train", 64000), ("validation", 12000), ("test", 12000)]}
    seen = set()
    for name, rows in datasets.items():
        datasets[name] = [row for row in rows if row["input"] not in seen]
        seen.update(row["input"] for row in datasets[name])
        (HERE / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in datasets[name]))
    print(json.dumps({"stage": "features", "counts": {name: len(rows) for name, rows in datasets.items()}}), flush=True)
    matrices = {name: original.matrix(rows) for name, rows in datasets.items()}
    labels = {name: np.asarray([row["label"] for row in rows]) for name, rows in datasets.items()}
    rng = np.random.default_rng(SEED)
    params = [rng.normal(0, .12, (FEATURES, HIDDEN)).astype(np.float32), np.full(HIDDEN, .02, np.float32), rng.normal(0, .15, (HIDDEN, 2)).astype(np.float32), np.zeros(2, np.float32)]
    means, variances = [[np.zeros_like(p) for p in params] for _ in range(2)]
    best, best_loss, history, step = None, float("inf"), [], 0
    for epoch in range(28):
        order = rng.permutation(len(labels["train"]))
        loss = 0.0
        for offset in range(0, len(order), 256):
            indices = order[offset:offset+256]
            x, y = matrices["train"][indices], labels["train"][indices]
            w1, b1, w2, b2 = params
            probability, hidden = original.predict(x, *params)
            loss -= float(np.log(np.maximum(probability[np.arange(len(indices)), y], 1e-12)).sum())
            delta = probability.copy()
            delta[np.arange(len(indices)), y] -= 1
            delta /= len(indices)
            delta *= np.where(y == 0, 1.5, 1).astype(np.float32)[:, None]
            dh = (delta @ w2.T) * (hidden > 0)
            gradients = [x.T @ dh + 1e-5 * w1, dh.sum(axis=0), hidden.T @ delta + 1e-5 * w2, delta.sum(axis=0)]
            step += 1
            rate = .006 if epoch < 12 else (.003 if epoch < 22 else .0015)
            for p, g, mean, variance in zip(params, gradients, means, variances):
                mean *= .9
                mean += .1 * g
                variance *= .999
                variance += .001 * g * g
                p -= rate * (mean / (1-.9**step)) / (np.sqrt(variance / (1-.999**step)) + 1e-8)
        probability, _ = original.predict(matrices["validation"], *params)
        y = labels["validation"]
        validation_loss = float(-np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12)).mean())
        metrics = original.scores(datasets["validation"], probability, .99)
        row = {"epoch": epoch+1, "train_loss": loss/len(order), "validation_loss": validation_loss, "recall_0.99": metrics["recall"], "fp_0.99": metrics["fp"]}
        history.append(row)
        if validation_loss < best_loss:
            best, best_loss = [p.copy() for p in params], validation_loss
        print(json.dumps(row), flush=True)
    validation, _ = original.predict(matrices["validation"], *best)
    candidates = [original.scores(datasets["validation"], validation, t) for t in [.98, .99, .995, .999, .9995, .9999]]
    safe = [m for m in candidates if m["fp"] == 0]
    threshold = safe[0]["threshold"] if safe else .9999
    metrics = {name: original.scores(rows, original.predict(matrices[name], *best)[0], threshold) for name, rows in datasets.items()}
    header = b"OMNIRTR2" + struct.pack("<8IQ", 2, FEATURES, HIDDEN, 2, 1, 4, 1, 1, SEED)
    w1, b1, w2, b2 = best
    weights = HERE / "router.bin"
    weights.write_bytes(header + w1.T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + w2.T.astype("<f4").tobytes() + b2.astype("<f4").tobytes())
    curated = original.CURATED + [("what is 1 + 1", 1), ("1 + 1", 1), ("0.7 - 0.4 - 0.3", 1), ("sin ( 3 ) ^ 2", 1), ("what is 86 degrees Fahrenheit to Celsius", 1), ("\t WHAT IS 4 + 9?\n", 1), ("💀 weather 𐐀", 0)]
    parity = []
    for text, label in curated:
        indices = original.feature_indices(text)
        parity.append({"input": text, "label": label, "indices": indices, "scale_bits": int(np.asarray(original.feature_scale(len(indices))).view(np.uint32)), "confidence": original.scalar_probability(text, *best)})
    original.save_json(HERE / "parity.json", parity)
    metadata = {"model": "omni-reference-router-v2", "format": "OMNIRTR2", "seed": SEED, "threshold": threshold, "architecture": {"features": FEATURES, "hidden": HIDDEN, "outputs": 2}, "training": {"epochs": 28, "optimizer": "Adam", "batch_size": 256, "selected_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"], "cpu_only": True, "elapsed_seconds": time.monotonic()-start}, "split_policy": {"train": "integers0..999, decimals, scientific notation, random legal token spacing, depths1..2", "validation": "disjoint integers2000..8999, heldout wrappers/negative templates, depth3; selects epoch and threshold", "test": "disjoint integers10001..999999, heldout wrappers/negative templates, depth3..5; no selection", "curated": "v1 regression cases plus discovered spacing and decimal failures; excluded from gradients and validation, not pristine holdout", "duplicates_between_splits": 0}, "hashes": {"model_sha256": original.sha(weights), "script_sha256": original.sha(Path(__file__)), "shared_training_code_sha256": original.sha(Path(original.__file__)), **{f"{name}_sha256": original.sha(HERE/f"{name}.jsonl") for name in datasets}}, "model_bytes": weights.stat().st_size, "metrics": metrics, "validation_thresholds": candidates, "history": history}
    original.save_json(HERE / "metadata.json", metadata)
    print(json.dumps({"threshold": threshold, "metrics": {name: {k: v for k, v in m.items() if k != "false_positives"} for name, m in metrics.items()}, "curated": [{k: p[k] for k in ["input", "label", "confidence"]} for p in parity]}), flush=True)


if __name__ == "__main__":
    main()
