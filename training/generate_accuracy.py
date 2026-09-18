#!/usr/bin/env python3
"""Bounded Needle 2 continuation curriculum; never train on evaluation queries."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    rng = random.Random(20260918)
    tools = json.loads((ROOT / "assets/tools.json").read_text())
    system = (ROOT / "assets/system.txt").read_text().strip()
    frozen = set()
    sources = list((ROOT / "tests/cases").glob("*.json")) + args.exclude
    for source in sources:
        data = json.loads(source.read_text())
        if isinstance(data, list):
            frozen.update(row["input"].lower().strip() for row in data if "input" in row)
    used = set(frozen)
    rows = []

    def append(query, function, operands, family):
        query = rng.choice(["", "calculate ", "what is ", "please find ", "compute "]) + query
        key = query.lower().strip()
        if key in used:
            return
        used.add(key)
        rows.append({"query": query, "tools": tools, "system": system,
                     "answers": [{"name": "function", "arguments": {
                         "function": function, "arguments": operands}}],
                     "reasoning": f"Use {function} with the source values.", "_family": family})

    while len(rows) < 1800:
        a, b, c, d = [str(rng.randint(1, 60)) for _ in range(4)]
        if rng.random() < .25:
            a = "-" + a
        family = rng.randrange(4)
        if family == 0:
            exponent = rng.choice(["2", "3"])
            noun = "squares" if exponent == "2" else "cubes"
            op, prefix = rng.choice([("add", "add the"), ("sum", "sum of"),
                                     ("subtract", "difference of"), ("multiply", "product of"),
                                     ("mean", "mean of")])
            values = [a, b, c] if op == "sum" and rng.random() < .5 else [a, b]
            group = " and ".join(values) if len(values) == 2 else f"{a}, {b}, and {c}"
            powers = [f"({v})^{exponent}" for v in values]
            if op in ("sum", "mean") and rng.random() < .5:
                powers = ["[" + ",".join(powers) + "]"]
            append(f"{prefix} {noun} of {group}", op, powers, "plural_powers")
        elif family == 1:
            query, op, values = rng.choice([
                (f"{a} less than {b}", "subtract", [b, a]),
                (f"{a} fewer than {b}", "subtract", [b, a]),
                (f"take {a} away from {b}", "subtract", [b, a]),
                (f"subtract negative {b} from {c}", "subtract", [c, "-" + b]),
                (f"divide {a} into {b}", "divide", [b, a]),
            ])
            append(query, op, values, "order")
        elif family == 2:
            query, op, values = rng.choice([
                (f"subtract the product of {a} and {b} from {c}", "subtract", [c, f"({a}*{b})"]),
                (f"divide {c} by the product of {a} and {b}", "divide", [c, f"({a}*{b})"]),
                (f"add the product of {a} and {b} to {c}", "add", [f"({a}*{b})", c]),
                (f"subtract the sum of {a} and {b} from {c}", "subtract", [c, f"({a}+{b})"]),
                (f"multiply the difference of {a} and {b} by the sum of {c} and {d}", "multiply", [f"({a}-{b})", f"({c}+{d})"]),
                (f"multiply the sum of {a} and {b} by the difference of {c} and {d}", "multiply", [f"({a}+{b})", f"({c}-{d})"]),
                (f"divide the sum of {a} and {b} by the difference of {c} and {d}", "divide", [f"({a}+{b})", f"({c}-{d})"]),
                (f"add the square root of {b} and the cube root of {c}", "add", [f"sqrt({b})", f"cbrt({c})"]),
                (f"the square of the difference of {a} and {b}", "power", [f"({a}-{b})", "2"]),
                (f"add {a} squared and {b} cubed", "add", [f"({a})^2", f"({b})^3"]),
            ])
            append(query, op, values, "composition")
        else:
            words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
            x, y = rng.sample(range(1, 20), 2)
            query, op, values = rng.choice([
                (f"{words[x-1]} times negative {words[y-1]}", "multiply", [str(x), str(-y)]),
                (f"subtract negative {words[x-1]} from {words[y-1]}", "subtract", [str(y), str(-x)]),
                (f"add {words[x-1]} and {words[y-1]}", "add", [str(x), str(y)]),
            ])
            append(query, op, values, "word_numbers")

    curated = list(rows)
    replay = [json.loads(line) for line in args.replay.read_text().splitlines() if line]
    rng.shuffle(replay)
    for row in replay:
        key = row["query"].lower().strip()
        # Retry wrappers contain a source query as their first line.
        source = key.split("\n", 1)[0].removeprefix("calculation request: ")
        if key not in used and source not in used:
            rows.append(row)
            used.add(key)
        if len(rows) == 3000:
            break
    rng.shuffle(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("curated.jsonl", curated), ("mixed.jsonl", rows)]:
        path = args.out_dir / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data))
        print(name, len(data), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
