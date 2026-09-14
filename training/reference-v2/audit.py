#!/usr/bin/env python3
"""Frozen-model audit written after selecting reference-v2 weights and threshold."""
import json
from pathlib import Path
import random
import train
from train import original as core
import numpy as np

HERE = Path(__file__).resolve().parent
POSITIVE = [
    "({a} + {b}) / ({c} - 7)", "{a}  +\t{b} - 0.005", "- .{a} + .{b}",
    "sqrt ( {a}^2 + {b} ^ 2 )", "(sin({a}) + cos({b}))^2", "hypot ( {a} , {b} )",
    "log10({a}) + log2({b})", "asinh({a}) - acosh({b})", "round({a}/7,3)",
    "normal_quantile(0.91,{a},2)", "gamma_cdf({a},2,{b})", "sample_stdev([{a},{b},{c}])",
    "percentile([{a},{b},{c}],75)", "covariance([{a},{b}],[{c},7])",
    "cross([{a},0,1],[0,{b},2])", "trace([[{a},0],[0,{b}]])", "matrix_power([[1,{a}],[0,1]],3)",
    "least_squares([[1,{a}],[1,{b}],[1,{c}]],[2,3,4])", "{a} psi to Pa", "({a} m)^3 to L",
    "substitute(x^2+2*x,x,{a})", "diff(exp(x)*sin(x),x)", "factor(x^4-{a}^4)",
    "integrate(1/(1+x^2),x,0,{a})", "product_range(k,k,1,7)", "(1/{a} + 1/{b})^(-2)",
]
NEGATIVE = [
    "why would {a} + {b} be correct", "turn {a}+{b} into a joke", "{a} + {b} but never compute it",
    "don't add {a} and {b}", "what is the latest weather at {a}", "show {a}+{b} on a graph",
    "send me the result of {a}+{b} by email", "{a} / {b} is the date", "there are {a} m of rope and {b} nails",
    "please explain matrix_power([[1,{a}],[0,1]],3)", "read_file({a})", "sqrt({a}) then remove files",
    "convert {a} kg to pounds", "{a} plus {b} divided by {c}", "the sum of {a} and the root of {b}",
    "which is larger, {a} or {b}", "{a} + {b} is printed on my T-shirt", "find train {a} arriving at {b}",
    "ncr({a},{b}) is a function name", "what's the number on invoice {a}", "choose {a} songs from album {b}",
    "describe {a} science facts", "{a}^2 as a password", "how fast is the VPS with {a} CPUs",
]


def main():
    metadata = json.loads((HERE / "metadata.json").read_text())
    weights = HERE / "router.bin"
    assert core.sha(weights) == metadata["hashes"]["model_sha256"]
    values = np.frombuffer(weights.read_bytes()[48:], dtype="<f4")
    a, b, c = 2048 * 48, 2048 * 48 + 48, 2048 * 48 + 48 + 96
    params = [values[:a].reshape(48,2048).T, values[a:b], values[b:c].reshape(2,48).T, values[c:]]
    rows, seen = [], set()
    for name in ["train", "validation", "test"]:
        seen.update(json.loads(line)["input"] for line in (HERE/f"{name}.jsonl").read_text().splitlines())
    rng = random.Random(20260919)
    for label, templates in [(1, POSITIVE), (0, NEGATIVE)]:
        for index in range(2600):
            a, b, c = (rng.randrange(10_000_000,99_999_999) for _ in range(3))
            query = templates[index % len(templates)].format(a=a,b=b,c=c)
            if label:
                query = rng.choice(["{}", "how much is {}", "wot is {}", "could you compute {}?", "wut is {}?"]).format(query)
            if query in seen:
                continue
            seen.add(query)
            rows.append({"input":query,"label":label,"category":"post_selection","split":"audit"})
    probabilities, _ = core.predict(core.matrix(rows), *params)
    metrics = core.scores(rows, probabilities, metadata["threshold"])
    data = HERE / "audit.jsonl"
    data.write_text("".join(json.dumps({**row,"confidence":float(probabilities[i,1])})+"\n" for i,row in enumerate(rows)))
    report = {"model_sha256":core.sha(weights),"script_sha256":core.sha(Path(__file__)),"data_sha256":core.sha(data),"policy":"Development regression audit, first written after attempt02 and reused after scientific-vocabulary augmentation; exact strings excluded from training; not a pristine final-model holdout.","overall":metrics}
    core.save_json(HERE / "audit-report.json",report)
    print(json.dumps({key:value for key,value in metrics.items() if key != "false_positives"},indent=2))


if __name__ == "__main__":
    main()
