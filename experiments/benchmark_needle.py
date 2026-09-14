#!/usr/bin/env python3
"""Reproducible, offline Needle tool-routing benchmark (standard library only)."""

import argparse
import ast
import ctypes
import hashlib
import json
import os
import pathlib
import re
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def normalize(expression):
    expression = str(expression)
    try:
        return ast.dump(ast.parse(expression.replace("^", "**"), mode="eval"), include_attributes=False)
    except (ValueError, SyntaxError):
        return tuple(re.findall(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|[^\s]", expression))


def expression_from_call(call, source=""):
    name = call.get("name")
    args = call.get("arguments") or {}
    if name == "evaluate" and not args:
        return re.sub(r"^(?:what is|what's|wat iz|calculate|evaluate|compute|convert)\s+", "", source,
                      flags=re.IGNORECASE).removesuffix("?")
    if name in ("calculate", "evaluate"):
        return args.get("expression")
    if name == "convert":
        return f"{args.get('value')} to {args.get('unit')}"
    if name == "function":
        function = args.get("function")
        operands = args.get("arguments") or []
        if not isinstance(operands, list):
            return None
        operands = list(map(str, operands))
        if function in ("add", "subtract", "multiply", "divide", "power"):
            operator = {"add": "+", "subtract": "-", "multiply": "*",
                        "divide": "/", "power": "^"}[function]
            return operator.join(f"({operand})" for operand in operands)
        if function == "percent" and len(operands) == 2:
            return f"{operands[0]}/100*{operands[1]}"
        if function == "factorial" and len(operands) == 1:
            return f"{operands[0]}!"
        return f"{function}({','.join(operands)})"
    if name == "algebra":
        operation = args.get("operation")
        if "variable" in args and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", str(args["variable"])):
            return None
        if operation in ("integrate", "root") and (("lower" in args) != ("upper" in args)):
            return None
        operands = [args.get("expression", "")]
        if operation in ("diff", "integrate", "solve", "substitute", "root"):
            operands.append(args.get("variable", "x"))
        for field in ("lower", "upper", "order"):
            if field in args:
                operands.append(str(args[field]))
        if operation == "substitute" and "value" in args:
            operands.append(args["value"])
        return f"{operation}({','.join(operands)})"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", default="experiments/tools-baseline.json")
    parser.add_argument("--system", default="experiments/system-baseline.txt")
    parser.add_argument("--cases", default="experiments/heldout.jsonl")
    parser.add_argument("--output", default="experiments/results-baseline.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--weights", help="Optional local .cact fine-tuned weights")
    parser.add_argument("--quiet", action="store_true", help="Print summaries rather than each response")
    parser.add_argument("--cpu-cores", type=int, help="Pin to this many last available logical CPU cores")
    args = parser.parse_args()
    if args.cpu_cores:
        if args.cpu_cores<1:
            raise ValueError("--cpu-cores must be positive")
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-args.cpu_cores:])

    library = ctypes.CDLL(str(ROOT / ".cache/needle/libneedle.so"))
    library.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    library.needle_init.restype = ctypes.c_int
    library.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    library.needle_complete.restype = ctypes.c_int
    library.needle_reset.argtypes = []
    library.needle_reset.restype = None
    library.needle_load.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    library.needle_load.restype = ctypes.c_int
    output = ctypes.create_string_buffer(65536)
    tools = (ROOT / args.tools).read_bytes()
    system = (ROOT / args.system).read_bytes().strip() if args.system else b""
    start = time.perf_counter()
    if args.weights:
        weights = (ROOT / args.weights).read_bytes()
        weight_buffer = ctypes.create_string_buffer(weights)
        if library.needle_load(weight_buffer, len(weights)) < 0:
            raise RuntimeError("needle_load failed")
    rc = library.needle_init(system, tools, None)
    initialization_ms = (time.perf_counter() - start) * 1000
    if rc < 0:
        raise RuntimeError(f"needle_init failed: {rc}")
    cases = [json.loads(line) for line in (ROOT / args.cases).read_text().splitlines() if line]
    if args.limit:
        cases = cases[:args.limit]
    results = []
    for case in cases:
        library.needle_reset()
        start = time.perf_counter()
        source = case.get("input", case.get("query", ""))
        rc = library.needle_complete(source.encode(), args.tokens, output, len(output))
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            response = json.loads(output.value.decode())
        except Exception:
            response = {"raw": output.value.decode(errors="replace")}
        calls = response.get("function_calls") or []
        call = calls[0] if len(calls) == 1 else {}
        expression = expression_from_call(call, source)
        if "answers" in case:
            correct = response.get("success") is True and calls == case["answers"]
        elif case.get("reject"):
            correct = bool(response.get("success") and
                           (not calls or call.get("name") == "unsupported"))
        else:
            correct = response.get("success") is True and expression is not None and normalize(expression) in {
                normalize(expected) for expected in case["expressions"]
            }
        results.append(dict(case, elapsed_ms=elapsed_ms, rc=rc, response=response,
                            exact_accepted=correct))
        if not args.quiet:
            print(json.dumps({"input": source, "ms": round(elapsed_ms, 2),
                              "accepted": correct, "calls": calls}), flush=True)
    latencies = sorted(case["elapsed_ms"] for case in results)
    report = {"tools": args.tools, "system": args.system, "cases": args.cases, "weights": args.weights,
              "tools_sha256":hashlib.sha256(tools).hexdigest(),
              "system_sha256":hashlib.sha256(system).hexdigest(),
              "cases_sha256":hashlib.sha256((ROOT / args.cases).read_bytes()).hexdigest(),
              "weights_sha256":hashlib.sha256(weights).hexdigest() if args.weights else None,
              "cpu_affinity":sorted(os.sched_getaffinity(0)),
              "native_environment":{k:v for k,v in os.environ.items() if k.startswith("NEEDLE_")},
              "initialization_ms": initialization_ms,
              "samples": len(results), "exact_accepted": sum(r["exact_accepted"] for r in results),
              "warm_median_ms": statistics.median(latencies),
              "warm_p95_ms": latencies[max(0, int(0.95 * len(latencies)) - 1)],
              "results": results}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
