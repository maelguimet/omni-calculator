#!/usr/bin/env python3
"""Summarize raw routing evidence, distinguishing exact and structural matches."""

import argparse
import ast
import collections
import json
import pathlib
import re


def canonical(value):
    if isinstance(value, dict):
        return tuple(sorted((key, canonical(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(canonical(item) for item in value)
    if not isinstance(value, str):
        return value
    try:
        return ("expression", ast.dump(ast.parse(value.replace("^", "**"), mode="eval"), include_attributes=False))
    except SyntaxError:
        # Keep token boundaries: "2 3" must never equal "23".
        return ("tokens", tuple(re.findall(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|[^\s]", value)))


def equivalent_calls(calls):
    """Allow only known commutative outer operations, preserving nested ASTs.

    This is intentionally narrower than general mathematical equivalence and
    does not exempt calls from the real source/intent guard.
    """
    result=[]
    for call in calls:
        args=call.get("arguments")
        if (call.get("name")=="function" and isinstance(args,dict)
                and args.get("function") in ("add","multiply","gcd","lcm")
                and isinstance(args.get("arguments"),list)):
            operands=tuple(sorted((canonical(v) for v in args["arguments"]),key=repr))
            rest={k:v for k,v in args.items() if k!="arguments"}
            result.append(("commutative_function",canonical(rest),operands))
        else:
            result.append(canonical(call))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    result = json.loads(pathlib.Path(args.result).read_text())
    families = collections.defaultdict(lambda: {"samples": 0, "exact": 0, "structural": 0, "call_equivalent":0, "native_errors": 0})
    for case in result["results"]:
        score = families[case.get("_family", "curated")]
        score["samples"] += 1
        score["exact"] += bool(case["exact_accepted"])
        response = case["response"]
        score["native_errors"] += response.get("success") is not True
        if "answers" in case:
            structural = response.get("success") is True and canonical(case["answers"]) == canonical(response.get("function_calls") or [])
            equivalent=response.get("success") is True and equivalent_calls(case["answers"])==equivalent_calls(response.get("function_calls") or [])
        else:
            structural = case["exact_accepted"]
            equivalent=structural
        score["structural"] += structural
        score["call_equivalent"] += equivalent
    print(json.dumps({"result": args.result, "samples": result["samples"],
                      "exact": sum(x["exact"] for x in families.values()),
                      "structural": sum(x["structural"] for x in families.values()),
                      "call_equivalent": sum(x["call_equivalent"] for x in families.values()),
                      "families": dict(sorted(families.items())),
                      "warm_median_ms": result["warm_median_ms"], "warm_p95_ms": result["warm_p95_ms"]}, indent=2))


if __name__ == "__main__":
    main()
