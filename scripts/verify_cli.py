#!/usr/bin/env python3
"""Run independent golden requests through the real CLI, model, and math engine."""
import argparse
import fractions
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
import time


def equal_value(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int) and isinstance(actual, int) and not isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal_value(actual[key], expected[key]) for key in expected)
    return actual == expected


def check_result(expect, response, exit_code):
    if "error" in expect:
        if response.get("ok") is not False or exit_code == 0:
            return "expected an error"
        allowed = expect["error"]
        if allowed and response.get("error", {}).get("code") not in allowed:
            return "wrong error category"
        return None
    if not response.get("ok") or exit_code != 0:
        return "calculation rejected"
    result = response.get("result", {})
    if "number" in expect:
        actual = result.get("value")
        try:
            if isinstance(actual, bool):
                raise ValueError("boolean is not a number")
            if isinstance(expect["number"], int) and result.get("exact") is True:
                if fractions.Fraction(actual) != expect["number"]:
                    return "wrong exact numerical answer"
            actual = float(fractions.Fraction(actual)) if isinstance(actual, str) else float(actual)
        except (ValueError, TypeError, ZeroDivisionError):
            return "result is not a numeric scalar"
        if not math.isclose(actual, expect["number"], rel_tol=expect.get("rtol", 1e-9), abs_tol=expect.get("atol", 1e-10)):
            return "wrong numerical answer"
    if "display" in expect and result.get("display") != expect["display"]:
        return "wrong exact display"
    if "value" in expect and not equal_value(result.get("value"), expect["value"]):
        return "wrong structured value"
    if "exact" in expect and result.get("exact") != expect["exact"]:
        return "wrong exactness claim"
    if "kind" in expect and result.get("kind") != expect["kind"]:
        return "wrong result kind"
    if not response.get("expression") or not response.get("tool_call"):
        return "successful response lacks auditable interpretation"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("--cases", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "tests/cases/acceptance.json")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--backend", choices=["auto", "needle"], default="auto")
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    cases = json.loads(args.cases.read_text())
    environment = os.environ.copy()
    environment.pop("OMNI_MODEL", None)
    environment.pop("OMNI_BACKEND", None)
    prefix = [str(binary), "--json", "--threads", "2", "--backend", args.backend, "--retry", str(args.retry)]
    if args.model:
        prefix += ["--model", str(args.model.resolve(strict=True))]
    os.nice(10)
    records = []
    with tempfile.TemporaryDirectory(prefix="omni-acceptance-") as runtime:
        environment["OMNI_RUNTIME_DIR"] = runtime
        try:
            for index, case in enumerate(cases):
                start = time.perf_counter_ns()
                try:
                    process = subprocess.run(prefix + [case["input"]], env=environment, capture_output=True, timeout=9, check=False)
                    response = json.loads(process.stdout)
                    failure = check_result(case["expect"], response, process.returncode)
                    if process.stderr:
                        failure = "unexpected stderr in JSON mode"
                    record = {**case, "response": response, "exit_code": process.returncode, "failure": failure}
                except (subprocess.TimeoutExpired, json.JSONDecodeError, UnicodeDecodeError) as error:
                    record = {**case, "failure": str(error)}
                record["wall_ms"] = (time.perf_counter_ns() - start) / 1e6
                records.append(record)
                if record["failure"]:
                    print(f"FAIL {index + 1}: {case['input']!r}: {record['failure']}", flush=True)
        finally:
            subprocess.run(prefix + ["--stop"], env=environment, capture_output=True, timeout=5, check=False)
    groups = {}
    for record in records:
        group = groups.setdefault(record["category"], {"passed": 0, "total": 0, "wrong_answers": 0, "rejected": 0})
        group["total"] += 1
        group["passed"] += record["failure"] is None
        group["wrong_answers"] += bool(record["failure"] and record.get("response", {}).get("ok"))
        group["rejected"] += record["failure"] == "calculation rejected"
    report = {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
        "backend": args.backend, "retry": args.retry, "groups": groups,
        "passed": sum(r["failure"] is None for r in records), "total": len(records), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "total", "groups"]}, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
