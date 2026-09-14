#!/usr/bin/env python3
"""Bounded, sequential CPU-only end-to-end benchmark. No external Python packages."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time


def percentile(samples, p):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p + 0.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--one-cpu", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--backend", choices=["auto", "needle"], default="auto")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or not 1 <= args.threads <= 8:
        parser.error("runs must be 1..100 and threads 1..8")
    binary = args.binary.resolve(strict=True)
    if args.one_cpu:
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    environment = os.environ.copy()
    environment.pop("OMNI_MODEL", None)
    environment.pop("OMNI_BACKEND", None)
    prefix = [str(binary), "--json", "--threads", str(args.threads), "--backend", args.backend]
    if args.model:
        prefix += ["--model", str(args.model.resolve(strict=True))]
    records = []
    with tempfile.TemporaryDirectory(prefix="omni-benchmark-") as runtime:
        environment["OMNI_RUNTIME_DIR"] = runtime

        def invoke(query, kind, expected=None, expect_error=False):
            start = time.perf_counter_ns()
            process = subprocess.run(prefix + ["--retry", "0", query], capture_output=True,
                                     env=environment, timeout=12, check=False)
            wall = (time.perf_counter_ns() - start) / 1e6
            response = json.loads(process.stdout)
            correct = None if expected is None else response.get("result", {}).get("display") == expected
            if expect_error:
                correct = response.get("ok") is False and process.returncode != 0
            elif expected is not None:
                correct = correct and response.get("ok") is True and process.returncode == 0
            record = {"kind": kind, "query": query, "wall_ms": wall, "exit_code": process.returncode,
                      "correct": correct, "response": response}
            records.append(record)
            return record

        def status():
            process = subprocess.run(prefix + ["--status"], capture_output=True, env=environment, timeout=3)
            result = json.loads(process.stdout)
            pid = result.get("pid")
            if isinstance(pid, int):
                try:
                    process_status = pathlib.Path(f"/proc/{pid}/status").read_text()
                    result["memory_kib"] = {line.split(":")[0]: int(line.split()[1])
                                            for line in process_status.splitlines()
                                            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:"))}
                except (FileNotFoundError, PermissionError):
                    pass
            return result

        try:
            cold = invoke("what is 1 + 1", "cold_literal", "2")
            router_status = status()
            # This wording can use the tiny intent model. Do not call it Needle latency.
            cold_language = invoke("what is the square root of 1089", "first_language", "33")
            for i in range(args.runs):
                a, b = 37 + i, 19 + i
                invoke(f"wat iz {a}+{b}", "warm_literal", str(a + b))
                invoke(f"what is the square root of {(i+11)**2}", "warm_language", str(i + 11))
            fast_status = status()
            # Unit-bearing verbal conversions require the generative model in auto mode.
            # Every request has a new value; no response cache or repeated-query shortcut.
            cold_generative = invoke("convert 2.75 km to m", "first_generative", "2750 m")
            for i in range(args.runs):
                value = 37 + i
                invoke(f"convert {value} km to m", "warm_generative", f"{value * 1000} m")
            invoke("what is (13+7)*5", "grouping", "100")
            invoke("what is the weather", "off_topic", expect_error=True)
            invoke("integrate(x^2,x,0,3)", "calculus", "9")
            worker_status = status()
        finally:
            subprocess.run(prefix + ["--stop"], capture_output=True, env=environment, timeout=5, check=False)
    summary = {}
    for kind in ("warm_literal", "warm_language", "warm_generative"):
        group = [r for r in records if r["kind"] == kind]
        timings = [r["wall_ms"] for r in group]
        summary[kind] = {"requests": len(group), "correct": sum(r["correct"] is True for r in group),
                         "median_ms": statistics.median(timings), "p95_ms": percentile(timings, 0.95),
                         "min_ms": min(timings), "max_ms": max(timings),
                         "backends": {name: sum(r["response"].get("model") == name for r in group)
                                      for name in sorted({r["response"].get("model", "unknown") for r in group})}}
    report = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "binary_bytes": binary.stat().st_size, "platform": platform.platform(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "model_threads": args.threads,
              "backend": args.backend,
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
              "cold_ms": cold["wall_ms"], "first_language_ms": cold_language["wall_ms"],
              "first_generative_ms": cold_generative["wall_ms"],
              "router_status": router_status, "fast_status": fast_status, "worker_status": worker_status,
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("cold_ms", "first_language_ms", "first_generative_ms", "summary", "worker_status")}, indent=2))


if __name__ == "__main__":
    main()
