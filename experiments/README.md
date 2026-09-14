# Needle calculator routing experiments

These files measure actual local Needle 2 outputs. They do not count a model
call as a correct calculation merely because it parses as JSON.

The pinned official native library and runner are prepared by the project setup
in `.cache/needle`. The experiments use Python's standard-library `ctypes`; the
calculator application itself remains Rust.

```sh
env NEEDLE_THREADS=2 python3 experiments/benchmark_needle.py \
  --tools experiments/tools.json --system experiments/system.txt \
  --cases experiments/heldout.jsonl --output experiments/results-base.json
```

Add `--weights training/runs/v1/epoch-01.cact` to evaluate an exported local
fine-tune. The benchmark resets the native conversation before every request
and records initialization, warm latency, success, calls, confidence, native
validation, and the original prompt. Raw native confidence is recorded only;
it is not a correctness threshold, and the upstream head is not recalibrated
by fine-tuning.

For synthetic JSONL cases, acceptance requires the exact tool name and
arguments and a successful native response. `summarize.py` additionally reports
structural equivalence, allowing redundant parentheses and whitespace without
changing operators or token boundaries. For `experiments/heldout.jsonl`, the
benchmark compares the resulting expression's structure with listed spellings.
These are conservative routing checks, not an independent mathematical
equivalence prover. The final application also validates source grounding and
executes the mathematics in Rust. Equivalent-looking function calls on an
explicit source may still be rejected by the application's immutable-reference
contract; native routing counts are therefore not application pass counts.

`build_auditor.py` links `audit_calls.rs` against an existing release application
library without modifying application code. It records the library hash and
checks expected/predicted calls against the real planner. `audit-calls --math`
also executes both accepted expressions with a 50 ms mathematical deadline.
An equal error is not proof of a correct calculation, and equal results for
one input are not proof of mathematical equivalence.

Schema variants are versioned; original response dumps are retained in the local development archive. The initial
single-expression schema silently dropped parentheses, roots, and factorials.
It accepted only 18 of 53 curated requests/rejections. Accordingly, the chosen
schema uses an immutable `evaluate()` input reference and separate structured
function, conversion, and algebra tools. Natural-language extraction still
requires fine-tuning and validation.

Latency recorded while training or other agents build the application is
contended and must not be presented as a clean release benchmark. Final VPS
benchmarks are the authority for the deployment target.

For comparable measurements, use the same available physical CPU cores for both
candidates. On a hybrid CPU, inspect the topology before choosing affinity:
`--cpu-cores 2` selects the last two logical CPUs, which may be efficiency cores. Reports now retain CPU affinity, native
environment settings, and SHA-256 hashes of the weights, schema, system text,
and case file. One early V3 timing run overlapped another benchmark; both raw
artifacts are explicitly marked and excluded from latency conclusions.

`NEEDLE_CONFIDENCE` set to `0`, `1`, or unset yielded identical raw calls and
confidence values on the same 32-case V3 test. `NEEDLE_CONF_RESCORE` is not a
disable flag: merely setting it (including `0`) enabled expensive behavior in
the official library. Neither setting is used by the application. The frozen
confidence head is uncalibrated after specialization and must not be a safety
threshold.

`parity-investigation.md` documents the controlled JAX/native discrepancy and
the answer-boundary alignment experiment. The expanded direct-call model had
better raw routing scores but regressed the real CLI acceptance suite, so raw
tool accuracy alone did not determine the selected application model. All
failed candidates and raw responses remain available locally.

Final deployment uses the original V2 epoch-3 checkpoint in `models/omni.cact`
(SHA-256 `1852739bfbb36f945eed2bca5c3ddfd364fab276cf7914ccbe793df4746e46e4`).
Parent validation reports 75/75 development CLI cases for that model, compared
with 74/75 for the final V2 alignment candidate. Alternatives' higher synthetic
native scores did not compensate for observed composition regressions in the
complete application. Large checkpoints, generated result dumps, and training environments are not
part of the public repository. Published model hashes and selected comparison
metadata preserve the selection record.
