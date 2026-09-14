# Tiny learned reference router

`assets/router.bin` contains trained weights, not cached answers or a table of
input patterns. A 2048-feature, 24-hidden-unit ReLU MLP chooses between the typed
`evaluate()` input-reference tool and deferring to Needle. Rust implements its
inference using only `std`. Rust still executes all mathematics, and the planner
validates every selected tool call. The router never constructs new arithmetic
or extracts numbers from a request.

The caller must require both `reference_confidence(input) >= REFERENCE_THRESHOLD`
and `planner::literal_expression(input)` to succeed. Selected calls must then go
through `expression_for_call` like every other model call. In particular, the
validator rejects explicit instructions not to calculate, even if the classifier
misclassifies them. Confidence is a model score, not a guarantee of correctness.

## Reproduce

Prepare the isolated NumPy/SciPy environment described in the parent
[training README](../README.md), then run from the repository root. These
commands regenerate local model artifacts; the shipped reference model is V2.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/router/train.py --epochs 28
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/router/audit.py
cargo test --no-default-features --features fast-router --test router --lib
```

Training uses CPU only, seed `20260914`, Adam, batches of 256, 24,000 synthetic
training rows, and 28 epochs. The checkpoint is selected by validation cross
entropy (epoch 2). The generated train/validation/test JSONL corpus occupies
3,453,781 bytes. No GPU runtime, fine-tuning package, or download is used.

Weights and all reproducibility inputs have SHA-256 entries in `metadata.json`.
Model SHA-256:
`a5d6e9d61a232e96b07280ffa458607b6b79dd32aed33ca29db715c3a7204f29`.

## Evaluation and limits

The operating threshold is **0.98**. The initial 0.99 threshold deferred the
required typo `wat iz 1 + 1` (score 0.9856). We retained the learned weights and
checkpoint selection, added 0.98 to the validation threshold candidates, and
disclosed this adjustment in `metadata.json`. The strongest validation negative
scored 0.8281. Original test and curated results below were reviewed before this
adjustment; they are regression evidence, not pristine evidence for the final
threshold. A separate, freshly written audit was frozen and evaluated afterwards.

| Split | Rows | Literal recall | Raw classifier false positives | Precision |
| --- | ---: | ---: | ---: | ---: |
| Train | 24,000 | 98.40% | 2 / 12,000 negatives | 99.98% |
| Validation | 4,926 | 85.32% | 0 / 2,432 negatives | 100% observed |
| Original test | 4,926 | 93.70% | 0 / 2,432 negatives | 100% observed |
| Original curated | 33 | 100% | 0 / 20 negatives | 100% observed |
| Fresh post-selection audit | 2,442 | 65.79% | 7 / 1,226 negatives | 99.13% |

The two training false positives are incomplete requests `calculate` and
`wat iz`; the literal-reference guard rejects both. All seven fresh audit false
positives explicitly negate the calculation (for example,
`5099+5077 but do not calculate it`). The existing tool validator rejects every
one. `tests/router.rs` independently runs Rust inference and checks that all fresh
negative cases are either deferred or rejected by that validator. The raw model
failures remain in `audit-report.json`; they are not removed from its metrics.

The split generator excludes duplicate texts across train/validation/test and
curated sets. Train uses integers 0–99 and expression depths 1–2; validation and
test use separate larger prime pools, depths 3 and 3–5, respectively, held-out
conversational wrappers, and held-out verbal-operation templates. DSL categories
and function signatures intentionally overlap. Fresh audit uses another number
pool, different templates, depth 6 examples, and 42 additional curated cases.
Malformed explicit expressions can select `evaluate()`: rejecting their syntax
or mathematical domain belongs to the Rust engine. Broad natural language,
unseen functions, and typo wrappers may defer and incur Needle's latency.

## Features and binary layout

Input is UTF-8, with ASCII space/tab/CR/LF trimmed and ASCII letters lowercased.
Boundary bytes `ff` and `fe` are added. Unique byte 1–4-grams are hashed with FNV-1a
(width-salted), mapped to 2048 bins, and normalized by `1/sqrt(active_bins)` in
`f32`. Rust then accumulates sparse linear products in `f64`, applies ReLU, and
computes the two-class softmax probability. `parity.json` contains exact active
indices and normalization bits, plus an independently accumulated Python scalar
reference. Tests require matching features and predictions within `1e-12`.

The 196,952-byte file starts with a 48-byte header:

| Offset | Encoding | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | `OMNIRTR1` |
| 8 | 8 little-endian `u32` | version 1, features 2048, hidden 24, outputs 2, min gram 1, max gram 4, normalization 1, reserved 0 |
| 40 | little-endian `u64` | training seed |
| 48 | little-endian `f32` values | W1 `[24][2048]`, b1 `[24]`, W2 `[2][24]`, b2 `[2]` |

Inputs over 4096 bytes defer immediately. Inference performs no I/O and caches
only decoded immutable weights. Each request computes fresh features and neural
activations.

## Inference-only timing

```sh
rustc --edition 2024 -O training/router/bench.rs -o /tmp/omni-router-bench
nice -n 10 /tmp/omni-router-bench
```

A local development-machine run over 10,000 unique expressions measured 140.735µs
for the first inference including weight decoding, 2.182µs warm p50, and 3.552µs
warm p95. Of those inputs 8,520 selected the reference tool; the others deferred.
These measurements exclude CLI startup, IPC, validation, mathematical execution,
and Needle fallback. They do not substitute for the complete VPS benchmark.
