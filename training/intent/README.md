# Bounded learned verbal intent model

This is a separate CPU neural model from the reference router and Needle. It
learns an operation and operand order, then copies original numeric literals
into an existing typed `function` call. There is no answer cache, arithmetic in
the model, or mapping from keywords directly to a function result.

The model is a 2048-feature, 32-hidden-unit ReLU MLP with 20 softmax classes and
264,960 bytes of weights. Classes are defer, add, subtract (both orders),
multiply, divide (both orders), power, sqrt, cbrt, factorial, mod, gcd, lcm, ncr
(both orders), npr (both orders), percent, and abs. `metadata.json` records the
exact class indices. The learned confidence threshold is **0.98**.

The caller must first exclude literal expressions. It must also run every
returned candidate through `planner::expression_for_call`, and immediately
defer to Needle if validation rejects the call. Such deferral should not consume
a user retry. The model's high score alone never authorizes mathematical
execution. In particular, reverse division can run only if the semantic
validator supports its wording and order.

## Input boundary

The class-independent lexer accepts at most 512 ASCII bytes and exactly one or
two decimal/scientific-notation scalar spans. Direct signs and original decimal
precision are preserved byte-for-byte. Numeric values are replaced by `#` only
in the network's features, so the network cannot invent or round operands.

A fixed vocabulary gate excludes units, number words, unknown nonmathematical
words, and negation. A separate cue-count gate requires one operation cue; it
does not pick the operation. Brackets, arithmetic expressions, more than two
numbers, malformed exponents, ambiguous thousands commas, and compositions
defer. A small set of punctuation supports ordinary questions, argument commas,
and percentages. Some valid compound wording is conservatively excluded—for
example multiple synonymous power cues. More general requests remain Needle's
responsibility. Square/cube requests without an explicit exponent also defer,
because this model is allowed to copy numbers, not insert implicit constants.

## Reproduce

Prepare the isolated NumPy/SciPy environment described in the parent
[training README](../README.md), then run from the project root. Training
regenerates datasets, metadata, and `assets/intent.bin`; it is optional for builds
using the published weights.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/intent/train.py --epochs 32
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/intent/audit.py
cargo test --no-default-features --features fast-router --test intent --lib
```

Training uses CPU only, seed `20260918`, Adam, 32 epochs, and batches of 256.
Checkpoint selection prefers zero validation routing mistakes, then the most
correct accepted calls at 0.98, then lower cross entropy. Epoch 32 was selected.
The initial loss-selected checkpoint is preserved in the author's local
`baseline-loss-selection/` development archive, not the public checkout; its
conservative minimum-100-correct validation rule deferred everything. The objective was revised from validation behavior.
Its initial test metrics had already been visible; this history is disclosed in
metadata, and a new audit was written after freezing the final weights.

Training, validation, and test use disjoint operation templates, wrapper groups,
and number pools, with spacing/case/tab variations. Duplicates are excluded
across splits **after numeric masking, trimming, and lowercasing**, since raw
number differences do not make a new neural input. The test split does not
select model weights or thresholds. No external acceptance corpus is used.

| Split | Rows | Correct accepted calls | Routing mistakes | Positive recall |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,657 | 6,373 | 0 | 97.63% |
| Validation | 970 | 109 | 0 | 16.93% |
| Test | 1,015 | 234 | 0 | 36.00% |
| Fresh audit | 216 | 64 | 0 | 84.21% |
| Audit inputs unseen after masking | 177 | 48 | 0 | 80.00% |

These figures apply the lexical/arity gates and threshold. Actual accepted
coverage can be lower after the mandatory semantic validator. Fresh audit
includes 54 new curated sentences and four typography variants per sentence;
the variants are correlated, not independent evidence. Thirty-nine variants
overlap earlier inputs after masking and are reported separately. Zero observed
mistakes does not prove perfect precision, and the low held-out recall shows
that this is a limited acceleration path rather than broad language support.

`tests/intent.rs` verifies target examples, unchanged signed numeric arguments,
malformed/unsupported shape deferral, artifact hashes, and that no fresh audit
request can execute a wrong operation after semantic validation. The internal
parity test compares lexer spans, masked text, exact feature indices and `f32`
normalization bits, and Python/Rust scalar inference within `1e-12`.

## Format and timing

Weights SHA-256:
`97d0df5d619f5f8badfb1aa6d5c6d72a77ba4b248b63b6caad18c05321e46dd2`.

The 48-byte header is magic `OMNIINT1`, eight little-endian `u32` fields
`[1,2048,32,20,1,4,1,0]`, then the little-endian `u64` seed. Remaining little-endian
`f32` arrays are W1 `[32][2048]`, b1 `[32]`, W2 `[20][32]`, and b2 `[20]`.
Features use the reference model's width-salted FNV-1a byte 1–4-gram hashing,
binary presence, and L2 normalization, applied to the masked text. Inference
accumulates in `f64` and caches only immutable decoded weights.

To time fresh unique requests through the neural model and semantic validator:

```sh
cargo test --release --no-default-features --features fast-router --test intent -- --ignored --nocapture
```

This benchmark excludes mathematical execution, CLI startup, IPC, and Needle.
Use the complete application benchmark for end-to-end VPS latency.

A local release run measured 310.163µs for the first model call, then 31.589µs
p50 and 48.269µs p95 for model plus semantic validation over 10,000 unique numeric
requests across four fixed wordings (root, reverse subtraction, percent, and
addition). All 10,000 produced validated calls. This is a microbenchmark of those
wordings, not a measurement of language coverage or full application latency.
