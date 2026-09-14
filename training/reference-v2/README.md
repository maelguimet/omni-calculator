# Reference model v2

`assets/router-v2.bin` is a 393,848-byte trained MLP with 2048 sparse hashed
features, 48 ReLU hidden units, and a two-way softmax. It selects the immutable
`evaluate()` tool or defers. It never computes an answer or generates a changed
expression. Its threshold is 0.98, selected using the validation set.

The first reference model is preserved in `training/router/` and
`assets/router.bin`. Real CLI benchmarking found that v1 deferred the required
spaced input `what is 1 + 1`, despite accepting its unspaced spelling. V2 adds
systematic token-boundary whitespace variation, scientific notation, decimals,
the documented scientific vocabulary, and numeric masking in model features.
The original text, including its numbers and whitespace, always goes to the
source validator and mathematical engine.

## Training and reproducibility

Prepare the isolated NumPy/SciPy environment described in the parent
[training README](../README.md). Training regenerates datasets, metadata, and
`assets/router-v2.bin`; it is optional for builds using the published weights.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/reference-v2/train.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/reference-v2/audit.py
cargo test --offline --lib --test router
```

The examples use one BLAS thread and no machine-specific CPU affinity.
Training does not use a GPU or a teacher API. The deterministic seed is 20260915. Adam trains 28 epochs on
64,000 generated examples, batch 256; validation cross entropy selects epoch18.
Training took about 9.3 seconds on four desktop CPU cores. The script shares
documented feature/training helpers with the frozen v1 script, whose hash is
also recorded. Earlier `attempt-01` and `attempt-02` candidate artifacts remain in the
author's local development archive; they are not part of the public checkout.

Current weight SHA-256:
`d3b0c2c0d30daa38dd9f3cc2411f7b1cc842c4a1b2c07b56dfdcc57a8cf5cc14`.

| Set | Rows | Reference recall | Raw false positives |
| --- | ---: | ---: | ---: |
| Train | 64,000 | 99.994% | 0 / 32,000 negatives |
| Validation | 11,856 | 90.15% | 0 / 5,856 negatives |
| Generated test | 11,858 | 99.95% | 0 / 5,858 negatives |
| Development audit | 5,010 | 88.22% | 108 / 2,600 negatives |

The 108 development false positives are variants of a request to explain a
matrix-power expression. The complete-expression gate rejects every one before
a reference tool executes. Their raw classifier scores remain in the report;
they are not relabeled as correct model predictions.

Train, validation, and generated test use different numeric ranges and some
different wrappers/templates. Numeric features are masked, so changing only
numbers does **not** create an independent model-feature holdout. Expression
families and mathematical vocabulary intentionally overlap. These metrics
measure the generated distributions and do not imply universal language recall.
The development audit was written after attempt02, then reused after the
scientific-vocabulary augmentation. It is regression evidence, not a pristine
final-model holdout. End-to-end CLI results and separately authored final
requests are the more relevant application checks.

## Feature parity and binary layout

Numeric tokens, including decimals and scientific exponents, become `0` only
for feature extraction. Signs remain, and digits inside identifiers such as
`log10` remain. Rust and Python use the same scanner behavior, ASCII lowercase,
UTF-8 byte 1–4-grams, FNV-1a width salt, 2048 bins, binary presence, and L2
normalization. The classifier's output is invariant to quantities in the same
textual structure. Mathematical values remain independent of this preprocessing.

The file has an eight-byte `OMNIRTR2` magic; eight little-endian `u32` values
`[2,2048,48,2,1,4,1,1]`; a little-endian `u64` seed; then `f32` W1 `[48][2048]`,
b1 `[48]`, W2 `[2][48]`, b2 `[2]`. The last header flag indicates numeric masking.
`parity.json` contains exact feature indices and normalization bits plus
independently accumulated Python scalar predictions. Rust tests compare every
fixture within `1e-12`, verify weight/script/data hashes, and run all development
negative requests through the actual source validator.
