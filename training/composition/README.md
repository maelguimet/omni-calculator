# Bounded composition model

`omni-composition-v1` is a 2048-feature, 32-hidden-unit, 20-class CPU MLP.
Its 264,960-byte weights are embedded in the calculator. It predicts a complete
typed-call structure, including operand order and implicit square/cube exponents,
then copies scalar text into that structure. It never calculates an answer.
Every call must pass the independent validator against the original request.
Rejected or low-confidence candidates defer to the unchanged Needle 2 model.

The lexical boundary accepts two to four signed decimal/scientific scalar spans,
up to 512 ASCII bytes, and a fixed vocabulary. Units, arbitrary expressions,
ambiguous thousands separators, and unfamiliar words defer. Number-word requests
may reach this model through the shared, source-validated digit spelling pass.

Features combine the existing hashed character grams with grams annotated by
the number of preceding scalar placeholders. This distinguishes a difference
before a sum from a sum before a difference; unpositioned character bags could
not represent that distinction. Numeric values are masked and cannot act as an
answer cache. Python/Rust extraction and scalar inference are checked by parity
fixtures. Artifact hashes and held-out executable-call structure are tested.

Training uses only CPU NumPy/SciPy, seed 20260918, Adam, and batches of 256:

```sh
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  training/.venv/bin/python training/composition/train.py --epochs 40
```

The selected checkpoint is epoch 40. Selection maximizes correct validation
coverage at 0.98 with zero validation classification errors; the final threshold
is the lowest tested value with zero validation errors and at least 100 correct
classifications. It is **0.90**. Confidence alone never authorizes execution.

| Split | Rows | Correct accepted classifications | Wrong classifications |
| --- | ---: | ---: | ---: |
| Training | 12,816 | 7,344 | 0 |
| Validation | 4,272 | 2,435 | 0 |
| Test | 4,272 | 2,448 | 0 |

All 12,240 positive targets across the three splits pass source validation.
Rows outside the lexical boundary are labeled defer, including some otherwise
valid language. Runtime coverage can therefore differ from classification recall.
Splits use distinct wrapper groups and disjoint normalized numeric-masked text;
operation templates are shared. These are synthetic family tests, not estimates
of general English accuracy. Initial bag-feature test metrics were visible before
the positional encoding revision. The final test does not choose checkpoints or
thresholds. Application audits and their development exposure are documented in
[the accuracy report](../../docs/accuracy-0.1.1.md).

`classes.json`, vocabulary, corpora, weights, parity fixtures, generator hashes,
and training history are published. The feature/optimizer helper is the existing
intent training script, whose hash is included without altering its artifacts.
Training is optional; source builds need no Python ML environment.
