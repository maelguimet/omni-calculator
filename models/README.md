# Bundled local models

The release embeds all four models; it does not download weights at runtime.
`manifest.json` records their exact hashes and provenance.

- `omni-reference-router-v2`: a 2048 → 48 → 2 MLP selects the original-expression
  `evaluate()` tool. Weights: 393,848 bytes; Rust inference; no GPU.
- `omni-intent-v1`: a 2048 → 32 → 20 MLP selects a narrow set of verbal arithmetic
  tool calls, or defers. Weights: 264,960 bytes; Rust inference; no GPU.
- `omni-composition-v1`: a 2048 → 32 → 20 MLP selects bounded composition
  structures and reversed subtraction. Weights: 264,960 bytes; Rust inference;
  positional features retain operand roles. Every call is source-validated.
- `omni.cact`: a calculator specialization of Cactus Compute's Needle 2,
  exported from the V2 third-epoch adapter. Quantized weights: 13,737,807 bytes;
  45,211,383 base parameters. Its native runtime is bundled separately inside
  the executable. `--backend needle` forces this model for diagnostic comparisons.

The three MLPs were trained from scratch on generated data. The Needle model uses
LoRA continuation on the pinned public checkpoint, with generated calculator
requests and validation-feedback examples. No hosted teacher API, personal data,
or user files were used for training. The public checkout includes the selected
weights, generators, MLP test corpora, metadata, and selected development results.
Large Needle training datasets, intermediate adapters/checkpoints, and the
original training environment remain in the author's local development archive;
they are not required to build or run Omni. Paths under `training/runs/` in the
manifest identify that original provenance, rather than downloadable files in
this repository. See [training notes](../training/README.md).

The fast models preserve numeric text and use conservative thresholds and
source validation. Needle can still reject valid language or produce an incorrect
interpretation. Validation does not prove every possible sentence's meaning.
For complex formulas, explicit mathematical syntax offers the clearest input.
All authoritative answers are calculated by Rust, never by model text.

The default Needle weights were selected using actual CLI answers on the
development corpus, not training loss alone. The independently written audit is
reported separately. Development sets have been inspected repeatedly and are
not pristine measures of general English understanding. The MLP feature
normalization intentionally hides numeric values, so changing numbers alone
does not create an independent model-level holdout.

The original v0.1.0 application verification had 199/200 expected outcomes, with no
accepted wrong answers and one conservative language rejection. See
[verification](../docs/verification.md) and the public evidence copies for the scope and the initial
failures that led to validator and retry repairs. Experimental answer-boundary
alignment increased synthetic native accuracy but retained CLI regressions;
the original V2 third-epoch checkpoint remains bundled. Version 0.1.1 adds the
composition model and stricter validation after a broader audit found dropped
plural powers. The original 200 cases and a further 66 language cases now pass;
the [expanded accuracy report](../docs/accuracy-0.1.1.md) records the remaining
language limitations, rejected Needle fine-tunes, and VPS measurements.

The MLPs and Omni training code are MIT licensed. Needle's original
runtime, base checkpoint and specialized weights are Apache 2.0; see
[`third_party/needle/`](../third_party/needle/). A source build compiles Omni's Rust
code and embeds the pinned, prebuilt native Needle runtime; it does not compile
that runtime from native engine source. The original confidence head is not
calibrated for the specialized weights and is not reported as their confidence.
