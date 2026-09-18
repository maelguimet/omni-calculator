# v0.1.1 accuracy and performance

`add the squares of 6 and 8` now returns **100**. The previous release could
drop both powers and return 14. Source validation now checks each power,
operand, sign, grouping, and the outer operation for supported plural-power
requests. Ambiguous forms are rejected rather than guessed.

The original Needle 2 weights and runtime are unchanged. A new 264,960-byte
CPU classifier supplies bounded composition candidates, and existing fast
models can receive equivalent digit spellings of number words. Every candidate
still has to pass validation against the original request before execution.

## Real CLI results

Both executables used the same expected answers and retry budget of three.
The baseline is the original v0.1.0 development executable. The final run uses
v0.1.1 with the selected composition model.

| Measure | Baseline | v0.1.1 |
| --- | ---: | ---: |
| All cases | 306/352 (86.9%) | 346/352 (98.3%) |
| Correct calculations | 244/290 (84.1%) | 284/290 (97.9%) |
| Correct expected errors | 62/62 | 62/62 |
| Accepted wrong answers | 8 | 0 |
| Previously passing cases that regressed | — | 0 |

The existing 200 cases and 66 language regressions all pass. The broader
62-case expansion passes 60, and the 24 reserved cases pass 20. CI now requires
all 266 existing and language regression cases to pass. The other 86 remain
diagnostic suites with their actual expected answers, including the six failures.

These are development and regression results, **not an estimate of universal
English accuracy**. The expansion was written before fine-tuning, but its
failures informed development. The reserved set was written before the new
composition model and first evaluated after its initial training; it remained
visible during the subsequent positional-feature repair. Some wording families
and numeric-masked requests overlap synthetic training or validation examples.
It is not an untouched semantic holdout.

Evidence: [summary](evidence/accuracy-0.1.1/summary.json),
[baseline responses](evidence/accuracy-0.1.1/baseline-all.json),
[final responses](evidence/accuracy-0.1.1/final-all.json), and
[Rust and lint checks](evidence/accuracy-0.1.1/checks.json).
The development executable's SHA-256 is
`023bb21787fb7870f25267f06b514b5050d5baecf7bcd501bfb814df504e0ad4`.
GitHub rebuilds the tagged source with remapped build paths; the release
manifest and fresh packaged CI reports identify that executable separately.

### Remaining valid requests that are refused

All six remaining failures are refusals, not accepted wrong answers. Explicit
expressions provide these operations:

| Refused wording | Explicit form |
| --- | --- |
| add the square root of 81 and the cube root of 125 | `sqrt(81)+cbrt(125)` |
| find the smallest of -8, 3, -12, 9 | `min([-8,3,-12,9])` |
| could you calculate the product of the sum of 12 and 8 and 5 | `(12+8)*5` |
| the square root of 121 plus the cube root of 216 | `sqrt(121)+cbrt(216)` |
| divide the difference of 68 and 20 by the sum of 7 and 5 | `(68-20)/(7+5)` |
| express 850 cm as m | `850 cm to m` |

## Live VPS comparison

Two sequential runs of each executable used the same 40 requests on the live
Debian 13 VPS. Kernel-enforced limits were one CPU, 512 MiB memory, no swap,
64 tasks, and a four-minute lifetime, with `nice 10`. These are warm end-to-end
CLI wall times, including process startup and IPC. They are a small live-server
sample, not a latency guarantee.

Both baseline runs passed 32/40; both final runs passed **40/40**.

| Request family | Baseline median | v0.1.1 median |
| --- | ---: | ---: |
| Literal arithmetic | 2.31 ms | 2.62 ms |
| Existing intent classifier | 2.95 ms | 2.43 ms |
| Number words | 430.57 ms | 3.14 ms |
| New powers, ordering, compositions, signed words | no correct baseline results | 2.37–2.57 ms |
| Unit conversion | 361.16 ms | 363.10 ms |
| Existing generative compositions | 465.76 ms | 459.17 ms |
| Statistics | 462.68 ms | 461.87 ms |
| Calculus | 598.98 ms | 607.39 ms |
| Temperature conversion | 1288.51 ms | 1292.85 ms |

The largest observed increase in a family median was 8.40 ms, below the
100 ms tradeoff threshold. The largest gain came from routing number words
through the existing small model. The broader generative paths still take
hundreds of milliseconds or more. Incorrect baseline attempts are not counted
as successful latency measurements.

[Aggregated measurements](evidence/accuracy-0.1.1/vps-summary.json) link the
binary identities, resource limits, and cold-start observations. Individual
responses and timings are preserved for
[baseline A](evidence/accuracy-0.1.1/vps-baseline-a.json),
[baseline B](evidence/accuracy-0.1.1/vps-baseline-b.json),
[final A](evidence/accuracy-0.1.1/vps-final-a.json), and
[final B](evidence/accuracy-0.1.1/vps-final-b.json).
The transient workers stopped and only the marked, task-owned benchmark
directory was removed. No server services or configuration were changed.
[Cleanup record](evidence/accuracy-0.1.1/vps-cleanup.json).

## Model selection and training

The [composition model card](../training/composition/README.md) describes its
training data, feature representation, threshold selection, limitations, and
reproduction command. All 12,240 positive synthetic targets passed the native
source validator. Python/Rust inference parity and full operation-structure
checks over the 4,272 synthetic test examples are covered by Rust tests.
Synthetic templates are shared across splits; the model card explains which
wrappers and masked strings are separated.

A separate Needle continuation experiment used 3,000 rows: 1,800 targeted
examples and 1,200 replay examples, generated by
[generate_accuracy.py](../training/generate_accuracy.py). The new targets
passed the [native auditor](../examples/audit_training.rs) before training.
Two epochs took 431 seconds with bounded resources; the longest training
sequence was 635 tokens under a 672-token cap. On the 328-case development
suite, the epoch-one and epoch-two exports passed 298 and 302 cases;
float32 export passed 303, and two blended adapters passed 303 and 306.
They caused regressions and were rejected. The original Needle model remains
bundled. Earlier Needle 3 evaluation also did not justify a switch.
