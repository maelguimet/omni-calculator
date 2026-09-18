# Development verification — 2026-09-14

This report records v0.1.0. See the [v0.1.1 accuracy and performance report](accuracy-0.1.1.md) for current results.

These historical results identify the original development build, before the
public GitHub release pipeline was added. GitHub release binaries are rebuilt
from their tagged source, use remapped compilation paths, and have different
hashes. Their fresh CLI reports and source revision are included in the release
archive; CI also runs Rust checks and verifies the extracted executable.

The development Linux x86-64 executable is 31,585,552 bytes. Its SHA-256 is
`bb8d4a5cc7cff5783fc7967a32ef1fd3e2750baff8f70b7a4edf46fa650cd07a`.
The Rust checks, application audits, local benchmark, VPS benchmark, and network
trace below all refer to these same development executable bytes. Public evidence
copies redact local paths and machine identity while retaining results and
original artifact hashes; see [the evidence index](evidence/README.md).

## Correctness

| Check | Outcome |
| --- | --- |
| Rust tests, default debug and release | 125 passed in each configuration |
| Rust tests, without bundled models | 115 passed |
| Formatting and all-target clippy, both feature configurations | Passed, warnings denied |
| Development CLI corpus | 75/75 expected outcomes |
| Independently authored primary CLI corpus | 100/100 expected outcomes after fixes |
| Follow-up language and scope corpus | 24/25 expected outcomes |

The 200 CLI cases contain **no accepted wrong answers or accepted out-of-scope
tasks** in the final runs. One valid phrase, `what is 9 less than 42`, still
exhausts its retries and returns an error because the model cannot produce the
correct grounded call. The equivalent literal input `42-9` is supported. This
is a known language-model limitation, not a claim of universal English coverage.

The initial independent 100-case run passed 98 cases. It exposed reversed
`divide A into B` operands and arithmetic accepted inside a request to write a
poem. Both were fixed and retained as regressions. The follow-up cases initially
passed 22/25; improved generic feedback, retry identity, and wrapper normalization
recovered two model failures. No audit request was added to model training.

One primary golden expectation was updated solely for the safer JSON contract:
an integer beyond `2^53-1` is now the identical decimal string. The original
request, mathematical answer, original corpus, original reports, and the
one-row representation change are preserved in the source checkout. These
audits informed validator repairs; the final reruns are regression evidence,
not an untouched estimate of general language accuracy.

The Rust tests cover exact arithmetic with an independent rational-expression
oracle, numerical and symbolic identities and domains, malformed inputs, JSON
precision and IDs, changed model operands/operations, bounded retries, private
IPC, absolute deadlines, worker recovery, and broken pipes. The vendored scalar
engine's separate conversion-fix validation is documented in its patch notice.

Full results: [Rust checks](evidence/rust-checks-final.json),
[development cases](evidence/acceptance-final.json),
[primary cases](evidence/independent-audit-final.json),
[follow-up cases](evidence/order-ood-audit-final.json), and
[independent review](evidence/final-verification-review.json).

## VPS performance

Measured on the authorized Debian 13 VPS, with one of its four CPUs selected,
one inference thread, sequential requests, and scheduling priority `nice 10`.
Each warm group contains 20 requests with different numeric inputs. All 60
warm requests and all six startup/scope/scientific checks were correct.

| Request family | Median wall time | p95 | Selected model |
| --- | ---: | ---: | --- |
| Warm `wat iz A+B` | 1.80 ms | 3.29 ms | Reference MLP |
| Warm `what is the square root of N` | 2.75 ms | 3.20 ms | Intent MLP |
| Warm `convert N km to m` | 377.49 ms | 386.79 ms | Tuned Needle |

The first literal request started a fresh worker in **6.71 ms**. The first
generative conversion, including loading Needle, took **2,977.56 ms**. A cold
worker does not imply a flushed operating-system disk cache. These wall times
include CLI process startup, IPC, model selection, calculation, and JSON output.
They are not isolated neural-network timings.

Worker resident memory was approximately **6.4 MiB** after the two fast paths
and **59.8 MiB** after loading Needle. The sub-100-ms target is met for the
measured fast paths; broader generative requests do not yet meet it. Difficult
mathematical work also has its own computational cost. Do not extrapolate these
specific wording families to all natural-language or scientific requests.

The same one-CPU local benchmark measured medians of 1.88 ms, 1.94 ms, and
252.33 ms for these groups. Reports preserve all individual requests, answers,
timings, model names, process memory, executable hashes, and platform details:
[VPS](evidence/vps-final.json), [local](evidence/local-final.json).

The VPS benchmark worker stopped, and only the known files in our temporary
benchmark directory were removed. No server services or configuration were
changed. The final load average was `0.01 0.03 0.00`, with about 2.39 GiB
available memory. [Cleanup record](evidence/vps-cleanup.json).

## Local execution and model selection

A network-syscall trace of a real forced-Needle conversion, using a fresh
disposable worker, observed no Internet socket operations. The source contains
no hosted inference client or runtime model downloader. The trace corroborates
the exercised path; it is not a proof about every possible future modification.
[Trace result](evidence/offline-verification.json).

The package script verifies that the executable literally contains the three
documented model files and the pinned native runtime, then verifies every
archived file's hash. There are no sidecar weights or Python/GPU requirements
for the executable.

Original V2 epoch 3 remains the selected Needle model. Answer-token alignment
experiments improved synthetic native tool-call accuracy, but retained real CLI
composition regressions. The selected candidate passed the full 75-case
development corpus; the final aligned V2 candidate passed 74. Alternative
weights and full experiment records remain in the local development archive;
public source includes the selected weights, generators, and curated metadata. The model card
and manifest identify the exact selected weights and their provenance.
