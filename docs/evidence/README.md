# Original verification evidence

These are publication copies of the local development reports from 2026-09-14.
The originals remain unchanged in the author's local `reports/` directory.
This directory is a historical baseline, not a report that GitHub CI has rebuilt
or benchmarked exactly those same executable bytes.

The final baseline executable was Linux x86-64, 31,585,552 bytes, SHA-256:

```text
bb8d4a5cc7cff5783fc7967a32ef1fd3e2750baff8f70b7a4edf46fa650cd07a
```

The final Rust checks, final CLI audits, CLI smoke checks, local/VPS benchmarks,
offline trace result, and original package verification identify that binary.
`independent-audit-initial.json` and `order-ood-audit-initial.json` concern earlier
builds: they preserve the original failures behind the subsequent validator and
retry fixes. Consult each report's own `binary_sha256`. The cleanup record
records the benchmark environment after the worker stopped; it has no binary
identity of its own.

## Publication changes

Absolute author workspace and home prefixes are replaced with `{workspace}` and
`{home}`. References to included reports, logs, and traces are rewritten to their
repository-relative `docs/evidence/` locations. No VPS address, SSH login, token,
or private key was present in the selected originals. Machine platform details,
process IDs, CPU affinity, memory figures, times, requests, expected answers,
actual results, and recorded artifact hashes are preserved.

[manifest.json](manifest.json) records the original and public SHA-256 for every
copy, along with its transformations. Recorded hashes inside reports still
identify the original measured artifacts or original evidence bytes; they are
not silently recomputed after redaction. Use the manifest's `public_sha256` for
these published copies. [SHA256SUMS](SHA256SUMS) covers all public files here
except itself. Verify from this directory with:

```sh
sha256sum --check SHA256SUMS
```

The original `package-verification.json` describes a local pre-publication
archive and its historical checksum. That archive checksum is not the checksum
of a later GitHub release. Source-path remapping, toolchains, documentation,
packaging, or later source changes can produce a different release binary.
Use the GitHub release's checksums and workflow logs to identify and verify its
artifacts. Performance numbers here remain measurements of the original binary
on the documented machines, not new benchmarks of every subsequent build.

## Contents

- [Rust checks](rust-checks-final.json) and their six attached logs.
- [Development CLI cases](acceptance-final.json), [primary audit](independent-audit-final.json),
  [follow-up audit](order-ood-audit-final.json), and [classified final review](final-verification-review.json).
- [Initial primary audit](independent-audit-initial.json) and [initial follow-up](order-ood-audit-initial.json).
- [CLI/JSONL smoke checks](final-cli-smoke.json).
- [VPS benchmark](vps-final.json), [local benchmark](local-final.json), and [VPS cleanup](vps-cleanup.json).
- [One forced-Needle offline check](offline-verification.json) and its [network-syscall trace](offline-network.trace).
- [Original package verification](package-verification.json).

The baseline final CLI total is 199/200 expected outcomes, with one valid
subtraction phrase rejected and no accepted wrong answer in those cases.
These are scoped regression results; they do not prove universal language
coverage, numerical correctness for all inputs, or zero network activity for
all future modifications. See the [verification narrative](../verification.md)
for the test scope, known limitation, and performance interpretation.
