# Contributing to Omni

Issues and pull requests are welcome at
<https://github.com/maelguimet/omni-calculator>. Include the exact request,
expected result, actual JSON output, platform, and `calc --version` when reporting
a calculator problem. Do not include private inputs or credentials.

Follow the source setup in [README.md](README.md). The runtime application is
Rust; Python is used for build setup, verification, packaging, and optional
model training. RTK and a GPU are not required.

## Checks

CI pins Rust 1.97.1. Install its formatter and linter if needed:

```sh
rustup component add --toolchain 1.97.1 rustfmt clippy
cargo +1.97.1 fmt --all -- --check
cargo +1.97.1 clippy --locked --all-targets -- -D warnings
cargo +1.97.1 clippy --locked --all-targets --no-default-features -- -D warnings
cargo +1.97.1 test --locked --release
cargo +1.97.1 test --locked --no-default-features
python3 scripts/verify_cli.py target/release/calc --output target/acceptance.json
python3 scripts/verify_cli.py target/release/calc \
  --cases tests/cases/final-audit-json-safe.json --output target/independent-audit.json
```

Keep model interpretation separate from mathematical execution. A model call
must pass source and schema validation before evaluation. Include meaningful
regressions for changed quantities, grouping, operand order, malformed inputs,
and error behavior when those areas change. CI also runs
`tests/cases/order-ood-audit.json` and `tests/cases/accuracy-regressions.json`.
The broader `accuracy-expansion.json` and `accuracy-holdout.json` remain
diagnostic audits: their expected mathematical answers are retained even for
known language rejections. See [accuracy evidence](docs/accuracy-0.1.1.md).

## Models and licenses

The deployed weights and MLP test fixtures are versioned with the source. See
[the model card](models/README.md) and [training notes](training/README.md).
Large base checkpoints, alternate training runs, environments, and generated
Needle curricula are excluded from Git. Do not edit a hash-pinned fixture or
training script without updating its provenance and independently validating
the resulting model behavior.

Omni code is MIT licensed. Needle components are Apache-2.0; dependency notices
are in `third_party/`. Keep attribution and modification notices when changing
vendored code. If dependencies or the toolchain change, refresh the notices
using `python3 third_party/rust/collect.py` after fetching dependencies. The
collector needs the toolchain's `rust-docs` component for standard-library notices.

## Releases

The [GitHub workflow](https://github.com/maelguimet/omni-calculator/blob/main/.github/workflows/ci.yml) builds from a clean checkout on
Ubuntu 24.04, checks Rust code and real model-backed CLI cases, and packages the
result with hashes and licenses. Source paths are remapped during compilation.
Main-branch and pull-request builds upload CI artifacts. A version tag matching
`Cargo.toml` publishes a GitHub release only after the same checks succeed.
Maintainers update `.github/release-notes.md` before tagging a version.

To inspect a local package after building and running the checks above:

```sh
python3 scripts/package.py \
  --report target/acceptance.json --report target/independent-audit.json
```

The output is in `dist/`. Each supplied report must identify the exact executable
being packaged. Historical development results in `docs/evidence/` retain their
own executable hashes and are not measurements of a later rebuild.

## File-size cap

Each file stays under about 5,000 tokens, counted as 20,000 bytes. CI is the gate: the File size workflow runs `scripts/check-file-size --all` on every push and pull request. On pull requests it runs the base branch's copy of the script, and also rejects new or raised `.file-size-baseline` entries and any over-cap file that is exempt through `.gitattributes` but was not an exempt file on the base branch, including a new file under an existing exemption. Narrowing an exemption passes. The pre-commit hook runs the same check on staged files before CI does. Enable it once per clone:

```sh
git config core.hooksPath .githooks
```

`scripts/check-file-size --all` checks the whole tree. When the check fails, split the file by responsibility, move tests to their own file, or extract data and fixtures. Some files were already over the cap when the check was added. They are listed in `.file-size-baseline` with their size at that time; they may shrink but not grow, and their entries may only be lowered or removed. Lockfiles (by exact name), files with binary content, and paths marked `linguist-generated`, `linguist-vendored`, `binary` or `-diff` in the committed `.gitattributes` are exempt.
