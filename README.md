# Omni calculator

A Linux CLI scientific calculator written in Rust. Small trained local models
map requests to typed tools; Rust validates calls and performs the mathematics.
There is no hosted API, account, telemetry, Python runtime, or GPU requirement.

[Download a release](https://github.com/maelguimet/omni-calculator/releases/latest)
· [Build and test status](https://github.com/maelguimet/omni-calculator/actions/workflows/ci.yml)
· [Mathematical reference](docs/mathematics.md)

```sh
calc what is 1 + 1
calc wat iz 1+1 --json
calc 'what is (3+4)*5?' --retry 3
calc 'what is the square root of 144'
calc 'add the squares of 6 and 8'
calc 'integrate(sin(x),x,0,pi)'
```

Plain words, numbers, `+`, `/`, and `^` can usually be passed without quotes.
Your shell interprets `*`, parentheses, redirects, apostrophes, and sometimes
`!` before an app can read them. Quote those inputs, or pipe the text into `calc`.

## Install a released binary

The executable includes the trained weights and native inference runtime. It
targets **Linux x86-64 with glibc 2.39 or newer** (for example Ubuntu 24.04 or
Debian 13). Other architectures and musl-based distributions are not packaged.
Download the archive and checksums from the
[release page](https://github.com/maelguimet/omni-calculator/releases/latest), or:

```sh
curl -fLO https://github.com/maelguimet/omni-calculator/releases/download/v0.1.1/omni-calc-0.1.1-linux-x86_64.tar.gz
curl -fLO https://github.com/maelguimet/omni-calculator/releases/download/v0.1.1/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf omni-calc-0.1.1-linux-x86_64.tar.gz
install -Dm755 omni-calc-0.1.1-linux-x86_64/calc "$HOME/.local/bin/calc"
```

Ensure `$HOME/.local/bin` is on your `PATH`, then run `calc what is 1 + 1`.
The archive also contains documentation, model provenance, and license notices.

## Build from source

Use Linux x86-64, a C linker (`cc`, usually supplied by `build-essential`),
Rust/Cargo, and Python 3.11 or newer **during setup**. CI uses Rust 1.97.1.
The selected model weights are included in Git; no Git LFS or training step is
needed. The build compiles Omni and its Rust dependencies from source and
bundles the checksum-pinned, prebuilt Needle 2.0.4 native engine.

```sh
git clone https://github.com/maelguimet/omni-calculator.git
cd omni-calculator
rustup toolchain install 1.97.1 --profile minimal
python3 scripts/setup_model.py
cargo +1.97.1 build --release --locked
./target/release/calc what is 1 + 1
install -Dm755 target/release/calc "$HOME/.local/bin/calc"
```

The setup script downloads about 13 MB and verifies both the package and native
library hashes. It only writes `.cache/needle/` inside this checkout. Subsequent
builds can run offline once Cargo dependencies are cached. The executable embeds the native model engine and loads
it through a sealed anonymous Linux file; it does not extract an executable
into a shared directory or download anything at runtime.

The default build includes `models/omni.cact` and the three tiny Rust models. Their
[model card and hashes](models/README.md) describe the selected weights.
Use `--model path/to/custom.cact` or `OMNI_MODEL` to override the Needle weights.
Training procedures, datasets, and deployment evaluations are in
[training/README.md](training/README.md). Development builds without the model
are available with `--no-default-features`; their mathematical library and
validation tests still run, while inference returns `model_unavailable`.

For an offline rebuild, verify the cached engine with
`python3 scripts/setup_model.py --offline` and add `--offline` to Cargo.
See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and release packaging.

## Humans and agents

Human output contains only the result on stdout. Errors go to stderr.
`--explain` adds the interpreted expression, attempts, and timing to stderr.
`--json` emits exactly one JSON object on stdout, including on failure:

```json
{"schema_version":1,"ok":true,"input":"wat iz 1+1","expression":"1+1","tool_call":{"name":"evaluate","arguments":{}},"result":{"display":"2","value":2,"kind":"integer","exact":true},"attempts":1,"retries":0,"timing":{"total_ms":0,"model_ms":0,"compute_ms":0},"model":"omni-reference-router-v2"}
```

The zeros above illustrate the schema, not measured performance. `result.value`
is a JSON number, boolean, lossless string, array, or structured object depending on the
calculation. Large integers, exact fractions, units, and symbolic expressions
are retained without rounding through a JSON floating-point value. Exact rational
values use an improper fraction string even when human display uses mixed numbers.
Exact integers outside `[-(2^53-1), 2^53-1]` use strings so JavaScript and other
IEEE-754 JSON clients cannot silently round them.
Always check
`ok`; inspect `expression` when the intended interpretation matters.

For a stream, use `--jsonl`. Each line is a JSON object with `input` and optional
`id`, `retry`, and `timeout_ms`, or a JSON string. Requests are independent;
variables and conversation context do not leak between records. Malformed and
oversized lines produce errors and the next record is still processed.

```sh
printf '%s\n' \
  '{"id":"a","input":"what is 1+1"}' \
  '{"id":"b","input":"what is the weather"}' | calc --jsonl
```

Exit codes: **0** success, **2** invalid/unsupported input, **3** mathematical
error/unsupported mathematical operation, **4** model failure, **5** timeout or
resource limit, **6** local service/I/O error. JSONL returns the highest error
code encountered, after processing the stream.

## Local model and retries

The first invocation starts a private local worker. Two trained neural networks
implemented in Rust handle literal expressions and a subset of simple verbal
operations. Broader language lazily loads the local Needle model. Every accepted
request still requires a model-selected tool call; no answer cache is used.
Later invocations reuse loaded weights and cached tools. `--warmup` starts and checks the fast path;
`--backend needle --warmup` also loads and checks the generative model.
`--status` reports it; `--stop` stops it. It exits after two idle minutes.
`--no-daemon` uses a fresh disposable process for one request. `--threads N`
sets model threads (default 2; range 1–8). There is no TCP listener.
`--backend needle` bypasses both tiny models; the default is `--backend auto`.
JSON identifies the selected model, with `none` when input validation ends the
request before inference. Classifier confidence is a score, not a proof that a
language interpretation is correct.

`--retry N` means **N additional attempts**, up to 20, with three by default.
Retries retain the loaded model and give it validation feedback; there is no
backoff sleep. Identical failures stop early. Mathematical domain errors do not
cause the model to alter the question. `--timeout-ms N` bounds the entire
request, including retries (default 5000 ms; maximum 30000 ms).

Literal expressions use an `evaluate()` reference tool: Rust reads the complete
original expression, so generated text cannot change parentheses or operators.
Natural-language operations use explicit function, conversion, and algebra
tools. Calls are checked for allowed names, argument shape, preserved numbers,
and supported mathematical meaning before execution. Rejection is preferable
to silently supplying a different calculation. The model is fallible; these
checks do not prove the interpretation of every possible English sentence.

Workers use a private owner-only Unix socket, a 768 MiB address-space limit,
bounded input/output, and a process watchdog for native hangs. Each calculation
has a deadline. The application evaluates mathematical expressions, never shell,
Python, Rust, or filesystem operations from a model. `OMNI_RUNTIME_DIR` can
select an existing/creatable private runtime directory; its mode must be 0700.
The [architecture notes](docs/architecture.md) describe the process and model boundaries.

## Mathematics

Use the explicit expression syntax for complex formulas and functions outside
the model's current natural-language vocabulary. These still pass through the
model's `evaluate()` tool. See [the mathematical reference](docs/mathematics.md)
for conventions and supported operations.

The scope includes exact arithmetic, units, complex numbers, trigonometry,
statistics and distributions, vectors and matrices, numerical calculus,
polynomial algebra, symbolic derivatives and common integrals. This is a broad
scientific calculator, not a general solution to every computer-algebra problem.
Unsupported symbolic cases return an explicit error.

Use `ln(x)` for natural logarithms and `log(x)` for base 10. Statistics and matrix
operations use finite-precision real numbers; exact scalar integers and
fractions retain their precision. Numerical integration has bounded work and
convergence checks, while `derivative_at` uses automatic differentiation.

## Verify and benchmark

The development VPS benchmark measured **1.8–2.7 ms** median for the tested fast
literal/verbal requests and **377 ms** for Needle-backed conversions, with all
60 warm answers correct on one CPU. The 200 CLI audit cases had 199 expected
outcomes and no accepted wrong answers; one valid English phrase was safely
rejected. See the [verification report](docs/verification.md) for exact scopes,
startup costs, test results, limitations, and artifact hashes. GitHub release
binaries are rebuilt and tested in CI; the historical VPS measurements identify
their original development executable separately.

```sh
cargo test --locked
cargo test --release --locked
cargo clippy --all-targets -- -D warnings
python3 scripts/benchmark.py target/release/calc \
  --one-cpu --threads 1 --runs 10 --output reports/local.json
```

The benchmark measures a cold worker and unique warm requests separately, checks
correctness, and records the executable's SHA-256. It runs sequentially at a
lower scheduling priority and stops its own worker afterward. Timing alone is
not a passing result. Fast verbal arithmetic and generative-model requests are
reported separately, including the actual selected backend. A cold worker does
not imply that the operating system's disk cache was flushed. VPS measurements
must be taken on the final executable.

Rust code: MIT. Needle model and engine: Apache 2.0, from
[Cactus Compute](https://github.com/cactus-compute/needle) and the
[pinned model repository](https://huggingface.co/Cactus-Compute/needle2/tree/32e9e3a93b205f786929697446ae669cf0a84579).
Core scalar arithmetic uses [fend-core](https://github.com/printfn/fend), with a
[local numerical conversion fix](vendor/fend-core/OMNI-PATCH.md), alongside
nalgebra, statrs, and the Rust num crates for the additional mathematics.
