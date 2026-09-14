Omni Calculator's first public release: a fully local Rust scientific calculator for humans and agents. Small bundled models turn natural language into validated tool calls; Rust executes the mathematics.

- Human CLI output, JSON and JSONL, stable exit codes, bounded retries, and request timeouts.
- Exact arithmetic, units, complex numbers, scientific functions, statistics, distributions, matrices, numerical calculus, and symbolic algebra.
- No account, network access, Python installation, or GPU required to run the released executable.

The binary targets **Linux x86_64 with glibc 2.39 or newer**. Download the archive, check its SHA-256 checksum, and unpack it:

```sh
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf omni-calc-0.1.0-linux-x86_64.tar.gz
cd omni-calc-0.1.0-linux-x86_64
./calc what is 1 + 1
./calc wat iz 1+1 --json
./calc 'integrate(sin(x),x,0,pi)'
```

Download `SHA256SUMS` beside the archive before running the first command. The archive includes documentation, model provenance, licenses, an artifact manifest, and acceptance reports for this exact executable. A standalone `calc` executable is also available; mark it executable with `chmod +x calc` after downloading.

Common requests use fast trained classifiers. Broader language uses a generative model and takes longer, especially on its first request. Language coverage and symbolic mathematics have limits: for example, `what is 9 less than 42` is conservatively rejected, while `42-9` works. The mathematical reference documents supported operations and numerical limits.

See the [README](https://github.com/maelguimet/omni-calculator#readme) for building from source and CLI usage. The [verification report](https://github.com/maelguimet/omni-calculator/blob/v0.1.0/docs/verification.md) describes local testing, VPS benchmarks, and known limitations. Every release is built from its tagged source and checked with Rust tests and two real-CLI acceptance suites before publication.
