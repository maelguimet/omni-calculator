`add the squares of 6 and 8` now returns **100**. This release improves natural-language accuracy while keeping the original Needle 2 model and runtime.

- A small trained CPU classifier handles more powers, nested arithmetic, and reversed subtraction. Number words can use the existing fast model.
- Stricter validation checks powers, signs, grouping, quantities, and operand order against the original request.
- The expanded 352-case CLI suite improves from **306/352 (86.9%) to 346/352 (98.3%)**, with **zero observed accepted wrong answers** and no previously passing cases lost. Six valid requests are still refused. These are development/regression results, not universal language accuracy.
- Two live-VPS runs each improve from **32/40 to 40/40**. Newly supported arithmetic has 2–3 ms family medians; the largest existing-family median increase is 8.4 ms. Number-word requests improve from about 431 ms to 3 ms.
- CI now checks 266 CLI regressions, both Rust feature configurations, formatting, linting, and the actual extracted release executable.

See the [accuracy report and remaining limitations](https://github.com/maelguimet/omni-calculator/blob/v0.1.1/docs/accuracy-0.1.1.md) for the corpora, data-exposure caveats, raw responses, training provenance, and VPS measurements. Needle 3 and further generative fine-tuning were evaluated; neither justified replacing the selected Needle 2 model.

The executable runs fully locally without Python, a GPU, an account, or network access. It targets **Linux x86_64 with glibc 2.39 or newer**. Download the archive and `SHA256SUMS` together, then run:

```sh
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf omni-calc-0.1.1-linux-x86_64.tar.gz
cd omni-calc-0.1.1-linux-x86_64
./calc 'add the squares of 6 and 8'
./calc 'multiply negative six by eleven' --json
```

The archive includes documentation, licenses, model provenance, an artifact manifest, and fresh acceptance reports for its executable. A standalone `calc` is also available; run `chmod +x calc` after downloading. Broader generative requests still take hundreds of milliseconds or more. See the [README](https://github.com/maelguimet/omni-calculator/blob/v0.1.1/README.md) for usage and building from source.
