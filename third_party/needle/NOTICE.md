# Needle runtime and model

Omni includes Cactus Compute's Needle 2 runtime and a calculator specialization
of its model, under Apache License 2.0. The original license is in `LICENSE`.

- Upstream: <https://github.com/cactus-compute/needle>
- Model: <https://huggingface.co/Cactus-Compute/needle2>
- Pinned model repository revision: `32e9e3a93b205f786929697446ae669cf0a84579`
- Native package: `cactus_needle-2.0.4-py3-none-manylinux2014_x86_64.whl`
- Package SHA-256: `13a84e6c73095fd175b11d46a30a984b62123d94421b769c107074aff7f65c2b`
- Embedded `libneedle.so` SHA-256: `9fa5386d3e3a8ee17914fb23643bc5f5c906b683fa33561e79c5445dd78bc389`

Omni modifications: synthetic calculator-specific adapter training and exported
quantized weights, separate tiny reference and intent models, a Rust inference wrapper,
typed mathematical tools, validation, and process isolation. The native shared
library is used without modifications. Training provenance and candidate
evaluations are preserved in `training/` and `experiments/`.

The Apache license does not imply endorsement by Cactus Compute.
