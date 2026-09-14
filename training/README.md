# Training and model provenance

Training is optional. The public checkout includes the three selected model
files, so building or running Omni does not require a training environment,
Python ML packages, or a GPU. See the root [README](../README.md) for source builds.

The included generators produce calculator requests and typed tool calls from
mathematical templates. They do not call a hosted teacher or use personal files.
The model selects a tool; Rust validates the source interpretation and computes
the answer. An `evaluate()` call refers to the original expression and therefore
does not ask the model to reproduce its operators or parentheses.

## What is published

The public source includes training/evaluation scripts, seeds and model
metadata, the MLP corpora and parity fixtures used by the Rust tests, and selected
development summaries. `models/manifest.json` records the selected model hashes
and the original Needle training-data, adapter, and run-record hashes.

Large generated Needle datasets, the downloaded base checkpoint, intermediate
adapters and exported candidates, package caches, and the original virtual
environment are retained in the author's local development archive. They are
not in the public checkout. Historical paths such as `training/runs/v2/` in
metadata refer to that archive; they are not links to published checkpoints.
The public generators can create new datasets and train new candidates. This is
not a claim that the selected Needle weights can be reproduced bit for bit from
the public files alone.

Public application evidence is in [docs/evidence](../docs/evidence/README.md).
Those reports identify the original tested binary; later GitHub release builds
have their own artifact identities and CI results.

## Tiny CPU models

The reference and intent models train from scratch with NumPy and SciPy.
Create an isolated environment from the repository root:

```sh
python3 -m venv training/.venv
training/.venv/bin/python -m pip install numpy scipy
```

The training scripts regenerate datasets, metadata, and the selected files under
`assets/`; use a separate checkout when experimenting if you want to retain the
published weights. Start with one BLAS thread on a small machine:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/reference-v2/train.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 training/.venv/bin/python training/intent/train.py --epochs 32
```

Exact numerical reproducibility can depend on dependency versions and BLAS.
The committed weights and their hashes are the source-build inputs; retraining
is a separate development activity. Model-specific details, audits, and formats:
[reference V2](reference-v2/README.md), [intent V1](intent/README.md), and the
[earlier reference experiment](router/README.md).

## Needle specialization

The selected model is the original V2 third-epoch export, committed as
`models/omni.cact`, SHA-256
`1852739bfbb36f945eed2bca5c3ddfd364fab276cf7914ccbe793df4746e46e4`.
Its base has 45,211,383 parameters and comes from `Cactus-Compute/needle2` revision
`32e9e3a93b205f786929697446ae669cf0a84579`.

Training followed the official [Needle fine-tuning implementation](https://github.com/cactus-compute/needle/blob/main/doc/finetuning.md)
with `cactus-needle==2.0.12`, JAX, LoRA, and deployment-aware quantization. The
observed package versions are in `environment-versions.json`; that record is
not a complete portable lockfile or an environment installer. The native engine
embedded in the CLI is separately pinned to version 2.0.4 by
`scripts/setup_model.py`.

For new training, install the upstream training dependencies in your own
environment. The local scripts expect these files from the pinned model
repository under `training/base/`:

| File | SHA-256 |
| --- | --- |
| `checkpoints/needle2.pkl` | `4b0a972d163ffc7678fb3c36bace508114872e9d2ce9e10f225825752d3795bc` |
| `tokenizer/tokenizer.model` | `0871b8e3df78e0dd7ccc236e187fc055f75cd57ebe3bec76ca8103030b8fe659` |
| `tokenizer/tokenizer.vocab` | `ed835aff5b2a3ca67b5896906dd471ab0ce12626d699d90e736e3c7e4ba9b1e3` |

The scripts are development tools, not a one-command reconstruction of the
original continuation chain. For example, after preparing the upstream
environment and base files, create a fresh V3 curriculum and candidate:

```sh
python3 training/generate_v3.py --out-dir training/data/new-v3
nice -n 10 training/.venv/bin/python training/audit_dataset.py training/data/new-v3 --heldout heldout-source-disjoint.jsonl --tokens
nice -n 10 training/.venv/bin/python training/run_training.py --data training/data/new-v3/train.jsonl --out-dir training/runs/new-v3 --epochs 1 --batch-size 4 --max-len 672
nice -n 10 training/.venv/bin/python training/export.py training/runs/new-v3/best.pkl training/runs/new-v3/best.cact
```

The generator excludes any available frozen development queries; omitting old
unpublished datasets can therefore change its output. The dataset auditor checks
tool schemas, required fields, arity, source separation including retry wrappers,
and rendered token lengths. Check every candidate with the native runtime and
the full CLI; low training loss does not establish deployable correctness.

The examples use low scheduling priority. The runner restricts itself to up
to four logical CPUs, disables JAX GPU preallocation, caps its GPU pool at 25%,
and defaults to a
90-minute wall limit. CUDA context overhead is additional. It defaults to rank
16, alpha 32, learning rate 0.0002, and the base checkpoint's mixed
`embedding=4,mhc=4,default=2` weight quantization with 8-bit activation/KV
quantization. `--adapter` continues adapter weights with a fresh optimizer and
schedule. `SIGTERM`, `SIGINT`, or a `STOP` file in the run directory requests a
checkpointed stop. CPU-only MLP training is much lighter than Needle training.

## Selection and known limitations

The original development sequence used 7,680 V1 examples, then 10,240 mixed V2
examples including retry feedback, then a 24,576-example V3 curriculum. V1 and
V2 included reasoning targets; V3 trained direct tool calls and broader
compositions. The selected V2 candidate passed 75/75 real CLI development cases.
An answer-token alignment candidate scored better on synthetic native routing
but passed 74/75 CLI cases, so it was not selected.

An investigation found differences between separately encoded answer tokens
and contiguous prompt/answer encoding, along with residual JAX/native inference
differences. The retained experiment notes discuss the evidence and its limits.
The issue was not declared fully resolved. The native confidence head was not
recalibrated after specialization and is not used as its confidence estimate.

Development distributions deliberately share mathematical families and have
been inspected repeatedly. Numeric masking in the MLPs also means new numbers
alone do not create an independent model-level holdout. The separate application
audits were not added to training, but their failures informed validator fixes;
final reruns are regression evidence, not untouched estimates of general English
accuracy. The [verification report](../docs/verification.md) includes the remaining
valid-language rejection and measured latency limits.

Original Omni training code and the two MLPs are MIT licensed. The Needle base,
specialized weights, and runtime are Apache 2.0; see
[third-party notices](../third_party/needle/NOTICE.md).
