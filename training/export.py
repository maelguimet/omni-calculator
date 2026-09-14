#!/usr/bin/env python3
"""Export an epoch adapter into a local Needle 2 .cact, on CPU only."""

import argparse
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])

parser = argparse.ArgumentParser()
parser.add_argument("adapter")
parser.add_argument("output")
parser.add_argument("--float32", action="store_true", help="Merge on float32 base values, matching training exactly")
args = parser.parse_args()

from needle.model import tokenizer
from needle.model.finetune import build_main

if args.float32:
    import jax
    import numpy as np
    from needle.model import run
    original_load = run.load_checkpoint

    def float32_load(*load_args, **load_kwargs):
        loaded = list(original_load(*load_args, **load_kwargs))
        loaded[0] = jax.tree.map(lambda value: np.asarray(value).astype(np.float32), loaded[0])
        return tuple(loaded)

    run.load_checkpoint = float32_load

tokenizer.TOKENIZER_DIR = str(ROOT / "base/tokenizer")
tokenizer.TOKENIZER_PREFIX = str(ROOT / "base/tokenizer/tokenizer")
build_main(argparse.Namespace(checkpoint=str(ROOT / "base/checkpoints/needle2.pkl"),
                              lora=args.adapter, out=args.output, bits=None, upload=False))
