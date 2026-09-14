#!/usr/bin/env python3
"""Bounded, reproducible Needle LoRA/QAT training with epoch checkpoints.

Uses the installed upstream training primitives. Model optimization follows
needle.model.finetune, adding deterministic shuffles, resumable epoch artifacts,
held-out validation early stopping, resource settings, and durable logs.
"""

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import signal
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "train.jsonl"))
    parser.add_argument("--checkpoint", default=str(ROOT / "base/checkpoints/needle2.pkl"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs/v1"))
    parser.add_argument("--adapter", help="Resume adapter weights on the same frozen base; optimizer starts fresh")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=640)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=2714)
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save-steps", type=int, default=0,
                        help="Also save immutable intermediate adapters at this step interval")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop and save after this many optimizer steps; zero means full epochs")
    parser.add_argument("--encoding", choices=("upstream", "contiguous"), default="upstream",
                        help="Diagnostic answer-boundary encoding; upstream remains the reference")
    args = parser.parse_args()
    if args.max_steps < 0 or args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs/batch-size must be positive and max-steps nonnegative")
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log = (out_dir / "training.log").open("a", buffering=1)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def emit(message):
        line = f"[{time.perf_counter()-started:8.1f}s] {message}"
        print(line, flush=True); log.write(line + "\n")

    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from needle.model import tokenizer
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.finetune import fit_max_len, load_jsonl, render_example, lora_target_paths, init_lora, merge_lora
    from needle.model.quantize import configure_deploy, cq_ste_params, cq_ste_mixed_params, parse_bits_map

    tokenizer.TOKENIZER_DIR = str(ROOT / "base/tokenizer")
    tokenizer.TOKENIZER_PREFIX = str(ROOT / "base/tokenizer/tokenizer")
    params, config = load_checkpoint(args.checkpoint)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params))
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(f"Expected bounded local GPU training, got {backend}")
    tok = tokenizer.get_tokenizer(config.vocab_size)
    max_len = fit_max_len(args.data, tok, args.max_len)
    if args.encoding == "upstream":
        seqs, masks = load_jsonl(args.data, tok, max_len)
    else:
        seqs, masks = [], []
        with pathlib.Path(args.data).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "query" not in row:
                    continue
                prompt, target = render_example(row)
                prompt_ids = tok.encode(prompt)
                combined = tok.encode(prompt + target)
                if combined[:len(prompt_ids)] != prompt_ids:
                    raise ValueError("Contiguous encoding changed a prompt token")
                ids = [tokenizer.BOS_ID] + combined + [tokenizer.EOS_ID]
                start = 1 + len(prompt_ids)
                mask = [0.0] * start + [1.0] * (len(ids) - start)
                if len(ids) > max_len:
                    raise ValueError(f"Contiguous example would be truncated: {len(ids)} > {max_len}")
                pad = max_len - len(ids)
                seqs.append(ids + [tokenizer.PAD_ID] * pad)
                masks.append(mask + [0.0] * pad)
        seqs, masks = np.asarray(seqs, np.int32), np.asarray(masks, np.float32)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(seqs)); seqs, masks = seqs[order], masks[order]
    n_val = int(len(seqs) * args.val_split)
    val_seqs, val_masks = seqs[:n_val], masks[:n_val]
    seqs, masks = seqs[n_val:], masks[n_val:]
    emit(f"backend={backend} train={len(seqs)} val={n_val} seq_len={max_len} batch={args.batch_size}")
    model = SimpleAttentionNetwork(config)
    bits_map = getattr(config, "weight_bits", "") or None
    bits = None if bits_map else 4
    parsed = parse_bits_map(bits_map) if bits_map else None
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    paths = lora_target_paths(params)
    scale = args.alpha / args.rank
    lora = init_lora(params, paths, args.rank, jax.random.PRNGKey(args.seed))
    if args.adapter:
        with pathlib.Path(args.adapter).open("rb") as handle:
            previous = pickle.load(handle)
        if previous.get("rank") != args.rank or previous.get("qat_bits_map") != bits_map:
            raise ValueError("Adapter rank or quantization scheme does not match this run")
        if float(previous["scale"]) != scale:
            raise ValueError("Adapter alpha/rank scaling does not match this run")
        lora = {tuple(key.split("/")): {"A": jnp.asarray(value["A"]), "B": jnp.asarray(value["B"])}
                for key, value in previous["lora"].items()}
        emit(f"Resuming adapter {args.adapter}; fresh optimizer and schedule")
    batch = args.batch_size
    steps_per_epoch = (len(seqs) + batch - 1) // batch
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = min(max(1, total_steps // 20), total_steps - 1)
    schedule = optax.warmup_cosine_decay_schedule(init_value=0, peak_value=args.lr,
                                                warmup_steps=warmup, decay_steps=total_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(schedule))
    state = optimizer.init(lora)

    def loss_fn(adapter, ids, mask):
        merged = merge_lora(params, adapter, scale)
        if parsed:
            merged = cq_ste_mixed_params(merged, *parsed)
        else:
            merged = cq_ste_params(merged, bits)
        logits = model.apply({"params": merged}, ids, quant=True)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:])
        target_mask = mask[:, 1:]
        return (ce * target_mask).sum() / jnp.maximum(target_mask.sum(), 1)

    @jax.jit
    def train_step(adapter, opt_state, ids, mask):
        loss, grads = jax.value_and_grad(loss_fn)(adapter, ids, mask)
        updates, opt_state = optimizer.update(grads, opt_state, adapter)
        return optax.apply_updates(adapter, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)
    history, step, best, stale = [], 0, float("inf"), 0
    metadata = {"args": vars(args), "backend": backend, "device": str(jax.devices()[0]),
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "qat_bits": bits, "qat_bits_map": bits_map,
                "model_config": str(config), "train_count": len(seqs), "val_count": n_val,
                "data_sha256": hashlib.sha256(pathlib.Path(args.data).read_bytes()).hexdigest(),
                "parent_adapter_sha256": hashlib.sha256(pathlib.Path(args.adapter).read_bytes()).hexdigest() if args.adapter else None}
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    emit(f"rank={args.rank} alpha={args.alpha} groups={len(paths)} QAT={bits_map or bits}; total_steps={total_steps}; compiling")

    def save(path, epoch, validation):
        data = {"lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])}
                         for p, v in lora.items()}, "scale": float(scale), "base": args.checkpoint,
                "rank": args.rank, "qat_bits": bits, "qat_bits_map": bits_map,
                "omni_epoch": epoch, "omni_validation_loss": validation, "omni_seed": args.seed,
                "omni_step": step, "omni_data_sha256": metadata["data_sha256"],
                "omni_encoding": args.encoding,
                "omni_parent_adapter_sha256": metadata["parent_adapter_sha256"]}
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle: pickle.dump(data, handle)
        tmp.replace(path)

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(seqs)); losses = []
        for start in range(0, len(seqs), batch):
            idx = order[start:start + batch]
            lora, state, loss = train_step(lora, state, jnp.asarray(seqs[idx]), jnp.asarray(masks[idx]))
            scalar = float(loss); losses.append(scalar); step += 1
            if not np.isfinite(scalar):
                raise RuntimeError("non-finite training loss; stopping without replacing best checkpoint")
            if step % 100 == 0 or step == 1:
                emit(f"step={step}/{total_steps} epoch={epoch} loss={scalar:.4f}")
            if args.save_steps and step % args.save_steps == 0:
                save(out_dir / f"step-{step:05d}.pkl", epoch, None)
            if args.max_steps and step >= args.max_steps:
                save(out_dir / "last.pkl", epoch, None)
                emit("Step limit reached; saved final adapter and retained intermediate checkpoints.")
                return
            if stopping or (out_dir / "STOP").exists():
                save(out_dir / "interrupted.pkl", epoch, None)
                emit("Stop requested; saved current adapter and retained completed checkpoints.")
                return
            if time.perf_counter() - started > args.max_minutes * 60:
                save(out_dir / "timeout.pkl", epoch, None)
                emit("Time limit reached; saved current adapter and retained best checkpoint.")
                return
        values = [float(eval_step(lora, jnp.asarray(val_seqs[i:i+batch]), jnp.asarray(val_masks[i:i+batch])))
                  for i in range(0, n_val, batch)]
        validation = float(np.mean(values)); train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "step": step, "train_loss": train_loss,
                        "validation_loss": validation, "elapsed_s": time.perf_counter()-started})
        save(out_dir / f"epoch-{epoch:02d}.pkl", epoch, validation)
        save(out_dir / "last.pkl", epoch, validation)
        if validation < best:
            best, stale = validation, 0
            save(out_dir / "best.pkl", epoch, validation)
        else:
            stale += 1
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        emit(f"epoch={epoch}/{args.epochs} train={train_loss:.4f} val={validation:.4f} best={best:.4f} stale={stale}")
        if stale >= args.patience:
            emit("Validation stopped improving; retaining best checkpoint."); break
    emit(f"Finished. Best adapter: {out_dir / 'best.pkl'}")


if __name__ == "__main__":
    main()
