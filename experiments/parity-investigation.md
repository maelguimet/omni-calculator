# Needle 2 training/native parity investigation

The final deployment selection is the original V2 epoch-3 checkpoint, packaged as `models/omni.cact`, SHA-256 `1852739bfbb36f945eed2bca5c3ddfd364fab276cf7914ccbe793df4746e46e4`. Parent validation reports 75/75 development CLI cases passing and retains the independent 100-case validation for this selection. Alternatives achieved higher synthetic native routing rates but showed composition, units, and other CLI regressions. The final V2 alignment continuation reached 74/75 CLI cases, so it did not replace the original checkpoint. All alternative artifacts remain preserved.

The V3 epoch-1 adapter has low teacher-forced validation loss, but native calls disagree with actual autoregressive JAX inference. This is a deployment discrepancy, and V3 remains an experimental candidate. The selected model is based on complete-application acceptance rather than training loss or synthetic native scores alone.

All probes use the frozen calculator schema and trimmed system string in this directory. No independent final-acceptance cases were read for training or this investigation. This four-case set is a deliberately small diagnostic for already observed failure categories, not an accuracy estimate.

| Request | JAX greedy call | Native call |
| --- | --- | --- |
| `1 + 1` | `evaluate()` | `function(multiply,[1,1])` |
| `what is 2 to the power of 10` | `function(power,[2,10])` | `function(add,[2,10])` |
| `natural logarithm of e^3` | `function(ln,[e^3])` | `function(lcm,[e^3])` |
| `square root of 144` | `function(sqrt,[144])` | `function(sqrt,[144])` |

`results-teacher-v3-epoch1-greedy.json` records the exact greedy sequences and teacher-forced probabilities. `results-teacher-v3-epoch1-actual-encoding.json` has zero incorrectly predicted target tokens across all four cases. `results-native-teacher-v3.json` records every native response, weights hash, schema hash, system hash, and native environment. Native confidence is high even on wrong calls; it is not a correctness guarantee. Rust checks reject these changed-operation calls before evaluating mathematics.

The official Python package version is `cactus-needle==2.0.12`; its own `needle/agent/fetch.py` dispatches generation 2 to native `2.0.4`. There is no alternate 2.0.12 native library bundled in that wheel. The pinned 2.0.4 runtime is the generation-2 runtime selected by upstream, so this difference is not evidence of an accidental package-version mismatch.

The following controlled variants retained the same 1/4 native result:

- Merge LoRA into a float32 base before native export: `results-native-teacher-v3-fp32.json`.
- Disable KV rebasing with `NEEDLE_NO_REBASE=1`: `results-native-teacher-v3-no-rebase.json`.
- Increase KV window to 2048 and keep 8-bit KV: `results-native-teacher-v3-full-kv.json`.

An initial teacher probe encoded the complete prompt and answer together. That probe is retained as `results-teacher-v3-epoch1.json`, but is not the upstream encoding and must not be quoted as teacher-forced accuracy. Upstream `_encode` instead concatenates separately tokenized prompt and answer. The answer-alone encoding adds leading-space token 8042 before the tool-call token; correcting the probe removes its first-token mismatch, and unconstrained autoregressive JAX rollout is also correct.

The native runtime applies a byte grammar during generation. The extra answer-boundary space is a plausible interaction to test. The bounded `training/runs/encoding-aligned/` continuation changes only training tokenization to encode prompt and answer together, retaining the same data, adapter architecture, schema, system string, and resource limits. It finished normally after 1,024 steps in 339 seconds. No native runtime or application assets are changed by this experiment.

At step 256 the native four-case score improves to 3/4; at step 512 it reaches 4/4. On all 768 frozen V2 diagnostic requests, step 512 gets 623 exact calls (81.1%), or 631 under narrow structural/commutative equivalence, compared with V2 epoch 3's 476 exact calls (62.0%). Its direct native median is 123.8 ms on two local P cores; this is a language-model diagnostic, not a VPS or complete-CLI timing. The corresponding artifact is `results-encoding-aligned-512-v2-heldout.json`. Literal requests reach 206/210 and off-topic rejection 111/111, while algebra and nested composition remain harder.

This does not establish complete native parity. Step 1,024 again produces `lcm` for the logarithm probe despite JAX greedy producing `ln`; JAX is 4/4 and native 3/4. Both exact trajectories are retained in `results-teacher-encoding-aligned-1024-greedy.json` and `results-native-encoding-aligned-1024.json`. The answer-boundary change has a strong measured benefit, but at least one residual inference difference remains. The final selection therefore used real native and application outcomes.

The application-level benchmark determined the deployment decision. Four successful probes alone did not justify replacing the broader-tested original V2 checkpoint. In the parent's development CLI comparison, V3 epoch 1 scored 70/75, V3-aligned step 512 scored 71/75, V3-aligned step 1,024 scored 70/75, and V2-aligned step 1,024 scored 74/75; original V2 epoch 3 scored 75/75. The observed composition regressions were preserved as failures rather than hidden by the stronger synthetic routing scores.

The final controlled continuation uses the original V2 epoch-3 adapter and its unchanged 10,240-row mixed training file, including reasoning targets. It completed 1,024 steps in 328 seconds; snapshots at 512 and 1,024 are retained under `training/runs/v2-encoding-aligned/`. The step-512 export improves the frozen 768-case diagnostic from 476 to 544 exact calls (549 structural). Step 1,024 reaches 555 exact calls (72.3%), with a local native median of 176.5 ms. Both retain weaknesses in units and nested composition. This is the final training experiment for this delivery; deployment selection is based on the complete CLI tests. No training, export, GPU verification, or native benchmark workers remain running. All sessions were reaped, and completed checkpoints plus failed-attempt evidence remain available.

`development-model-comparison.json` collects the exact artifact hashes and routing counts across retained candidates. These repeatedly inspected diagnostic cases remain separate from the application's independent final corpus, whose contents were not used by this training work.
