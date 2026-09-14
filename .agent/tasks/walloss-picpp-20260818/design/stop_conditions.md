# Stop conditions, escalation, and rollback

## 1. Phase 1 boundary

After this packet is statically validated, stop at the human `design-review`
gate. Before approval, do not:

- download the 8.33 GB model weight;
- clone Wall-X or create an implementation worktree;
- install a Mac or AGX model environment;
- create an AGX artifact directory;
- export ONNX or build TensorRT engines/plugins;
- allocate AGX GPU memory;
- start or restart a service;
- run server/client or closed-loop evaluation.

## 2. Identity and preprocessing stops

Stop if:

- source/checkpoint/tokenizer/normalizer identity differs from the pinned
  revision or a local hash conflicts with Hub metadata;
- checkpoint loading has unreviewed missing/unexpected keys;
- official camera order, resolution, state/action order, horizon, or D10 differs;
- the normalizer key is selected by fallback rather than manifest;
- external `x0` cannot reproduce the original PyTorch path;
- explicit left-padding to 768 changes meaningful tokens, M-RoPE, masks, KV,
  or action output outside the seam tolerance;
- a real approved prompt/state overflows 768 or truncates meaningful content;
- processor capture disagrees with the frozen tensor contract.

An interface mismatch requires `contract_revision + 1`; it is not patched
silently during exporter work.

## 3. Export and engine stops

Stop the affected candidate if:

- ONNX needs Python control flow, DynamicCache objects, `torchdiffeq`, or
  internal randomness at runtime;
- an exported stage cannot pass its PyTorch wrapper parity;
- postfix export retains VLM expert-0 weights or mixed-token routing;
- TensorRT uses an unsupported operation, silent precision downgrade, or
  hidden PyTorch/CUDA-extension fallback;
- a tensor dtype is unknown or any runtime dimension is `-1`;
- a plugin cannot serialize/version/deserialize deterministically;
- any one of 36 K/V layers is missing, wrong-shaped, non-finite, unexpectedly
  zero, overwritten, or numerically outside tolerance;
- captured CUDA Graph output differs or capture falls back to naive execution;
- rootfs, `/tmp`, or `/dev/shm` would receive task artifacts.

## 4. Resource stops

Perform a fresh read-only audit before every remote phase. Stop without
cleaning or killing anything if:

- another GPU/process owner appears;
- the exact NVMe revision lacks the approved headroom;
- rootfs pressure worsens enough to threaten the OS;
- available unified memory falls below an approved safety margin or any OOM,
  swap storm, thermal throttle, or repeated allocation failure occurs;
- the task would need to stop/restart another service or reuse an occupied
  port;
- a dependency would be installed into system Python/rootfs rather than the
  task-owned NVMe environment.

## 5. Correctness and latency stops

Reject the candidate if:

- any output is non-finite or not exactly `[32,26]`;
- any stage, KV layer, or `x1..x10` drift gate fails;
- only a synthetic fixture passes while real held-out fixtures fail;
- a faster result changes cameras, resolution, state/action dimensions,
  prompt, padding semantics, flow steps, x0, timing boundary, or output adapter;
- C0/C1 is not materially faster than same-contract R0-AGX at complete-policy
  level, or p95 is worse beyond run noise;
- a kernel/engine win disappears after preprocessing, transfers, wrapper, or
  server overhead;
- the plugin and maintenance surface is disproportionate to the measured
  deployment benefit.

A result may still be recorded as export research or a mechanical prototype,
but it is not promoted as a persistent production backend.

## 6. Server and closed-loop stops

Stop and reject/escalate if:

- server errors are converted into nominal response dictionaries;
- backend, precision, package hash, or camera mapping changes between requests;
- fixed-noise repeated requests drift;
- memory grows monotonically after warmup;
- reset does not restore declared RNG/state semantics;
- client RTT is reported as policy latency or vice versa;
- closed-loop wiring changes task, prompt, observation, seed, initial state,
  action extraction, or episode accounting;
- smoke, Gate40, or final400 fails its preregistered threshold;
- physical robot commands would be required without a new safety contract and
  explicit human authorization.

## 7. Escalation decisions

Human approval or contract revision is mandatory for:

- changing any fixed model/tensor/API contract;
- choosing FP16 after BF16 failure;
- adding a custom TensorRT plugin;
- increasing storage/memory/time budget;
- creating the AGX revision and staging the checkpoint;
- launching a persistent candidate service;
- each closed-loop rung;
- final promotion or baseline/default-model change.

## 8. Rollback

Rollback is selection-based and recoverable:

1. stop only a task-owned Wall-OSS process if it was approved and launched;
2. select the prior pi.cpp binary/model package/service configuration;
3. retain candidate logs and hashes in its fresh revision directory;
4. do not delete or overwrite accepted assets;
5. record failure cause and the last passed gate.

The current accepted backends remain unchanged throughout this task.
