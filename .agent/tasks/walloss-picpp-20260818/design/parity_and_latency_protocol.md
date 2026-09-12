# Parity, latency, resource, and evaluation protocol

## 1. Evidence lanes

The following are separate claims and are never collapsed into one result:

1. source/export build;
2. engine load and shape/finite smoke;
3. stage/KV/offline action parity;
4. standalone engine and complete `picpp latency`;
5. persistent server policy, serialization, transport, and client RTT;
6. small closed-loop smoke;
7. Gate40;
8. final400.

Only the final closed-loop gate can establish deployment-quality policy
success. A successful export, finite output, low action error, or fast server
request is not closed-loop success.

## 2. Comparison groups

| ID | Definition | Purpose |
|---|---|---|
| R0-Mac | pinned CPU/MPS-compatible reference where available | format and deterministic oracle only |
| R0-AGX | pinned upstream PyTorch BF16 on agx-test | same-device model baseline |
| C0 | three-stage TensorRT BF16 persistent runner | production candidate |
| C1 | three-stage TensorRT FP16 persistent runner | explicit fallback candidate only |

PI0.5 and other model latency rows may provide product context but are not a
speedup denominator because their architecture and external contract differ.

## 3. Fixture separation

Each fixture records:

- dataset/split/source ID without embedding private raw data in Git;
- exact two image hashes and preprocessing output hashes;
- prompt bytes and hash;
- raw and normalized state hash;
- state-bin string and token tensor hash;
- `x0` hash and seed provenance;
- flow-time/delta hash;
- reference package/source identity.

Splits are disjoint:

- calibration/export-debug;
- held-out offline parity;
- latency fixtures;
- closed-loop episode manifest.

Synthetic zero/random cases are edge-case diagnostics, not final correctness
or latency evidence.

## 4. Stage parity ladder

Run in this order:

1. original internal-random path versus external-`x0` PyTorch seam;
2. upstream dynamic tokenization versus fixed left-padded-768 wrapper;
3. processor image, token, mask, M-RoPE, token-type, and schedule assets;
4. PyTorch `prefix_embed` wrapper versus original model;
5. PyTorch `prefill` wrapper: velocity, `x1`, and all 72 KV tensors;
6. PyTorch single postfix step;
7. PyTorch full nine-step postfix loop and final normalized action;
8. ONNX stage parity;
9. TensorRT stage parity on AGX;
10. final unnormalized `[32,26]` action parity.

For every compared tensor record shape, dtype, finite count, max absolute
error, RMS error, relative L2, and cosine similarity. For KV, record every
layer separately. For the flow loop, record `x1` through `x10` so accumulated
drift is visible.

## 5. Provisional numerical gates

These thresholds were approved in design revision 1 and are inherited unchanged
by revision 2. They are intentionally expressed on normalized action/state
space where units are comparable.

| Boundary | max_abs | RMS | relative L2 | cosine |
|---|---:|---:|---:|---:|
| external-`x0` seam, same PyTorch dtype | `1e-6` | `1e-7` | `1e-6` | `>=0.999999` |
| processor/token/mask IDs | exact | exact | exact | exact |
| BF16/FP16 embeddings and each KV tensor | `<=0.10` | `<=0.02` | `<=0.02` | `>=0.999` |
| each normalized flow state `x1..x10` | `<=0.15` | `<=0.03` | `<=0.03` | `>=0.998` |
| final normalized action chunk | `<=0.20` | `<=0.05` | `<=0.05` | `>=0.995` |

All shapes must be exact and every value finite. Passing final action while a
stage, KV layer, or drift trend fails is not acceptance. Threshold changes
after observing candidate results require a contract revision and cannot be
used to retroactively pass a candidate.

## 6. Engine and cache smoke

Before timing:

- validate source/asset/engine/plugin hashes;
- inspect actual engine tensors, dtypes, fixed dimensions, tactics, and
  precision;
- run one real held-out fixture uncaptured;
- check `[32,26]`, finite output, and all 36 cache layers;
- verify cache addresses remain fixed and values unchanged across all nine
  postfix steps;
- compare captured and uncaptured outputs exactly or within the same
  candidate tolerance;
- prove no hidden backend or CUDA Graph fallback occurred.

## 7. Standalone AGX latency

Preserve the existing formal `picpp latency` boundary: 10 warmups and 10
measured complete wrapper calls, written to the candidate model directory.
Add a detailed benchmark; it does not replace the formal command:

- at least 20 warmups;
- five interleaved rounds per candidate;
- at least 100 complete-policy samples per candidate;
- identical fixed real latency fixture and `x0` sequence;
- CUDA events for device stages and host monotonic clock for wrapper work;
- synchronized final output;
- mean, median, p95, standard deviation, min, and max;
- first-load/cold-start reported separately from warm inference.

Report separately:

```text
image resize/processor
state normalize/bin/tokenize
H2D
prefix_embed
prefill
postfix_step_1 ... postfix_step_9
postfix_loop_total
D2H
action unnormalize
C++ infer total
Python wrapper total
```

Do not hide image/tokenization, H2D/D2H, synchronization, action
unnormalization, or failed/cold requests outside the declared complete-policy
number.

## 8. Power, thermal, memory, and profiler evidence

Before every AGX benchmark refresh process ownership and storage. Record:

- JetPack/L4T, CUDA, TensorRT, engine ABI, power mode, GPU/EMC clocks;
- temperature before/after, throttling, power, GR3D and EMC from tegrastats;
- resident engine/package size, unified-memory use, peak available RAM, and
  task NVMe usage;
- kernel launch counts and tactic/layer profile when tools permit.

Nsight Compute and Nsight Systems were not in PATH during Phase 0. Unless a
fresh check finds them, the report must state that occupancy, register spill,
warp stalls, and hardware DRAM traffic were not measured. Tegrastats is
system-level support evidence, not a substitute for kernel profiler counters.

## 9. Server/client protocol

After separate approval, start only a new candidate service on an approved
unused port. Measure:

- load/startup and health;
- warm repeated requests with one and multiple queued clients;
- queue, preprocessing, policy, serialization, transport, and client RTT;
- exact same-request repeat with fixed `x0`;
- reset and post-reset determinism;
- memory after warmup and after repeated requests;
- graceful termination of only the task-owned process.

Server/client timing is not substituted for standalone `picpp latency`.

## 10. Closed-loop ladder

After offline, engine, latency, and server gates pass:

```text
small smoke -> Gate40 -> final400
```

Each stage requires a separately frozen episode/task/seed/initial-state/prompt/
camera/noise/step manifest. Missing episodes count as failures unless a
preapproved partial protocol says otherwise. The final non-inferiority margin
must be approved against the same-model Wall-OSS reference; it cannot be
borrowed from PI0.5.

## 11. Promotion decision

Promotion requires:

- every blocking mechanical/parity/cache gate passes;
- TensorRT complete-policy latency is materially better than same-contract
  AGX PyTorch R0, with p95 not worse;
- server is stable with no fallback or memory growth;
- final closed-loop result satisfies the approved non-inferiority boundary;
- maintenance cost and plugin surface are accepted.

Until R0-AGX exists, there is no valid Wall-OSS speedup percentage.
