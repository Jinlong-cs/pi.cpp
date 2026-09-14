# Revision 6 strictly bounded mixed-EP diagnostic delta

Status: design and exact placement reviews approved; the authorized
single-fixture mixed-EP diagnostic ran and failed the prefix/KV numerical gate.
Revision 6 is stopped and does not unlock AGX work.

Revision 6 does not change the frozen Wall-OSS model, v18 ONNX files,
checkpoint, preprocessing, tensor names, static shapes, BF16 precision, cache
layout, external `x0`, request-constant `v_padding`, D10 schedule,
normalization, or output `[32,26]`. It introduces a diagnostic-only ONNX
Runtime lane after Revision 5 proved that strict CUDA session creation is not
possible for two stages with ORT 1.23.2.

## Revision 5 evidence inherited unchanged

With `SimplifiedLayerNormFusion` disabled, session-only placement reported:

| Stage | CUDA nodes | CPU nodes | Memcpy nodes observed |
|---|---:|---:|---:|
| `prefix_embed` | 1,815 | 8 | 4 |
| `prefill` | 13,613 | 3,481 | 4 |
| `postfix_step` | 4,184 | 0 | 0 expected |

The eight prefix CPU nodes are exactly:

```text
Pad /visual/Pad
Gather /Gather
Unsqueeze /Unsqueeze_1
Equal /Equal_4
Where /Where_2
Concat /Concat_1
Gather /Gather_3
Unsqueeze /Unsqueeze_7
```

Prefill CPU placement repeats shape/index/control families across 36 layers,
including Gather, Equal, Where, Slice, Concat, Unsqueeze, ScatterElements and
Squeeze. Revision 6 does not approve this set by count or operator family
alone; every node name, operator type and tensor dtype must be materialized in
an exact manifest and reviewed before inference.

## Remote and artifact boundary

All Revision 6 writes must stay under:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/
```

The frozen Revision 5 v18 package may be mounted read-only as input. It must
not be copied over, edited, re-saved, optimized in place, or used as an output
directory. No AGX host, TensorRT builder, latency harness, service, client,
closed-loop evaluator, baseline or accepted package is in scope.

## Two-approval execution topology

### Gate A: design and placement extraction

After `revision6-design-review`, the Agent may:

1. re-audit 5090 GPU/process/disk state;
2. revalidate the v18 manifest and environment hashes;
3. create sessions only, with inference disabled;
4. disable only `SimplifiedLayerNormFusion`;
5. capture verbose provider placement and generated Memcpy boundaries;
6. emit an exact sorted placement manifest containing stage, optimized node
   name, operator type, provider, input/output dtype classes and a SHA256;
7. stop for human `revision6-placement-review`.

### Gate B: mixed-EP numerical diagnostic

Inference may run only after the exact manifest is approved. Sessions use the
ordered providers `CUDAExecutionProvider, CPUExecutionProvider`. No other EP,
provider option, optimizer disablement or graph change is allowed.

## CPU allowlist and CUDA denylist

The placement manifest passes only if all conditions hold:

1. Counts remain exactly `8/3481/0` CPU nodes for
   prefix/prefill/postfix; any drift stops.
2. Every CPU node is explicitly named in the reviewed manifest. Operator
   family wildcards are insufficient.
3. CPU nodes may consume and produce only BOOL/INT32/INT64 shape, index, mask
   or control tensors. Any FP32/FP16/BF16 activation, weight, cache, embedding,
   action or reduction tensor on CPU stops the lane.
4. These operators are forbidden on CPU regardless of name:
   `MatMul`, `Gemm`, `Conv`, `Attention`, `MultiHeadAttention`, `Softmax`,
   `LayerNormalization`, `SimplifiedLayerNormalization`,
   `SkipSimplifiedLayerNormalization`, `RMSNormalization`, `BatchNormalization`,
   `GroupNormalization`, `InstanceNormalization`, `ReduceMean`, `ReduceSum`,
   `Pow`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos` and all quantized compute ops.
5. All 72 KV tensors, embedding tensors, `x0`, `x1..x10`, action projection,
   visual patch projection and decoder numerical paths must stay on CUDA.
6. Memcpy boundaries must match the reviewed manifest; new or missing copies
   stop the lane.
7. Provider profile must contain only the two approved EP names and match the
   pre-inference placement hash.

## Numerical diagnostic

The inherited thresholds are unchanged and cannot be relaxed:

| Boundary | max_abs | RMS | relative L2 | cosine |
|---|---:|---:|---:|---:|
| embedding and each KV | `0.10` | `0.02` | `0.02` | `>=0.999` |
| normalized `x1..x10` | `0.15` | `0.03` | `0.03` | `>=0.998` |
| final normalized action | `0.20` | `0.05` | `0.05` | `>=0.995` |

The diagnostic must run prefix output parity, prefill `x1 + 72 KV`, final-step
teacher-forced postfix, recursive D10, final normalized `x10`, finite/shape
checks, nonzero KV checks and KV input byte-hash immutability.

The existing fixture has no PyTorch `x2..x9` and no held-out sample. Therefore
even a numerical pass is labelled `single-fixture mixed-EP mechanical parity`.
It cannot pass `export-parity`, unlock `agx-build-approval`, justify latency,
or support deployment acceptance. A later fixture/export revision must add
held-out inputs and per-step PyTorch references before those gates can move.

## Stop conditions

Stop without inference if placement count, exact node set, dtype class,
Memcpy boundary, optimizer set, artifact hash, GPU ownership or disk state
differs from the approved manifest. Stop during inference on any unexpected
provider, CPU floating tensor, NaN/Inf, shape/name mismatch, zero KV layer,
KV mutation or numerical threshold failure. Do not respond to failure by
enabling more fallback, changing precision, editing v18, rebuilding ONNX,
running AGX TensorRT, or weakening thresholds.

## Claim boundary

This route answers only: “Does the frozen v18 graph preserve PyTorch numerical
behavior when ORT executes its registered shape/control subgraphs on CPU and
the numerical model path on CUDA?” It does not answer whether the graph is
pure CUDA, TensorRT-buildable, fast on 5090, deployable on AGX, service-ready,
or closed-loop correct.
