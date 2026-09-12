# Wall-OSS Revision 7: reference recovery and AGX engine feasibility

Revision 7 is a new candidate lane. It preserves all Revision 5/6 artifacts
and failure conclusions; those lanes are diagnostic history and are not used as
the production golden. The primary golden for this revision is a pinned
PyTorch BF16 reference fixture. ONNX Runtime remains a secondary diagnostic
tool only.

## Objective

Recover a self-consistent calibration and held-out PyTorch reference, isolate
the v18 prefix/prefill numerical boundary with teacher-forced inputs, freeze
one candidate (v18 or a minimally repaired v19), and prove or reject an
SM87/TensorRT 10.3 engine package on `agx-test`.

Passing Revision 7 means only that one candidate has engine-level feasibility
for a later pi.cpp runtime integration. It does not claim latency, server/client
stability, closed-loop success, or promotion.

## Frozen ABI

The two 448x448 camera contract, BF16 model, 768/736 sequence split, 36-layer
72-tensor KV bridge, external deterministic `x0`, request-constant `v_padding`,
ten flow intervals ending at 0.999, and `[32,26]` action output are unchanged.
No threshold, shape, prompt, normalization, schedule, or cache-layout change
is permitted in this revision.

## Execution order

1. Validate fixture lineage and BF16 bit-preserving round trips.
2. Run teacher-forced v18 prefill directly from saved PyTorch prefill inputs.
3. Build the same-host PyTorch original/export-patched/reference ladder and the
   PT/ORT 2x2 prefix-prefill matrix.
4. Locate the first numerical drift boundary before deciding whether v18 is
   retained, v19 is minimally repaired, or the candidate is rejected.
5. Add disjoint held-out fixtures and PyTorch `x1..x10` references.
6. Freeze one candidate and its complete hash tree.
7. After a separate AGX approval, ingest and build target TensorRT engines.
8. Compare TensorRT directly with the pinned PyTorch golden, including all KV
   layers and recursive D10.

## Hard stops

Do not create v19 without a located exporter defect. Do not enter AGX build
without held-out coverage and per-step references. Do not continue to runtime,
latency, server/client, or closed-loop work after a blocking parity failure.

