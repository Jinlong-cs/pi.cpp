# Revision 5 dual-host parity and engine-feasibility delta

Status: pending human `revision5-design-review` and
`5090-ort-cuda-approval`.

Revision 5 does not change the approved Wall-OSS model, preprocessing, tensor
ABI, three-stage split, BF16 reference precision, cache layout, external
initial-noise semantics, request-constant `v_padding`, D10 schedule,
normalization, or output shape. It changes only the evidence and execution
topology needed to resolve the revision 4 ONNX Runtime provider blocker.

## Current evidence and blocker

The valid v18 package remains read-only under:

```text
/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/onnx_real_legacy_v18/
```

All three ONNX stages passed path-based ONNX checker and static IO checks.
The real PyTorch seams, fixed vision decomposition, two-expert `x1 + 72 KV`
decomposition, and 36-layer nonzero-cache gates passed. AGX ONNX Runtime
1.23.2 exposes only Azure and CPU providers; CPU session creation cannot
execute BF16 `MatMul(13)` or BF16 `Expand(13)`. This is a provider blocker,
not observed numerical mismatch, and `export-parity` remains pending.

## Revision 5 execution topology

1. Revalidate both hosts read-only.
2. Preserve and manifest the v18 ONNX, 434 external-data files, frozen real
   fixture, and source/checkpoint/export lineage.
3. After `5090-ort-cuda-approval`, stage the package only under:

   ```text
   /home/zem/wujinlong/picpp_walloss_20260818/revision5_ort_cuda/
   ```

4. Require actual strict CUDA EP execution on 5090, with CPU fallback disabled
   and node placement/profile evidence. CUDA EP visibility alone is not a
   pass.
5. Run prefix, prefill, teacher-forced postfix, recursive D10, final action,
   and all 72 KV comparisons using the inherited thresholds.
6. Create v19 only after a confirmed export-graph defect and a separate human
   `5090-v19-export-approval`; provider or fixture failures do not justify
   re-export.
7. Only after the 5090 stop gate and independent `agx-build-approval`, stage
   the unique parity-approved package under:

   ```text
   /root/wujinlong/nvme/picpp_walloss_20260818/revision5_engine/
   ```

8. On AGX, revalidate external-data lineage, run TensorRT parser preflight,
   build stages sequentially, and run engine-level stage/KV/D10 correctness.
9. Stop after engine correctness. Runtime integration, formal latency,
   server/client, closed loop, promotion, and baseline selection remain later
   independent gates.

## Work-package correction

Revision 4 coupled engine construction to the future C++ and Python runtime.
Revision 5 splits the dependency graph:

- `wp-02b-5090-ort-cuda-parity` depends on the frozen export package;
- `wp-05a-agx-engine-feasibility` depends on successful 5090 parity and the
  AGX human approval;
- the existing runtime/latency work package depends on engine feasibility plus
  C++ and Python integration.

This permits target engine feasibility without silently bypassing the task
contract.

## Inherited numerical gates

All names, counts, order, shapes, dtypes and control tensors are exact; every
floating value must be finite.

| Boundary | max_abs | RMS | relative L2 | cosine |
|---|---:|---:|---:|---:|
| external `x0`, same PyTorch dtype | `1e-6` | `1e-7` | `1e-6` | `>=0.999999` |
| BF16/FP16 embedding and each KV | `<=0.10` | `<=0.02` | `<=0.02` | `>=0.999` |
| normalized `x1..x10` | `<=0.15` | `<=0.03` | `<=0.03` | `>=0.998` |
| final normalized action | `<=0.20` | `<=0.05` | `<=0.05` | `>=0.995` |

Thresholds cannot be changed after observing v18 or v19 output. Zero-norm
relative/cosine metrics are explicitly undefined and cannot automatically
pass. Unnormalized `[32,26]` action is reported separately and does not borrow
the normalized threshold.

## Safety and claim boundaries

- Preserve v7-v18, baseline, accepted, HF, verify, and all other task assets.
- Do not stop, signal, restart, or interfere with another process or service.
- All 5090 and AGX environment, cache, temporary, log, builder, timing-cache,
  and artifact files remain inside their revision directories.
- Do not use AGX rootfs, `/tmp`, or `/dev/shm` for task data.
- Preserve external-data relative paths and require a complete per-file
  manifest plus atomic completion marker before consumption.
- Do not copy a 5090 engine, timing cache, plugin, tactic, or latency result to
  AGX as deployment evidence.
- ONNX checker is not ORT numerical parity. 5090 ORT parity is not AGX
  TensorRT parity. Engine correctness is not picpp runtime, latency, service,
  closed-loop, or deployment acceptance.

No remote host may be contacted until the two pending human gates named at the
top are approved.
