# Revision 5 ORT CUDA decision

Verified on `5090-zem-lan` (`zem-pc`, RTX 5090 D v2) on 2026-08-20 CST.

## Identity and environment

- v18 transfer manifest: 462 entries, verified before consumption.
- ORT: 1.23.2 in the isolated revision directory.
- CUDA capability preflight: BF16 `MatMul(13)` and BF16 `Expand(13)` executed on `CUDAExecutionProvider` with CPU fallback disabled.
- Full-stage harness SHA256: `243b9163ae34285c21e2820302b0d527d979db6f7a437947d2d8ce688349ff4e`.
- Postfix-only harness SHA256: `5601dba1aba6d63aa5d97c1858c063a95bf4f97d20fc9492fa020971c8ec776e`.

## Full three-stage provider gate

The first full-stage session attempt failed because ORT fused a BF16 visual
normalization into `SimplifiedLayerNormalization`, whose CUDA kernel did not
accept the fused BF16 type combination. Disabling only
`SimplifiedLayerNormFusion` removed that optimizer-generated failure, but the
strict provider gate still rejected two stages because ORT assigned nodes to
the default CPU EP:

| Stage | CUDA nodes | CPU nodes | Strict CUDA session |
|---|---:|---:|---|
| `prefix_embed` | 1,815 | 8 | rejected |
| `prefill` | 13,613 | 3,481 | rejected |
| `postfix_step` | 4,184 | 0 | accepted |

The CPU assignments include ORT `fallback_cpu_capability` decisions for
shape/control operations such as Gather, Slice, Concat, Equal, Where,
Unsqueeze and ScatterElements. Because Revision 5 requires no unapproved CPU
fallback, mixed-EP prefix/prefill inference was not run and was not relabelled
as CUDA parity.

Raw remote evidence:

- `logs/provider_placement_session_only.log`, SHA256
  `b2a920d68107d68ec858f2105c0d3b3d732ede0ff264f5e479265c097c0f1bee`.
- `logs/ort_cuda_parity.log` for the original optimizer failure.

## Strict CUDA postfix evidence

The fully CUDA-assigned `postfix_step` was executed with the real PyTorch
fixture, 72 KV tensors and the frozen D10 schedule.

| Check | max abs | RMS | relative L2 | cosine | Result |
|---|---:|---:|---:|---:|---|
| final-step teacher-forced `x10` | 0.00299701 | 0.000708630 | 0.00151218 | 0.999998857 | pass |
| recursive `x1 + KV -> x10` | 0.01039924 | 0.000921380 | 0.00196617 | 0.999998067 | pass |

- Output shape and finite checks passed.
- All 72 KV input byte hashes were unchanged before and after both paths.
- Provider profile contains only `CUDAExecutionProvider` node events.
- The profile is correctness-only; host-bound BF16 outputs make its wall time
  unsuitable as latency evidence.
- `x2..x9` are finite, but the fixture has no PyTorch tensors for per-step
  numerical comparison. Only `x1` and final `x10` have PyTorch references.
- There is one real export fixture and no registered held-out fixture.

Result artifact:

- Remote `results/ort_cuda_postfix_teacher_d10_parity.json`, SHA256
  `c24d37027d369698407cb220ad7665eac4b2ee1e317f92e9a07fa5957e91b72f`.
- Remote provider profile SHA256
  `c98954e6ee94f6b39f635c5bfafbbd8fe2a5943778bf0d4dbe08ea502c2b216b`.

## Decision

`5090-ort-cuda-parity` fails as a blocking full-stage gate. The postfix stage
has strong single-fixture mechanical parity, but prefix and prefill cannot
satisfy the strict CUDA-only contract, and held-out plus per-step fixture
coverage is incomplete. Therefore `export-parity` stays pending and no AGX
TensorRT build, latency, server/client, closed-loop or promotion work is
authorized.

The next design decision must be human-approved: either allow a precisely
enumerated mixed CUDA/CPU diagnostic lane, or authorize a new graph/export
revision that removes the CPU-assigned control subgraphs and adds held-out
plus `x2..x9` PyTorch fixtures.
