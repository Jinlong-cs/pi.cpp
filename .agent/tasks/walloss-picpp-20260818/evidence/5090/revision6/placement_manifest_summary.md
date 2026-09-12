# Revision 6 session-only placement manifest

Verified: 2026-08-20 CST. Scope is limited to ONNX Runtime session creation and
placement inspection on `5090-zem`; no model inference, latency, AGX build,
server/client, or closed-loop action was run.

Remote artifact root:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/
```

## Resource and v18 identity audit

Read-only live check at `2026-08-20T17:18:41+08:00`:

- host: `zem-pc` (`5090-zem`)
- GPU: `NVIDIA GeForce RTX 5090 D v2`, driver `580.173.02`
- GPU state: 15 MiB used, 23959 MiB free, 0% utilization
- compute processes: none reported by `nvidia-smi`
- project filesystem: 103 GiB available
- Revision 5 transfer completion marker: `2026-08-20T14:36:36+08:00`
- frozen v18 source-manifest SHA256:
  `5eec1d2dcf062d0899e7e1ea1d893aac37ee6ced1e9a96a1f60a6854cf296d06`

The Revision 5 v18 package remained a read-only input. No service or process
was stopped, and no baseline or accepted artifact was modified.

## Session contract

- ORT: `1.23.2`
- Provider order: `CUDAExecutionProvider, CPUExecutionProvider`
- Disabled optimizer: `SimplifiedLayerNormFusion` only
- `inference_called`: `false`
- Sessions created: `prefix_embed`, `prefill`, `postfix_step`
- Available providers also included TensorRT, but TensorRT was not selected.

## Placement counts

| Stage | CUDA nodes | CPU nodes | Count check |
|---|---:|---:|---|
| `prefix_embed` | 1815 | 8 | pass |
| `prefill` | 13613 | 3481 | pass |
| `postfix_step` | 4184 | 0 | pass |

The exact manifest parser reported `allowlist_pass=true`,
`count_match=true`, `cpu_control_only=true`, and 8 observed Memcpy boundaries.

## CPU allowlist evidence

`prefix_embed` CPU nodes are limited to `Concat(1), Equal(1), Gather(2),
Pad(1), Unsqueeze(2), Where(1)`. Observed CPU tensor classes are BOOL (2)
and INT64 (8).

`prefill` CPU nodes are limited to `Concat(725), Equal(399), Gather(906),
ScatterElements(109), Slice(617), Squeeze(109), Unsqueeze(217), Where(399)`.
Observed CPU tensor classes are BOOL (798) and INT64 (3481).

`postfix_step` has no CPU nodes. No UNKNOWN, floating activation, or denied
numerical CPU operator was found in the parsed manifest.

## Memcpy boundaries

The 8 observed boundaries are:

- `prefix_embed`: one `MemcpyFromHost` for `/visual/Pad_output_0`; three
  `MemcpyToHost` outputs (`/visual/CumSum_output_0`,
  `/visual/Where_1_output_0`, `/visual/Where_output_0`).
- `prefill`: four `MemcpyToHost` outputs under
  `/model/layers.0/Unsqueeze_{12,13,4,5}_output_0`.

## Immutable artifact hashes

```text
placement_manifest.json:
0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2
cpu_allowlist.json:
474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341
placement_session_only.log:
8763bb2482a663fd7250b109294e041ce251e8b71260a4ec165e5361252036a2
placement_session_only_summary.json:
74f04501a5f1879eb2ef36e0a147814e8466c2e9dafb2b7d3cd88d287639e3a6
```

This evidence completes the machine placement gate only. It does not approve
the manifest for inference and does not establish numerical parity, export
parity, TensorRT buildability, latency, or deployment readiness.
