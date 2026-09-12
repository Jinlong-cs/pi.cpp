# Revision 6 mixed-EP diagnostic decision

Verified: 2026-08-20 CST.

## Authorized scope

The human owner approved one frozen v18 fixture for mixed
`CUDAExecutionProvider, CPUExecutionProvider` prefix, prefill, postfix, KV and
D10 parity under these immutable artifacts:

```text
placement_manifest SHA256:
0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2
cpu_allowlist SHA256:
474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341
```

AGX TensorRT build, latency, server/client and closed-loop remained excluded.

## Execution preconditions

- Live GPU audit showed no compute process and 23959 MiB free.
- v18 source-manifest SHA remained
  `5eec1d2dcf062d0899e7e1ea1d893aac37ee6ced1e9a96a1f60a6854cf296d06`.
- A fresh session-only recheck created all three sessions with
  `inference_called=false` and regenerated both approved manifest hashes
  exactly.
- Two earlier session-only attempts failed before session creation/inference:
  the first omitted the image cuDNN path; the second selected an incompatible
  CUDA compat library. Both failed directories are retained under the
  Revision 6 artifact root. The successful environment removed the compat
  override and used the image's driver-mounted NVIDIA libraries.

## Provider placement result

Runtime profiles exactly matched the approved CPU node allowlist:

| Stage | Expected CPU | Profiled CPU | Exact |
|---|---:|---:|---|
| `prefix_embed` | 8 | 8 | yes |
| `prefill` | 3481 | 3481 | yes |
| `postfix_step` | 0 | 0 | yes |

There were no missing or unexpected CPU rows and no unexpected provider rows.
The numerical graph path stayed on CUDA; CPU execution remained limited to the
approved control/index nodes.

## Numerical result

The registered diagnostic failed because prefix embedding and KV thresholds
were exceeded, despite finite outputs and exact provider placement.

### Prefix

- 12 of 13 registered outputs passed.
- `prefill_inputs_embeds` failed the `embedding_kv` threshold:
  `max_abs=6.6875`, `RMS=0.1185363`, `relative_L2=0.0488974`,
  `cosine=0.9988038`.
- Limits were `max_abs<=0.10`, `RMS<=0.02`, `relative_L2<=0.02`,
  `cosine>=0.999`.

### Prefill and KV

- `x1` passed: `max_abs=0.00350833`, `RMS=0.00108005`,
  `relative_L2=0.00116552`, `cosine=0.99999935`.
- All 72 KV tensors existed, covered 36 layers, were finite and nonzero.
- 0 of 72 KV tensors passed the preregistered per-tensor threshold.
- Worst observed KV metrics were `max_abs=12.59375`, `RMS=1.62498057`,
  `relative_L2=0.59366822`, and minimum cosine `0.82229524`.

Because prefill consumed the mixed-EP prefix output, these KV measurements
include accumulated prefix embedding drift. The approved stop rule forbids a
new teacher-forced isolation run after numerical failure, so this experiment
does not attribute the KV drift separately to prefix versus prefill.

### Postfix and D10

- Teacher-forced final postfix passed with immutable KV input hashes:
  `max_abs=0.00299701`, `RMS=0.00070863`,
  `relative_L2=0.00151218`, `cosine=0.99999886`.
- Recursive D10 final action passed and every intermediate output was finite:
  `max_abs=0.13758576`, `RMS=0.01628188`,
  `relative_L2=0.03474459`, `cosine=0.99939628`.
- PyTorch `x2..x9` references remain unavailable.

## Decision

`5090-mixed-ep-diagnostic` is **failed** under the preregistered contract.
Exact control-only CPU placement is mechanically feasible, and final action
checks passed for this single fixture, but the prefix embedding and every KV
threshold did not. The route therefore cannot pass `export-parity`, cannot
unlock AGX TensorRT work, and is not deployment evidence.

## Remote evidence

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/results/mixed_ep_stage_kv_d10_parity.json
SHA256 0d8ae8a10f4dfcac58a8b1d3b42c22aecef46fdcfc1c294da3bef3f023fc44f6

/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/logs/mixed_ep_parity.log
SHA256 381d3203c1e7975c038fe3824dc9c088e65b6825669d8acd514f0ee08ac1d97f

/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/manifests/mixed_ep_parity.py
SHA256 0cca6e6d2a7cb137ba4a304710f9a48fa31d8698a45b6f604381e01570935a9e
```

The three provider profiles are retained under
`results/mixed_profiles/`; their hashes are recorded in the remote artifact
directory and referenced by the JSON report.
