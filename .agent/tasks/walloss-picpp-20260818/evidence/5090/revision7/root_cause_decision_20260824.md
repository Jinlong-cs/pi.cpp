# Wall-OSS Revision 7 5090 same-host root-cause decision

Date: 2026-08-24 CST

Decision: **reject V18 as a candidate and block candidate freeze**.

This decision is limited to the approved single-calibration-fixture 5090
diagnostic. It is not AGX, TensorRT, latency, server/client, closed-loop, or
promotion evidence.

## Authorized scope

The human owner authorized only the following work on `5090-zem` under:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/
```

- calibration fixture recovery;
- teacher-forced V18 prefill;
- same-host PyTorch unpatched/export-patched ladders;
- a PT/ORT 2x2 prefix/prefill diagnostic.

No AGX access, Revision 8 work, latency measurement, server/client run, or
closed-loop evaluation was performed.

## Frozen identity

- Wall-X commit: `6764e8f12f320819e86d5476b3ed68fb8f45b43c`
- checkpoint revision: `b7dc23a82dbf70a65859c9fdd596703234597028`
- `model.safetensors`: `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`
- V18 `prefix_embed.onnx`: `f367251038bae0f1543de475caaf066e94c8ed163186706a9ba6efe4a2142a6d`
- V18 `prefill.onnx`: `f0c7fa01919e10a82b15fa7b98d82230c917fa2def441bc62e298faf9b41c169`
- V18 `postfix_step.onnx`: `034118f77bd996fc2e6a0e13d6087a394be15fb6a657ca554b34903cdbf711e1`
- approved placement manifest: `0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2`
- approved CPU allowlist: `474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341`
- current 5090 calibration PyTorch golden: `6bb757fbdf40f8af92d3107af4cda65d8cd7b4cd7add18eed271efddd2e02709`
- diagnostic harness: `44b91f69cd39a302114c39da1f1a3f1e3caad62005932603430ca4e8f6d094cd`
- execution wrapper: `1e9a26c8111e394eec58284314cab5cbeb8579f7d493b64b8989f2e0fc541ce0`

The harness argument named `legacy_fixture` pointed to the current 5090
calibration golden above. It did not point to the older AGX stage fixture.

## Environment and execution

- host/GPU: `zem-pc`, NVIDIA GeForce RTX 5090 D v2, 24,455 MiB, SM120;
- Python: 3.11.15;
- ORT-GPU: 1.23.2;
- lane 1: Torch 2.11.0+cu128;
- lane 2: Torch 2.8.0+cu128;
- both lanes used the same checkpoint, source, V18 ONNX tree, ORT build,
  placement manifest, CPU allowlist, fixture, thresholds, and device;
- the lanes ran sequentially in independent output directories;
- both produced a complete JSON and intentionally exited with code 2 because
  their registered numerical gates failed;
- final audit: no remaining model process, GPU memory 15 MiB, zero compute
  processes, and 262,098,153,472 bytes free under `/home/zem/wujinlong`.

The host had no `tmux`, so execution used task-local `nohup`, PID files,
launcher logs, complete run logs, and `run.rc` evidence. No new files appeared
under the default Hugging Face cache during either run.

## Fixture recovery result

The calibration fixture is mechanically complete:

- all tensors finite;
- 36 KV layers and 72 KV tensors present and nonzero;
- `x1..x10` present;
- `step1..step9` postfix inputs present;
- internal-random and external-`x0` full generation are bit exact;
- external-`x0` generation and wrapper action are bit exact;
- 146 BF16 tensors pass a bit-preserving round trip;
- all fixed ABI and stage bridge checks pass.

This does **not** pass the contract's fixture-completeness gate because no
independent real held-out/paired fixture exists. The fixture checker ran with
`paired_fixture.requested=false`.

## Teacher-forced V18 prefill

Two saved PyTorch prefill inputs were tested directly, without consuming an
ORT prefix output. The older AGX-derived fixture and the recovered 5090 golden
show the same failure pattern.

| Input reference | x1 | KV passed | Worst KV relative L2 | Worst KV cosine |
|---|---:|---:|---:|---:|
| Older AGX-derived fixture | pass | 2/72 | 0.591010 | 0.825150 |
| Current 5090 golden | pass | 2/72 | 0.590079 | 0.824476 |

Only `prefix_kv.layer_00.key/value` pass. Layers 1 through 35 fail. The
current-golden `x1` remains close (`relative_L2=0.001258`,
`cosine=0.99999924`), demonstrating why checking only the action tensor would
hide the cache defect.

Evidence:

- `diagnostic/calibration_teacher_forced_prefill.json`, SHA256
  `698829d522087ed4e638750f9e83dc3cd2f53f85fae55d6bb29e5da61da6e936`;
- `diagnostic/calibration_teacher_forced_prefill_5090golden.json`, SHA256
  `2e22d01f16c17b2282e5f34478578645fcdce98b652c89dbc6b73c5223a10d90`.

## Same-host ladder

### Torch 2.11 control

- current golden to fresh unpatched PyTorch: prefix 13/13 and prefill 73/73;
- unpatched PyTorch to export-patched PyTorch: prefix 13/13 and prefill 73/73,
  with exact `prefill_inputs_embeds` and exact `x1`;
- external `x0` seam: bit exact.

This establishes that the approved export monkey patches are mathematically
equivalent inside PyTorch for this fixture.

### Torch 2.8 control

Unpatched to export-patched PyTorch is again exact (13/13 prefix and 73/73
prefill), so the patch remains exonerated in the exporter-era Torch version.

Torch 2.8 does differ from the current Torch 2.11 golden:

- prefix embedding: `max_abs=3.1328125`, `RMS=0.0714574`,
  `relative_L2=0.0294634`, `cosine=0.999567`;
- `x1`: `relative_L2=0.0002071`, `cosine=0.99999998`;
- normalized action: `relative_L2=0.0019656`, `cosine=0.99999807`;
- final action fails the intentionally exact external-action seam
  (`max_abs=0.0201821`, `relative_L2=0.0031585`).

Framework-version drift therefore exists, but it cannot explain the ORT
failure: the matched Torch 2.8 2x2 result has essentially the same failure
magnitude as the Torch 2.11 result.

## Prefix boundary

For Torch 2.11 PyTorch prefix versus V18 ORT prefix, 12 of 13 outputs pass.
`prefill_inputs_embeds` fails:

```text
max_abs     13.125
RMS          0.122772
relative L2  0.0506213
cosine       0.998718
```

The token-region split localizes the large drift:

- image tokens: `relative_L2=0.0549676`, `cosine=0.998488`;
- prefix non-image tokens: bit exact;
- action suffix: small drift (`relative_L2=0.0008731`,
  `cosine=0.99999962`) but fails the frozen max-absolute threshold at 0.125.

Torch 2.8 gives the same qualitative result (`relative_L2=0.0480023`,
`cosine=0.998848`). This is a V18 ONNX/ORT prefix image-token boundary, not a
general tokenizer/control-tensor mismatch.

## PT/ORT 2x2 prefill matrix

Cells are:

- A: PyTorch prefill(PyTorch prefix);
- B: ORT prefill(PyTorch prefix);
- C: PyTorch prefill(ORT prefix);
- D: ORT prefill(ORT prefix).

Torch 2.11 results:

| Comparison | Outputs passed | KV passed | Worst KV relative L2 | Worst KV cosine |
|---|---:|---:|---:|---:|
| A vs B: prefill backend on PT prefix | 3/73 | 2/72 | 0.590079 | 0.824476 |
| A vs C: prefix backend with PT prefill | 1/73 | 0/72 | 0.115114 | 0.993374 |
| A vs D: end to end | 1/73 | 0/72 | 0.593908 | 0.822331 |
| B vs D: prefix backend with ORT prefill | 1/73 | 0/72 | 0.184111 | 0.982964 |
| C vs D: prefill backend on ORT prefix | 3/73 | 2/72 | 0.589258 | 0.824925 |

Every `x1` comparison passes (`relative_L2` between 0.000236 and 0.001258),
while the KV bridge fails. In the backend-isolating A-vs-B and C-vs-D rows,
only layer 0 key/value pass and the worst error is around layer 16 value. The
same pattern and magnitude reproduce under Torch 2.8; its A-vs-D worst KV is
`relative_L2=0.592300`, `cosine=0.823135`.

Provider placement is not the differentiator. Both lanes exactly reproduce
the approved mixed placement: all 8 approved prefix CPU nodes and all 3,481
approved prefill CPU nodes are present, with zero missing or unexpected CPU
nodes.

## Decision and stop boundary

The evidence supports two independent V18 defects/boundaries:

1. `prefix_embed.onnx` changes the vision/image-token embedding path while
   leaving non-image prefix controls exact.
2. `prefill.onnx` produces large layer-1-through-layer-35 KV drift even when
   fed the correct PyTorch prefix output.

The PyTorch export-patched implementation is exact to unpatched PyTorch in
both framework versions, so the failure lies after that Python-level patch:
the ONNX export/lowering/runtime boundary. The experiment locates the failing
stages but not a specific exporter operator or rewrite. Under the Revision 7
hard stop, that is insufficient authority to create a minimal V19.

Therefore:

- do not retain V18;
- do not create V19 in this revision;
- do not freeze or transfer a candidate;
- do not access AGX or enter Revision 8;
- mark Revision 7 blocked pending a separately reviewed operator-level export
  localization plan and a real disjoint held-out fixture.

## Primary artifacts

Torch 2.11:

```text
remote: diagnostic/same_host/torch211/attempt1/
report: c22aa5dd626ccf42f0479564d61aaff41c585b80548ab16cc29d92ea389cb43f
recovered fixture: bf2b30b3511c3b4ca66b6353517cd81c3724ee7f3e4db6890848ec840fc04887
run log: a374bad50514a87d28fac851070da8bfcdd82ea5892d35196960c6b3488d0efc
return code: 2
```

Torch 2.8:

```text
remote: diagnostic/same_host/torch28/attempt1/
report: cfeda32601edc2aec6aa21a8aaa8e47643676f7a856b990bc64fc4ce0b99dd93
recovered fixture: 0b5e277398fa041bca6213abc6a3f66cd5937557a3816677340ce1058942145d
run log: 11df26dd5b75a3bbef1717259a84b1b30d0c37dea539e014a63cadfc6d5a9949
return code: 2
```

No accepted baseline, V18 package, Revision 5/6 artifact, or service was
modified.
