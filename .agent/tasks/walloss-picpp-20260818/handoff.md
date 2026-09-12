# Wall-OSS pi.cpp Revision 7 handoff

Updated: 2026-08-24 CST

## Current decision

- task: `walloss-picpp-20260818-revision7`
- contract revision: 7
- lifecycle: `blocked`
- acceptance decision: `blocked`
- V18 candidate: rejected by 5090 prefix/prefill numerical parity
- V19 candidate: not created because no specific exporter/operator defect is
  localized
- candidate freeze: skipped
- AGX and Revision 8: not unlocked and not executed

The detailed decision is in
`evidence/5090/revision7/root_cause_decision_20260824.md`.

## Authorization boundary

The human owner approved only fixture recovery, teacher-forced prefill,
same-host PyTorch ladders, and PT/ORT 2x2 prefix/prefill diagnostics on
`5090-zem` under:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/
```

AGX transfer/build/parity, Revision 8, latency, server/client, and closed-loop
were explicitly excluded. Those operations did not run.

## Frozen inputs

- Wall-X commit: `6764e8f12f320819e86d5476b3ed68fb8f45b43c`
- checkpoint revision: `b7dc23a82dbf70a65859c9fdd596703234597028`
- checkpoint SHA256:
  `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`
- V18 prefix SHA256:
  `f367251038bae0f1543de475caaf066e94c8ed163186706a9ba6efe4a2142a6d`
- V18 prefill SHA256:
  `f0c7fa01919e10a82b15fa7b98d82230c917fa2def441bc62e298faf9b41c169`
- V18 postfix SHA256:
  `034118f77bd996fc2e6a0e13d6087a394be15fb6a657ca554b34903cdbf711e1`
- approved placement SHA256:
  `0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2`
- approved allowlist SHA256:
  `474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341`
- current 5090 PyTorch calibration golden SHA256:
  `6bb757fbdf40f8af92d3107af4cda65d8cd7b4cd7add18eed271efddd2e02709`

## Completed 5090 evidence

The calibration fixture passes mechanical integrity:

- 36 layers / 72 nonzero KV tensors;
- complete `x1..x10` and `step1..step9`;
- finite tensors and exact stage bridges;
- internal/external `x0` and wrapper action exact within each framework lane;
- 146 BF16 bit-preserving round trips.

The full fixture-completeness gate still fails because no disjoint real
held-out/paired fixture exists.

Teacher-forced V18 prefill fails independently of ORT prefix propagation:

- `x1` passes;
- only layer 0 key/value pass (`2/72` KV);
- current-golden worst KV `relative_L2=0.590079`, `cosine=0.824476`.

Torch 2.11 same-host controls:

- golden to unpatched PyTorch: prefix 13/13, prefill 73/73;
- unpatched to export-patched PyTorch: exact 13/13 and 73/73;
- ORT prefix: 12/13, with image-token `prefill_inputs_embeds` drift
  (`max_abs=13.125`, `relative_L2=0.0506213`, `cosine=0.998718`);
- PT/ORT prefill on the same PT prefix: 3/73, with only layer 0 KV passing.

Torch 2.8 reproduces the same ORT failure magnitude while retaining exact
unpatched-to-export-patched PyTorch outputs. It also shows separate
framework-version drift relative to the current Torch 2.11 golden. Therefore
the Python export patch is not the primary defect and framework version does
not explain the V18 ONNX/ORT boundary.

## Primary artifacts

Torch 2.11 report:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/diagnostic/same_host/torch211/attempt1/same_host_prefix_prefill_diagnostics.json
SHA256 c22aa5dd626ccf42f0479564d61aaff41c585b80548ab16cc29d92ea389cb43f
run.rc 2
```

Torch 2.8 report:

```text
/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/diagnostic/same_host/torch28/attempt1/same_host_prefix_prefill_diagnostics.json
SHA256 cfeda32601edc2aec6aa21a8aaa8e47643676f7a856b990bc64fc4ce0b99dd93
run.rc 2
```

Root-cause report exists locally and remotely with SHA256:

```text
66a442dfa7a8d93c09301b1e8ecda40ab3910a958d5ddfca2e2804a6e1f108bd
```

The final 5090 audit found no remaining task process and no compute process;
GPU memory returned to 15 MiB. Revision 7 occupies about 31 GiB and the project
filesystem had 262,098,153,472 bytes free.

## Gate state

- `revision7-design-review`: passed
- `revision7-5090-execution-approval`: passed
- `revision7-reference-fixture-completeness`: failed (held-out absent)
- `revision7-teacher-forced-prefill`: passed as a diagnostic, with V18
  numerical failure recorded
- `revision7-root-cause-decision`: passed
- `revision7-candidate-lineage`: skipped
- `revision7-candidate-freeze`: skipped
- `revision7-engine-feasibility-decision`: passed as an explicit early reject
- all AGX gates: pending and unexecuted

The Revision 7 contract/acceptance pair validates with 6 work packages and 12
gates. Validation evidence is in
`evidence/revision7/contract_validation_20260824.txt`.

## Exact next gate

Do not retry V18, relax thresholds, create V19, or begin AGX work from this
handoff. A continuation requires a new reviewed contract delta that provides:

1. a real disjoint held-out fixture;
2. operator-level localization inside the V18 prefix image-token path;
3. layerwise localization inside V18 prefill between layer 0 and layer 1;
4. one minimal exporter hypothesis with a registered parity test;
5. a new 5090 execution approval for that diagnostic only.

Only a parity-passing, fully hashed candidate could later request a separate
AGX transfer/build approval. Accepted pi.cpp baselines and all Revision 3/5/6
artifacts remain unchanged.
