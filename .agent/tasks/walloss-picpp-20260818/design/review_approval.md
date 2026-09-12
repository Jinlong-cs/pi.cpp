# Design review approval

## Revision 4 export-only execution approval

- Task: `walloss-picpp-20260818`
- Contract revision: `4`
- Decision: approved
- Decision owner: human owner
- Decision received: 2026-08-19 14:48:04 CST
- User statement: `你去AGX完成导出吧，agx-test这个服务器上`

This approval keeps the revision 3 model, tensor, cache, normalization, and
numerical contracts unchanged. It authorizes checkpoint staging, an isolated
task-local environment, real PyTorch reference and wrapper parity, ONNX export,
ONNX checking, and ONNX Runtime parity on `agx-test` only under:

```text
/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/
```

It does not authorize TensorRT engine build, service launch or restart,
latency measurement, server/client validation, closed-loop evaluation, or
promotion. The separate `agx-build-approval` gate remains pending.

## Revision 3

- Task: `walloss-picpp-20260818`
- Contract revision: `3`
- Decision: approved
- Decision owner: human owner
- Decision received: 2026-08-19 13:26:27 CST
- User statement: `批准 walloss-picpp-20260818 revision 3 design-review`

This approval accepts the request-constant
`v_padding = padding_action - x0` semantics, the corrected prefill handoff,
the specialized action-only postfix inputs, and the explicit Transformers
cache bridge requirement. It authorizes corrected Mac exporter/static/parity
implementation only. AGX artifact creation, target environment installation,
engine build, service, closed-loop evaluation, and promotion remain controlled
by separate gates.

## Revision 2

- Task: `walloss-picpp-20260818`
- Contract revision: `2`
- Decision: approved
- Decision owner: human owner
- Decision received: 2026-08-19 10:34:35 CST
- User statement: `批准 walloss-picpp-20260818 revision 2 design-review`

This approval accepts the corrected `dof_mask [1,32,26]` engine ABI, the
separate tokenizer/config/checkpoint vocabulary identities, the exact flow
schedule, and the unchanged revision 1 numerical thresholds. It authorizes
dependent Mac exporter/static/parity implementation. AGX build, service,
closed-loop evaluation, and promotion remain controlled by separate gates.

## Revision 1 history

This section preserves the revision 1 decision. A later implementation pass
proved that the upstream `dof_mask` ABI is
`[1,32,26]`, not the revision 1 draft's `[1,1,26]`. See
`revision2_delta.md`. The revision 1 approval alone did not authorize dependent
exporter/runtime work against the corrected ABI; revision 2 now does.

- Task: `walloss-picpp-20260818`
- Contract revision: `1`
- Decision: approved
- Decision owner: human owner
- Decision received: 2026-08-19 09:57:57 CST
- User statement: `批准 walloss-picpp-20260818 revision 1 design-review`

The approval authorizes implementation of the frozen revision 1 exporter and
runtime contract in an isolated Git worktree. It does not approve an AGX
artifact directory, target environment installation, TensorRT build, GPU
execution, persistent service launch, closed-loop evaluation, or promotion.

Those actions remain controlled by their separate acceptance gates, including
`agx-build-approval`, `server-client-smoke`, `closed-loop-smoke`, `gate40`,
`final400`, and `final-review`.
