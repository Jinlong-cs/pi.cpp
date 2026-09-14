# Revision 4 execution delta

Status: approved by the human owner on 2026-08-19.

Revision 4 does not change the approved revision 3 model contract, tensor ABI,
stage split, cache layout, deterministic noise semantics, normalization,
numerical thresholds, or output shape. It changes only the execution and
artifact boundary for `wp-02-export` because the 16 GiB Mac does not have safe
headroom for the 4.159B-parameter real-checkpoint export.

## Approved AGX export-only lane

Host:

```text
agx-test
```

Only authorized remote artifact root:

```text
/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/
```

Allowed actions are limited to:

- stage the pinned Wall-X source, Wall-OSS checkpoint, normalizers, processor
  manifest, and task-owned exporter source;
- create a task-local Python environment and package/cache/temp directories on
  NVMe;
- reproduce processor/config/checkpoint identity and strict checkpoint load;
- run the pinned PyTorch reference and revision 3 wrapper parity ladder with
  deterministic external `x0`;
- export the three approved ONNX stages;
- run ONNX checker, exact manifest/hash checks, and ONNX Runtime parity.

## Storage and process constraints

- Do not write task assets, Python environments, package caches, or temporary
  export data to rootfs, `/tmp`, or `/dev/shm`.
- Do not stop, restart, signal, or otherwise interfere with any process or
  service.
- Recheck GPU/process ownership and NVMe headroom immediately before staging
  and before real model execution.
- Stop if parent identity, source/checkpoint hash, available storage, or process
  ownership differs from the recorded contract.

## Explicitly excluded

Revision 4 does not authorize TensorRT engine build, plugin work, C++ runtime
integration, formal latency, service launch or restart, server/client testing,
closed-loop evaluation, or promotion. `agx-build-approval` remains pending.
