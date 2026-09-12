# Revision 7 Execution Preflight

## Immutable Inputs

- Source commit: `6764e8f12f320819e86d5476b3ed68fb8f45b43c`
- Checkpoint revision: `b7dc23a82dbf70a65859c9fdd596703234597028`
- Checkpoint payload SHA256:
  `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`
- Existing V18 source root on AGX:
  `/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/`
- Fresh 5090 root:
  `/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/`
- Fresh AGX root:
  `/root/wujinlong/nvme/picpp_walloss_20260818/revision7_engine/`

## Read-only Size Budget

- Checkpoint: 7.8G
- Pinned Wall-X/export source: 3.9M
- V18 ONNX package including external data: 9.2G
- Prior PyTorch tensor artifacts: 388M
- Existing export environment: 1.4G (not copied as a portable environment)
- 5090 free storage at audit: about 314G
- AGX NVMe free storage at audit: 70,223,773,696 bytes (about 65.4 GiB)
- AGX rootfs free storage: about 3.7G; forbidden for artifacts and scratch

The 5090 budget is sufficient for additive checkpoint, source, two fixtures,
one candidate, logs and profiles. The AGX budget must be rechecked after the
candidate is frozen. Reserve at least 9.2G for the ONNX package, the serialized
engine sizes, timing cache, build logs, and a target-local TensorRT temporary
directory; stop before build if the final candidate-specific budget does not
fit with a safety margin.

## Execution Order After the GPU Gate Opens

1. Freshly audit 5090 GPU owner, processes, storage and runtime.
2. Create only the named Rev7 directory hierarchy.
3. Transfer the pinned checkpoint, Wall-X/export source, V18 package, existing
   calibration fixture and Rev7 scripts; verify every SHA256 before use.
4. Generate one calibration/debug and one disjoint real held-out processor
   fixture. Synthetic images are not sufficient for the held-out gate.
5. Run the patched parity capture twice and save complete `x1..x10`, 72 KV,
   normalized action and final action for each split.
6. Run `fixture_integrity.py --require-trajectory` on both fixtures.
7. Run teacher-forced prefill from saved PyTorch prefill inputs.
8. Run the same-host original/export-patched/ORT ladder and the 2x2
   prefix/prefill matrix. Decide retain-v18, minimal-v19 or reject.
9. Freeze one complete relative-path size/SHA256 tree. No AGX build may begin
   before this immutable candidate exists.

## AGX Tool Facts

- Target `trtexec`: TensorRT v10.3 (`v100300`).
- The installed CLI exposes `--onnx`, `--bf16`, `--saveEngine`,
  `--timingCacheFile`, `--tempdir`, `--skipInference`, `--dumpLayerInfo`,
  `--profilingVerbosity` and `--memPoolSize`.
- Parser/build commands will be frozen from the final candidate manifest and
  target-local `trtexec --help`; no guessed flag or CPU/GPU fallback is allowed.

## Current Stop

At 2026-08-21T18:26:29+08:00, PID 365367 still owned 9496 MiB on the 5090 and
belonged to the non-task DStereo training. No Rev7 inference or transfer was
started. A fresh audit and the explicit `revision7-5090-execution-approval`
remain required.
