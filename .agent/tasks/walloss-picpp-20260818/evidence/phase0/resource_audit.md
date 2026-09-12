# Phase 0 resource audit

Checked: 2026-08-18 18:01 CST from the Mac through `ssh -o BatchMode=yes agx-test`.

## Current state

- Host is reachable as `agx-test` / `root@100.64.17.47` and reports `NVIDIA Jetson AGX Orin Developer Kit`.
- The board is in `MAXN`; the read-only clock snapshot showed GPU 306 MHz of 1300.5 MHz maximum and EMC 2.133 GHz of 3.199 GHz maximum. This is not a locked benchmark state.
- Rootfs has only 3.7G free (94% used). NVMe has 166G free. All future large artifacts must go under a fresh NVMe revision directory; no rootfs, `/tmp`, or `/dev/shm` artifact is allowed.
- `fuser` on the main GPU device nodes returned no owner, and the scoped process search found no `trtexec`, `picpp`, Wall-X, serving, or LIBERO process. This proves only that no owner was visible at this instant; it is not a reservation.
- `tegrastats` is available, but Nsight Compute and Nsight Systems are not in PATH.
- System Python has no torch/transformers/flash-attn/onnxruntime. CUDA 12.6 packages and TensorRT 10.3 are installed, but the CUDA compiler is only discoverable at `/usr/local/cuda-12.6/bin/nvcc` and `ninja` is absent.
- No task-owned `/root/wujinlong/nvme/picpp_walloss_20260818*` directory exists yet.
- No Wall-OSS checkpoint or local Wall-X checkout exists on the Mac, and no Wall-OSS AGX package exists in the scoped NVMe search.

## Commands used

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 agx-test '<read-only host, filesystem, package, power, thermal, fuser, and process audit>'
ssh -o BatchMode=yes agx-test 'python3 import probe; dpkg-query CUDA/TensorRT probe; find /root/wujinlong/nvme -maxdepth 3 ...'
```

The command surface did not stop a process, change a service, alter clocks, change power mode, write a remote file, or allocate GPU memory for model inference.

## Consequence

Phase 0 can freeze identity and resource constraints, but cannot claim Wall-OSS latency or deployment readiness. The first non-read-only action requires a new isolated environment and the `agx-build-approval` gate.
