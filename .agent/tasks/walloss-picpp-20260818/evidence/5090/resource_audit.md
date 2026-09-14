# Revision 5 5090-zem resource audit

Verified: 2026-08-20 12:05-12:37 CST

- Host: `zem-pc`, user `zem`, Ubuntu 24.04 x86_64.
- GPU: NVIDIA GeForce RTX 5090 D v2, driver 580.173.02, 24,455 MiB.
- Storage: `/home/zem/wujinlong` is on a 1.8 TiB filesystem with about
  118-119 GiB available (94% used).
- RAM: 60 GiB total, approximately 46 GiB available at the first audit.
- System Python: 3.12.3; `onnxruntime` is not installed.
- Docker is available. No system Python or global CUDA environment was changed.
- The approved target directory did not exist and was not created.

At 12:05 the GPU had only display usage. At 12:37 a separate root-owned
DStereo recovery training process appeared:

```text
PID 54170
python -u /task/src/train_recovery.py ... --target-step 5000 --max-steps 30000
GPU memory about 10,202 MiB
GPU utilization 74-90 percent
power about 267 W
```

Per Revision 5 resource stop conditions, no ORT environment, capability probe,
artifact staging, or parity workload was started. The process was not stopped,
signaled, or modified. Exact process evidence is in
`resource_conflict_20260820.txt`.

## Transfer observation

AGX-to-Mac transfer over the current Tailscale route had effectively unusable
throughput. The task-owned rsync was stopped without touching the AGX source.
Approximately 15 MiB of partial data remains in the Mac task-local relay at
`/private/tmp/walloss_r5_transfer_20260820/`; it is not a complete package and
was not sent to 5090.

