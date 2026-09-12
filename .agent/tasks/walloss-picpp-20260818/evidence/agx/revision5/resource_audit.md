# Revision 5 agx-test read-only audit

Verified: 2026-08-20 12:06 CST

- Host: `agx-test`, user `root`, aarch64, Jetson AGX Orin.
- Kernel: `5.15.185-tegra`.
- TensorRT Python: 10.3.0.
- System Python: 3.10.12; system `onnxruntime` is not installed.
- NVMe: 916 GiB total, approximately 108 GiB available.
- RAM: 61 GiB total, approximately 51 GiB available.
- The frozen revision-3 export directory exists and remains untouched.
- `/root/wujinlong/nvme/picpp_walloss_20260818/revision5_engine/` is absent.
- No TensorRT build, latency, service, or closed-loop action was executed.

The v18 ONNX directory contains 438 files and occupies approximately 9.2 GiB.
The approved transfer set plus the real explicit-cache stage fixture is about
9.3 GiB. A 462-line SHA256 source manifest was generated at:

```text
/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/evidence/revision5_v18_source_manifest.sha256
```

Manifest SHA256:

```text
5eec1d2dcf062d0899e7e1ea1d893aac37ee6ced1e9a96a1f60a6854cf296d06
```

