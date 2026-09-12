# AGX Wall-OSS export evidence

Verified: 2026-08-20 CST

## Contract and source

- Host: `agx-test`, Jetson AGX Orin, CUDA 12.6, PyTorch 2.8.0, ONNX 1.21.0.
- Checkpoint SHA256: `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`.
- Runner SHA256: `04ca73531d9f1fc7fe9daa7b99d916485a21d0c829b4f1111000467286a5c460`.
- Exporter SHA256: `7500ce82d17c8e23b1994e6cfaf2f89a2a06f4c24388c0220f8945d0ae3d0af7`.
- Wrapper SHA256: `320dc247a0814ad9529b3a6a96b3c0021a3bd5d002743ffa4e7809d46112841e`.
- Artifact directory: `/root/wujinlong/nvme/picpp_walloss_20260818/revision3_export/onnx_real_legacy_v18/`.

## Export and checker

- Real checkpoint load completed with only the intentional missing `lm_head.weight`; no unexpected keys.
- Fixed vision decomposition is exact; rotary/window/cu-window checks are exact.
- Two-expert MoT decomposition is exact for `x1` plus all 72 KV tensors.
- `prefix_embed.onnx`: SHA256 `f367251038bae0f1543de475caaf066e94c8ed163186706a9ba6efe4a2142a6d`, 8 inputs, 13 outputs, 3431 nodes, static IO.
- `prefill.onnx`: SHA256 `f0c7fa01919e10a82b15fa7b98d82230c917fa2def441bc62e298faf9b41c169`, 11 inputs, 73 outputs, 44824 nodes, static IO, 434 external initializers.
- `postfix_step.onnx`: SHA256 `034118f77bd996fc2e6a0e13d6087a394be15fb6a657ca554b34903cdbf711e1`, 79 inputs, 1 output, 14381 nodes, static IO.
- External data: 434 files, 6,795,845,632 bytes, aggregate SHA256 `857b5609c607a30b04cf587ce54a05ad81c2a0d156abc80a974b873023235387`.
- `onnx.checker.check_model(path)` passed for all three stages. Full IO and hash evidence is in `onnx_validation_v18.json`.

## ORT CPU limitation

AGX environment exposes only `AzureExecutionProvider` and `CPUExecutionProvider` in ONNX Runtime 1.23.2. Session creation failed for all three stages before execution:

- prefix: CPU EP has no implementation for BF16 `MatMul(13)`.
- prefill/postfix: CPU EP has no implementation for BF16 `Expand(13)`.

Therefore ONNX Runtime CPU numerical parity is not claimed. This is an environment/provider limitation, not a reported model mismatch. Evidence is in `ort_cpu_v18.json`; `export-parity` remains pending/blocked and no TensorRT work is authorized by revision 4.
