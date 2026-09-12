# V18 Actual IO Audit

The V18 ONNX protobufs were opened read-only on `agx-test` with external data
loading disabled. The actual public input ABI is:

- `prefix_embed/pixel_values`: ONNX `FLOAT` `[2048,1176]`.
- `prefill/prefill_inputs_embeds`: ONNX `BFLOAT16` `[1,768,2048]`.
- `prefill` KV outputs: 72 ONNX `BFLOAT16` tensors
  `[1,2,736,128]`.
- `postfix_step` consumes the same 72 BF16 KV tensors.

The earlier source `stage_io_manifest.json` described `pixel_values` as
`BF16_OR_FP16`; that metadata did not match the exported graph. The draft
runtime and source manifest were corrected to FP32 for `pixel_values`. No ONNX
graph, checkpoint, fixture or prior evidence was modified.
