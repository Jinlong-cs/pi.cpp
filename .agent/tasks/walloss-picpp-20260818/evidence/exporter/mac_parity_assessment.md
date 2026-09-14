# Mac export-parity assessment

Status: `export-parity` remains pending.

## Evidence obtained

A real Wall-X meta-device construction with the pinned JSON/YAML configuration
completed and confirmed:

- 4,158,803,200 parameters;
- approximately 8,317,606,400 bytes for parameters if uniformly BF16;
- 36 decoder layers;
- action hidden size 1024;
- `use_adarms=false`;
- `use_x_pred=false`;
- SDPA structural lane can be selected for a Mac-only probe.

The postfix stage's outer export boundary is viable in isolation: a synthetic
model with the exact fixed shapes exported through Torch ONNX, retained all 79
public inputs including all 72 flat KV bindings, and returned `x_next`.

## Why full parity did not run on this Mac

The Mac has 16 GiB unified memory. The BF16 parameter body alone is about 8.32
GB. A full reference load and export additionally need checkpoint/state-dict
or mapped tensor residency, model buffers, two-image vision activations, full
768-token decoder activations, 36-layer KV tensors, ONNX capture state, and
host/application headroom. The machine was already using compressed memory.

Running the 8.33 GB checkpoint through original Wall-X, three wrappers, and
three ONNX stages on this host therefore lacks a safe memory margin and risks
system-wide memory pressure. No full checkpoint forward was attempted.

## Remaining proof required

The following remain unproven and must not be inferred from static tests:

1. original internal-random versus external-`x0` full-model equivalence;
2. original model versus prefix/prefill/postfix wrappers on real fixtures;
3. per-layer KV and `x1..x10` numerical metrics;
4. prefix_embed and prefill ONNX export with real weights;
5. all-stage ONNX Runtime parity;
6. absence of expert-0 weights from the real postfix ONNX initializer set.

The next safe route is a larger-memory CPU/CUDA export host or an explicitly
approved AGX staging/build phase. Mac format, manifest, cache, and synthetic
ONNX results are not AGX or TensorRT evidence.
