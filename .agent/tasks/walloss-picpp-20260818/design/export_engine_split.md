# Export and TensorRT engine split

## 1. Export principle

Do not export `generate_flow_action()` as one graph. It contains Python
objects, dynamic cache mutation, `torchdiffeq`, routing bookkeeping, and
internal randomness. Export three pure-tensor wrappers with fixed batch and
sequence contracts.

The exporter belongs under `python/pi_cpp/walloss`; the pinned Wall-X checkout
is an input dependency, not copied production code. Every wrapper has a
PyTorch reference mode, ONNX export mode, IO manifest, and focused parity
fixture.

## 2. Stage A: `prefix_embed.engine`

Responsibilities:

- execute the 32-layer Qwen2.5-VL vision encoder for two 448 images;
- embed padded text/special/action token IDs;
- insert vision and initial action embeddings at their exact masks;
- compute fixed token-type, padding, joint-attention, action, and M-RoPE
  tensors needed by prefill;
- apply the action preprocessor for `x0` and `t0`.

Inputs:

| Name | Dtype | Shape |
|---|---|---|
| `input_ids` | INT64 | `[1,768]` |
| `attention_mask` | BOOL | `[1,768]` |
| `pixel_values` | FP32 | expected `[2048,1176]` |
| `image_grid_thw` | INT64 | `[2,3]` |
| `moe_token_types` | BOOL | `[1,768]` |
| `x0` | FP32 | `[1,32,26]` |
| `t0` | FP32 | `[1]` |
| `dof_mask` | FP32 | `[1,32,26]` |

Outputs are fixed-shape embedded states, M-RoPE positions, masks,
action-selection metadata, the first-step condition, and
`v_padding = padding_action - x0` as FP32 `[1,32,26]`. Their exact
names and dtypes are generated into `prefix_embed_io.json`; no consumer binds
by numeric position alone.

If TensorRT cannot represent image rotary/window attention, M-RoPE, or scatter
semantics, the exporter first decomposes them into standard fixed-shape ops.
A plugin is introduced only after parser/build evidence identifies the exact
unsupported or unfused boundary.

## 3. Stage B: `prefill.engine`

Responsibilities:

- run all 36 mixed-token decoder layers on the full padded sequence;
- preserve deterministic token-type permute/compute/unpermute;
- execute joint GQA attention with 16 query heads and 2 KV heads;
- produce the first action velocity;
- perform Euler interval 0 and return `x1`;
- return prefix-only K/V for all 36 layers.

In addition to the embedded decoder tensors, Stage B explicitly consumes
`x0 [1,32,26]`, `dt0 [1]`, `dof_mask [1,32,26]`, and
`v_padding [1,32,26]`. These are required by the first Euler update and may not
be hidden in a Python context object.

Outputs:

```text
x1                             FP32 [1,32,26]
prefix_kv.layer_00.key/value   cache [1,2,736,128]
...
prefix_kv.layer_35.key/value   cache [1,2,736,128]
```

The export wrapper crops K/V to 736. The C++ runtime must not receive a
full-sequence cache and infer the crop point itself.

## 4. Stage C: `postfix_step.engine`

Responsibilities:

- accept one `x_i`, `t_i`, `dt_i`, `dof_mask`, the request-constant
  `v_padding`, and the read-only 36-layer prefix cache;
- embed exactly 32 action tokens and build action AdaRMS conditioning;
- execute only action-expert norm, QKV/attention, MLP, and projection paths;
- perform one Euler update and return `x_(i+1)`;
- never update or reserialize prefix KV.

All 32 postfix positions are action tokens. The production graph therefore
builds its inputs directly from the current action embedding and does not bind
`postfix_inputs_embeds`, `postfix_action_mask`, `postfix_moe_token_types`, or
expert ranges as request tensors. Those values are action-only constants.
Only prompt-dependent postfix position IDs and attention mask cross the Stage
A boundary.

Inactive dimensions use the same `v_padding = padding_action - x0` at every
postfix step, exactly as upstream. Recomputing `padding_action - x_i` is a
semantic change and fails parity.

The graph is specialized to token type 1. The exporter must prove from the
ONNX initializer inventory that no VLM expert-0 MLP/QKV weights are included.
If constant folding retains both experts or runtime permute/unpermute, this
stage fails the production stop gate even if it builds.

The C++ runtime invokes the stage exactly nine times for indices 1 through 9.
Two fixed `[1,32,26]` device buffers alternate as input/output. CUDA Graph
capture is an optimization after uncaptured parity passes; capture mismatch
must fail visibly rather than fall back to naive forward.

## 5. Unsupported-operator decision order

For each failure, use this order:

1. native TensorRT layer/tactic;
2. fixed-shape ONNX decomposition or constant folding;
3. a minimal versioned plugin for the measured boundary;
4. reject the TensorRT route if the plugin would reimplement a large unstable
   model region or erase latency/memory benefit.

Candidate plugins, only when proven necessary:

- `WallMropePlugin`: three-axis positions and sections `[16,24,24]`;
- `WallJointGqaAttentionPlugin`: mixed prefill or prefix-KV postfix attention;
- `WallMotPermutePlugin`: deterministic mixed-token gather/scatter in prefill;
- `WallAdaRmsNormPlugin`: conditional action-expert norm;
- a vision rotary/window plugin if the 32-layer vision graph cannot build.

RMSNorm, SwiGLU, Linear/GEMM, scatter, and Euler update are not preemptively
pluginized. Every plugin requires a reference kernel, dtype/shape guards,
serialization version, TensorRT namespace, SM87 build, and plugin-level parity.

## 6. TensorRT build lanes

Build only on agx-test under an approved fresh NVMe revision. Initial lanes:

```text
C0: strongly typed BF16/FP32 boundary
C1: strongly typed FP16/FP32 boundary, only if C0 is unsupported
```

The build command, builder flags, optimization profile, tactic sources,
workspace, engine inspector output, plugin hashes, timing cache identity, and
TensorRT/CUDA/SM version are recorded. Unknown TensorRT dtypes and any `-1`
runtime dimension are hard errors; they must not be treated as FP32 or
allocated by unsigned-size multiplication.

## 7. Artifact package

```text
walloss_model/
  export_manifest.json
  tokenizer/
  normalizers/
    proprio.npz
    action.npz
  schedule/
    flow_times.npy
    flow_deltas.npy
  engines/
    prefix_embed.engine
    prefill.engine
    postfix_step.engine
  io/
    prefix_embed_io.json
    prefill_io.json
    postfix_step_io.json
    kv_manifest.json
  fixtures/
    fixture_manifest.json
    calibration/
    heldout/
    latency/
  plugins/
    libpicpp_walloss_plugins.so
```

The plugin file is required only if evidence proves a plugin is necessary.
`export_manifest.json` records every source/checkpoint/asset/engine/plugin
SHA-256, architecture and input contract, precision, stage contract version,
TensorRT ABI, and exact build command. Absolute source-machine paths are not
part of the portable package.

## 8. Export gates

The exporter cannot advance unless:

- the checkpoint tensor inventory matches the expected model without an
  unreviewed `strict=False` discrepancy;
- processor capture confirms the 768/736/32 sequence contract;
- internal-random and external-`x0` reference paths agree;
- each pure PyTorch wrapper agrees with the original model;
- each ONNX stage agrees with its wrapper on calibration and held-out data;
- all 72 KV outputs pass shape, finite, nonzero, and numerical checks;
- postfix ONNX excludes expert 0;
- no backend or operator fallback is present.
