# Preprocessing, tokenization, normalization, and noise

## 1. Ownership boundary

The first production version keeps processor/tokenizer logic in the Python
wrapper and model execution in the C++ TensorRT runner. This avoids embedding
Transformers or Python model execution in the pi.cpp C++ runtime while keeping
the upstream text/image semantics reproducible.

The Python path must be deterministic, directly tested, and included in
`wrapper_ms`. It is not excluded from complete-policy latency.

## 2. Camera and image contract

Canonical order:

```text
0: face_view
1: right_wrist_view
```

Each request must provide both RGB images. The approved processor is the
checkpoint Qwen2.5-VL processor with:

- output resolution 448x448 for both cameras;
- patch size 14;
- temporal patch size 2;
- merge size 2;
- mean `[0.48145466, 0.4578275, 0.40821073]`;
- standard deviation `[0.26862954, 0.26130258, 0.27577711]`.

Expected capture for two fixed images:

```text
pixel_values:   [2048,1176]
image_grid_thw: [[1,32,32],[1,32,32]]
merged vision tokens: 512 total
```

These are expected processor outputs, not permission to hand-reimplement the
processor from arithmetic alone. The exporter gate captures the actual pinned
processor output and compares it bytewise or with the approved dtype tolerance
before freezing the engine IO manifest.

The wrapper rejects missing cameras, extra cameras, alpha/gray images,
non-finite floats, and ambiguous BGR input. Resize/interpolation, rescaling,
normalization, and patch flattening must match the upstream processor.

## 3. Prompt skeleton

The flow prompt preserves the pinned source template. It contains the system
message, the two named camera placeholders, the instruction, the line
`Predict the next action in robot action.`, the `Proprioception: <|propri|>`
placeholder, and the assistant prefix. The 32 `<|action|>` tokens form the
postfix.

The exact bytes, newlines, public camera labels, special tokens, and tokenizer
IDs are captured from the pinned source. Only the immutable skeleton and
normalized task text may be cached. The full token sequence cannot be cached
because `<|propri|>` is replaced on every request by state-derived tokens.

## 4. State normalization and string representation

The raw state ordering is the 26-D `agent_pos_config` order. The selected
normalizer key must be provided by the package manifest; automatic unique-key
or prefix-match fallback is forbidden.

For each active state dimension:

```text
delta_safe = 1 if delta == 0 else delta
normalized = clamp(2 * (raw - min) / delta_safe - 1, -1, 1)
bin = digitize(normalized, linspace(-1, 1, 513)[:-1]) - 1
```

The 26 integer bins are joined by a single ASCII space and replace
`<|propri|>` before tokenization. State masks and virtual-tail behavior are
explicit assets. No dimension may be silently dropped or filled with default
normalizer values.

The package stores the converted 26-element proprio min/delta arrays as a
Torch-free NPZ or JSON asset plus SHA-256. The exporter verifies that this
asset produces the same normalized values and state string as the pinned
PyTorch normalizer for zero, min, max, random, masked, and out-of-range inputs.

## 5. Fixed sequence profile

The checkpoint resolves `max_length` to 768. Production tokenization uses:

```text
padding_side = left
truncation = true
max_length = 768
total_length = 768
postfix_action_tokens = 32
prefix_length = 736
```

The implementation adds explicit left padding to obtain a fixed TensorRT
profile. Before export, it must prove that this padded representation matches
the upstream unpadded request for input IDs, meaningful attention region,
vision-token expansion, M-RoPE positions, action mask, and final action.

The fixture generator enumerates all approved task prompts and adversarial
state-bin patterns (all zero, all 511, alternating, and per-coordinate boundary
values). It fails if meaningful content is truncated, the 32 action tokens are
not the final 32 non-padding positions, the first action index is not 736, or
the full tensor does not have length 768.

## 6. Token and mask assets

The package manifest records:

- tokenizer directory hash and vocabulary size;
- IDs for pad, image pad, vision start/end, `<|propri|>`, and `<|action|>`;
- `input_ids` int64 `[1,768]`;
- `attention_mask` bool `[1,768]`;
- `moe_token_types` bool `[1,768]`, with exactly 32 true/action positions;
- action and padding masks;
- M-RoPE position IDs int64 `[3,1,768]`;
- image-grid and placeholder-expansion counts.

No tokenizer addition or ID assignment may occur at runtime after the engine
package is built.

Revision 2 freezes the captured identities:

```text
pad token ID:                151643
<|propri|> token ID:         151665
<|action|> token ID:         151666
tokenizer length:            151667
checkpoint config vocab:     151936
checkpoint embedding rows:   155765
```

These three vocabulary sizes have different meanings and must remain separate
manifest fields. The model embedding table is not truncated to tokenizer
length.

## 7. Action normalization

The 26-D action normalizer uses checkpoint `min_key=q001` and
`delta_key=delta_001`:

```text
normalized = clamp(2 * (raw - min) / delta_safe - 1, -1, 1)
raw = ((normalized + 1) / 2) * delta + min
```

The C++ runtime receives the exported min/delta arrays. Revision 1 performs
unnormalization after the final D2H copy for ease of audit, records
`postprocess_ms`, and returns float32 `[32,26]`. A later fused GPU epilogue is a
separate optimization with the same output gate.

## 8. External initial noise

The original generation method calls `torch.randn` internally. The export
wrapper must refactor this into an explicit `x0` tensor without changing any
subsequent model operation.

Required checks:

1. Capture the original internally generated tensor.
2. Re-run through the new external-`x0` seam with that exact tensor.
3. Require exact `x0`, identical flow times, identical first prefill result,
   and dtype-appropriate final action parity.
4. Record shape, dtype, finite status, mean/std, seed provenance, and SHA-256.

Eval and latency fixtures store `x0` separately from calibration data. The
optimized path never generates or mutates its own correctness input.
