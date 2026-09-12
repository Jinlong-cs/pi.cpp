# Revision 3 design delta

Status: approved by the human owner on 2026-08-19.

Revision 2 was approved on 2026-08-19. The first real processor capture and
reference-wrapper audit then exposed an incomplete engine handoff and one
incorrect Euler-mask implementation. Export was paused until this delta was
approved; the human owner approved revision 3 on 2026-08-19.

## Evidence retained from revision 2

- real processor output is `input_ids int64 [1,768]`, processor
  `attention_mask int64 [1,768]`, `pixel_values float32 [2048,1176]`, and
  `image_grid_thw [[1,32,32],[1,32,32]]`;
- action tokens occupy exactly positions `736..767`;
- the engine converts the processor mask explicitly to BOOL;
- checkpoint structural loading has 1061 checkpoint tensors, no unexpected
  keys, and only the reviewed tied `lm_head.weight` omission;
- normalizer NPZ assets and source hashes remain unchanged.

## Blocking Euler-state correction

The upstream path computes this once after prefill:

```text
v_padding = padding_action - x0
```

and reuses the same `v_padding` for all nine cached postfix steps. It does not
recompute `padding_action - x_i` at each step. The revision-2 reference wrapper
did the latter and was therefore semantically wrong for inactive action
dimensions.

The corrected equations are:

```text
v0_masked = v_padding * (1 - dof_mask) + v0 * dof_mask
x1 = x0 + dt0 * v0_masked

for i = 1..9:
    vi_masked = v_padding * (1 - dof_mask) + vi * dof_mask
    x_(i+1) = x_i + dt_i * vi_masked
```

`v_padding` is FP32 `[1,32,26]`; `dof_mask` remains FP32 `[1,32,26]`.

## Corrected stage handoff

`prefix_embed` additionally returns `v_padding`. The prefill engine consumes
`x0`, `dt0`, `dof_mask`, and `v_padding`; these values were missing from its
revision-2 public input manifest even though Euler interval 0 requires them.

The 32-token postfix contains only action tokens. Therefore the production
postfix engine does not consume a stale `postfix_inputs_embeds`, a dynamic
action mask, token types, or expert ranges. It creates the current action
embedding directly from `x_i`, `t_i`, and `dof_mask`; action mask/type/range are
fixed constants. Its variable inputs are:

```text
x_i, t_i, dt_i, dof_mask, v_padding,
postfix_position_ids, postfix_attention_mask,
36 layers of prefix key/value tensors
```

`postfix_position_ids` and `postfix_attention_mask` remain outputs of
`prefix_embed` because they depend on the actual padded prompt and visual
layout.

## Cache bridge requirement

The exporter must reconstruct the exact upstream cache object from all 72 flat
prefix-KV tensors before calling the decoder. Tuple substitution is not
accepted. The bridge must pass a focused round-trip test against the installed
Transformers cache API before ONNX export is attempted.

## Unchanged boundaries

The source/checkpoint pins, two cameras, 448 resolution, 768/736/32 sequence,
36 decoder layers, KV layout, horizon 32, D10, output `[32,26]`, BF16-first
lane, AGX approval gate, service gate, and closed-loop gates are unchanged.

Approving revision 3 authorizes only the corrected Mac exporter/static/parity
implementation. It does not authorize AGX artifact creation, environment
installation, engine build, service startup, or closed-loop execution.
