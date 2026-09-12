# Wall-OSS-0.5 model contract

## 1. Identity and claim boundary

The only accepted reference is Wall-X commit
`6764e8f12f320819e86d5476b3ed68fb8f45b43c` with Hugging Face revision
`b7dc23a82dbf70a65859c9fdd596703234597028`. The current model identity is the
immutable Hub revision and the Phase 0 file manifest. A local
`model.safetensors` SHA-256 remains pending until an approved download.

No PI0.5, FastWAM, StarVLA, or other pi.cpp latency/success number is a
Wall-OSS baseline. The first same-model deployment baseline must be the pinned
upstream BF16 model on the same AGX and the same input/noise contract.

## 2. Fixed architecture

| Field | Contract |
|---|---|
| decoder | Qwen2.5-VL-derived mixed-token decoder |
| decoder layers | 36 |
| VLM hidden/intermediate | 2048 / 11008 |
| action hidden/intermediate | 1024 / 2048 |
| query/KV heads | 16 / 2 |
| head dimension | 128 |
| vision encoder | 32 layers, hidden 1280, output 2048 |
| token experts | exactly 2 |
| routing | deterministic token type, not learned top-k |
| VLM token type | 0 |
| action token type | 1 |
| reference dtype | BF16 |
| cameras | `face_view`, then `right_wrist_view` |
| image size | 448x448 RGB for each camera |
| raw state | float32 `[26]` |
| action latent | float32 `[1, 32, 26]` |
| action output | float32 `[32, 26]` |
| flow steps | exactly 10 Euler intervals |
| flow schedule | float32 `linspace(0,1,11) * 0.999` |
| full padded sequence | 768 tokens |
| action postfix | 32 tokens |
| prefix cache length | 736 tokens |
| dof mask | float32 `[1,32,26]` |
| tokenizer/config/checkpoint vocab | 151667 / 151936 / 155765 rows |

The action expert is not a conventional learned MoE route. Token type 0 uses
the VLM expert and token type 1 uses the action expert. The exporter must
preserve permute/compute/unpermute semantics in mixed-token prefill. The
postfix graph is action-only and must not retain VLM-expert weights.

## 3. External policy request

The Python policy boundary accepts:

```text
images:
  face_view:        uint8 [H,W,3]
  right_wrist_view: uint8 [H,W,3]
state:              float32 [26]
prompt:             UTF-8 non-empty string
x0:                 optional float32 [1,32,26]
request_id:         non-empty string
noise_seed:         optional uint64, mutually exclusive with x0
```

Rules:

- Both named cameras are required. No duplication, missing-camera padding, or
  implicit positional remapping is permitted.
- The wrapper may accept CHW float images only through a separately tested
  conversion path; the canonical stored fixture is RGB uint8 HWC.
- State must contain exactly 26 finite values in `agent_pos_config` order.
- Prompt whitespace normalization is explicit and must not alter instruction
  text beyond the approved canonicalization function.
- For parity, `x0` is mandatory. For production, the policy may generate `x0`
  from a declared standard-normal RNG when only `noise_seed` is supplied. The
  generated tensor and its SHA-256 become request metadata.
- Supplying neither `x0` nor `noise_seed` is allowed only in a separately
  declared nondeterministic service mode. It is forbidden in eval and
  benchmark commands.

## 4. Flow semantics

The upstream flow path is preserved as:

```text
x0 ~ N(0, I)
full mixed-token prefill with use_cache=True
-> velocity_0 and 36-layer cache
x1 = Euler(x0, velocity_0, t0, dt0)
crop every layer's K/V to prefix length 736
v_padding = padding_action - x0
for i = 1..9:
    x_(i+1) = postfix_step(x_i, t_i, dt_i, v_padding, prefix_kv)
unnormalize x10
```

The exact upstream time array ends at `0.999` and is exported as a manifest asset and hashed. A
generic `linspace`, a reduced step count, a different ODE solver, or a changed
Euler ordering is a contract violation even if the final output is finite.
For inactive dimensions, the initial `v_padding` is reused for all nine
postfix steps; it is not recomputed from the evolving `x_i`.

The 36-layer prefix cache is read-only during the nine postfix steps. It is
valid only for one observation/prompt/state request; images and state change
each control cycle, so the cache cannot be reused across requests.

## 5. Model action layout

The base runner returns all 26 dimensions in the order from checkpoint
`dof_config`:

```text
follow_left_ee_cartesian_pos_relative        3
follow_left_ee_rotation_6D_relative          6
follow_left_gripper                          1
follow_right_ee_cartesian_pos_relative       3
follow_right_ee_rotation_6D_relative         6
follow_right_gripper                         1
velocity_decomposed                          3
height                                       1
head_actions                                 2
```

Environment-specific extraction, such as a LIBERO right-arm 7-D adapter, is a
named policy adapter after the base output. It cannot be hidden inside the
TensorRT runner or used to change the `[32,26]` model correctness gate.

## 6. Prefix KV ABI

For each layer `00` through `35`:

```text
prefix_kv.layer_XX.key    BF16-or-FP16 [1,2,736,128]
prefix_kv.layer_XX.value  BF16-or-FP16 [1,2,736,128]
```

The selected cache dtype must equal the engine manifest. K already includes
the model's M-RoPE transformation. Every tensor must be present, have fixed
dimensions, be finite, have a nonzero count consistent with the reference,
and remain at a stable device address for the postfix loop.

The gate compares all 72 tensors. It must reject missing layers, a layer-0-only
cache, dynamic `-1` dimensions, wrong head order, a transposed sequence/head
layout, an unknown dtype, or a cache that is rewritten by postfix execution.

## 7. Precision lanes

- `R0-AGX-Torch-BF16`: pinned upstream same-device reference.
- `C0-TRT-BF16`: preferred production candidate.
- `C1-TRT-FP16`: allowed only as a separately identified fallback candidate
  if BF16 build/tactic support fails and FP16 passes all numerical gates.

No INT8, FP8, weight-only quantization, reduced layer count, reduced image
resolution, prompt caching that omits dynamic state, or reduced D10 schedule is
part of revision 2.

## 8. Required manifest assertions

Load must fail if any of these disagree:

- source/checkpoint/tokenizer/normalizer hashes;
- camera names/order/resolution;
- batch, sequence, horizon, state/action dimensions, or flow steps;
- token IDs, padding side, prompt template, or state-bin count;
- engine names, tensor names/order, dtypes, shapes, or TensorRT ABI;
- layer count, KV layout, M-RoPE sections `[16,24,24]`, heads, or head dim;
- backend or precision label.

`strict=False` checkpoint loading is not accepted as evidence. The exporter
must record missing/unexpected keys and fail unless an explicitly reviewed
allowlist is empty or exact.
