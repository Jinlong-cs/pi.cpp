# PI0.6 Airbot runtime

`pi06_airbot` adapts the Airbot LeRobot contract to the existing split
TensorRT runner. The adapter keeps the model's internal 32-dimensional action
space and exposes the physical 14-dimensional action space only after
unnormalization.

## Input and output

The runner accepts three RGB images in this order:

```text
base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
```

Images are raw `uint8` HWC arrays in RGB order. Each image is resized with
aspect-ratio-preserving black padding to `224x224`, converted to float32 in
`[-1, 1]`, and sent as `[1, 3, 3, 224, 224]`. The state is raw `float32[14]`:

```text
left arm 6 + left gripper 1 + right arm 6 + right gripper 1
```

The dataset metadata fixes this ordering but does not declare per-dimension
physical units. A robot client must therefore preserve the joint and gripper
units used by the source Airbot recorder; unit parity remains a required gate
before closed-loop execution.

The adapter zero-pads state to 32 dimensions, applies the checkpoint's q01/q99
normalization, and emits the PI0.6 prompt form:

```text
Task: <task>, State: <32 bins>, Advantage: Positive;
Action:
```

The `state` and `actions` entries in `norm_stats.json` are both 32-dimensional;
their padded dimensions (14 through 31) must have zero q01/q99 values.

The native runner receives Gaussian noise with shape `[1, 50, 32]` and returns
the normalized `[50, 32]` denoising result. The adapter applies q01/q99 action
unnormalization and returns `[50, 14]` absolute actions. The last 18 dimensions
must remain in the model path and are cropped only at the final boundary.

## WebSocket request

The Airbot client sends one msgpack observation with these top-level keys:

```text
base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, state, prompt, advantage
```

The server consumes the camera keys in the order above; it does not accept an
`images` list or a `task` alias. `advantage` is a boolean and is included in
the prompt conditioning: `true` selects Positive and `false` selects Negative.
The current Airbot clients send `true`. The handshake metadata repeats this
contract so a client can inspect it before sending observations.

Noise is generated from the manifest seed as a continuous per-process stream.
Passing an explicit `[1, 50, 32]` noise array to `run_once` is reserved for
same-input parity checks.

The native engine ABI uses float32 `image` and `x_t`, boolean `image_mask` and
`tokenized_prompt_mask`, and int64 `tokenized_prompt`. The current native API
exposes startup shape checks; exact engine dtypes are rejected on the first
inference call and remain part of the real-engine ABI gate.

## Asset package

An asset directory passed to `picpp server --model pi06_airbot` must contain:

```text
export_manifest.json
norm_stats.json
tokenizer.model
engines/prefix_embed.engine
engines/prefix_lm.engine
engines/suffix_step.engine
```

`export_manifest.json` is the source of the fixed contract. The required shape
is:

```json
{
  "schema": "pi_cpp.pi06_airbot.v1",
  "model_family": "pi06",
  "runtime": {"runner": "Pi05OfflineRunner", "abi": "pi05_offline_v1"},
  "assets": {"tokenizer": "tokenizer.model", "norm_stats": "norm_stats.json"},
  "normalization": {"method": "quantile", "state_key": "state", "action_key": "actions"},
  "contract": {
    "camera_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    "image_size": [224, 224],
    "raw_state_dim": 14,
    "raw_action_dim": 14,
    "token_length": 200,
    "action_semantics": "absolute",
    "advantage_conditioning": true,
    "default_advantage": true
  },
  "action": {"horizon": 50, "dim": 32},
  "sampling": {"steps": 10, "dt": -0.1, "noise_seed": 1234}
}
```

Start the websocket server with:

```bash
picpp server --model pi06_airbot --model-dir assets/pi06_airbot
```

The fixed-shape runner can be smoke-profiled after engines are installed:

```bash
picpp latency --model pi06_airbot --model-dir assets/pi06_airbot
```

The reported native `infer_ms` begins after host-to-device copies. Compare it
separately from wrapper time, preprocessing and transfer time, WebSocket
round-trip time, and robot closed-loop cadence.

The policy output is a 50-step chunk. The source Airbot sync client executes
25 steps before replanning; an async client may use a different queue policy.
The runtime returns the full chunk and does not silently reinterpret it as
servo-rate commands.

The runtime uses Pillow bilinear resize for the CPU-side adapter. The source
training path uses JAX linear resize, so pixel-level preprocessing parity is a
separate golden test before claiming end-to-end numerical parity.

This change supplies the runtime adapter and package contract only. Checkpoint
conversion, ONNX export, engine building, quantization, AGX measurements, and
robot closed-loop validation remain separate gates. The final package must also
record tokenizer, norm-stat, engine, checkpoint, and exporter source hashes
before numerical parity is claimed.
