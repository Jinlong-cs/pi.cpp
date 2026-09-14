# Persistent pi.cpp runtime API

## 1. Integration style

Wall-OSS is added through the existing explicit pi.cpp model surfaces. It does
not introduce a second dynamic plugin registry or a generic compatibility
layer. The initial implementation owns:

- `WallOssOfflineRunner` in C++ and pybind;
- `WallOssRunnerWrapper` in Python;
- `WallOssPolicy` in the existing server policy module;
- explicit `walloss` branches in CLI, latency, eval, server/client, TUI,
  CMake, and Makefile;
- a model-specific contract header and recipe directory.

The stable deployment interfaces are the WebSocket request/response schema,
the versioned model-package manifest, and named engine tensor ABI. The task
does not promise a cross-version C++ binary ABI.

## 2. C++ runner boundary

Illustrative interface:

```cpp
struct WallOssPreparedRequest {
  Tensor input_ids;
  Tensor attention_mask;
  Tensor pixel_values;
  Tensor image_grid_thw;
  Tensor moe_token_types;
  Tensor x0;
  Tensor flow_times;
  Tensor flow_deltas;
  Tensor dof_mask;
};

class WallOssOfflineRunner {
 public:
  Status Load();
  Status RunOnce(const WallOssPreparedRequest&, WallOssResult*);
  WallOssInputShapes InputShapes() const;
};
```

The actual implementation follows existing pi.cpp `Status` and TensorRT
utility patterns. Inputs are named and validated before enqueue. The runner
does not tokenize, normalize raw state, choose a camera, generate noise,
select a backend, or catch an engine error and continue.

## 3. Load lifecycle

`Load()` performs, once:

1. parse and validate the package manifest;
2. verify required asset, engine, and plugin hashes;
3. verify target TensorRT/CUDA/SM and precision ABI;
4. load any versioned task plugin library;
5. deserialize the three engines;
6. validate every named tensor, dtype, fixed shape, layer count, and KV layout;
7. create one execution context per engine and one CUDA stream;
8. allocate persistent input, output, KV, workspace, and action ping-pong
   buffers;
9. bind stable device addresses;
10. run an uncaptured mechanical warmup and optional CUDA Graph capture;
11. record resident memory and selected backend/precision metadata.

There is no request-time engine/context construction, allocator growth, model
download, tokenizer mutation, or plugin discovery.

## 4. Request execution

```text
Python image/state/prompt preprocessing and x0 selection
-> validate prepared tensor hashes/shapes
-> H2D changed inputs
-> enqueue prefix_embed
-> enqueue prefill, producing x1 and persistent prefix KV
-> enqueue postfix_step nine times with ping-pong x buffers
-> D2H x10
-> float32 action unnormalize
-> return [32,26] plus timing metadata
```

The runner synchronizes only at correctness-required stage boundaries during
debug/parity mode and once before final D2H completion in production mode.
Device-side CUDA events measure stages. Host `perf_counter` measures
preprocess, wrapper, serialization, and request RTT boundaries separately.

## 5. Lifetime, concurrency, and reset

- One runner supports exactly one in-flight inference.
- The server may queue multiple clients, but it must not execute two requests
  concurrently on shared KV/action buffers.
- Queue time is reported separately from policy inference.
- Prefix KV is overwritten at the start of every request and never reused
  across observations.
- `reset` clears request counters, queued work, policy-owned RNG sequence, and
  any future temporal adapter state. It does not unload engines.
- Repeated requests must show bounded resident memory after warmup; monotonic
  growth is a failure.
- Shutdown stops only the task-owned server/process and releases contexts,
  buffers, and stream in deterministic order.

## 6. Python policy API

```python
runner = picpp.build_walloss_runner(model_dir=...)
actions = runner.run_once(
    images={"face_view": ..., "right_wrist_view": ...},
    prompt=...,
    state=...,
    x0=...,
    noise_seed=None,
)
```

Properties exposed to existing commands:

```text
num_cameras = 2
camera_names = [face_view, right_wrist_view]
image_size = (448,448)
state_dim = 26
action_horizon = 32
action_dim = 26
flow_steps = 10
backend = tensorrt
precision = bf16 or fp16
```

The wrapper metadata includes model/package/engine hashes, input and noise
hashes, selected backend, precision, preprocessing and stage timings, output
shape, finite status, and cache gate status.

## 7. WebSocket request/response

The existing observation envelope is extended only through the named Wall-OSS
policy. Required request fields are task, 26-D state, and the two named images.
Eval/benchmark requests also include a seed or explicit fixture `x0` identity.

Response:

```text
actions: float32 [32,26]
model: walloss
request_id
backend: tensorrt
precision
policy_ms
preprocess_ms
queue_ms
serialization_ms
server_elapsed_ms
```

Client round-trip time is measured at the client and is not copied into
`policy_ms`. An exception is a failed request, not an HTTP/WebSocket success
containing an error dictionary.

## 8. Fail-fast rules

Startup or request execution fails on:

- missing or hash-mismatched asset;
- unsupported backend/precision/ABI;
- unknown TensorRT dtype or dynamic dimension;
- wrong camera count/order, image/state/noise shape, or non-finite input;
- tokenizer/profile overflow or meaningful truncation;
- missing/wrong/non-finite KV layer;
- CUDA Graph shape/address mismatch;
- enqueue, CUDA, plugin, or output finite failure.

No path silently selects PyTorch, a CUDA extension fallback, naive forward,
another normalizer key, another camera mapping, fewer flow steps, or zero
actions.

## 9. Rollback

The integration is additive. Rollback means selecting an existing model and
binary/package, stopping only the task-owned Wall-OSS service if one was
approved and started, and leaving all accepted baselines untouched. No
baseline artifact is overwritten or deleted.
