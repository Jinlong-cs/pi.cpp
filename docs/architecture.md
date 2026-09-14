# pi.cpp Architecture

TensorRT-first VLA/WAM inference for edge GPUs: a C++ runtime + a Python
CLI, with the quantization / graph / deployment strategies shipped as
recipes and as tool commands. The internal acceptance protocol (package /
verify / lock + the agent contract) is deliberately **not** part of this
repo; it is recorded in the embodied-ai skill under
`03-deploy/recipes/picpp-acceptance.md` and runs offline on the boards.

## Runtime layer

- One generic multi-stage TensorRT runner per engine topology; per-model
  differences stop at the Python wrapper layer:
  - `src/runtime/split_denoise_runner.cpp` — the pi05/pi06 multi-stage
    prefix/suffix host-denoise loop (CUDA-graph capture included);
  - `src/runtime/*_offline_tensorrt.cpp` — fastwam / smolvla / starvla /
    dit4dit / evo1 / semanticvla / groot model-specific paths;
  - `include/pi_cpp/core/` — tensor, host tensor, engine, logger, CUDA
    buffer primitives.
- Python wrappers (`pi06.py`, `pi06_heterogeneous.py`, `pi06_airbot.py`,
  `pi06_rtc.py`, `pi05.py`, `runner_wrapper.py`) adapt the LeRobot-style
  contracts: image layout, state normalization, prompt caching, action
  denormalization. The pi06 ABI variants (airbot / heterogeneous / rtc)
  are **manifest-driven** — the wrapper reads `deployment_manifest.json`
  (or the export manifest) and never needs a caller-chosen class.

## CLI surface

```text
picpp eval       --model ... --dataset ... [--progress]    offline action-trace eval
picpp latency    --model ... --model-dir ...               end-to-end timing profile
picpp infer      --model ... --case ...                    single-case inference
picpp server     --model ... --host ... --port ... [--authorize]
picpp client     --host ... --port ... [--async-supervision]
picpp graph      cast-caches | qdq                          ONNX rewrites (strategy tools)
picpp calibrate  --onnx ... --cases ...                     ORT activation-scale collection
picpp build      --recipe ... [--workspace-gb 4]            recipe-driven TensorRT build
```

`picpp-opt` (the Optimization SDK) covers the inspection / selective-QDQ /
precision-guard / comparison workflow for producing portable assets;
`picpp-tui` is the interactive Docker front end.

## Strategies live in the repo (knowledge as recipes + tool behavior)

- `recipes/` — reproducible deployment recipes with evidence
  (`recipes/fastwam/agx-orin-int8.json`: which nodes stay FP16, which
  islands go QDQ, the closed-loop result it was accepted on).
- `docs/` — strategy write-ups (`fastwam-detail.md`, `pi06-airbot.md`,
  `optimization-sdk.md`) and the benchmarks.
- The toolchain commands encode the durable traps as **sentinels**, not
  prose: `picpp build` defaults to a 4 GB workspace and refuses engines
  whose size pattern means a silent fp32 fallback; `picpp graph` refuses
  width trims below `768 + max_lang`; the INT8 calibrator trap fails
  fast on JetPack before any build work.

## Design principles

1. **Data over code.** A new model = a manifest + a thin adapter, not a
   new C++ class. The generic runner owns everything mechanical.
2. **Knowledge as sentinels.** Traps become tool behavior so they cannot
   be skipped by reading the docs wrong.
3. **One-layer adaptation.** Model differences stop at the wrapper layer;
   loop shape and capture behavior are manifest data.
4. **Engines and code stay separate.** The wheel is pure code; engines
   and weights never enter git.

## Acceptance (out of this repo by design)

Deploying an engine package is gated by the offline acceptance protocol —
package assembly, the parity ladder (scaled triplet vs the JAX golden +
raw gate vs the direct-TensorRT reference), the D10 latency band, and the
sha256 lock. The protocol, the gate thresholds, the golden-data contract,
and the board evidence live in the embodied-ai skill:
`03-deploy/recipes/picpp-acceptance.md`.
