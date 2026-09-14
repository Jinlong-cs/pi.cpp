# pi.cpp Toolchain Architecture

Status: **DRAFT v1** (2026-09-14) — the target architecture for extending
pi.cpp from a multi-model inference runtime into the company's VLA
deployment toolchain. Grounded in the deployed experience: PI0.5 Safe4-P25
(NX + AGX), FastWAM selective PTQ (AGX), PI0.6* heterogeneous_3view
(non-RTC 180000 + RTC cloth 120000, AGX), StarVLA/DiT4DiT/SmolVLA/Evo-1.

## Vision

One installable bundle. One CLI. Every model is a **package** (engines +
manifest + adapter); the runtime, the build/quantize toolchain, and the
verification ladder all read the same contract. An agent (or a human) can
run:

```text
picpp package install <model-package>
picpp verify   --model pi06_rtc            # parity ladder + D10
picpp infer    --model pi06_rtc --case ... # open loop
picpp serve && picpp client                # closed loop
```

with JSON output and fail-closed exit codes at every step.

## What exists today

- **Runtime layer (already multi-model):** `src/runtime/*_offline_tensorrt.cpp`
  for pi05 / fastwam / smolvla / starvla / dit4dit / evo1 / semanticvla /
  groot; pi06 rides `pi05_offline_tensorrt.cpp` (branches
  `feat/pi06-airbot-runtime`, `feat/pi06-heterogeneous-runtime`,
  `feat/pi06-rtc-runtime`, the last adds CUDA-graph replay of the denoise
  loop). Spec-driven engine binding already exists (the pi06_rtc delta was
  two bound tensors + a post-loop pin).
- **CLI layer:** `python/pi_cpp/cli.py` (tyro) + `commands/{infer,eval,
  latency,server,client}` + per-model adapters (`pi06_airbot.py`,
  `pi06_heterogeneous.py`, `pi06_rtc.py`, ...).
- **Optimization SDK (branch `feat/v0.2-optimization-sdk`):** the picpp-opt
  subcommands (compact-sequence / audit-sequence / rewrite-io / edit-qdq /
  narrow-ffn / apply-delta) with JSON recipes.
- **Outside the repo today** (the gap): the deployment-side toolchain that
  made the pi06 rounds work — ONNX surgery (width trim, RoPE bake,
  bf16→fp16 cache rewrite, hybrid precision, QDQ), ORT range calibration,
  trtexec/python engine builds with the 4 GB-workspace recipe, the parity
  ladder (`agx_harness.py`) and the D10 locked-clock protocol. These live
  in per-task directories on the boards; the durable traps (builder
  workspace, width trim, CUDA-graph host pointers) live only in docs.

## Target architecture: four layers + one contract center

```text
L4 distribution   one wheel (runtime + CLI + toolchain) + versioned model
                  packages (device-sealed engines + manifest + adapter)
L3 verification   picpp verify (parity ladder + D10, gates from manifest)
                  picpp eval (frozen cases, LIBERO gate40/full400)
                  picpp serve/client (closed loop), real-arm = authorization
                  boundary (fail-closed, owned by the arm skill)
L2 toolchain      picpp graph (trim/bake/cast/hybrid/QDQ)
                  picpp calibrate (ORT range collection -> scales)
                  picpp build (recipe-driven, 4 GB default, size sentinels)
L1 runtime        one generic multi-stage TRT runner (load/bind/stages/loop/
                  CUDA graph) + thin model front-ends
L0 contract       deployment manifest v1 = ABI/stages/gates/lock sha256 —
                  the single source of truth
```

The contract center is the load-bearing idea: every layer reads the
manifest instead of re-encoding model knowledge in code. pi06_rtc already
proved the pattern — its runtime delta over the non-RTC adapter is two
bound tensors plus one post-loop pin, everything else is spec-driven.

## Design principles (elegant + light)

1. **Data over code.** A new model = a manifest + a thin host adapter
   (unnorm / action maps / RTC pin — the `pi06_rtc.py` size of thing), not
   a new C++ class. The generic runner owns everything mechanical.
2. **Knowledge as sentinels, not prose.** The durable traps become tool
   behavior: `picpp build` defaults to 4 GB workspace and refuses an
   engine whose size pattern says "silent fp32 fallback" (the size
   sentinel); `picpp graph trim` computes `768 + max_lang` from the
   deployment prompts and refuses to cut below it; CUDA-graph capture code
   forbids stack-address host sources by construction (member buffers
   only).
3. **One-layer adaptation.** Model differences stop at the adapter layer;
   the loop shape (host-denoise, steps, capture) is manifest data.
4. **Rejected routes live in the repo.** `docs/rejected-routes.md` + a
   `picpp build` preflight warning, so future agents don't re-run dead
   ends (TRT 10.3 INT8 calibrator on JetPack, W8A8 QDQ for the pi06
   suffix at the current gates, plain-fp16/all-bf16 prefix, ...).
5. **Engines and code stay separate.** The wheel is pure toolchain; model
   packages are versioned artifacts with a sha256 lock. Weights never
   enter git.
6. **Progressive merge.** Each branch lands on mainline only after its
   board-level ladder passes; the offline ladder protocol doubles as the
   release gate.

## CLI surface (target)

```text
picpp package    install/list/info/verify/export/lock   (L4/L0)
picpp build      --recipe <json> [--workspace-gb 4]     (L2)
picpp graph      trim|bake|cast-caches|hybrid|qdq       (L2)
picpp calibrate  --onnx ... --cases ... --layers ...    (L2)
picpp verify     parity|latency [--protocol d10]        (L3)
picpp infer      --model ... --case ... [--delay N --action-prefix ...]
picpp eval       --model ... --suite gate40|full400
picpp latency    --model ... --protocol d10
picpp serve / client                                     (L3 closed loop)
```

Agent contract: every command writes JSON on stdout, uses exit codes
0/1/2 (pass / fail / unauthorized), and never actuates a real arm without
the authorization flag matching the manifest contract.

## Model package format

```text
<model-package>/
  deployment_manifest.json     # picpp.deployment-manifest.v1 (see schema doc)
  export_manifest.json         # the supernova exporter manifest, unchanged
  engines/*.engine             # device-sealed, sha256-locked
  adapter.py                   # host-side mapping only (or a registered name)
  results/                     # ladder + D10 evidence produced by picpp verify
```

## Roadmap

| Phase | Scope | Deliverable |
|---|---|---|
| 1 converge | merge the 5 branches, one wheel, CI smoke | installable `pi_cpp` bundle |
| 2 contract | manifest v1 + package commands; generic runner refactor; pi06 ABI variants collapse to one adapter | `picpp package *` |
| 3 toolchain | port the board scripts to `picpp graph/calibrate/build` with sentinels | the traps become tool behavior |
| 4 verify | `picpp verify/eval` with manifest gates; closed-loop smoke; arm authorization | the acceptance protocol as a command |
| 5 agent loop | JSON/exit-code contract hardening, package export/import, docs | agents run deploy+verify+eval chains |

## Out of scope (by design)

- Real-arm actuation (the arm skill owns the authorization boundary; pi.cpp
  is fail-closed without it).
- Training-side quantization (QAT belongs to the training line; pi.cpp
  consumes exported artifacts).
- A new inference engine: TensorRT stays the backend for the board
  runtimes; ggml stays the portable/lightweight side.
