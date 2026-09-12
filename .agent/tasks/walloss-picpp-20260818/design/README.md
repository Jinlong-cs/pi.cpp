# Wall-OSS pi.cpp Phase 1 design packet

Status: revision 3 was approved by the human owner on 2026-08-19 after a
blocking Euler-state and stage-handoff correction.

This packet freezes the first implementation contract for integrating
Wall-OSS-0.5 as a persistent pi.cpp backend. It is derived from the pinned
Wall-X source, the pinned Hugging Face checkpoint metadata, the current pi.cpp
runtime structure, and the Phase 0 AGX audit.

## Frozen decisions

1. The production backend is explicit TensorRT. A PyTorch sidecar may be used
   only as the same-device reference baseline; it is never a transparent
   fallback from TensorRT.
2. The fixed engine split is:

   ```text
   host preprocessing/tokenization
   -> prefix_embed.engine
   -> prefill.engine (full sequence plus Euler interval 0)
   -> postfix_step.engine x 9
   -> host unnormalize/output adapter
   ```

3. D10 means exactly one full prefill plus nine KV-cached postfix steps. It
   does not mean one prefill plus ten postfix steps.
4. The inference API receives the initial Gaussian action tensor `x0`
   explicitly. Parity runs must provide and hash it.
5. The full token sequence is left-padded to 768 tokens. The final 32 tokens
   are action tokens, so the prefix cache length is 736. Padding-equivalence
   must pass against the unmodified upstream path before export proceeds.
6. Prefix KV contains key and value tensors for all 36 decoder layers. A
   layer-0-only check is invalid.
7. Batch size is 1, there are exactly two named 448x448 camera inputs, state
   and action dimensions are 26, action horizon is 32, and the output is
   float32 `[32, 26]` in the model action layout.
8. The persistent server has one in-flight request per runner. Engines,
   contexts, stream, KV, workspace, and action ping-pong buffers are allocated
   during load and remain resident.
9. BF16 is the first TensorRT precision candidate. FP16 is a separately named
   candidate and must pass the same parity gates. INT8 is outside the first
   integration scope.

## Documents

- `model_contract.md`: source identity, model semantics, fixed external and
  internal tensor contract.
- `preprocessing_tokenization.md`: image, prompt, state-string,
  normalization, padding, and initial-noise rules.
- `export_engine_split.md`: exporter wrappers, engine responsibilities, KV
  ABI, unsupported-operator plan, and package layout.
- `runtime_api.md`: C++/Python/server API, lifetime, concurrency, failure, and
  timing behavior.
- `parity_and_latency_protocol.md`: staged correctness, AGX timing, resource,
  and closed-loop evidence lanes.
- `stop_conditions.md`: mandatory stop, rejection, escalation, and rollback
  rules.

## Captured facts and remaining target facts

The real processor capture now confirms:

- `pixel_values float32 [2048,1176]` for two fixed 448 images;
- `image_grid_thw = [[1,32,32],[1,32,32]]`;
- processor `attention_mask` is INT64 and the engine binding is an explicit
  BOOL conversion;
- action tokens occupy positions `736..767`.

Remaining target facts are not guessed into an engine:

- whether TensorRT 10.3 on SM87 accepts the BF16 graph without an unsupported
  operation or precision fallback.

Any mismatch changes a model or tensor interface and therefore requires a
contract revision before dependent implementation continues.

## Mac implementation status after revision 3 approval

- checkpoint header inventory and strict inventory checks are implemented under
  `python/pi_cpp/walloss/checkpoint.py`;
- the fixed three-stage IO manifest is generated under
  `recipes/walloss/manifests/stage_io_manifest.json`;
- processor contract capture, strict checkpoint-load helpers, reference stage
  wrappers, and a fail-fast ONNX export entry point are implemented;
- the selected normalizer key and source/array identities are recorded in
  `recipes/walloss/manifests/normalizer_selection.json`.

The revision 3 stage manifest now exposes 13 prefix outputs, 11 prefill inputs,
73 prefill outputs, and 79 postfix inputs. Prefill and postfix accept only their
named public tensor bindings; they no longer recover Euler or cache state from
a hidden Python context. A synthetic fixed-shape postfix graph passed
`torch.export` and ONNX export while retaining all 72 KV inputs.

The `exporter-static` gate passed 29 focused tests. Full real-checkpoint
PyTorch/wrapper/ONNX parity remains pending because the 4.159B-parameter model
does not have a safe full-load/export memory margin on the 16 GiB Mac. See
`../evidence/exporter/`.

These Mac-side artifacts do not prove ONNX validity, TensorRT buildability, AGX
latency, server behavior, or closed-loop quality.

## Source anchors

- pi.cpp: `a1b1571e5f26692463412cf9065ebd3d547a2f6d`
- Wall-X: `6764e8f12f320819e86d5476b3ed68fb8f45b43c`
- Wall-OSS-0.5: `b7dc23a82dbf70a65859c9fdd596703234597028`
- Phase 0 evidence: `../evidence/phase0/`

Revision 2 freezes token IDs, vocabulary identities, the exact flow schedule,
the checkpoint SHA256, and corrected `dof_mask` shape in
`revision2_delta.md`. A local implementation worktree and checkpoint now
exist; no AGX environment, ONNX, TensorRT engine, service, or GPU process has
been created.

Revision 3 corrects the request-constant inactive-dimension velocity and the
prefill/postfix tensor handoff in `revision3_delta.md`. Corrected Mac
exporter/static/parity work may proceed; AGX and deployment gates remain
closed.
