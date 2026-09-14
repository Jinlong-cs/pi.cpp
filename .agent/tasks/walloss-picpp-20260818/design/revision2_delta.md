# Revision 2 design delta

Status: approved by the human owner on 2026-08-19.

Revision 1 was approved on 2026-08-19. During the first implementation batch,
the pinned source and locally verified checkpoint exposed one engine-ABI
mismatch and three identities that had previously been listed as post-review
captures. Dependent exporter/runtime implementation was stopped until this
delta was approved.

## Blocking ABI correction

`dof_mask` is fixed as FP32 `[1,32,26]`, not `[1,1,26]`.

Evidence in pinned Wall-X commit `6764e8f12f320819e86d5476b3ed68fb8f45b43c`:

- `_vendor/harrix/envs/libero_common.py::encode_proprio()` constructs
  `[1, action_horizon, total_dof]`;
- `_vendor/harrix/serving/_wallx_infer/model_wrapper.py` forwards that tensor;
- `model/core/action/processor.py::ActionProcessor.step()` concatenates it
  directly with `noisy_action [1,32,26]` on the last axis, so there is no
  sequence-axis broadcast.

The corrected tensor remains semantically identical to upstream. It does not
change horizon, action layout, active dimensions, or masking behavior.

## Newly frozen identities

- tokenizer length after adding `<|propri|>` and `<|action|>`: `151667`;
- `<|propri|>` ID: `151665`;
- `<|action|>` ID: `151666`;
- checkpoint config `vocab_size`: `151936`;
- checkpoint embedding rows: `155765`;
- flow times: `float32 linspace(0,1,11) * 0.999`, ending at `0.999`;
- downloaded `model.safetensors`: `8,331,486,672` bytes, SHA256
  `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`.

Tokenizer length, config vocabulary size, and checkpoint embedding rows are
three different fields. The exporter must preserve all three and must not
resize the checkpoint embedding table to the tokenizer length. The pinned
upstream loader likewise resizes the model back to the checkpoint row count
before loading weights.

## Non-changes

The three-engine split, two camera inputs, 448 resolution, batch 1, state and
action dimension 26, horizon 32, D10 execution, 768/736/32 sequence contract,
36-layer KV ABI, BF16-first precision lane, correctness thresholds, AGX gates,
and service/closed-loop boundaries remain unchanged.

No AGX directory, environment, GPU work, engine build, service, or closed-loop
run is authorized by approving this revision.
