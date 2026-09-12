# Revision 3 source structural probe

Verified at 2026-08-19 14:08 CST against:

- Wall-X checkout `/private/tmp/wall-x-picpp-export-r2` at
  `6764e8f12f320819e86d5476b3ed68fb8f45b43c` plus the reviewed
  `external_x0.patch`;
- Torch `2.13.0`;
- Transformers `5.2.0`;
- pi.cpp worktree base `a1b1571e5f26692463412cf9065ebd3d547a2f6d`.

The real Wall-X classes were imported and inspected. The probe confirmed:

- `_prepare_flow_action_inputs` has the explicit `initial_noise` parameter;
- upstream computes `v_padding = padding_action - noisy_action` once before
  the cached loop and reuses it for every postfix velocity mask;
- postfix position IDs and embeddings are the `prefix_length:` action-only
  slice;
- postfix expert ranges are equivalent to `[0,0]` and `[0,32]` for 32 action
  tokens;
- the SM-independent decoder path with `use_cache=False` reads Transformers
  5.x cache tensors through `past_key_value.layers[layer].keys/values`;
- a real Transformers `DynamicCache` built from the 72 flat bindings contains
  36 layers and sequence length 736.

The Mac base environment has unrelated optional dependency conflicts:

- installed scikit-learn is binary-incompatible with the active NumPy;
- installed torchvision lacks a matching `torchvision::nms` operator.

For this source-only probe, Transformers' optional sklearn and torchvision
availability checks were explicitly disabled before importing Wall-X. Neither
assisted generation nor torchvision transforms were called. This workaround is
not part of repository code, the ONNX graph, or a production package, and it
does not count as a full model import/forward proof.

The FlashAttention Mac stub was import-only and throws if invoked. It was not
called by this probe.
