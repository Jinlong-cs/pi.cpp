# Deployment Manifest Schema v1 (draft)

Status: **DRAFT v1** (2026-09-14) — the pi.cpp-side contract that turns a
sealed model package into something the runtime, the toolchain, and the
verification ladder all read. The exporter-side manifest
(`export_manifest.json`, schema `supernova.pi06_heterogeneous_3view.*.v1`)
stays unchanged and is referenced, not merged.

Design goal: one file per package carries everything a fresh board needs —
stage ABIs, engine identity, loop shape, gates, runtime options, and the
acceptance lock. Nothing model-specific in the C++ runner.

## File layout

```text
<package>/
  deployment_manifest.json   # this schema
  export_manifest.json       # exporter output, unchanged (has contract/rtc/provenance)
  engines/
    prefix_embed.engine
    prefix_lm.engine
    suffix_step.engine
  adapter.py                 # host-side mapping (unnorm / action maps / RTC pin)
```

## Schema

```jsonc
{
  "schema": "picpp.deployment-manifest.v1",

  "model": {
    "family": "pi06",                 // pi05 | pi06 | fastwam | smolvla | ...
    "variant": "rtc",                 // airbot | heterogeneous | rtc | ...
    "display_name": "pi06_cloth_v3_v4_hang_v1_120000_rtc_848",
    "runtime": "pi06_offline"         // the runner front-end that can execute it
  },

  // The exporter manifest, identity-locked. Its contract/rtc/precision/
  // provenance sections are authoritative for ABI semantics.
  "export_manifest": {
    "path": "export_manifest.json",
    "sha256": "<hex>"
  },

  // Engine identity + build recipe provenance. `recipe` is how the engine
  // must be rebuilt; `size_bytes` is the sentinel that catches the silent
  // fp32-pattern fallback (the workspace trap) at verify time.
  "engines": {
    "prefix_embed": {
      "path": "engines/prefix_embed.engine",
      "size_bytes": 1889327756,
      "sha256": "<hex>",
      "recipe": {
        "tool": "trtexec",
        "profile": "fp16",            // fp16 | bf16 | hybrid | qdq | ...
        "workspace_gb": 4,            // the 4 GB recipe is mandatory for fp16/bf16
        "builder_optimization_level": 3,
        "skip_inference": true
      }
    },
    "prefix_lm":   { "path": "...", "size_bytes": 3749172284, "sha256": "...",
                     "recipe": { "tool": "trtexec", "profile": "bf16", "workspace_gb": 4,
                                 "builder_optimization_level": 3, "skip_inference": true } },
    "suffix_step": { "path": "...", "size_bytes": 868300540, "sha256": "...",
                     "recipe": { "tool": "python", "profile": "hybrid",
                                 "keep_patterns": ["/mlp/"],  // bf16 islands; everything else fp32
                                 "workspace_gb": 4 } }
  },

  // The loop shape is data, not code. The generic runner executes any
  // host-denoise loop from these fields.
  "loop": {
    "kind": "host_denoise",
    "step_stage": "suffix_step",
    "steps": 10,
    "graph_capture": "auto"           // off | auto (capture after first warmup)
  },

  // Runtime options: fail-closed semantics come from here.
  "runtime": {
    "adapter": "pi_cpp.pi06_rtc",     // host mapping module (or a bundled adapter.py)
    "rtc": {
      "enabled": true,
      "max_delay": 4,
      "pin_after_last_step": true,    // PinRtcActionPrefix on the final action
      "require_training_time_rtc": true   // fail if export_manifest.rtc.training_time_rtc != true
    },
    "device": ["orin_agx"]            // allowed boards; verify refuses others
  },

  // Acceptance gates. picpp verify reads these; a package with no `lock`
  // section is a candidate, not an accepted artifact.
  "gates": {
    "parity": {
      "rel_l2": 0.02,
      "cosine": 0.9998,
      "max_abs_scaled": 0.2,          // per-dim quantile-scaled, worst case
      "raw": 0.1                      // picpp vs direct_tensorrt raw max-abs
    },
    "latency": {
      "protocol": "d10",              // MAXN + performance governor + jetson_clocks
      "warmup": 10,
      "runs": 100,
      "budget_p50_ms": 320            // informational budget; the anchor is recorded in lock
    }
  },

  // Written by `picpp package lock` after a green verify; frozen afterwards.
  "lock": {
    "sha256": "<package hash>",
    "accepted_at": "2026-09-13",
    "accepted_d10_p50_ms": 265.6,
    "parity": { "rel_l2": 0.0064936, "cosine": 0.9999789, "max_abs_scaled": 0.1002219, "raw": 0.0 },
    "evidence": {
      "ladder": "results/triplet_report.json",
      "d10": "results_lat/latency/picpp/latency.json"
    }
  }
}
```

## Rules

1. **Export vs deployment manifest stay split.** The exporter owns the ABI
   semantics and provenance; pi.cpp owns engine identity, loop execution,
   gates, and the lock. `picpp verify` cross-checks the locked sha256 of
   the export manifest.
2. **Fail-closed by default.** `rtc.require_training_time_rtc` (and any
   future precondition flags) refuse to run when the export manifest
   contradicts the request; a missing gate or lock makes a package a
   candidate (verifyable, not deployable).
3. **Engine size sentinels.** `verify` compares `size_bytes` (within a
   tolerance band) and warns on the fp32-pattern signature — this encodes
   the builder-workspace trap as a check instead of a memory.
4. **Lock is append-only.** Changing engines, gates, or the adapter
   invalidates the lock (sha256 over the package minus `lock.evidence`).
5. **No weights in the manifest.** Only identity (size + sha256) and
   recipe provenance; the bytes stay device-sealed.
6. **Extension discipline.** New optional sections (e.g. `closed_loop`
   suites) must be additive and default to fail-closed when absent.

## Open questions for review

- Merge `gates` into a shared defaults table (per-family) with per-package
  overrides, or keep every gate explicit per package?
- `adapter`: register by python module name (current) vs bundle a
  `adapter.py` file in the package? (Bundle is safer for sealed artifacts.)
- Should `lock` pin the engine sha256 individually (it does via the
  package hash) or also keep a per-engine table for faster invalidation?
