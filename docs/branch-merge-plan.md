# Phase 1 — Branch Convergence Plan (draft)

Status: **DRAFT v1** (2026-09-14). Goal: land the five experiment branches
on `main` as one installable bundle, mechanically — **no refactor during
the merge**. The generic-runner refactor (Phase 2) happens on mainline
afterwards.

## Branch inventory

| Branch | Content | Depends on |
|---|---|---|
| `fix/trt-fast-engine-read` | fast engine-file reader (PR #3) | main |
| `feat/pi06-airbot-runtime` | airbot 14-dim contract adapter + CLI | main |
| `feat/pi06-heterogeneous-runtime` | pi06 3-stage split runner + CLI + eval | airbot |
| `feat/pi06-rtc-runtime` | RTC ABI (delay/action_prefix + pin) + CUDA-graph loop replay | heterogeneous |
| `feat/v0.2-optimization-sdk` | picpp-opt subcommands + JSON recipes | main (parallel) |

Local-only codex task branches (`codex/pi05-umi-deploy-20260814`,
`codex/walloss-picpp-20260818`) are NOT merged; their outcomes are already
upstream or superseded.

## Merge order

```text
main ── fix/trt-fast-engine-read          (1) isolated file-touch, safest first
    └─ feat/pi06-airbot-runtime           (2)
         └─ feat/pi06-heterogeneous-runtime (3) builds on airbot
              └─ feat/pi06-rtc-runtime    (4) builds on heterogeneous
    └─ feat/v0.2-optimization-sdk         (5) independent python layer, last
```

Steps 2-4 stack chronologically on shared files
(`src/runtime/pi05_offline_tensorrt.cpp`, `python/pi_cpp/cli.py`, the
per-model adapters), so rebase each onto its parent's merged head and
resolve once, in order.

## Gate per merge

Each merge must be green before the next starts:

1. **Python layer:** importable package, existing tests pass
   (`tests/`), CLI entrypoints render help for every registered model.
2. **Board build (the authoritative gate):** the C++ runtime only builds
   against Jetson TRT/CUDA — rebuild the `.so` on the affected board with
   the documented `CMAKE_ARGS` recipe and rerun the affected model's
   ladder:
   - `fix/trt-fast-engine-read`: pi05 ladder on AGX/NX (the fix must not
     change engine load results — bytewise-identical outputs).
   - pi06 merges (2-4): the frozen 36-case parity ladder
     (`agx_harness.py --mode parity`) + a D10 spot-check; the RTC merge
     additionally re-runs the delay>0 cases and the closed-loop smoke.
   - SDK merge (5): the compact712 recipe regression on the pi05 package.
3. **Tag each merge point** (`merge-{branch}`) so any regression can
   roll back one step without unwinding the rest.

## Non-goals during Phase 1

- No refactor of pi06 out of `pi05_offline_tensorrt.cpp` (Phase 2).
- No porting of board-side build/quantize scripts into the repo
  (Phase 3 does that deliberately — port, don't copy).
- No manifest adoption in the runtime yet: Phase 1 ships the current
  adapter code as-is; the manifest contract lands in Phase 2.
- Keep the per-commit history (do not squash): the CUDA-graph trap-fix
  commits (`2be986a`, `edb1e89`, `24a1b08`, `ee17f94`) document a durable
  rule and must survive the merge.

## CI (minimal, Phase 1)

GitHub Actions: lint + import-check + unit tests on the python layer only
(the native runtime needs Jetson TRT/CUDA and is gated on the board, not
in CI). The board ladder remains the release gate and is run manually from
the deployment skill's protocol.

## Rollback

`git revert -m 1 <merge-commit>` + restore the tagged point; engine
packages are untouched by any of this (they are board-side artifacts, not
repo content).
