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

> **Executed 2026-09-14** — see the log below; the doc order was adjusted:
> `fix/trt-fast-engine-read` originally branched from the SDK commit
> `a1b1571`, so the SDK merged first to keep the fix PR to its single
> 7-line change.

```text
main ── feat/v0.2-optimization-sdk          (1) PR #1 — pure python layer, self-contained
    ├─ fix/trt-fast-engine-read             (2) PR #3 — rebased to the 1-commit engine read fix
    ├─ feat/pi06-airbot-runtime             (3) PR #2
    │    └─ feat/pi06-heterogeneous-runtime (4) PR #4 — builds on airbot
    │         └─ feat/pi06-rtc-runtime      (5) PR #5 — builds on heterogeneous
```

The pi06 steps stack chronologically on shared files
(`src/runtime/pi05_offline_tensorrt.cpp`, `python/pi_cpp/cli.py`, the
per-model adapters), so rebase each onto main's merged head and resolve
once, in order. Rebase drops the duplicated predecessor commits (airbot's
commit is contained in heterogeneous, heterogeneous in rtc).

### Executed log (2026-09-14)

| # | Branch | PR | Merge commit | Tag |
|---|---|---|---|---|
| — | docs (architecture/schema/plan) | direct push | `9cea37e` | — |
| 1 | feat/v0.2-optimization-sdk | #1 (was DRAFT) | `6b7406d` | `merge-opt-sdk` |
| 2 | fix/trt-fast-engine-read | **#6** | `a34ba0a` | `merge-engine-read-fix` |
| 3 | feat/pi06-airbot-runtime | #2 | `1b751b2` | `merge-pi06-airbot` |
| 4 | feat/pi06-heterogeneous-runtime | #4 (new) | `e90a831` | `merge-pi06-heterogeneous` |
| 5 | feat/pi06-rtc-runtime | #5 (new) | `02ab124` | `merge-pi06-rtc` |
| — | test: tolerance-compare sparse-delta gate scales | direct push | `ee30abc` | — |
| — | merge-plan executed log | direct push | `6011d77` | — |

> **Incident (recorded):** PR #3 carried the fix but its base was
> `feat/v0.2-optimization-sdk` (the fix branch was originally cut from the
> SDK commit), so `gh pr merge 3` merged the fix into the SDK branch
> (`cc71bd5`), not main — discovered by checking `--contains` against
> origin/main after the "MERGED" state. Remediation: the polluted SDK
> branch was deleted, the fix re-submitted as PR #6 onto main, and the
> mis-tagged `merge-engine-read-fix` (on `6b7406d`) was moved to the real
> merge commit. Lesson: verify the PR's `baseRefName` (and the resulting
> main ancestry) before merging, not after.

The trap-fix history (`cudaGraphInstantiate` signature, timestep host
pointer, debug-print cleanup) survived the rebase as individual commits on
main — not squashed.

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
