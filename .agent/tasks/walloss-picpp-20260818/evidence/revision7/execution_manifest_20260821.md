# Revision 7 bounded execution manifest

Status: **bounded 5090 scope executed; AGX scope not executed**

This manifest records the exact command surface to review after the named
human approvals. It does not authorize execution by itself.

The human owner later approved only the 5090 fixture-recovery,
teacher-forced-prefill, same-host PyTorch ladder, and PT/ORT 2x2 portions.
Those portions completed on 2026-08-24 and rejected V18 numerical parity. See
`evidence/5090/revision7/root_cause_decision_20260824.md`. Candidate freeze,
AGX transfer/build/parity, Revision 8, latency, server/client, and closed-loop
work remain unexecuted.

## Artifact roots

```text
5090-zem: /home/zem/wujinlong/picpp_walloss_20260818/revision7_export
agx-test: /root/wujinlong/nvme/picpp_walloss_20260818/revision7_engine
```

Only these fresh roots may receive Revision 7 artifacts. Existing Rev5/Rev6,
Rev3, baseline, HF, verify, and accepted directories remain read-only.

## Frozen remote layout for the approved 5090 lane

The following absolute paths are the only paths referenced by the bounded
reference commands below. They are created only after the human execution gate
is approved:

```text
R7_ROOT=/home/zem/wujinlong/picpp_walloss_20260818/revision7_export
R7_SCRIPTS=$R7_ROOT/scripts
R7_INPUT=$R7_ROOT/input
CHECKPOINT=$R7_INPUT/checkpoint
V18_ROOT=$R7_INPUT/v18/onnx_real_legacy_v18
FIXTURE_CALIBRATION=$R7_ROOT/reference/calibration/fixture
FIXTURE_HELDOUT=$R7_ROOT/reference/heldout/fixture
```

The script bundle must be copied from the pinned worktree into
`$R7_SCRIPTS` and its hash recorded before use:

```text
recipes/walloss/agx_export_runner.py -> $R7_SCRIPTS/agx_export_runner.py
evidence/revision7/reference/fixture_integrity.py -> $R7_SCRIPTS/fixture_integrity.py
evidence/5090/revision7/teacher_forced_prefill.py -> $R7_SCRIPTS/teacher_forced_prefill.py
```

The commands must run with an explicit task-local `PYTHONPATH` containing
`$R7_INPUT/source/wall-x` and `$R7_INPUT/source/pi_cpp_walloss`; no system-wide
package installation or implicit current-directory import is allowed.

## Preconditions

```bash
hostname
date -Is
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
df -B1 /home/zem/wujinlong
```

The 5090 command set is allowed only when the GPU owner list is empty of
non-task work and the human `revision7-5090-execution-approval` records this
host, path, and command manifest.

AGX preconditions are read-only:

```bash
hostname
date -Is
trtexec --version
tegrastats --interval 1000
df -B1 /root/wujinlong/nvme /root
ps -eo user,pid,ppid,etime,args
```

The AGX build commands below are forbidden until the separate
`agx-revision7-build-approval` is approved against a frozen candidate hash.

## 5090 reference sequence

The following commands are run inside the task-local environment and use only
pinned source/checkpoint paths after their manifests have been verified:

```bash
  python "$R7_SCRIPTS/agx_export_runner.py" probe \
    --checkpoint "$CHECKPOINT" \
    --output "$R7_ROOT/reference/probe.json" \
    --device cuda

python "$R7_SCRIPTS/agx_export_runner.py" parity \
    --checkpoint "$CHECKPOINT" \
    --fixture "$FIXTURE_CALIBRATION" \
    --output "$R7_ROOT/reference/calibration/pytorch_parity.json" \
    --tensors "$R7_ROOT/reference/calibration/pytorch_tensors.pt" \
    --device cuda

python "$R7_SCRIPTS/fixture_integrity.py" \
    --fixture "$R7_ROOT/reference/calibration/pytorch_tensors.pt" \
    --require-trajectory \
    --output "$R7_ROOT/reference/calibration/fixture_integrity.json"

python "$R7_SCRIPTS/fixture_integrity.py" \
    --fixture "$R7_ROOT/reference/heldout/pytorch_tensors.pt" \
    --require-trajectory \
    --output "$R7_ROOT/reference/heldout/fixture_integrity.json"

python "$R7_SCRIPTS/teacher_forced_prefill.py" \
    --model "$V18_ROOT/prefill.onnx" \
    --fixture "$R7_ROOT/reference/calibration/pytorch_tensors.pt" \
    --output "$R7_ROOT/diagnostic/calibration_teacher_forced_prefill.json" \
    --profile-prefix "$R7_ROOT/diagnostic/calibration_prefill_profile"
```

The existing `agx_export_runner.py processor` command creates synthetic
diagnostic images and is not a held-out fixture command. A real held-out
capture must be supplied by the approved fixture-generation path before these
commands can pass the completeness gate.

## Candidate freeze

After the root-cause decision, retain v18 or create exactly one minimal v19.
Record the sorted relative-path, byte-size, and SHA256 tree and an aggregate
SHA256 in `candidate/candidate_manifest.json`. No AGX transfer is valid before
that file is immutable and independently rechecked.

## AGX parser/build sequence after approval

```bash
mkdir -p "$R7_AGX_ROOT"/{candidate,build,parity,logs}
sha256sum -c candidate/transfer_manifest.sha256

trtexec --onnx=candidate/postfix_step.onnx \
  --bf16 --skipInference --dumpLayerInfo \
  --saveEngine=build/postfix_step.engine \
  --timingCacheFile=build/postfix_step.timing.cache \
  --tempdir=build/postfix_step.tmp \
  --profilingVerbosity=detailed

trtexec --onnx=candidate/prefix_embed.onnx \
  --bf16 --skipInference --dumpLayerInfo \
  --saveEngine=build/prefix_embed.engine \
  --timingCacheFile=build/prefix_embed.timing.cache \
  --tempdir=build/prefix_embed.tmp \
  --profilingVerbosity=detailed

trtexec --onnx=candidate/prefill.onnx \
  --bf16 --skipInference --dumpLayerInfo \
  --saveEngine=build/prefill.engine \
  --timingCacheFile=build/prefill.timing.cache \
  --tempdir=build/prefill.tmp \
  --profilingVerbosity=detailed
```

These commands are a review surface, not an execution authorization. Exact
flags must be compared with the target-local `trtexec --help` output at the
approved build time; no guessed fallback is permitted.
