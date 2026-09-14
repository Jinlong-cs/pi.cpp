# picpp agent contract

How a coding agent (or any script) drives the pi.cpp VLA toolchain. Everything
here is a stable interface: JSON on stdout, diagnostics on stderr, exit codes
with fixed meanings. Command help (`picpp <cmd> --help`) is authoritative for
flags; this document is authoritative for the interpretation of outputs.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success; the stdout JSON carries the result. |
| 1 | The command ran and the *result failed its gates* (verify parity/latency checks, package integrity), or a usage/contract error was raised. |
| 2 | Unauthorized: the package's `authorization.required` is true and the closed loop was started without `--authorize`. Nothing was served. |

Crashes (Python exceptions) also exit nonzero and print a rich traceback to
**stderr**; treat any nonzero exit as a failure of the step and read the
report/error text before deciding the next step.

## Output contract

- Every one-shot command writes a single JSON object to **stdout**. The
  `verify` modes write to `--out` when it is set (stdout stays empty then);
  `picpp package export` writes the archive to `--out` (default
  `./<display_name>.tar.gz`) and the report to stdout.
- `eval --progress` and `infer`/`latency` stream one JSON object per line to
  stdout; each line parses standalone, and the final line is the summary.
- Progress, logs, warnings, and tracebacks go to **stderr**. Agents may show
  stderr to the user but must parse stdout only.
- Timings are in milliseconds; surfaces are [action_horizon, action_dim]
  float32; all numeric fields are plain JSON numbers.

## The chain

The acceptance flow an agent runs end to end:

```
picpp package install --package <dir-or-tar.gz> --root ~/.picpp/packages   # exit 0/1
picpp package verify  --package <installed-dir>                            # integrity + lock
picpp verify parity  --package <pkg> --golden <golden-root> \
        --norm-stats <norm-stats.json> --contract <model_contract.json> \
        --reference <direct-trt-surfaces> --out <report.json>              # exit 0/1
picpp verify latency --package <pkg> --golden <golden-root> \
        --out <latency.json>                                               # exit 0/1
picpp eval --model <m> --dataset <ds> --progress                            # task metrics
```

- **verify parity** compares the package engines against the JAX golden
  (per-dim quantile-scaled triplet) and the saved direct-TensorRT reference
  surfaces (raw max-abs), then evaluates the manifest gates (family defaults
  merged with the package's per-package overrides). The report's
  `gates.checks.<key>` entries carry `observed`/`threshold`/`passed`; the
  command exits 0 only when every declared gate passes.
- **verify latency** runs the D10 protocol from the gates (warmup + measure
  runs) and reports p50/p95/mean; numeric bounds (`p95_ms_max` etc.) are
  checked only when declared, otherwise the run is report-only (exit 0).
- Golden payloads are sha256-checked against the golden manifest before use;
  a mismatch crashes the run (fail-fast).
- The gates live in the package's `deployment_manifest.json` (`gates` section,
  deep-merged over `FAMILY_GATES` defaults). A package without a deployment
  manifest uses the family defaults derived from its `export_manifest.json`.

## Package transport

`picpp package export` bundles a package directory into
`<display_name>.tar.gz`; `picpp package install` accepts either the directory
or that archive (the archive must contain exactly one package with a
`deployment_manifest.json` at its root). Install runs the integrity check
(engine sha256 + the package lock hash) before copying; a failed check exits 1
unless `--force` is passed.

## Authorization boundary

A package whose deployment manifest declares

```json
"authorization": { "required": true }
```

refuses to serve without an explicit grant: `picpp server ... --authorize`.
Starting it without the flag exits **2** before any policy is constructed.
Packages without the section have no requirement.

## Notes

- `picpp graph` / `calibrate` / `build` are the offline toolchain (graph
  rewrites, scale collection, engine build); they take `--out` paths and
  return 0 with the report at that path. `build` requires TensorRT and is
  board-side; `calibrate` requires onnxruntime.
- `picpp server`/`client` are the closed-loop pair; the client reads the
  server's metadata handshake (image size, camera order, request keys, action
  semantics) before running the loop.
- Long-running loops (server, client) exit nonzero on the first
  contract/connection violation — do not treat them as retry-safe without
  reading the error.
