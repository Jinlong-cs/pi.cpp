# Revision 7 helper static review

Checked: 2026-08-24 CST

## Fixed issue

`fixture_integrity.py` previously used lexical sorting for trajectory keys. That
orders `x10` before `x2`, so a valid `x1..x10` fixture could never satisfy the
`--require-trajectory` gate. The helper now sorts the registered `x1..x10`
keys numerically and rejects extra/non-contract keys by requiring exact ordered
equality.

The bridge comparison also now reports key mismatches and makes missing bridge
tensors fail the `shared_exact` gate rather than silently comparing only the
intersection. Missing sections are represented in the report instead of being
dereferenced immediately.

## Local checks

```text
Revision 7 contract validator: PASS (6 work packages, 12 gates)
Revision 8 contract validator: PASS (7 work packages, 14 gates)
fixture_integrity.py py_compile: PASS
teacher_forced_prefill.py py_compile: PASS
numeric x1..x10 ordering AST check: PASS
```

No model, ONNX, remote, GPU, TensorRT, service, latency, or closed-loop action
was performed for this fix.
