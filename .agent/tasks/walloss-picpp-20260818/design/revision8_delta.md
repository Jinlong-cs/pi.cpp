# Wall-OSS Revision 8: first-class pi.cpp runtime and deployment validation

Revision 8 is a conditional future draft. It cannot execute until Revision 7
engine feasibility passes and a separate Revision 8 design review is approved.

## Objective

Add the frozen Wall-OSS TensorRT candidate as an explicit pi.cpp backend:
C++ persistent runner, pybind and Python wrapper, CLI/eval/latency/policy/
server/client/TUI dispatch, AGX runtime validation, same-contract latency, and
the approved closed-loop ladder.

## Dependency and evidence boundaries

Revision 8 consumes a frozen Revision 7 engine and never rebuilds it. C++ build,
offline parity, standalone latency, server/client timing, closed-loop smoke,
Gate40, and final400 remain separate claims. A passing engine or fast standalone
call is not deployment acceptance.

The runtime contract follows the actual V18 ONNX ABI: `pixel_values` is FP32
`[2048,1176]`; the prefill embedding and all 72 KV tensors are BF16 for the
preferred candidate. “BF16 model” does not imply a BF16 image-patch input.

## Required comparison

The latency denominator is the same Wall-OSS pinned upstream BF16 model on the
same AGX, same inputs, same external `x0`, same ten-step schedule, and same
timing boundary. If that R0 cannot run, report only absolute candidate latency;
do not claim a speedup percentage.

## Promotion boundary

Formal latency thresholds must be frozen before observing candidate results.
Server/client and every closed-loop rung require their own human approval.
Promotion requires all blocking gates, no fallback, exact artifact hashes,
stable resident memory, and final human acceptance.
