# FastWAM AGX Optimization Details

This note records the FastWAM sweep behind the main README acceleration row.
The promoted route is the lowest-latency route that improved the accepted
success-preserving QDQ baseline.

Platform and contract:

| Item | Value |
| --- | --- |
| Target | NVIDIA Jetson AGX Orin |
| Runtime stack | TensorRT 10.3.0, CUDA 12.6 driver stack |
| Docker package | TensorRT 10.3.0.30-1+cuda12.5 |
| Output contract | `[32,7]` |
| Accuracy-aligned baseline | AGX `infer_ms=297.16`; LIBERO `390/400 = 97.50%` |

## Selected Routes

| Route | Operations | Runtime | Result |
| --- | --- | --- | --- |
| Success-preserving QDQ baseline | Video/action fixed-weight Linear/Gemm QDQ-INT8; VAE/context and five action boundary nodes stay FP16. | 10-step loop CUDA Graph | AGX `infer_ms=182.53`; LIBERO `375/400 = 93.75%` |
| Final accelerated route | QDQ baseline plus 8-step distillation and P1-12.5% action-FFN structured pruning QAT. | 8-step loop CUDA Graph with variable-step runtime fix | AGX `infer_ms=161.88`; LIBERO `379/400 = 94.75%` |
| Success/size route | QDQ baseline plus P1-25% action-FFN structured pruning QAT6000. | 10-step loop CUDA Graph | AGX `infer_ms=178.81-180.32`; LIBERO `381/400 = 95.25%` |

## Denoising Step Controls

Plain step reduction stayed above 90% without QAT, but QAT did not reliably
improve the step-only controls. The final accelerated route is therefore tied
to step reduction plus structural pruning, not step reduction alone.

| Route | Control result | Trained variant | Decision |
| --- | --- | --- | --- |
| 10-step | AGX `infer_ms=182.53`; LIBERO `375/400 = 93.75%` | P1-25% QAT6000: AGX `infer_ms=178.81-180.32`; LIBERO `381/400 = 95.25%` | Baseline success-preserving QDQ route; P1-25 is the success/size variant. |
| 8-step | AGX `infer_ms=163.35`; LIBERO `371/400 = 92.75%` | S1 8-step QAT-only: LIBERO `373/400 = 93.25%` | QAT-only gain is small. |
| 6-step | AGX `infer_ms=144.50`; LIBERO `371/400 = 92.75%` | QAT1000: AGX `infer_ms=145.48`, LIBERO `371/400 = 92.75%`; QAT4000: LIBERO `369/400 = 92.25%` | Fast speed tradeoff, but QAT did not recover more success. |
| 1-step | AGX `infer_ms=95.85`; LIBERO `369/400 = 92.25%` | Not promoted as the main route. | Useful speed/accuracy reference, not the selected README result. |

## What Actually Moved Latency

| Lever | Outcome |
| --- | --- |
| Fixed-weight Linear/Gemm QDQ-INT8 | Accepted for video/action heavy paths. |
| Sensitive FP16 rollback | Required for VAE/context and five action boundary nodes: `time_embedding.0`, `time_embedding.2`, `time_projection.1`, `action_encoder`, and `head`. |
| CUDA Graph action loop | Accepted after exact same-input parity: `max_abs=0`, `rms=0`, gripper flip `0/32`. |
| Variable-step loop graph | Required for 8-step runtime capture; steady-state final route reached `infer_ms=161.88`. |
| Structural QAT | Useful when it produces deployable graph reduction: S1 8-step distillation plus P1-12.5% FFN pruning. |

## Rejected Routes

| Route | Result |
| --- | --- |
| Dynamic attention QDQ | Slower: action attention `13.8731 -> 18.2080 ms`; video attention `46.8706 -> 53.5606 ms`. |
| VAE QDQ | Neutral, slower, or build-risky; VAE stays FP16. |
| FFN plugins | FP16/cuBLAS plugin regressed full runtime `219.2464 -> 225.5867 ms`; rowwise/static INT8 plugins did not beat TensorRT QDQ. |
| Boundary-node INT8 sweep | G1-G7 over the five boundary nodes did not produce stable end-to-end latency wins. |
| Boundary-only QAT for more QDQ | Trainable, but not enough latency benefit and broad boundary-QDQ closed eval reached only `351/400 = 87.75%`. |
| Longer 6-step QAT | QAT4000 did not beat the no-QAT/QAT1000 `371/400` result and dropped to `369/400`. |

## Takeaway

For FastWAM on AGX, the reliable route is not "make everything INT8." The
winning stack is measured QDQ-INT8 on fixed-weight Linear/Gemm islands, native
precision guards for fragile paths, CUDA Graph scheduling for the repeated
action loop, and QAT only when it enables a smaller deployed graph.
