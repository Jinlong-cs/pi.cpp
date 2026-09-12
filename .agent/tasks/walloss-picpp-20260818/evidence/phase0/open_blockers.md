# Open Phase 0 blockers

These are explicit blockers, not failures hidden behind fallback paths.

1. **No local checkpoint hash yet.** The Hub revision and LFS metadata are frozen, but `model.safetensors` (8,331,486,672 bytes) has not been downloaded, so a local SHA256 and safetensors tensor inventory are pending.
2. **No Wall-X reference environment.** The Mac has no Wall-X checkout and the AGX system Python lacks the pinned PyTorch/Transformers stack. An isolated task-owned environment is needed before reference execution or export.
3. **External initial-noise seam is unresolved.** The pinned source creates `torch.randn` inside both flow and DLLM generation paths. A deterministic exporter/oracle seam must be designed and validated without changing the production stochastic semantics.
4. **TensorRT stage split is unresolved.** The upstream model uses MoE token routing, custom CUDA rope/permute/unpermute, SDPA/FlashAttention, prefix KV caching, and a ten-step Euler flow loop. Which pieces can be represented by TensorRT and which remain a custom plugin must be proven, not assumed.
5. **Production fallback policy is unresolved.** Wall-X's operator proxy and CUDA graph wrapper can fall back to PyTorch/naive execution; pi.cpp must expose the selected backend and fail fast when the target backend is unavailable.
6. **Baseline Wall-OSS deployment evidence does not exist.** There is no accepted Wall-OSS `picpp latency`, server/client timing, or closed-loop result to compare against. Existing PI0.5/FastWAM baselines are not Wall-OSS baselines.
7. **AGX build resources are constrained.** Rootfs has 3.7G free and NVMe 166G free; build caches, checkpoint staging, engines, and logs must be bounded and redirected to NVMe.
8. **Profiler coverage is limited.** `tegrastats` is available, but Nsight Compute/Systems are not in PATH. Any latency report must state the profiler limitation and cannot invent bandwidth or occupancy evidence.
9. **No server/closed-loop authorization is active.** Starting a persistent Wall-OSS service, client smoke, Gate40, or final400 requires its separate acceptance gate and exact fresh port/process/episode manifest.
