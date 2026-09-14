#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/pi05_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct Pi05StageTimers {
  runtime::CudaEventTimer prefix_embed;
  runtime::CudaEventTimer prefix_lm;
  runtime::CudaEventTimer suffix_loop;

  Status Init();
};

struct Pi05OfflineRequest {
  HostTensor image;
  HostTensor image_mask;
  HostTensor tokenized_prompt;
  HostTensor tokenized_prompt_mask;
  HostTensor x_t;
  HostTensor state;
  HostTensor embodiment_id;
  // RTC (real-time chunking): the executed action prefix and its length.
  // delay [1] int32, action_prefix [1, horizon, action_dim] float32 in the
  // same quantile-normalized space as x_t. Empty for non-RTC models.
  HostTensor delay;
  HostTensor action_prefix;
};

struct Pi05RunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double prefix_embed_ms = 0.0;
  double prefix_lm_ms = 0.0;
  double suffix_loop_ms = 0.0;
  std::vector<float> action;
};

class Pi05OfflineRunner {
 public:
  explicit Pi05OfflineRunner(std::filesystem::path engine_dir);
  ~Pi05OfflineRunner();

  Pi05OfflineRunner(Pi05OfflineRunner&&) = delete;
  Pi05OfflineRunner& operator=(Pi05OfflineRunner&&) = delete;
  Pi05OfflineRunner(const Pi05OfflineRunner&) = delete;
  Pi05OfflineRunner& operator=(const Pi05OfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const Pi05OfflineRequest& request, Pi05RunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine prefix_embed_;
  TrtEngine prefix_lm_;
  TrtEngine suffix_step_;
  runtime::StageWorkspace prefix_embed_workspace_;
  runtime::StageWorkspace prefix_lm_workspace_;
  runtime::StageWorkspace suffix_step_workspace_;
  runtime::StagePlan prefix_embed_plan_;
  runtime::StagePlan prefix_lm_plan_;
  runtime::StagePlan suffix_step_plan_;
  Pi05StageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor image_;
  runtime::DeviceTensor image_mask_;
  runtime::DeviceTensor tokenized_prompt_;
  runtime::DeviceTensor tokenized_prompt_mask_;
  std::array<runtime::DeviceTensor, 2> x_t_buffers_;
  runtime::DeviceTensor timestep_;
  runtime::DeviceTensor dt_;
  runtime::DeviceTensor state_;
  runtime::DeviceTensor embodiment_id_;
  runtime::DeviceTensor delay_;
  runtime::DeviceTensor action_prefix_;
  bool has_state_input_ = false;
  bool has_embodiment_input_ = false;
  bool has_delay_input_ = false;
  bool has_action_prefix_input_ = false;
  std::size_t suffix_x_t_input_index_ = 0;
  std::size_t suffix_x_t_next_output_index_ = 0;

  // CUDA-graph replay of the 10-step suffix loop. Captured once after the
  // first successful RunOnce (the warmup); stays on the eager path if the
  // capture fails. The per-step timestep values are held in stable host
  // storage so the H2D scalar memcpys bake constant values into the graph.
  bool suffix_graph_ready_ = false;
  cudaGraphExec_t suffix_graph_exec_ = nullptr;
  std::array<float, pi05::kDefaultDenoiseSteps> timestep_values_{};

  Status CaptureSuffixGraph();
};

}  // namespace pi_cpp
