#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
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
  std::size_t suffix_x_t_input_index_ = 0;
  std::size_t suffix_x_t_next_output_index_ = 0;
};

}  // namespace pi_cpp
