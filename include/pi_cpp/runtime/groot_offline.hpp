#pragma once

#include <filesystem>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/groot_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct GrootStageTimers {
  runtime::CudaEventTimer backbone;
  runtime::CudaEventTimer action_loop;

  Status Init();
};

struct GrootOfflineRequest {
  HostTensor input_ids;
  HostTensor attention_mask;
  HostTensor pixel_values_0;
  HostTensor pixel_values_1;
  HostTensor state;
  HostTensor embodiment_id;
  HostTensor initial_actions;
};

struct GrootRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double backbone_ms = 0.0;
  double action_loop_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class GrootOfflineRunner {
 public:
  explicit GrootOfflineRunner(std::filesystem::path engine_dir);
  ~GrootOfflineRunner();

  GrootOfflineRunner(GrootOfflineRunner&&) = delete;
  GrootOfflineRunner& operator=(GrootOfflineRunner&&) = delete;
  GrootOfflineRunner(const GrootOfflineRunner&) = delete;
  GrootOfflineRunner& operator=(const GrootOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const GrootOfflineRequest& request, GrootRunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine backbone_;
  TrtEngine action_loop_;
  runtime::StageWorkspace backbone_workspace_;
  runtime::StageWorkspace action_workspace_;
  runtime::StagePlan backbone_plan_;
  runtime::StagePlan action_plan_;
  GrootStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor input_ids_;
  runtime::DeviceTensor attention_mask_;
  runtime::DeviceTensor pixel_values_0_;
  runtime::DeviceTensor pixel_values_1_;
  runtime::DeviceTensor state_;
  runtime::DeviceTensor embodiment_id_;
  runtime::DeviceTensor initial_actions_;
};

}  // namespace pi_cpp
