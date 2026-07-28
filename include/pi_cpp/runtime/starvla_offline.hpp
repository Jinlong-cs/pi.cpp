#pragma once

#include <filesystem>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/starvla_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct StarVlaStageTimers {
  runtime::CudaEventTimer policy;

  Status Init();
};

struct StarVlaOfflineRequest {
  HostTensor input_ids;
  HostTensor attention_mask;
  HostTensor position_ids;
  HostTensor visual_select;
  HostTensor action_select;
  HostTensor pixel_values;
};

struct StarVlaRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double policy_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class StarVlaOfflineRunner {
 public:
  explicit StarVlaOfflineRunner(std::filesystem::path engine_dir);
  ~StarVlaOfflineRunner();

  StarVlaOfflineRunner(StarVlaOfflineRunner&&) = delete;
  StarVlaOfflineRunner& operator=(StarVlaOfflineRunner&&) = delete;
  StarVlaOfflineRunner(const StarVlaOfflineRunner&) = delete;
  StarVlaOfflineRunner& operator=(const StarVlaOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const StarVlaOfflineRequest& request, StarVlaRunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine policy_;
  runtime::StageWorkspace policy_workspace_;
  runtime::StagePlan policy_plan_;
  StarVlaStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor input_ids_;
  runtime::DeviceTensor attention_mask_;
  runtime::DeviceTensor position_ids_;
  runtime::DeviceTensor visual_select_;
  runtime::DeviceTensor action_select_;
  runtime::DeviceTensor pixel_values_;
};

}  // namespace pi_cpp
