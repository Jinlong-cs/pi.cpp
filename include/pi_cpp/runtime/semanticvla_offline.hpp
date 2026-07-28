#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/semanticvla_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct SemanticVlaStageTimers {
  runtime::CudaEventTimer backbone;
  runtime::CudaEventTimer action_loop;

  Status Init();
};

struct SemanticVlaOfflineRequest {
  HostTensor input_ids;
  HostTensor attention_mask;
  HostTensor position_ids;
  HostTensor visual_select;
  HostTensor pixel_values;
  HostTensor actions;
  int steps = semanticvla::kDefaultInferenceSteps;
};

struct SemanticVlaRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double backbone_ms = 0.0;
  double action_loop_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class SemanticVlaOfflineRunner {
 public:
  explicit SemanticVlaOfflineRunner(std::filesystem::path engine_dir);
  ~SemanticVlaOfflineRunner();

  SemanticVlaOfflineRunner(SemanticVlaOfflineRunner&&) = delete;
  SemanticVlaOfflineRunner& operator=(SemanticVlaOfflineRunner&&) = delete;
  SemanticVlaOfflineRunner(const SemanticVlaOfflineRunner&) = delete;
  SemanticVlaOfflineRunner& operator=(const SemanticVlaOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const SemanticVlaOfflineRequest& request, SemanticVlaRunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine backbone_;
  TrtEngine action_step_;
  runtime::StageWorkspace backbone_workspace_;
  runtime::StageWorkspace action_workspace_;
  runtime::StagePlan backbone_plan_;
  runtime::StagePlan action_plan_;
  SemanticVlaStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor input_ids_;
  runtime::DeviceTensor attention_mask_;
  runtime::DeviceTensor position_ids_;
  runtime::DeviceTensor visual_select_;
  runtime::DeviceTensor pixel_values_;
  std::array<runtime::DeviceTensor, 2> action_buffers_;
  runtime::DeviceTensor timestep_;
  std::size_t action_input_index_ = 0;
  std::size_t action_output_index_ = 0;
};

}  // namespace pi_cpp
