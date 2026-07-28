#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/evo1_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct Evo1StageTimers {
  runtime::CudaEventTimer backbone;
  runtime::CudaEventTimer action_loop;

  Status Init();
};

struct Evo1OfflineRequest {
  HostTensor input_ids;
  HostTensor attention_mask;
  HostTensor attention_mask_2d;
  HostTensor position_ids;
  HostTensor pixel_values;
  HostTensor state;
  HostTensor actions;
  int steps = evo1::kInferenceSteps;
};

struct Evo1RunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double backbone_ms = 0.0;
  double action_loop_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class Evo1OfflineRunner {
 public:
  explicit Evo1OfflineRunner(std::filesystem::path engine_dir);
  ~Evo1OfflineRunner();

  Evo1OfflineRunner(Evo1OfflineRunner&&) = delete;
  Evo1OfflineRunner& operator=(Evo1OfflineRunner&&) = delete;
  Evo1OfflineRunner(const Evo1OfflineRunner&) = delete;
  Evo1OfflineRunner& operator=(const Evo1OfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const Evo1OfflineRequest& request, Evo1RunResult* result);

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
  Evo1StageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor input_ids_;
  runtime::DeviceTensor attention_mask_;
  runtime::DeviceTensor attention_mask_2d_;
  runtime::DeviceTensor position_ids_;
  runtime::DeviceTensor pixel_values_;
  runtime::DeviceTensor state_;
  std::array<runtime::DeviceTensor, 2> action_buffers_;
  runtime::DeviceTensor timestep_;
  std::size_t action_input_index_ = 0;
  std::size_t action_output_index_ = 0;
};

}  // namespace pi_cpp
