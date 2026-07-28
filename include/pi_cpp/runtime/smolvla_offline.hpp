#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/smolvla_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct SmolVlaStageTimers {
  runtime::CudaEventTimer prefix_embed;
  runtime::CudaEventTimer prefix_lm;
  runtime::CudaEventTimer suffix_loop;

  Status Init();
};

struct SmolVlaOfflineRequest {
  HostTensor image;
  HostTensor image_mask;
  HostTensor tokenized_prompt;
  HostTensor tokenized_prompt_mask;
  HostTensor state;
  HostTensor x_t;
  std::vector<float> action_mean;
  std::vector<float> action_std;
  int action_dim = smolvla::kDefaultActionDim;
};

struct SmolVlaRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double prefix_embed_ms = 0.0;
  double prefix_lm_ms = 0.0;
  double suffix_loop_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class SmolVlaOfflineRunner {
 public:
  explicit SmolVlaOfflineRunner(std::filesystem::path engine_dir);
  ~SmolVlaOfflineRunner();

  SmolVlaOfflineRunner(SmolVlaOfflineRunner&&) = delete;
  SmolVlaOfflineRunner& operator=(SmolVlaOfflineRunner&&) = delete;
  SmolVlaOfflineRunner(const SmolVlaOfflineRunner&) = delete;
  SmolVlaOfflineRunner& operator=(const SmolVlaOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const SmolVlaOfflineRequest& request, SmolVlaRunResult* result);

  [[nodiscard]] double load_ms() const;
  [[nodiscard]] std::vector<TensorSpec> input_specs() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine prefix_embed_;
  TrtEngine prefix_lm_;
  TrtEngine suffix_loop_;
  runtime::StageWorkspace prefix_embed_workspace_;
  runtime::StageWorkspace prefix_lm_workspace_;
  runtime::StageWorkspace suffix_loop_workspace_;
  runtime::StagePlan prefix_embed_plan_;
  runtime::StagePlan prefix_lm_plan_;
  runtime::StagePlan suffix_loop_plan_;
  SmolVlaStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor image_;
  runtime::DeviceTensor image_mask_;
  runtime::DeviceTensor tokenized_prompt_;
  runtime::DeviceTensor tokenized_prompt_mask_;
  runtime::DeviceTensor state_;
  runtime::DeviceTensor x_t_;
};

}  // namespace pi_cpp
