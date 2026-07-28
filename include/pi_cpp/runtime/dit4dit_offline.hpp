#pragma once

#include <filesystem>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/dit4dit_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct Dit4DitStageTimers {
  runtime::CudaEventTimer vae;
  runtime::CudaEventTimer feature;
  runtime::CudaEventTimer action_loop;

  Status Init();
};

struct Dit4DitOfflineRequest {
  HostTensor video_bcthw;
  HostTensor sample_noise;
  HostTensor latents;
  HostTensor cond_mask;
  HostTensor cond_indicator;
  HostTensor prompt_embeds;
  HostTensor padding_mask;
  HostTensor sigma_t;
  HostTensor state;
  HostTensor initial_actions;
};

struct Dit4DitRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double vae_ms = 0.0;
  double feature_ms = 0.0;
  double action_loop_ms = 0.0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class Dit4DitOfflineRunner {
 public:
  explicit Dit4DitOfflineRunner(std::filesystem::path engine_dir);
  ~Dit4DitOfflineRunner();

  Dit4DitOfflineRunner(Dit4DitOfflineRunner&&) = delete;
  Dit4DitOfflineRunner& operator=(Dit4DitOfflineRunner&&) = delete;
  Dit4DitOfflineRunner(const Dit4DitOfflineRunner&) = delete;
  Dit4DitOfflineRunner& operator=(const Dit4DitOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const Dit4DitOfflineRequest& request, Dit4DitRunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine vae_;
  TrtEngine feature_;
  TrtEngine action_loop_;
  runtime::StageWorkspace vae_workspace_;
  runtime::StageWorkspace feature_workspace_;
  runtime::StageWorkspace action_workspace_;
  runtime::StagePlan vae_plan_;
  runtime::StagePlan feature_plan_;
  runtime::StagePlan action_plan_;
  Dit4DitStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor video_bcthw_;
  runtime::DeviceTensor sample_noise_;
  runtime::DeviceTensor latents_;
  runtime::DeviceTensor cond_mask_;
  runtime::DeviceTensor cond_indicator_;
  runtime::DeviceTensor prompt_embeds_;
  runtime::DeviceTensor padding_mask_;
  runtime::DeviceTensor sigma_t_;
  runtime::DeviceTensor state_;
  runtime::DeviceTensor initial_actions_;
};

}  // namespace pi_cpp
