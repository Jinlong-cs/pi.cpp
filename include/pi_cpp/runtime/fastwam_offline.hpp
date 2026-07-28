#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/fastwam_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

struct FastWamStageTimers {
  runtime::CudaEventTimer vae_image_encoder;
  runtime::CudaEventTimer video_prefill;
  runtime::CudaEventTimer action_loop;

  Status Init();
};

struct FastWamOfflineRequest {
  HostTensor input_image;
  HostTensor context;
  HostTensor context_mask;
  HostTensor latents_action;
  HostTensor timestep_action;
  std::vector<float> scheduler_deltas;
  std::vector<float> action_mean;
  std::vector<float> action_std;
  int action_dim = fastwam::kDefaultActionDim;
  int steps = fastwam::kDefaultInferenceSteps;
};

struct FastWamRunResult {
  double load_ms = 0.0;
  double infer_ms = 0.0;
  double vae_image_encoder_ms = 0.0;
  double video_prefill_ms = 0.0;
  double action_loop_ms = 0.0;
  double action_decode_ms = 0.0;
  int kv_direct = 0;
  int kv_cast = 0;
  std::vector<int64_t> action_shape;
  std::vector<float> action;
};

class FastWamOfflineRunner {
 public:
  explicit FastWamOfflineRunner(std::filesystem::path engine_dir);
  ~FastWamOfflineRunner();

  FastWamOfflineRunner(FastWamOfflineRunner&&) = delete;
  FastWamOfflineRunner& operator=(FastWamOfflineRunner&&) = delete;
  FastWamOfflineRunner(const FastWamOfflineRunner&) = delete;
  FastWamOfflineRunner& operator=(const FastWamOfflineRunner&) = delete;

  Status Load();
  Status RunOnce(const FastWamOfflineRequest& request, FastWamRunResult* result);

  [[nodiscard]] std::vector<TensorSpec> input_specs() const;
  [[nodiscard]] double load_ms() const;

 private:
  std::filesystem::path engine_dir_;
  double load_ms_ = 0.0;
  TrtEngine vae_image_encoder_;
  TrtEngine video_prefill_;
  TrtEngine action_step_;
  runtime::StageWorkspace vae_workspace_;
  runtime::StageWorkspace video_workspace_;
  runtime::StageWorkspace action_workspace_;
  runtime::StagePlan vae_plan_;
  runtime::StagePlan video_plan_;
  runtime::StagePlan action_plan_;
  FastWamStageTimers timers_;
  runtime::CudaStream stream_;
  runtime::DeviceTensor input_image_;
  runtime::DeviceTensor context_;
  runtime::DeviceTensor context_mask_;
  std::array<runtime::DeviceTensor, 2> latents_action_buffers_;
  runtime::DeviceTensor timestep_action_;
  runtime::DeviceTensor scheduler_delta_;
  std::size_t action_latents_input_index_ = 0;
  std::size_t action_latents_next_output_index_ = 0;
  int kv_direct_ = 0;
};

Status RunFastWamOnce(const std::filesystem::path& engine_dir,
                      const FastWamOfflineRequest& request,
                      FastWamRunResult* result);

}  // namespace pi_cpp
