#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pi_cpp/core/host_tensor.hpp"
#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"
#include "pi_cpp/core/trt_engine.hpp"

namespace pi_cpp::runtime {

struct CudaFree {
  void operator()(void* ptr) const;
};

using DevicePtr = std::unique_ptr<void, CudaFree>;

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  Status Init();
  Status Synchronize() const;

  [[nodiscard]] cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

struct DeviceTensor {
  TensorSpec spec;
  DevicePtr data;

  [[nodiscard]] TensorView view() const { return {spec, data.get()}; }
};

class CudaEventTimer {
 public:
  CudaEventTimer() = default;
  ~CudaEventTimer();

  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;

  Status Init(std::string name);
  Status Start(cudaStream_t stream) const;
  Status Stop(cudaStream_t stream) const;
  Status ElapsedMs(double* elapsed_ms) const;

 private:
  std::string name_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

struct StageWorkspace {
  std::vector<std::unique_ptr<DeviceTensor>> output_storage;
  std::vector<TensorView> output_views;
  std::unordered_map<std::string, DeviceTensor*> named_outputs;
};

struct StagePlan {
  std::vector<TensorView> input_views;
  std::unordered_map<std::string, std::size_t> input_indices;
};

double ElapsedMs(std::chrono::steady_clock::time_point start);

Status CheckCuda(cudaError_t error, const std::string& message);
Status LoadEngine(TrtEngine* engine);
Status AllocateDeviceTensor(const TensorSpec& spec, DeviceTensor* tensor);
Status LoadHostToDevice(const HostTensor& host, const TensorSpec& expected, DeviceTensor* tensor, cudaStream_t stream);
Status CopyHostToDevice(const HostTensor& host, DeviceTensor* tensor, cudaStream_t stream);
Status SetFloat32Scalar(DeviceTensor* tensor, float value, cudaStream_t stream);
Status CopyDeviceToHost(const DeviceTensor& src, void* dst, std::size_t bytes, cudaStream_t stream);
Status CopyDeviceTensor(const DeviceTensor& src, DeviceTensor* dst, cudaStream_t stream);
Status PrepareStageWorkspace(const TrtEngine& engine, StageWorkspace* workspace);
Status PrepareStagePlan(const TrtEngine& engine, StagePlan* plan);
Status BindStageInput(StagePlan* plan, std::string_view name, const DeviceTensor& tensor);
Status FindStageInputIndex(const StagePlan& plan, std::string_view name, std::size_t* index);
Status RunStage(TrtEngine& engine, StagePlan* plan, StageWorkspace* workspace, cudaStream_t stream);

}  // namespace pi_cpp::runtime
