#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

#include <utility>

namespace pi_cpp::runtime {

void CudaFree::operator()(void* ptr) const {
  if (ptr != nullptr) cudaFree(ptr);
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

Status CudaStream::Init() {
  return CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
}

Status CudaStream::Synchronize() const {
  return CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");
}

CudaEventTimer::~CudaEventTimer() {
  if (start_ != nullptr) cudaEventDestroy(start_);
  if (stop_ != nullptr) cudaEventDestroy(stop_);
}

Status CudaEventTimer::Init(std::string name) {
  name_ = std::move(name);
  RETURN_IF_ERROR(CheckCuda(cudaEventCreate(&start_), "cudaEventCreate failed for " + name_ + " start"));
  return CheckCuda(cudaEventCreate(&stop_), "cudaEventCreate failed for " + name_ + " stop");
}

Status CudaEventTimer::Start(cudaStream_t stream) const {
  return CheckCuda(cudaEventRecord(start_, stream), "cudaEventRecord start failed for " + name_);
}

Status CudaEventTimer::Stop(cudaStream_t stream) const {
  return CheckCuda(cudaEventRecord(stop_, stream), "cudaEventRecord stop failed for " + name_);
}

Status CudaEventTimer::ElapsedMs(double* elapsed_ms) const {
  float value = 0.0F;
  RETURN_IF_ERROR(CheckCuda(cudaEventElapsedTime(&value, start_, stop_), "cudaEventElapsedTime failed for " + name_));
  *elapsed_ms = static_cast<double>(value);
  return Status::Ok();
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

Status CheckCuda(cudaError_t error, const std::string& message) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::RuntimeError(message + ": " + cudaGetErrorString(error));
}

Status LoadEngine(TrtEngine* engine) {
  RETURN_IF_ERROR(engine->Load());
  if (engine->inputs().empty() || engine->outputs().empty()) {
    return Status::RuntimeError("TensorRT engine has an empty IO contract: " + engine->path().string());
  }
  return Status::Ok();
}

Status AllocateDeviceTensor(const TensorSpec& spec, DeviceTensor* tensor) {
  if (spec.NumBytes() == 0) return Status::InvalidArgument("tensor has zero bytes: " + spec.name);
  void* raw = nullptr;
  Status status = CheckCuda(cudaMalloc(&raw, spec.NumBytes()), "cudaMalloc failed for " + spec.name);
  if (!status.ok()) return status;
  status = CheckCuda(cudaMemset(raw, 0, spec.NumBytes()), "cudaMemset failed for " + spec.name);
  if (!status.ok()) {
    cudaFree(raw);
    return status;
  }
  tensor->spec = spec;
  tensor->data.reset(raw);
  return Status::Ok();
}

Status LoadHostToDevice(const HostTensor& host, const TensorSpec& expected, DeviceTensor* tensor, cudaStream_t stream) {
  if (host.dtype != expected.dtype) return Status::InvalidArgument("dtype mismatch for tensor: " + expected.name);
  if (host.data.size() != expected.NumBytes()) {
    return Status::InvalidArgument("byte size mismatch for tensor: " + expected.name);
  }
  RETURN_IF_ERROR(AllocateDeviceTensor(expected, tensor));
  return CheckCuda(cudaMemcpyAsync(tensor->data.get(), host.data.data(), host.data.size(), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy H2D failed for " + expected.name);
}

Status CopyHostToDevice(const HostTensor& host, DeviceTensor* tensor, cudaStream_t stream) {
  if (host.dtype != tensor->spec.dtype) return Status::InvalidArgument("dtype mismatch for tensor: " + tensor->spec.name);
  if (host.data.size() != tensor->spec.NumBytes()) {
    return Status::InvalidArgument("byte size mismatch for tensor: " + tensor->spec.name);
  }
  return CheckCuda(cudaMemcpyAsync(tensor->data.get(), host.data.data(), host.data.size(), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy H2D failed for " + tensor->spec.name);
}

Status SetFloat32Scalar(DeviceTensor* tensor, float value, cudaStream_t stream) {
  if (tensor->spec.dtype != DType::kFloat32 || tensor->spec.shape.NumElements() != 1) {
    return Status::InvalidArgument("expected float32 scalar tensor: " + tensor->spec.name);
  }
  return CheckCuda(cudaMemcpyAsync(tensor->data.get(), &value, sizeof(value), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpy scalar H2D failed for " + tensor->spec.name);
}

Status CopyDeviceToHost(const DeviceTensor& src, void* dst, std::size_t bytes, cudaStream_t stream) {
  if (bytes != src.spec.NumBytes()) return Status::InvalidArgument("D2H copy size mismatch for " + src.spec.name);
  return CheckCuda(cudaMemcpyAsync(dst, src.data.get(), bytes, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpy D2H failed for " + src.spec.name);
}

Status CopyDeviceTensor(const DeviceTensor& src, DeviceTensor* dst, cudaStream_t stream) {
  if (src.spec.NumBytes() != dst->spec.NumBytes()) {
    return Status::InvalidArgument("copy size mismatch: " + src.spec.name + " -> " + dst->spec.name);
  }
  return CheckCuda(cudaMemcpyAsync(dst->data.get(), src.data.get(), src.spec.NumBytes(), cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy D2D failed for " + src.spec.name);
}

Status PrepareStageWorkspace(const TrtEngine& engine, StageWorkspace* workspace) {
  workspace->output_storage.clear();
  workspace->output_views.clear();
  workspace->named_outputs.clear();
  workspace->output_storage.reserve(engine.outputs().size());
  workspace->output_views.reserve(engine.outputs().size());
  workspace->named_outputs.reserve(engine.outputs().size());

  for (const auto& spec : engine.outputs()) {
    auto output = std::make_unique<DeviceTensor>();
    RETURN_IF_ERROR(AllocateDeviceTensor(spec, output.get()));
    DeviceTensor* output_ptr = output.get();
    workspace->output_views.push_back(output_ptr->view());
    workspace->named_outputs[spec.name] = output_ptr;
    workspace->output_storage.push_back(std::move(output));
  }
  return Status::Ok();
}

Status PrepareStagePlan(const TrtEngine& engine, StagePlan* plan) {
  plan->input_views.clear();
  plan->input_indices.clear();
  plan->input_views.reserve(engine.inputs().size());
  plan->input_indices.reserve(engine.inputs().size());

  for (std::size_t index = 0; index < engine.inputs().size(); ++index) {
    const auto& spec = engine.inputs()[index];
    plan->input_views.push_back({spec, nullptr});
    plan->input_indices[spec.name] = index;
  }
  return Status::Ok();
}

Status FindStageInputIndex(const StagePlan& plan, std::string_view name, std::size_t* index) {
  auto it = plan.input_indices.find(std::string(name));
  if (it == plan.input_indices.end()) return Status::InvalidArgument("missing stage input: " + std::string(name));
  *index = it->second;
  return Status::Ok();
}

Status BindStageInput(StagePlan* plan, std::string_view name, const DeviceTensor& tensor) {
  std::size_t index = 0;
  RETURN_IF_ERROR(FindStageInputIndex(*plan, name, &index));
  plan->input_views[index] = tensor.view();
  return Status::Ok();
}

Status RunStage(TrtEngine& engine, StagePlan* plan, StageWorkspace* workspace, cudaStream_t stream) {
  if (plan->input_views.size() != engine.inputs().size()) {
    return Status::InvalidArgument("stage plan input count does not match engine contract");
  }
  if (workspace->output_views.size() != engine.outputs().size()) {
    return Status::InvalidArgument("stage workspace output count does not match engine contract");
  }
  return engine.Enqueue(plan->input_views, workspace->output_views, stream);
}

}  // namespace pi_cpp::runtime
