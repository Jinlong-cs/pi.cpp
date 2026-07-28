#pragma once

#include <NvInfer.h>

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"

namespace pi_cpp {

class TrtEngine {
 public:
  TrtEngine() = default;
  explicit TrtEngine(std::filesystem::path path);
  ~TrtEngine();

  TrtEngine(TrtEngine&&) noexcept;
  TrtEngine& operator=(TrtEngine&&) noexcept;

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
  [[nodiscard]] const std::vector<TensorSpec>& inputs() const { return inputs_; }
  [[nodiscard]] const std::vector<TensorSpec>& outputs() const { return outputs_; }
  [[nodiscard]] bool loaded() const { return loaded_; }

  Status Load();
  Status Enqueue(std::span<const TensorView> inputs, std::span<TensorView> outputs, void* cuda_stream = nullptr);

 private:
  class NvLogger final : public nvinfer1::ILogger {
   public:
    void log(Severity severity, char const* message) noexcept override;
    [[nodiscard]] const std::string& last_message() const { return last_message_; }

   private:
    std::string last_message_;
  };

  template <typename T>
  struct TrtDestroy {
    void operator()(T* ptr) const { delete ptr; }
  };

  using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>>;
  using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>>;
  using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>>;

  std::filesystem::path path_;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
  bool loaded_ = false;
  std::unique_ptr<NvLogger> logger_;
  RuntimePtr runtime_;
  EnginePtr engine_;
  ContextPtr context_;
  std::vector<void*> bound_inputs_;
  std::vector<void*> bound_outputs_;
};

}  // namespace pi_cpp
