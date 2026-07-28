#include "pi_cpp/core/trt_engine.hpp"

#include <NvInfer.h>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace pi_cpp {
namespace {

Status ReadBinary(const std::filesystem::path& path, std::vector<char>* bytes) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return Status::NotFound("TensorRT engine not found: " + path.string());
  stream.unsetf(std::ios::skipws);
  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  bytes->reserve(static_cast<std::size_t>(size));
  bytes->insert(bytes->begin(), std::istream_iterator<char>(stream), std::istream_iterator<char>());
  return Status::Ok();
}

DType ConvertDataType(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
      return DType::kFloat16;
    case nvinfer1::DataType::kBF16:
      return DType::kBFloat16;
    case nvinfer1::DataType::kINT32:
      return DType::kInt32;
    case nvinfer1::DataType::kINT64:
      return DType::kInt64;
    case nvinfer1::DataType::kBOOL:
      return DType::kBool;
    default:
      return DType::kFloat32;
  }
}

TensorShape ConvertShape(nvinfer1::Dims dims) {
  TensorShape shape;
  shape.dims.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int index = 0; index < dims.nbDims; ++index) {
    shape.dims.push_back(dims.d[index]);
  }
  return shape;
}
}  // namespace

void TrtEngine::NvLogger::log(Severity severity, char const* message) noexcept {
  if (severity > Severity::kWARNING) return;
  last_message_ = message == nullptr ? "" : message;
}

Status TrtEngine::Load() {
  if (path_.empty()) return Status::InvalidArgument("TensorRT engine path is empty");

  std::vector<char> bytes;
  RETURN_IF_ERROR(ReadBinary(path_, &bytes));

  logger_ = std::make_unique<NvLogger>();
  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) return Status::RuntimeError("failed to create TensorRT runtime");

  engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!engine_) {
    const auto message = logger_->last_message().empty() ? "" : ": " + logger_->last_message();
    return Status::RuntimeError("failed to deserialize TensorRT engine: " + path_.string() + message);
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) return Status::RuntimeError("failed to create TensorRT execution context: " + path_.string());

  inputs_.clear();
  outputs_.clear();
  for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
    const char* name = engine_->getIOTensorName(index);
    TensorSpec spec;
    spec.name = name == nullptr ? "" : name;
    spec.dtype = ConvertDataType(engine_->getTensorDataType(name));
    spec.shape = ConvertShape(engine_->getTensorShape(name));
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      inputs_.push_back(std::move(spec));
    } else {
      outputs_.push_back(std::move(spec));
    }
  }

  bound_inputs_.assign(inputs_.size(), nullptr);
  bound_outputs_.assign(outputs_.size(), nullptr);
  loaded_ = true;
  return Status::Ok();
}

Status TrtEngine::Enqueue(std::span<const TensorView> inputs, std::span<TensorView> outputs, void* cuda_stream) {
  if (!loaded_ || !context_) return Status::RuntimeError("TensorRT engine is not loaded");
  if (inputs.size() != inputs_.size()) return Status::InvalidArgument("input tensor count does not match engine contract");
  if (outputs.size() != outputs_.size()) {
    return Status::InvalidArgument("output tensor count does not match engine contract");
  }

  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto& expected = inputs_[index];
    const auto& actual = inputs[index];
    if (actual.data == nullptr) return Status::InvalidArgument("input tensor data is null: " + expected.name);
    if (bound_inputs_[index] == actual.data) continue;
    if (!context_->setTensorAddress(expected.name.c_str(), actual.data)) {
      return Status::RuntimeError("failed to bind input tensor: " + expected.name);
    }
    bound_inputs_[index] = actual.data;
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const auto& expected = outputs_[index];
    const auto& actual = outputs[index];
    if (actual.data == nullptr) return Status::InvalidArgument("output tensor data is null: " + expected.name);
    if (bound_outputs_[index] == actual.data) continue;
    if (!context_->setTensorAddress(expected.name.c_str(), actual.data)) {
      return Status::RuntimeError("failed to bind output tensor: " + expected.name);
    }
    bound_outputs_[index] = actual.data;
  }

  if (!context_->enqueueV3(static_cast<cudaStream_t>(cuda_stream))) {
    return Status::RuntimeError("TensorRT enqueueV3 failed: " + path_.string());
  }
  return Status::Ok();
}
}  // namespace pi_cpp
