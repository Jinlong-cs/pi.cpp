#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace pi_cpp {

enum class DType {
  kBool,
  kInt32,
  kInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
};

inline constexpr std::size_t ByteSize(DType dtype) {
  switch (dtype) {
    case DType::kBool:
      return 1;
    case DType::kInt32:
    case DType::kFloat32:
      return 4;
    case DType::kInt64:
      return 8;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
  }
  return 0;
}

struct TensorShape {
  std::vector<int64_t> dims;

  [[nodiscard]] int64_t NumElements() const {
    if (dims.empty()) return 1;
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>{});
  }
};

struct TensorSpec {
  std::string name;
  TensorShape shape;
  DType dtype = DType::kFloat32;

  [[nodiscard]] std::size_t NumBytes() const {
    return static_cast<std::size_t>(shape.NumElements()) * ByteSize(dtype);
  }
};

inline const TensorSpec* FindTensorSpec(std::span<const TensorSpec> specs, std::string_view name) {
  for (const auto& spec : specs) {
    if (spec.name == name) return &spec;
  }
  return nullptr;
}

struct TensorView {
  TensorSpec spec;
  void* data = nullptr;
};

struct ConstTensorView {
  TensorSpec spec;
  const void* data = nullptr;
};

}  // namespace pi_cpp
