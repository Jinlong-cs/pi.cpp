#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pi_cpp/core/tensor.hpp"

namespace pi_cpp {

struct HostTensor {
  std::string name;
  TensorShape shape;
  DType dtype = DType::kFloat32;
  std::vector<std::uint8_t> data;

  [[nodiscard]] std::size_t NumBytes() const {
    return static_cast<std::size_t>(shape.NumElements()) * ByteSize(dtype);
  }
};

}  // namespace pi_cpp
