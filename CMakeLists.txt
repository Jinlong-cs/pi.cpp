cmake_minimum_required(VERSION 3.22)
project(pi_cpp LANGUAGES CXX)

option(PI_CPP_BUILD_TESTS "Build lightweight tests" ON)
option(PI_CPP_BUILD_PYTHON "Build Python bindings" ON)

add_library(pi_cpp_runtime
  src/engine/trt_engine.cpp
)
add_library(pi_cpp::runtime ALIAS pi_cpp_runtime)
set_target_properties(pi_cpp_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(pi_cpp_runtime PUBLIC cxx_std_20)
target_include_directories(pi_cpp_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(pi_cpp_runtime PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Werror>>
)

set(TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT include directory")
set(CUDA_INCLUDE_DIR "" CACHE PATH "CUDA include directory")
set(TENSORRT_LIBRARY "" CACHE FILEPATH "TensorRT nvinfer library")
set(CUDA_RUNTIME_LIBRARY "" CACHE FILEPATH "CUDA runtime library")

find_path(PI_CPP_FOUND_TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    ${TENSORRT_INCLUDE_DIR}
    $ENV{TENSORRT_INCLUDE_DIR}
)
find_path(PI_CPP_FOUND_CUDA_INCLUDE_DIR cuda_runtime_api.h
  HINTS
    ${CUDA_INCLUDE_DIR}
    $ENV{CUDA_INCLUDE_DIR}
    /usr/local/cuda/include
)
if(TENSORRT_LIBRARY AND EXISTS "${TENSORRT_LIBRARY}")
  set(PI_CPP_FOUND_TENSORRT_LIBRARY "${TENSORRT_LIBRARY}")
else()
  find_library(PI_CPP_FOUND_TENSORRT_LIBRARY
    NAMES nvinfer
    HINTS
      ${TENSORRT_LIBRARY}
      $ENV{TENSORRT_LIBRARY}
  )
endif()
if(CUDA_RUNTIME_LIBRARY AND EXISTS "${CUDA_RUNTIME_LIBRARY}")
  set(PI_CPP_FOUND_CUDA_RUNTIME_LIBRARY "${CUDA_RUNTIME_LIBRARY}")
else()
  find_library(PI_CPP_FOUND_CUDA_RUNTIME_LIBRARY
    NAMES cudart
    HINTS
      ${CUDA_RUNTIME_LIBRARY}
      $ENV{CUDA_RUNTIME_LIBRARY}
      /usr/local/cuda/lib64
  )
endif()

if(NOT PI_CPP_FOUND_TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers were not found. Set TENSORRT_INCLUDE_DIR to the directory containing NvInfer.h.")
endif()
if(NOT PI_CPP_FOUND_CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "CUDA headers were not found. Set CUDA_INCLUDE_DIR to the directory containing cuda_runtime_api.h.")
endif()
if(NOT PI_CPP_FOUND_TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT nvinfer library was not found. Set TENSORRT_LIBRARY to libnvinfer.so.")
endif()
if(NOT PI_CPP_FOUND_CUDA_RUNTIME_LIBRARY)
  message(FATAL_ERROR "CUDA runtime library was not found. Set CUDA_RUNTIME_LIBRARY to libcudart.so.")
endif()

set(TENSORRT_INCLUDE_DIR ${PI_CPP_FOUND_TENSORRT_INCLUDE_DIR})
set(CUDA_INCLUDE_DIR ${PI_CPP_FOUND_CUDA_INCLUDE_DIR})
set(TENSORRT_LIBRARY ${PI_CPP_FOUND_TENSORRT_LIBRARY})
set(CUDA_RUNTIME_LIBRARY ${PI_CPP_FOUND_CUDA_RUNTIME_LIBRARY})

message(STATUS "pi.cpp TensorRT runtime: ON")
target_sources(pi_cpp_runtime PRIVATE
  src/engine/trt_engine_tensorrt.cpp
  src/runtime/dit4dit_offline_tensorrt.cpp
  src/runtime/evo1_offline_tensorrt.cpp
  src/runtime/fastwam_offline_tensorrt.cpp
  src/runtime/groot_offline_tensorrt.cpp
  src/runtime/pi05_offline_tensorrt.cpp
  src/runtime/semanticvla_offline_tensorrt.cpp
  src/runtime/smolvla_offline_tensorrt.cpp
  src/runtime/starvla_offline_tensorrt.cpp
  src/runtime/utils/tensorrt_runtime.cpp
)
target_include_directories(pi_cpp_runtime SYSTEM PUBLIC
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
)
target_link_libraries(pi_cpp_runtime PUBLIC ${TENSORRT_LIBRARY} ${CUDA_RUNTIME_LIBRARY})

if(PI_CPP_BUILD_PYTHON)
  get_filename_component(PI_CPP_CUDA_RUNTIME_DIR ${CUDA_RUNTIME_LIBRARY} DIRECTORY)
endif()

if(PI_CPP_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/python/bindings.cpp)
  target_link_libraries(_native PRIVATE pi_cpp::runtime)
  set_target_properties(_native PROPERTIES
    BUILD_RPATH "${PI_CPP_CUDA_RUNTIME_DIR}"
    INSTALL_RPATH "${PI_CPP_CUDA_RUNTIME_DIR}"
  )
  install(TARGETS _native LIBRARY DESTINATION pi_cpp)
endif()
