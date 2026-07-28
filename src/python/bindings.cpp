#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pi_cpp/runtime/dit4dit_offline.hpp"
#include "pi_cpp/runtime/evo1_offline.hpp"
#include "pi_cpp/runtime/fastwam_offline.hpp"
#include "pi_cpp/runtime/groot_offline.hpp"
#include "pi_cpp/runtime/pi05_offline.hpp"
#include "pi_cpp/runtime/semanticvla_offline.hpp"
#include "pi_cpp/runtime/smolvla_offline.hpp"
#include "pi_cpp/runtime/starvla_offline.hpp"

namespace py = pybind11;

namespace {

bool IsCContiguous(const py::buffer_info& info) {
  py::ssize_t expected_stride = static_cast<py::ssize_t>(info.itemsize);
  for (py::ssize_t index = static_cast<py::ssize_t>(info.ndim) - 1; index >= 0; --index) {
    if (info.shape[static_cast<std::size_t>(index)] == 0) return true;
    if (info.strides[static_cast<std::size_t>(index)] != expected_stride) return false;
    expected_stride *= info.shape[static_cast<std::size_t>(index)];
  }
  return true;
}

pi_cpp::DType DTypeFromBuffer(const py::buffer_info& info) {
  if (info.format == py::format_descriptor<float>::format()) return pi_cpp::DType::kFloat32;
  if (info.format == py::format_descriptor<std::int32_t>::format()) return pi_cpp::DType::kInt32;
  if (info.format == py::format_descriptor<std::int64_t>::format()) return pi_cpp::DType::kInt64;
  if (info.format == py::format_descriptor<bool>::format()) return pi_cpp::DType::kBool;
  if (info.itemsize == 4 && (info.format == "i" || info.format == "l")) return pi_cpp::DType::kInt32;
  if (info.itemsize == 8 && (info.format == "q" || info.format == "l")) return pi_cpp::DType::kInt64;
  if (info.itemsize == 2) return pi_cpp::DType::kFloat16;
  throw std::invalid_argument("unsupported numpy dtype format for pi.cpp tensor: " + info.format);
}

pi_cpp::HostTensor HostTensorFromArray(const std::string& name, const py::array& array) {
  py::buffer_info info = array.request();
  if (!IsCContiguous(info)) throw std::invalid_argument("numpy array is not C-contiguous: " + name);

  pi_cpp::HostTensor tensor;
  tensor.name = name;
  tensor.dtype = DTypeFromBuffer(info);
  tensor.shape.dims.reserve(static_cast<std::size_t>(info.ndim));
  std::size_t bytes = static_cast<std::size_t>(info.itemsize);
  for (py::ssize_t dim : info.shape) {
    tensor.shape.dims.push_back(dim);
    bytes *= static_cast<std::size_t>(dim);
  }
  tensor.data.resize(bytes);
  std::memcpy(tensor.data.data(), info.ptr, bytes);
  return tensor;
}

py::array DictArray(const py::dict& tensors, const char* name) {
  if (!tensors.contains(name)) throw std::invalid_argument(std::string("missing tensor: ") + name);
  return py::cast<py::array>(tensors[py::str(name)]);
}

std::vector<float> FloatVectorFromArray(const py::array_t<float, py::array::c_style | py::array::forcecast>& array) {
  py::buffer_info info = array.request();
  const auto count = static_cast<std::size_t>(info.size);
  std::vector<float> values(count);
  std::memcpy(values.data(), info.ptr, count * sizeof(float));
  return values;
}

py::dict ResultToDict(const pi_cpp::Pi05RunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["prefix_embed_ms"] = result.prefix_embed_ms;
  out["prefix_lm_ms"] = result.prefix_lm_ms;
  out["suffix_loop_ms"] = result.suffix_loop_ms;
  return out;
}

py::dict InputShapesToDict(const pi_cpp::Pi05OfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::FastWamOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::GrootOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::SemanticVlaOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::Evo1OfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::SmolVlaOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::Dit4DitOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict InputShapesToDict(const pi_cpp::StarVlaOfflineRunner& runner) {
  py::dict out;
  for (const auto& spec : runner.input_specs()) {
    out[py::str(spec.name)] = spec.shape.dims;
  }
  return out;
}

py::dict ResultToDict(const pi_cpp::FastWamRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["vae_image_encoder_ms"] = result.vae_image_encoder_ms;
  out["video_prefill_ms"] = result.video_prefill_ms;
  out["action_loop_ms"] = result.action_loop_ms;
  out["action_decode_ms"] = result.action_decode_ms;
  out["kv_direct"] = result.kv_direct;
  out["kv_cast"] = result.kv_cast;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::GrootRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["backbone_ms"] = result.backbone_ms;
  out["action_loop_ms"] = result.action_loop_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::SemanticVlaRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["backbone_ms"] = result.backbone_ms;
  out["action_loop_ms"] = result.action_loop_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::Evo1RunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["backbone_ms"] = result.backbone_ms;
  out["action_loop_ms"] = result.action_loop_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::SmolVlaRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["prefix_embed_ms"] = result.prefix_embed_ms;
  out["prefix_lm_ms"] = result.prefix_lm_ms;
  out["suffix_loop_ms"] = result.suffix_loop_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::Dit4DitRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["vae_ms"] = result.vae_ms;
  out["feature_ms"] = result.feature_ms;
  out["action_loop_ms"] = result.action_loop_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

py::dict ResultToDict(const pi_cpp::StarVlaRunResult& result) {
  py::dict out;
  out["load_ms"] = result.load_ms;
  out["infer_ms"] = result.infer_ms;
  out["policy_ms"] = result.policy_ms;
  out["action_shape"] = result.action_shape;
  out["action_size"] = result.action.size();
  return out;
}

void ThrowIfError(const pi_cpp::Status& status) {
  if (!status.ok()) throw std::runtime_error(status.message());
}

pi_cpp::SmolVlaOfflineRequest SmolVlaRequestFromPy(
    const py::dict& tensors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& action_mean,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& action_std,
    int action_dim) {
  pi_cpp::SmolVlaOfflineRequest request;
  request.image = HostTensorFromArray("image", DictArray(tensors, "image"));
  request.image_mask = HostTensorFromArray("image_mask", DictArray(tensors, "image_mask"));
  request.tokenized_prompt = HostTensorFromArray("tokenized_prompt", DictArray(tensors, "tokenized_prompt"));
  request.tokenized_prompt_mask =
      HostTensorFromArray("tokenized_prompt_mask", DictArray(tensors, "tokenized_prompt_mask"));
  request.state = HostTensorFromArray("state", DictArray(tensors, "state"));
  request.x_t = HostTensorFromArray("x_t", DictArray(tensors, "x_t"));
  request.action_mean = FloatVectorFromArray(action_mean);
  request.action_std = FloatVectorFromArray(action_std);
  request.action_dim = action_dim;
  return request;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  py::class_<pi_cpp::Pi05RunResult>(m, "Pi05Result")
      .def_readonly("load_ms", &pi_cpp::Pi05RunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::Pi05RunResult::infer_ms)
      .def_readonly("prefix_embed_ms", &pi_cpp::Pi05RunResult::prefix_embed_ms)
      .def_readonly("prefix_lm_ms", &pi_cpp::Pi05RunResult::prefix_lm_ms)
      .def_readonly("suffix_loop_ms", &pi_cpp::Pi05RunResult::suffix_loop_ms)
      .def_readonly("action", &pi_cpp::Pi05RunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::Pi05RunResult&>(&ResultToDict));

  py::class_<pi_cpp::Pi05OfflineRunner>(m, "Pi05OfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::Pi05OfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::Pi05OfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::Pi05OfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::Pi05OfflineRunner& runner, const py::dict& tensors) {
        pi_cpp::Pi05OfflineRequest request;
        request.image = HostTensorFromArray("image", DictArray(tensors, "image"));
        request.image_mask = HostTensorFromArray("image_mask", DictArray(tensors, "image_mask"));
        request.tokenized_prompt =
            HostTensorFromArray("tokenized_prompt", DictArray(tensors, "tokenized_prompt"));
        request.tokenized_prompt_mask =
            HostTensorFromArray("tokenized_prompt_mask", DictArray(tensors, "tokenized_prompt_mask"));
        request.x_t = HostTensorFromArray("x_t", DictArray(tensors, "x_t"));
        pi_cpp::Pi05RunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::Dit4DitRunResult>(m, "Dit4DitResult")
      .def_readonly("load_ms", &pi_cpp::Dit4DitRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::Dit4DitRunResult::infer_ms)
      .def_readonly("vae_ms", &pi_cpp::Dit4DitRunResult::vae_ms)
      .def_readonly("feature_ms", &pi_cpp::Dit4DitRunResult::feature_ms)
      .def_readonly("action_loop_ms", &pi_cpp::Dit4DitRunResult::action_loop_ms)
      .def_readonly("action_shape", &pi_cpp::Dit4DitRunResult::action_shape)
      .def_readonly("action", &pi_cpp::Dit4DitRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::Dit4DitRunResult&>(&ResultToDict));

  py::class_<pi_cpp::Dit4DitOfflineRunner>(m, "Dit4DitOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::Dit4DitOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::Dit4DitOfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::Dit4DitOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::Dit4DitOfflineRunner& runner, const py::dict& tensors) {
        pi_cpp::Dit4DitOfflineRequest request;
        request.video_bcthw = HostTensorFromArray("video_bcthw", DictArray(tensors, "video_bcthw"));
        request.sample_noise = HostTensorFromArray("sample_noise", DictArray(tensors, "sample_noise"));
        request.latents = HostTensorFromArray("latents", DictArray(tensors, "latents"));
        request.cond_mask = HostTensorFromArray("cond_mask", DictArray(tensors, "cond_mask"));
        request.cond_indicator = HostTensorFromArray("cond_indicator", DictArray(tensors, "cond_indicator"));
        request.prompt_embeds = HostTensorFromArray("prompt_embeds", DictArray(tensors, "prompt_embeds"));
        request.padding_mask = HostTensorFromArray("padding_mask", DictArray(tensors, "padding_mask"));
        request.sigma_t = HostTensorFromArray("sigma_t", DictArray(tensors, "sigma_t"));
        request.state = HostTensorFromArray("state", DictArray(tensors, "state"));
        request.initial_actions = HostTensorFromArray("initial_actions", DictArray(tensors, "initial_actions"));
        pi_cpp::Dit4DitRunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::FastWamRunResult>(m, "FastWamResult")
      .def_readonly("load_ms", &pi_cpp::FastWamRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::FastWamRunResult::infer_ms)
      .def_readonly("vae_image_encoder_ms", &pi_cpp::FastWamRunResult::vae_image_encoder_ms)
      .def_readonly("video_prefill_ms", &pi_cpp::FastWamRunResult::video_prefill_ms)
      .def_readonly("action_loop_ms", &pi_cpp::FastWamRunResult::action_loop_ms)
      .def_readonly("action_decode_ms", &pi_cpp::FastWamRunResult::action_decode_ms)
      .def_readonly("kv_direct", &pi_cpp::FastWamRunResult::kv_direct)
      .def_readonly("kv_cast", &pi_cpp::FastWamRunResult::kv_cast)
      .def_readonly("action_shape", &pi_cpp::FastWamRunResult::action_shape)
      .def_readonly("action", &pi_cpp::FastWamRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::FastWamRunResult&>(&ResultToDict));

  py::class_<pi_cpp::FastWamOfflineRunner>(m, "FastWamOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::FastWamOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::FastWamOfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::FastWamOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::FastWamOfflineRunner& runner, const py::dict& tensors,
                          const py::array_t<float, py::array::c_style | py::array::forcecast>& scheduler_deltas,
                          const py::array_t<float, py::array::c_style | py::array::forcecast>& action_mean,
                          const py::array_t<float, py::array::c_style | py::array::forcecast>& action_std, int steps,
                          int action_dim) {
        pi_cpp::FastWamOfflineRequest request;
        request.input_image = HostTensorFromArray("input_image", DictArray(tensors, "input_image"));
        request.context = HostTensorFromArray("context", DictArray(tensors, "context"));
        request.context_mask = HostTensorFromArray("context_mask", DictArray(tensors, "context_mask"));
        request.latents_action = HostTensorFromArray("latents_action", DictArray(tensors, "latents_action"));
        request.timestep_action = HostTensorFromArray("timestep_action", DictArray(tensors, "timestep_action"));
        request.scheduler_deltas = FloatVectorFromArray(scheduler_deltas);
        request.action_mean = FloatVectorFromArray(action_mean);
        request.action_std = FloatVectorFromArray(action_std);
        request.steps = steps;
        request.action_dim = action_dim;
        pi_cpp::FastWamRunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::GrootRunResult>(m, "GrootResult")
      .def_readonly("load_ms", &pi_cpp::GrootRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::GrootRunResult::infer_ms)
      .def_readonly("backbone_ms", &pi_cpp::GrootRunResult::backbone_ms)
      .def_readonly("action_loop_ms", &pi_cpp::GrootRunResult::action_loop_ms)
      .def_readonly("action_shape", &pi_cpp::GrootRunResult::action_shape)
      .def_readonly("action", &pi_cpp::GrootRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::GrootRunResult&>(&ResultToDict));

  py::class_<pi_cpp::GrootOfflineRunner>(m, "GrootOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::GrootOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::GrootOfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::GrootOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::GrootOfflineRunner& runner, const py::dict& tensors) {
        pi_cpp::GrootOfflineRequest request;
        request.input_ids = HostTensorFromArray("input_ids", DictArray(tensors, "input_ids"));
        request.attention_mask = HostTensorFromArray("attention_mask", DictArray(tensors, "attention_mask"));
        request.pixel_values_0 = HostTensorFromArray("pixel_values_0", DictArray(tensors, "pixel_values_0"));
        request.pixel_values_1 = HostTensorFromArray("pixel_values_1", DictArray(tensors, "pixel_values_1"));
        request.state = HostTensorFromArray("state", DictArray(tensors, "state"));
        request.embodiment_id = HostTensorFromArray("embodiment_id", DictArray(tensors, "embodiment_id"));
        request.initial_actions = HostTensorFromArray("initial_actions", DictArray(tensors, "initial_actions"));
        pi_cpp::GrootRunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::SemanticVlaRunResult>(m, "SemanticVlaResult")
      .def_readonly("load_ms", &pi_cpp::SemanticVlaRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::SemanticVlaRunResult::infer_ms)
      .def_readonly("backbone_ms", &pi_cpp::SemanticVlaRunResult::backbone_ms)
      .def_readonly("action_loop_ms", &pi_cpp::SemanticVlaRunResult::action_loop_ms)
      .def_readonly("action_shape", &pi_cpp::SemanticVlaRunResult::action_shape)
      .def_readonly("action", &pi_cpp::SemanticVlaRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::SemanticVlaRunResult&>(&ResultToDict));

  py::class_<pi_cpp::SemanticVlaOfflineRunner>(m, "SemanticVlaOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::SemanticVlaOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::SemanticVlaOfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::SemanticVlaOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::SemanticVlaOfflineRunner& runner, const py::dict& tensors, int steps) {
        pi_cpp::SemanticVlaOfflineRequest request;
        request.input_ids = HostTensorFromArray("input_ids", DictArray(tensors, "input_ids"));
        request.attention_mask = HostTensorFromArray("attention_mask", DictArray(tensors, "attention_mask"));
        request.position_ids = HostTensorFromArray("position_ids", DictArray(tensors, "position_ids"));
        request.visual_select = HostTensorFromArray("visual_select", DictArray(tensors, "visual_select"));
        request.pixel_values = HostTensorFromArray("pixel_values", DictArray(tensors, "pixel_values"));
        request.actions = HostTensorFromArray("actions", DictArray(tensors, "actions"));
        request.steps = steps;
        pi_cpp::SemanticVlaRunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::Evo1RunResult>(m, "Evo1Result")
      .def_readonly("load_ms", &pi_cpp::Evo1RunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::Evo1RunResult::infer_ms)
      .def_readonly("backbone_ms", &pi_cpp::Evo1RunResult::backbone_ms)
      .def_readonly("action_loop_ms", &pi_cpp::Evo1RunResult::action_loop_ms)
      .def_readonly("action_shape", &pi_cpp::Evo1RunResult::action_shape)
      .def_readonly("action", &pi_cpp::Evo1RunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::Evo1RunResult&>(&ResultToDict));

  py::class_<pi_cpp::Evo1OfflineRunner>(m, "Evo1OfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::Evo1OfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::Evo1OfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::Evo1OfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::Evo1OfflineRunner& runner, const py::dict& tensors, int steps) {
        pi_cpp::Evo1OfflineRequest request;
        request.input_ids = HostTensorFromArray("input_ids", DictArray(tensors, "input_ids"));
        request.attention_mask = HostTensorFromArray("attention_mask", DictArray(tensors, "attention_mask"));
        request.attention_mask_2d =
            HostTensorFromArray("attention_mask_2d", DictArray(tensors, "attention_mask_2d"));
        request.position_ids = HostTensorFromArray("position_ids", DictArray(tensors, "position_ids"));
        request.pixel_values = HostTensorFromArray("pixel_values", DictArray(tensors, "pixel_values"));
        request.state = HostTensorFromArray("state", DictArray(tensors, "state"));
        request.actions = HostTensorFromArray("actions", DictArray(tensors, "actions"));
        request.steps = steps;
        pi_cpp::Evo1RunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::StarVlaRunResult>(m, "StarVlaResult")
      .def_readonly("load_ms", &pi_cpp::StarVlaRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::StarVlaRunResult::infer_ms)
      .def_readonly("policy_ms", &pi_cpp::StarVlaRunResult::policy_ms)
      .def_readonly("action_shape", &pi_cpp::StarVlaRunResult::action_shape)
      .def_readonly("action", &pi_cpp::StarVlaRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::StarVlaRunResult&>(&ResultToDict));

  py::class_<pi_cpp::StarVlaOfflineRunner>(m, "StarVlaOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::StarVlaOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::StarVlaOfflineRunner::load_ms)
      .def("input_shapes", py::overload_cast<const pi_cpp::StarVlaOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::StarVlaOfflineRunner& runner, const py::dict& tensors) {
        pi_cpp::StarVlaOfflineRequest request;
        request.input_ids = HostTensorFromArray("input_ids", DictArray(tensors, "input_ids"));
        request.attention_mask = HostTensorFromArray("attention_mask", DictArray(tensors, "attention_mask"));
        request.position_ids = HostTensorFromArray("position_ids", DictArray(tensors, "position_ids"));
        request.visual_select = HostTensorFromArray("visual_select", DictArray(tensors, "visual_select"));
        request.action_select = HostTensorFromArray("action_select", DictArray(tensors, "action_select"));
        request.pixel_values = HostTensorFromArray("pixel_values", DictArray(tensors, "pixel_values"));
        pi_cpp::StarVlaRunResult result;
        ThrowIfError(runner.RunOnce(request, &result));
        return result;
      });

  py::class_<pi_cpp::SmolVlaRunResult>(m, "SmolVlaResult")
      .def_readonly("load_ms", &pi_cpp::SmolVlaRunResult::load_ms)
      .def_readonly("infer_ms", &pi_cpp::SmolVlaRunResult::infer_ms)
      .def_readonly("prefix_embed_ms", &pi_cpp::SmolVlaRunResult::prefix_embed_ms)
      .def_readonly("prefix_lm_ms", &pi_cpp::SmolVlaRunResult::prefix_lm_ms)
      .def_readonly("suffix_loop_ms", &pi_cpp::SmolVlaRunResult::suffix_loop_ms)
      .def_readonly("action_shape", &pi_cpp::SmolVlaRunResult::action_shape)
      .def_readonly("action", &pi_cpp::SmolVlaRunResult::action)
      .def("to_dict", py::overload_cast<const pi_cpp::SmolVlaRunResult&>(&ResultToDict));

  py::class_<pi_cpp::SmolVlaOfflineRunner>(m, "SmolVlaOfflineRunner")
      .def(py::init<std::filesystem::path>())
      .def("load", [](pi_cpp::SmolVlaOfflineRunner& runner) {
        ThrowIfError(runner.Load());
      })
      .def_property_readonly("load_ms", &pi_cpp::SmolVlaOfflineRunner::load_ms)
      .def("input_shapes",
           py::overload_cast<const pi_cpp::SmolVlaOfflineRunner&>(&InputShapesToDict))
      .def("run_once", [](pi_cpp::SmolVlaOfflineRunner& runner, const py::dict& tensors,
                          const py::array_t<float, py::array::c_style | py::array::forcecast>& action_mean,
                          const py::array_t<float, py::array::c_style | py::array::forcecast>& action_std,
                          int action_dim) {
        pi_cpp::SmolVlaRunResult result;
        ThrowIfError(runner.RunOnce(SmolVlaRequestFromPy(tensors, action_mean, action_std, action_dim), &result));
        return result;
      });

}
