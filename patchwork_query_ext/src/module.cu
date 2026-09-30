#include <torch/custom_class.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include "kd_query.cuh"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace patchwork_query_ext {

struct KDTreeWrapper : torch::CustomClassHolder {
  std::variant<std::monostate, std::unique_ptr<KDTree<float, float3>>,
               std::unique_ptr<KDTree<float, float4>>>
      tree_;

  KDTreeWrapper(const torch::Tensor &A) {
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(A.size(0) > 2);
    TORCH_CHECK(A.size(-1) == 3 || A.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(A.device().type() == at::DeviceType::CUDA);
    const at::cuda::CUDAGuard device_guard{A.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (A.size(-1)) {
    case 3:
      tree_ = std::make_unique<KDTree<float, float3>>(stream, A.contiguous());
      break;
    case 4:
      tree_ = std::make_unique<KDTree<float, float4>>(stream, A.contiguous());
      break;
    }
  }

  torch::Tensor QueryClosest(const torch::Tensor &X) {
    TORCH_CHECK(X.is_cuda());
    TORCH_CHECK(X.size(-1) == 3 || X.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (X.size(-1)) {
    case 3:
      return std::get<std::unique_ptr<KDTree<float, float3>>>(tree_)
          ->QueryClosest(stream, X);
      break;
    case 4:
      return std::get<std::unique_ptr<KDTree<float, float4>>>(tree_)
          ->QueryClosest(stream, X);
      break;
    }
  }

  std::vector<torch::Tensor> QueryKNN2(const torch::Tensor &X) {
    TORCH_CHECK(X.is_cuda());
    TORCH_CHECK(X.size(-1) == 3 || X.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (X.size(-1)) {
    case 3:
      return std::get<std::unique_ptr<KDTree<float, float3>>>(tree_)
          ->QueryKNN<2>(stream, X);
      break;
    case 4:
      return std::get<std::unique_ptr<KDTree<float, float4>>>(tree_)
          ->QueryKNN<2>(stream, X);
      break;
    }
  }

  std::vector<torch::Tensor> QueryKNN3(const torch::Tensor &X) {
    TORCH_CHECK(X.is_cuda());
    TORCH_CHECK(X.size(-1) == 3 || X.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (X.size(-1)) {
    case 3:
      return std::get<std::unique_ptr<KDTree<float, float3>>>(tree_)
          ->QueryKNN<3>(stream, X);
      break;
    case 4:
      return std::get<std::unique_ptr<KDTree<float, float4>>>(tree_)
          ->QueryKNN<3>(stream, X);
      break;
    }
  }

  std::vector<torch::Tensor> QueryKNN4(const torch::Tensor &X) {
    TORCH_CHECK(X.is_cuda());
    TORCH_CHECK(X.size(-1) == 3 || X.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (X.size(-1)) {
    case 3:
      return std::get<std::unique_ptr<KDTree<float, float3>>>(tree_)
          ->QueryKNN<4>(stream, X);
      break;
    case 4:
      return std::get<std::unique_ptr<KDTree<float, float4>>>(tree_)
          ->QueryKNN<4>(stream, X);
      break;
    }
  }

  std::vector<torch::Tensor> QueryKNN5(const torch::Tensor &X) {
    TORCH_CHECK(X.is_cuda());
    TORCH_CHECK(X.size(-1) == 3 || X.size(-1) == 4);
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (X.size(-1)) {
    case 3:
      return std::get<std::unique_ptr<KDTree<float, float3>>>(tree_)
          ->QueryKNN<5>(stream, X);
      break;
    case 4:
      return std::get<std::unique_ptr<KDTree<float, float4>>>(tree_)
          ->QueryKNN<5>(stream, X);
      break;
    }
  }
};

TORCH_LIBRARY(patchwork_query_ext, m) {
  m.class_<KDTreeWrapper>("KDTree")
      .def(torch::init<torch::Tensor>())
      .def("query_closest", &KDTreeWrapper::QueryClosest)
      .def("query_knn2", &KDTreeWrapper::QueryKNN2)
      .def("query_knn3", &KDTreeWrapper::QueryKNN3)
      .def("query_knn4", &KDTreeWrapper::QueryKNN4)
      .def("query_knn5", &KDTreeWrapper::QueryKNN5);
}

} // namespace patchwork_query_ext
