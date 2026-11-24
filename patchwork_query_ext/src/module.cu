#include <pybind11/eigen.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#include "kd_query.cuh"
#include "tri_edge_collapse_quadric_masked.h"

namespace py = pybind11;

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

py::list
masked_quadric_edge_collapse(Eigen::Ref<const RowMatrixXf> V,
                             Eigen::Ref<const RowMatrixXi> F,
                             Eigen::Ref<const Eigen::VectorXi> F_label,
                             Eigen::Ref<const Eigen::VectorXi> F_label2) {

  // It should be unlikely to overflow 32 bit integer
  RowMatrixXf U;
  RowMatrixXi G;
  Eigen::VectorXi C1;
  Eigen::VectorXi C2;
  tri_edge_collapse_quadric_masked(V, F, F_label, F_label2, U, G, C1, C2);

  py::list return_list;
  return_list.append(U);
  return_list.append(G);
  return_list.append(C1);
  return_list.append(C2);
  return return_list;
}

} // namespace patchwork_query_ext

// TORCH_EXTENSION_NAME is set in torch.utils.cpp_extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_quadric_edge_collapse",
        &patchwork_query_ext::masked_quadric_edge_collapse,
        "Quadric edge collapse with face mask");
}
