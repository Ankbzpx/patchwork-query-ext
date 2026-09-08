// Heavily copied from https://github.com/tilmantroester/cudakdtree_jax_binding

#ifndef KD_QUERY_CUH
#define KD_QUERY_CUH

#include <torch/extension.h>

#include "cukd/knn.h"
#include <cuda_runtime.h>
#include <cukd/builder.h>

template <typename PointT> struct OrderedPoint {
  PointT position;
  int idx;
};

template <typename PointT>
struct OrderedPoint_traits : public cukd::default_data_traits<PointT> {
  using data_t = OrderedPoint<PointT>;
  using point_traits = cukd::point_traits<PointT>;
  using scalar_t = typename point_traits::scalar_t;

  static inline __device__ __host__ const PointT &
  get_point(const data_t &data) {
    return data.position;
  }

  static inline __device__ __host__ scalar_t get_coord(const data_t &data,
                                                       int dim) {
    return cukd::get_coord(get_point(data), dim);
  }

  enum { has_explicit_dim = false };
  static inline __device__ int get_dim(const data_t &) { return -1; }
};

template <typename PointT>
__global__ void CopyKernel(OrderedPoint<PointT> *points, PointT *positions,
                           int n_points) {
  int nid = threadIdx.x + blockIdx.x * blockDim.x;

  if (nid >= n_points)
    return;

  points[nid].position = positions[nid];
  points[nid].idx = nid;
}

template <typename PointT>
__global__ void QueryKNN2Kernel(int *d_indices, int *d_indices_sec,
                                PointT *d_queries, int n_queries,
                                const cukd::box_t<PointT> *d_bounds,
                                OrderedPoint<PointT> *d_nodes, int n_points) {
  int qid = threadIdx.x + blockIdx.x * blockDim.x;
  if (qid >= n_queries)
    return;

  PointT queryPos = d_queries[qid];
  cukd::FixedCandidateList<2> results(INFINITY);
  cukd::cct::knn<cukd::FixedCandidateList<2>, OrderedPoint<PointT>,
                 OrderedPoint_traits<PointT>>(results, queryPos, *d_bounds,
                                              d_nodes, n_points);
  d_indices[qid] = d_nodes[results.get_pointID(0)].idx;
  d_indices_sec[qid] = d_nodes[results.get_pointID(1)].idx;
}

template <typename PointT>
__global__ void
QueryClosestKernel(int *d_indices, PointT *d_queries, int n_queries,
                   const cukd::box_t<PointT> *d_bounds,
                   OrderedPoint<PointT> *d_nodes, int n_points) {
  int qid = threadIdx.x + blockIdx.x * blockDim.x;
  if (qid >= n_queries)
    return;

  PointT queryPos = d_queries[qid];
  cukd::FcpSearchParams params;
  // Local closest index
  int closestID =
      cukd::cct::fcp<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
          queryPos, *d_bounds, d_nodes, n_points, params);
  d_indices[qid] = d_nodes[closestID].idx;
}

template <typename T = float, typename PointT = float4,
          uint32_t BATCH_SIZE_N = 16, uint32_t BATCH_SIZE_M = 16>
class KDTree {
private:
  OrderedPoint<PointT> *d_A_;
  cukd::box_t<PointT> *d_bounds_;
  uint32_t M_;

public:
  KDTree() = delete;
  KDTree(const KDTree &) = delete;

  KDTree(cudaStream_t stream, const torch::Tensor &A) {
    M_ = A.size(0);

    CUKD_CUDA_CHECK(cudaMallocAsync((void **)&d_A_,
                                    M_ * sizeof(OrderedPoint<PointT>), stream));
    CUKD_CUDA_SYNC_CHECK();

    // **IMPORTANT** We cannot loop, as data is in device memory
    CopyKernel<<<dim3(cukd::divRoundUp(M_, BATCH_SIZE_M)), dim3(BATCH_SIZE_M),
                 0, stream>>>(d_A_, reinterpret_cast<PointT *>(A.data_ptr<T>()),
                              M_);
    CUKD_CUDA_SYNC_CHECK();

    CUKD_CUDA_CHECK(cudaMallocAsync((void **)&d_bounds_,
                                    sizeof(cukd::box_t<PointT>), stream));
    cukd::buildTree<OrderedPoint<PointT>, OrderedPoint_traits<PointT>>(
        d_A_, M_, d_bounds_, stream);
    CUKD_CUDA_SYNC_CHECK();
  }

  std::vector<torch::Tensor> QueryKNN2(cudaStream_t stream,
                                       const torch::Tensor &X) {
    uint32_t N = X.size(0);
    // Query
    PointT *d_X = reinterpret_cast<PointT *>(X.data_ptr<T>());
    const torch::TensorOptions idxOpts =
        torch::TensorOptions().dtype(torch::kInt32).device(X.device());
    torch::Tensor idx = torch::zeros({N}, idxOpts);
    torch::Tensor idx_sec = torch::zeros({N}, idxOpts);

    QueryKNN2Kernel<<<dim3(cukd::divRoundUp(N, BATCH_SIZE_N)),
                      dim3(BATCH_SIZE_N), 0, stream>>>(
        idx.data_ptr<int>(), idx_sec.data_ptr<int>(), d_X, N, d_bounds_, d_A_,
        M_);
    CUKD_CUDA_SYNC_CHECK();
    return {idx, idx_sec};
  }

  torch::Tensor QueryClosest(cudaStream_t stream, const torch::Tensor &X) {
    uint32_t N = X.size(0);
    // Query
    PointT *d_X = reinterpret_cast<PointT *>(X.data_ptr<T>());
    const torch::TensorOptions idxOpts =
        torch::TensorOptions().dtype(torch::kInt32).device(X.device());
    torch::Tensor idx = torch::zeros({N}, idxOpts);

    QueryClosestKernel<<<dim3(cukd::divRoundUp(N, BATCH_SIZE_N)),
                         dim3(BATCH_SIZE_N), 0, stream>>>(
        idx.data_ptr<int>(), d_X, N, d_bounds_, d_A_, M_);
    CUKD_CUDA_SYNC_CHECK();
    return idx;
  }
};

#endif // KD_QUERY_CUH