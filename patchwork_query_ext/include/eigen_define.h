#ifndef EIGEN_DEFINE_H
#define EIGEN_DEFINE_H

#include <Eigen/Core>

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrix2f = Eigen::Matrix<float, 2, 2, Eigen::RowMajor>;

#endif // EIGEN_DEFINE_H