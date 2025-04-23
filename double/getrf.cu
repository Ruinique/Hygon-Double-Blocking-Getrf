#include <cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "getrf.cuh"

#define SWAP_LEN_PANEL 2048

using data_type = double;

void trsm(cublasHandle_t cublasH, long m, long n, float alpha, float* A,
          long lda, float* B, long ldb, long nb) {
    float sonefloat = 1.0, snegonefloat = -1.0;
    if (m <= nb) {
        CUBLAS_CHECK(cublasStrsm(
            cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_UNIT, m, n, &alpha, A, lda, B, ldb));
        return;
    }

    trsm(cublasH, m / 2, n, alpha, A, lda, B, ldb, nb);

    long left = m - m / 2;
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left, n, m / 2,
                             &snegonefloat, A + m / 2, lda, B, ldb, &sonefloat,
                             B + m / 2, ldb));

    trsm(cublasH, left, n, alpha, A + m / 2 + m / 2 * lda, lda, B + m / 2, ldb,
         nb);
}

int parseArgs(int argc, char* argv[], size_t& n, size_t& k, size_t& nb,
              bool& verifyResult, bool& pivoting, bool& debug_mode,
              bool& compare_with_cusolver, bool& use_cublas_swp);

template <typename T>
void compareWithCusolver(thrust::device_vector<T>& A_device_vector, size_t n,
                         int roll_num, float& cusolver_lu_time, bool pivoting);

template <typename T>
void computeMinusOfPAandLU(thrust::device_vector<T>& A_device_vector,
                           thrust::device_vector<T>& oriA_device_vector,
                           thrust::device_vector<T>& P_device_vector, int n, cublasHandle_t cublas_handle);

template <typename T>
void launchSwapByPivotingKernel(int* source_raw_ptr, int* target_raw_ptr,
                                int swap_size, int swap_len, int stride, T* A_d,
                                T* temp_d, int blocksPerGrid,
                                int threadsPerBlock);

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelRead(int* source, int* target,
                                         int swap_size, int swap_len,
                                         int stride, T* A, T* temp);

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelWrite(int* source, int* target,
                                          int swap_size, int swap_len,
                                          int stride, T* A, T* temp);

template <typename T, size_t threadsPerRow>
__global__ void swapByPivotingKernel(int* __restrict__ source,
                                     int* __restrict__ target, int swap_size,
                                     int swap_len, int stride,
                                     T* __restrict__ A_d, T* __restrict__ temp);

struct DoubleBlockingLUBreakDownDetail {
    float tslu_time = 0.0f;
    float swap_panel_time = 0.0f;
    float trsm_panel_time = 0.0f;
    float gemm_panel_time = 0.0f;
    float gemm_panel_ops = 0.0f;
    float swap_kpanel_time = 0.0f;
    float trsm_kpanel_time = 0.0f;
    float gemm_kpanel_ops = 0.0f;
    float gemm_kpanel_time = 0.0f;
};

int main(int argc, char* argv[]) {
    size_t n, k, nb;
    bool verifyResult = false;
    bool pivoting = false;
    bool debug_mode = false;
    bool compare_with_cusolver = false;
    bool use_cublas_swp = false;
    int status = parseArgs(argc, argv, n, k, nb, verifyResult, pivoting,
                           debug_mode, compare_with_cusolver, use_cublas_swp);

    if (status != 0) {
        printf("Please execute the above command\n");
        return 1;
    }

    if constexpr (std::is_same_v<data_type, float>) {
        printf("data type using float\n");
    } else if constexpr (std::is_same_v<data_type, double>) {
        printf("data type using double\n");
    }

    // 打印当前显卡
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device name: %s\n", prop.name);

    int maxBlocksPerGrid = prop.multiProcessorCount;
    int threadsPerBlock = 256;
    int blocksPerGrid = maxBlocksPerGrid;

    // 初始化 A 矩阵，用来测试
    thrust::device_vector<data_type> A_device_vector(n * n);
    auto A_d = thrust::raw_pointer_cast(A_device_vector.data());
    generateNormalMatrix(A_device_vector, n, n);
    printMatrixDevice(A_device_vector, n, n, "A device");

    thrust::device_vector<data_type> P_device_vector;
    auto P_d = thrust::raw_pointer_cast(P_device_vector.data());
    thrust::device_vector<data_type> oriA_device_vector;
    auto oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());

    if (verifyResult) {
        P_device_vector.resize(n * n);
        P_d = thrust::raw_pointer_cast(P_device_vector.data());
        thrust::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(n),
            [P_d, n] __device__(size_t i) {
                if constexpr (std::is_same_v<data_type, float>)
                    P_d[i * n + i] = 1.0f;
                else if constexpr (std::is_same_v<data_type, double>)
                    P_d[i * n + i] = 1.0;
            });
        oriA_device_vector = A_device_vector;
        oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());
    }

    // 初始化 cublas 相关句柄
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // 初始化 cuSOLVER 相关句柄
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // 迭代轮次
    int roll_num = 3;
    if (verifyResult) roll_num = 1;

    float cusolver_lu_time = 0.0f;
    if (compare_with_cusolver) {
        compareWithCusolver(A_device_vector, n, roll_num, cusolver_lu_time,
                            pivoting);
    }

    // 为了交换准备空间
    thrust::device_vector<data_type> temp_device_vector(n * n);
    auto temp_d = thrust::raw_pointer_cast(temp_device_vector.data());

    // 预分配 TSLU 工作空间
    int buffer_size;
    auto A_raw_d = thrust::raw_pointer_cast(A_device_vector.data());
    if constexpr (std::is_same_v<data_type, float>) {
        CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
            cusolver_handle, n, nb, reinterpret_cast<float*>(A_raw_d), n,
            &buffer_size));
    } else if constexpr (std::is_same_v<data_type, double>) {
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
            cusolver_handle, n, nb, reinterpret_cast<double*>(A_raw_d), n,
            &buffer_size));
    }

    data_type* tslu_workspace_d;
    CUDA_CHECK(
        cudaMalloc((void**)&tslu_workspace_d, sizeof(data_type) * buffer_size));
    CUDA_CHECK(
        cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
    // 预分配 TSLU 状态
    int* devInfo_d;
    CUDA_CHECK(cudaMalloc((void**)&devInfo_d, sizeof(int)));
    CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
    // 预分配 TSLU ipiv 数组
    thrust::device_vector<int> ipiv_d_vector(n);
    auto ipiv_d = thrust::raw_pointer_cast(ipiv_d_vector.data());
    thrust::host_vector<int> ipiv_h_vector(n);

    DoubleBlockingLUBreakDownDetail detail;

    cudaEvent_t total_lu_begin, total_lu_end;

    startTimer(total_lu_begin, total_lu_end);

    for (int roll = 0; roll < roll_num; roll++) {
        //  执行 memset
        CUDA_CHECK(
            cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
        CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
        thrust::fill(ipiv_d_vector.begin(), ipiv_d_vector.end(), 0);
        thrust::fill(ipiv_h_vector.begin(), ipiv_h_vector.end(), 0);
        generateNormalMatrix(A_device_vector, n, n);
        //  长度为 k 的一组 panel 的 double-blocking
        for (size_t i = 0; i < n; i += k) {
            //  一组长度为 nb 的 panel 中执行对应操作
            for (size_t j = i; j < i + k; j += nb) {
                startTimer();
                // TSLU 分解
                data_type* A_d =
                    thrust::raw_pointer_cast(A_device_vector.data());
                if (pivoting) {
                    if constexpr (std::is_same_v<data_type, float>) {
                        CUSOLVER_CHECK(cusolverDnSgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<float*>(A_d + j + j * n), n,
                            reinterpret_cast<float*>(tslu_workspace_d),
                            ipiv_d + j, devInfo_d));
                    } else if constexpr (std::is_same_v<data_type, double>) {
                        CUSOLVER_CHECK(cusolverDnDgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<double*>(A_d + j + j * n), n,
                            reinterpret_cast<double*>(tslu_workspace_d),
                            ipiv_d + j, devInfo_d));
                    }
                } else {
                    if constexpr (std::is_same_v<data_type, float>) {
                        CUSOLVER_CHECK(cusolverDnSgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<float*>(A_d + j + j * n), n,
                            reinterpret_cast<float*>(tslu_workspace_d), NULL,
                            devInfo_d));
                    } else if constexpr (std::is_same_v<data_type, double>) {
                        CUSOLVER_CHECK(cusolverDnDgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<double*>(A_d + j + j * n), n,
                            reinterpret_cast<double*>(tslu_workspace_d), NULL,
                            devInfo_d));
                    }
                }
                detail.tslu_time += stopTimer();

                // 求 P 用来做校验
                if (pivoting && verifyResult) {
                    for (size_t idx = 0; idx < nb; idx++) {
                        ipiv_h_vector = ipiv_d_vector;
                        size_t global_idx = j + idx;
                        size_t local_pivot = ipiv_h_vector[global_idx] - 1;
                        size_t global_pivot = j + local_pivot;

                        if (global_idx != global_pivot) {
                            if constexpr (std::is_same_v<data_type, float>) {
                                CUBLAS_CHECK(cublasSswap(
                                    cublas_handle, n,
                                    reinterpret_cast<float*>(P_d + global_idx),
                                    n,
                                    reinterpret_cast<float*>(P_d +
                                                             global_pivot),
                                    n));
                            } else if constexpr (std::is_same_v<data_type,
                                                                double>) {
                                CUBLAS_CHECK(cublasDswap(
                                    cublas_handle, n,
                                    reinterpret_cast<double*>(P_d + global_idx),
                                    n,
                                    reinterpret_cast<double*>(P_d +
                                                              global_pivot),
                                    n));
                            }
                        }
                    }
                }

                int restnum_panel = i + k - j - nb;
                thrust::host_vector<int> source;
                thrust::host_vector<int> target;
                thrust::device_vector<int> source_d;
                thrust::device_vector<int> target_d;

                startTimer();

                if (pivoting) {
                    if (!use_cublas_swp) {
                        ipiv_h_vector = ipiv_d_vector;
                        std::vector<int> ipiv_h_idx(n);
                        for (int idx = 0; idx < ipiv_h_vector.size(); idx++) {
                            ipiv_h_idx[idx] = idx;
                        }

                        for (int idx = 0; idx < nb; idx++) {
                            std::swap(ipiv_h_idx[ipiv_h_vector[idx + j] - 1],
                                      ipiv_h_idx[idx]);
                        }

                        for (int idx = 0; idx < n; idx++) {
                            if (ipiv_h_idx[idx] != idx) {
                                source.push_back(ipiv_h_idx[idx]);
                                target.push_back(idx);
                            }
                        }

                        source_d = source;
                        target_d = target;

                        if (restnum_panel < k - nb) {
                            launchSwapByPivotingKernel(
                                thrust::raw_pointer_cast(source_d.data()),
                                thrust::raw_pointer_cast(target_d.data()),
                                source_d.size(), j - i, n, A_d + j + i * n,
                                temp_d, blocksPerGrid, threadsPerBlock);
                        }
                        if (restnum_panel > 0) {
                            launchSwapByPivotingKernel(
                                thrust::raw_pointer_cast(source_d.data()),
                                thrust::raw_pointer_cast(target_d.data()),
                                source_d.size(), restnum_panel, n,
                                A_d + j + (nb + j) * n, temp_d, blocksPerGrid,
                                threadsPerBlock);
                        }
                    } else {
                        if (restnum_panel < k - nb) {
                            for (int a = 0; a < nb; a++) {
                                if constexpr (std::is_same_v<data_type, float>)
                                    cublasSswap(cublas_handle, j - i,
                                                reinterpret_cast<float*>(
                                                    A_d + j + i * n + a),
                                                n,
                                                reinterpret_cast<float*>(
                                                    A_d + j + i * n +
                                                    ipiv_h_vector[j + a] - 1),
                                                n);
                                else if constexpr (std::is_same_v<data_type,
                                                                  double>)
                                    cublasDswap(cublas_handle, j - i,
                                                reinterpret_cast<double*>(
                                                    A_d + j + i * n + a),
                                                n,
                                                reinterpret_cast<double*>(
                                                    A_d + j + i * n +
                                                    ipiv_h_vector[j + a] - 1),
                                                n);
                            }
                        }
                        if (restnum_panel > 0) {
                            for (int i = j; i < j + nb; i++) {
                                if constexpr (std::is_same_v<data_type, float>)
                                    cublasSswap(
                                        cublas_handle, restnum_panel,
                                        reinterpret_cast<float*>(
                                            A_d + j + (nb + j) * n + i - j),
                                        n,
                                        reinterpret_cast<float*>(
                                            A_d + j + (nb + j) * n +
                                            ipiv_h_vector[i] - 1),
                                        n);
                                else if constexpr (std::is_same_v<data_type,
                                                                  double>)
                                    cublasDswap(
                                        cublas_handle, restnum_panel,
                                        reinterpret_cast<double*>(
                                            A_d + j + (nb + j) * n + i - j),
                                        n,
                                        reinterpret_cast<double*>(
                                            A_d + j + (nb + j) * n +
                                            ipiv_h_vector[i] - 1),
                                        n);
                            }
                        }
                    }
                }

                detail.swap_panel_time += stopTimer();
                // 更新完一组后，直接 continue
                if (nb + j - i >= k) {
                    continue;
                }
                // 解三角方程，求 U
                startTimer();
                data_type alpha = 1.0f;
                if constexpr (std::is_same_v<data_type, float>) {
                    CUBLAS_CHECK(cublasStrsm(
                        cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, nb + j - i, nb,
                        reinterpret_cast<float*>(&alpha),
                        reinterpret_cast<float*>(A_d + i + i * n), n,
                        reinterpret_cast<float*>(A_d + i + (nb + j) * n), n));
                } else if constexpr (std::is_same_v<data_type, double>) {
                    CUBLAS_CHECK(cublasDtrsm(
                        cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, nb + j - i, nb,
                        reinterpret_cast<double*>(&alpha),
                        reinterpret_cast<double*>(A_d + i + i * n), n,
                        reinterpret_cast<double*>(A_d + i + (nb + j) * n), n));
                }

                detail.trsm_panel_time += stopTimer();
                // 矩阵乘法更新
                alpha = -1.0f;
                data_type beta = 1.0f;
                startTimer();
                if constexpr (std::is_same_v<data_type, float>) {
                    CUBLAS_CHECK(cublasSgemm_v2(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - nb - j, nb,
                        nb + j - i, reinterpret_cast<float*>(&alpha),
                        reinterpret_cast<float*>(A_d + (nb + j) + i * n), n,
                        reinterpret_cast<float*>(A_d + i + (nb + j) * n), n,
                        reinterpret_cast<float*>(&beta),
                        reinterpret_cast<float*>(A_d + (nb + j) + (nb + j) * n),
                        n));
                } else if constexpr (std::is_same_v<data_type, double>) {
                    CUBLAS_CHECK(cublasDgemm_v2(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - nb - j, nb,
                        nb + j - i, reinterpret_cast<double*>(&alpha),
                        reinterpret_cast<double*>(A_d + (nb + j) + i * n), n,
                        reinterpret_cast<double*>(A_d + i + (nb + j) * n), n,
                        reinterpret_cast<double*>(&beta),
                        reinterpret_cast<double*>(A_d + (nb + j) +
                                                  (nb + j) * n),
                        n));
                }

                detail.gemm_panel_time += stopTimer();
                detail.gemm_panel_ops +=
                    2.0f * (n - nb - j) * nb * (nb + j - i);
            }

            int restnum_kpanel = n - i - k;

            thrust::host_vector<int> kpanel_source_i;
            thrust::host_vector<int> kpanel_target_i;
            thrust::device_vector<int> kpanel_source_i_d;
            thrust::device_vector<int> kpanel_target_i_d;
            thrust::host_vector<int> kpanel_source_r;
            thrust::host_vector<int> kpanel_target_r;
            thrust::device_vector<int> kpanel_source_r_d;
            thrust::device_vector<int> kpanel_target_r_d;

            startTimer();

            if (pivoting) {
                if (!use_cublas_swp) {
                    if (restnum_kpanel < n - k) {
                        std::vector<int> ipiv_h_idx(n);
                        for (int idx = 0; idx < n; idx++) {
                            ipiv_h_idx[idx] = idx;
                        }
                        for (int idx = 0; idx < k; idx++) {
                            int target_row = ipiv_h_vector[i + idx] - 1;
                            if (idx >= nb) {
                                target_row += nb * (idx / nb);
                            }
                            std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                        }
                        for (int idx = 0; idx < ipiv_h_idx.size(); idx++) {
                            if (idx != ipiv_h_idx[idx]) {
                                kpanel_source_i.push_back(ipiv_h_idx[idx]);
                                kpanel_target_i.push_back(idx);
                            }
                        }
                        kpanel_source_i_d = kpanel_source_i;
                        kpanel_target_i_d = kpanel_target_i;
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(kpanel_source_i_d.data()),
                            thrust::raw_pointer_cast(kpanel_target_i_d.data()),
                            kpanel_source_i_d.size(), i, n, A_d + i, temp_d,
                            blocksPerGrid, threadsPerBlock);
                    }
                    if (restnum_kpanel > 0) {
                        std::vector<int> ipiv_h_idx(n);
                        for (int idx = 0; idx < n; idx++) {
                            ipiv_h_idx[idx] = idx;
                        }
                        for (int idx = 0; idx < i + k; idx++) {
                            int target_row = ipiv_h_vector[idx] - 1;
                            if (idx >= nb) {
                                target_row += nb * (idx / nb);
                            }
                            std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                        }
                        for (int idx = 0; idx < ipiv_h_idx.size(); idx++) {
                            if (idx != ipiv_h_idx[idx]) {
                                kpanel_source_r.push_back(ipiv_h_idx[idx]);
                                kpanel_target_r.push_back(idx);
                            }
                        }
                        kpanel_source_r_d = kpanel_source_r;
                        kpanel_target_r_d = kpanel_target_r;
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(kpanel_source_r_d.data()),
                            thrust::raw_pointer_cast(kpanel_target_r_d.data()),
                            kpanel_source_r_d.size(), k, n, A_d + (k + i) * n,
                            temp_d, blocksPerGrid, threadsPerBlock);
                    }
                } else {
                    if (restnum_kpanel < n - k) {
                        for (int j = 0; j < k; j++) {
                            int target_row = ipiv_h_vector[i + j] - 1;
                            if ((j) >= nb) {
                                target_row += nb * (j / nb);
                            }
                            if constexpr (std::is_same_v<data_type, float>)
                                cublasSswap(
                                    cublas_handle, i,
                                    reinterpret_cast<float*>(A_d + i + j), n,
                                    reinterpret_cast<float*>(A_d + i +
                                                             target_row),
                                    n);
                            else if constexpr (std::is_same_v<data_type,
                                                              double>)
                                cublasDswap(
                                    cublas_handle, i,
                                    reinterpret_cast<double*>(A_d + i + j), n,
                                    reinterpret_cast<double*>(A_d + i +
                                                              target_row),
                                    n);
                        }
                    }

                    if (restnum_kpanel > 0) {
                        for (int j = 0; j < i + k; j++) {
                            int target_row = ipiv_h_vector[j] - 1;
                            if (j >= nb) {
                                target_row += nb * (j / nb);
                            }
                            if (target_row >= 0 && target_row < n) {
                                if constexpr (std::is_same_v<data_type, float>)
                                    cublasSswap(
                                        cublas_handle, k,
                                        reinterpret_cast<float*>(
                                            A_d + (k + i) * n + (j)),
                                        n,
                                        reinterpret_cast<float*>(
                                            A_d + (k + i) * n + target_row),
                                        n);
                                else if constexpr (std::is_same_v<data_type,
                                                                  double>)
                                    cublasDswap(
                                        cublas_handle, k,
                                        reinterpret_cast<double*>(
                                            A_d + (k + i) * n + (j)),
                                        n,
                                        reinterpret_cast<double*>(
                                            A_d + (k + i) * n + target_row),
                                        n);
                            }
                        }
                    }
                }
            }
            detail.swap_kpanel_time += stopTimer();
            if (k + i >= n) break;

            // 解三角方程，求 U
            data_type alpha = 1.0f;
            startTimer();
            if constexpr (std::is_same_v<data_type, float>) {
                trsm(cublas_handle, k + i, k, alpha,
                     reinterpret_cast<float*>(A_d), n,
                     reinterpret_cast<float*>(A_d + (k + i) * n), n, 512);
            } else if constexpr (std::is_same_v<data_type, double>) {
                CUBLAS_CHECK(cublasDtrsm(
                    cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT, k + i, k,
                    reinterpret_cast<double*>(&alpha),
                    reinterpret_cast<double*>(A_d), n,
                    reinterpret_cast<double*>(A_d + (k + i) * n), n));
            }

            detail.trsm_kpanel_time += stopTimer();
            startTimer();
            // 矩阵乘法更新
            alpha = -1.0f;
            data_type beta = 1.0f;
            if constexpr (std::is_same_v<data_type, float>) {
                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - i, k,
                    k + i, reinterpret_cast<float*>(&alpha),
                    reinterpret_cast<float*>(A_d + (k + i)), n,
                    reinterpret_cast<float*>(A_d + (k + i) * n), n,
                    reinterpret_cast<float*>(&beta),
                    reinterpret_cast<float*>(A_d + (k + i) + (k + i) * n), n));
            } else if constexpr (std::is_same_v<data_type, double>) {
                CUBLAS_CHECK(cublasDgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - i, k,
                    k + i, reinterpret_cast<double*>(&alpha),
                    reinterpret_cast<double*>(A_d + (k + i)), n,
                    reinterpret_cast<double*>(A_d + (k + i) * n), n,
                    reinterpret_cast<double*>(&beta),
                    reinterpret_cast<double*>(A_d + (k + i) + (k + i) * n), n));
            }

            detail.gemm_kpanel_time += stopTimer();
            detail.gemm_kpanel_ops += 2.0f * (n - k - i) * k * (k + i);
        }
    }

    printf("entire time: %f ms\n",
           stopTimer(total_lu_begin, total_lu_end) / roll_num);
    printf("TSLU time: %f ms\n", detail.tslu_time / roll_num);
    printf("Swap panel time: %f ms\n", detail.swap_panel_time / roll_num);
    printf("Trsm panel time: %f ms\n", detail.trsm_panel_time / roll_num);
    printf("Gemm panel time: %f ms\n", detail.gemm_panel_time / roll_num);
    printf("Gemm panel %f TFLOPS\n",
           detail.gemm_panel_ops / (detail.gemm_panel_time * 1e9));
    printf("Swap kpanel time: %f ms\n", detail.swap_kpanel_time / roll_num);
    printf("Trsm kpanel time: %f ms\n", detail.trsm_kpanel_time / roll_num);
    printf("Gemm kpanel time: %f ms\n", detail.gemm_kpanel_time / roll_num);
    printf("Gemm kpanel %f TFLOPS\n",
           detail.gemm_kpanel_ops / (detail.gemm_kpanel_time * 1e9));
    float total_lu_time = detail.tslu_time + detail.swap_panel_time +
                          detail.trsm_panel_time + detail.gemm_panel_time +
                          detail.swap_kpanel_time + detail.trsm_kpanel_time +
                          detail.gemm_kpanel_time;

    printf("double-blocking LU: %lf ms\n", total_lu_time / roll_num);
    printf(
        "double-blocking LU: %lf TFLOPS\n",
        double(2.0 / 3.0) * std::pow(n, 3) * roll_num / (total_lu_time * 1e9));

    if (compare_with_cusolver) {
        printf("Speedup: %f\n", cusolver_lu_time / total_lu_time);
    }

    if (verifyResult) {
        computeMinusOfPAandLU<data_type>(A_device_vector, oriA_device_vector,
                                         P_device_vector, n, cublas_handle);
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUDA_CHECK(cudaFree(devInfo_d));
    CUDA_CHECK(cudaFree(tslu_workspace_d));

    return 0;
}

/**
 * function: parse arguments
 * @param argc: number of arguments
 * @param argv: arguments
 * @param n: matrix size
 * @param k: matrix size
 * @param nb: block size
 */
int parseArgs(int argc, char* argv[], size_t& n, size_t& k, size_t& nb,
              bool& verifyResult, bool& pivoting, bool& debug_mode,
              bool& compare_with_cusolver, bool& use_cublas_swp) {
    if (argc == 2) {
        n = std::stoul(argv[1]);
        k = n / 2;
        nb = k / 2;
        compare_with_cusolver = true;
        pivoting = true;
    } else if (argc == 4) {
        n = std::stoul(argv[1]);
        k = std::stoul(argv[2]);
        nb = std::stoul(argv[3]);
    } else if (argc > 4) {
        n = std::stoul(argv[1]);
        k = std::stoul(argv[2]);
        nb = std::stoul(argv[3]);
        for (int i = 4; i < argc; i++)
            if (strcmp(argv[i], "-v") == 0) {
                verifyResult = true;
            } else if (strcmp(argv[i], "-p") == 0) {
                pivoting = true;
            } else if (strcmp(argv[i], "-d") == 0) {
                debug_mode = true;
            } else if (strcmp(argv[i], "-c") == 0) {
                compare_with_cusolver = true;
            } else if (strcmp(argv[i], "-s") == 0) {
                use_cublas_swp = true;
            }
    } else {
        printf("Default usage: %s <n>\n", argv[0]);
        printf("Usage: %s <n> <k> <nb>\n", argv[0]);
        return 1;
    }
    return 0;
}

template <typename T>
void computeMinusOfPAandLU(thrust::device_vector<T>& A_device_vector,
                           thrust::device_vector<T>& oriA_device_vector,
                           thrust::device_vector<T>& P_device_vector, int n, cublasHandle_t cublas_handle) {

    auto A_norm = computeFrobeniusNorm(n, n, oriA_device_vector);
    auto A_d = thrust::raw_pointer_cast(A_device_vector.data());
    auto oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());
    auto P_d = thrust::raw_pointer_cast(P_device_vector.data());
    // compute PA
    thrust::device_vector<T> PA_device_vector(n * n);
    auto PA_d = thrust::raw_pointer_cast(PA_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                    n, n, reinterpret_cast<float*>(&alpha),
                                    reinterpret_cast<float*>(P_d), n,
                                    reinterpret_cast<float*>(oriA_d), n,
                                    reinterpret_cast<float*>(&beta),
                                    reinterpret_cast<float*>(PA_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                    n, n, reinterpret_cast<double*>(&alpha),
                                    reinterpret_cast<double*>(P_d), n,
                                    reinterpret_cast<double*>(oriA_d), n,
                                    reinterpret_cast<double*>(&beta),
                                    reinterpret_cast<double*>(PA_d), n));
    }

    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    dim3 blockDim(16, 16);

    // compute LU
    thrust::device_vector<T> L_device_vector(n * n);
    thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                 L_device_vector.begin());
    auto L_d = thrust::raw_pointer_cast(L_device_vector.data());
    cleanMatrix<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(L_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, T* L_d, int lda) {
            T zero, one;
            if constexpr (std::is_same_v<T, float>) {
                zero = 0.0f;
                one = 1.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                zero = 0.0;
                one = 1.0;
            }
            if (i < n && j < n) {
                if (i < j) {
                    L_d[i + j * lda] = zero;
                }
                if (i == j) {
                    L_d[i + j * lda] = one;
                }
            }
        });
    thrust::device_vector<T> U_device_vector(n * n);
    thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                 U_device_vector.begin());
    auto U_d = thrust::raw_pointer_cast(U_device_vector.data());
    cleanMatrix<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(U_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, T* U_d, int lda) {
            T zero, one;
            if constexpr (std::is_same_v<T, float>) {
                zero = 0.0f;
                one = 1.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                zero = 0.0;
                one = 1.0;
            }
            if (i < n && j < n) {
                if (i > j) {
                    U_d[i + j * lda] = zero;
                }
            }
        });
    thrust::device_vector<T> LU_device_vector(n * n);
    auto LU_d = thrust::raw_pointer_cast(LU_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm_v2(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            reinterpret_cast<float*>(&alpha), reinterpret_cast<float*>(L_d), n,
            reinterpret_cast<float*>(U_d), n, reinterpret_cast<float*>(&beta),
            reinterpret_cast<float*>(LU_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n,
                                    n, n, reinterpret_cast<double*>(&alpha),
                                    reinterpret_cast<double*>(L_d), n,
                                    reinterpret_cast<double*>(U_d), n,
                                    reinterpret_cast<double*>(&beta),
                                    reinterpret_cast<double*>(LU_d), n));
    }

    // compute PA - LU
    thrust::device_vector<T> PAminusLU_device_vector(n * n);
    auto PAminusLU_d = thrust::raw_pointer_cast(PAminusLU_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        auto alpha = 1.0f;
        auto beta = -1.0f;
        CUBLAS_CHECK(cublasSgeam(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
            reinterpret_cast<float*>(&alpha), reinterpret_cast<float*>(PA_d), n,
            reinterpret_cast<float*>(&beta), reinterpret_cast<float*>(LU_d), n,
            reinterpret_cast<float*>(PAminusLU_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        auto alpha = 1.0;
        auto beta = -1.0;
        CUBLAS_CHECK(cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                                 reinterpret_cast<double*>(&alpha),
                                 reinterpret_cast<double*>(PA_d), n,
                                 reinterpret_cast<double*>(&beta),
                                 reinterpret_cast<double*>(LU_d), n,
                                 reinterpret_cast<double*>(PAminusLU_d), n));
    }

    T Minus_norm = computeFrobeniusNorm(n, n, PAminusLU_device_vector);
    printf("|PA - LU| Frobenius norm: %.16f\n", Minus_norm);
    printf("|A| Frobenius norm: %f\n", A_norm);
    printf("PA - LU / |A| Frobenius norm: %.16f\n", Minus_norm / A_norm);
}

template <typename T>
void compareWithCusolver(thrust::device_vector<T>& A_device_vector, size_t n,
                         int roll_num, float& cusolver_lu_time, bool pivoting) {
    printf("Starting cusolver LU decomposition...\n");
    thrust::device_vector<T> A_device_vector_copy(n * n);
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    for (int i = 0; i < roll_num; i++) {
        thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                     A_device_vector_copy.begin());
        auto A_cusolver = thrust::raw_pointer_cast(A_device_vector_copy.data());
        int* dinfo;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dinfo), sizeof(int)));
        // 调用 cuSolver 的 lu 分解，注意这些全部是在 device 上进行的
        int lwork;
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_handle, n, n,
                                                       A_cusolver, n, &lwork));
        } else if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolver_handle, n, n,
                                                       A_cusolver, n, &lwork))
        }
        T* dwork;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&dwork), lwork * sizeof(T)));
        int* dpiv;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&dpiv), n * sizeof(int)));
        startTimer();
        if (pivoting) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_CHECK(cusolverDnSgetrf(
                    cusolver_handle, n, n, A_cusolver, n, dwork, dpiv, dinfo));
            } else if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_CHECK(cusolverDnDgetrf(
                    cusolver_handle, n, n, A_cusolver, n, dwork, dpiv, dinfo))
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_CHECK(cusolverDnSgetrf(
                    cusolver_handle, n, n, A_cusolver, n, dwork, NULL, dinfo));
            } else if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_CHECK(cusolverDnDgetrf(
                    cusolver_handle, n, n, A_cusolver, n, dwork, NULL, dinfo))
            }
        }
        cusolver_lu_time += stopTimer();
    }

    printf("cusolver lu time: %f ms\n", cusolver_lu_time / roll_num);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelRead(int* source, int* target,
                                         int swap_size, int swap_len,
                                         int stride, T* A_d, T* temp) {
    // grid stride loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < swap_size;
         idx += blockDim.x * gridDim.x) {
        if (idx < swap_size) {
            auto temp4idx = temp + idx * swap_len;
            for (size_t j = 0; j < swap_len; j++) {
                temp4idx[j] = A_d[source[idx] + j * stride];
            }
        }
    }
}

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelWrite(int* source, int* target,
                                          int swap_size, int swap_len,
                                          int stride, T* A_d, T* temp) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < swap_size;
         idx += blockDim.x * gridDim.x) {
        if (idx < swap_size) {
            auto temp4idx = temp + idx * swap_len;
            for (size_t j = 0; j < swap_len; j++) {
                A_d[target[idx] + j * stride] = temp4idx[j];
            }
        }
    }
}

template <typename T, size_t threadsPerRow>
__global__ void swapByPivotingKernel(int* __restrict__ source,
                                     int* __restrict__ target, int swap_size,
                                     int swap_len, int stride,
                                     T* __restrict__ A_d,
                                     T* __restrict__ temp) {
    auto grid = cooperative_groups::this_grid();
    auto gridStride = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < swap_size * threadsPerRow; idx += gridStride) {
        if (idx < swap_size * threadsPerRow) {
            const int row_idx = idx / threadsPerRow;
            const int lane = idx % threadsPerRow;
            for (size_t j = lane; j < swap_len; j += threadsPerRow) {
                const size_t src_offset = source[row_idx] + j * stride;
                temp[row_idx * swap_len + j] = A_d[src_offset];
            }
        }
    }
    grid.sync();
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < swap_size * threadsPerRow; idx += gridStride) {
        if (idx < swap_size * threadsPerRow) {
            const int row_idx = idx / threadsPerRow;
            const int lane = idx % threadsPerRow;
            for (size_t j = lane; j < swap_len; j += threadsPerRow) {
                const size_t tgt_offset = target[row_idx] + j * stride;
                A_d[tgt_offset] = temp[row_idx * swap_len + j];
            }
        }
    }
}

template <typename T>
void launchSwapByPivotingKernel(int* source_raw_ptr, int* target_raw_ptr,
                                int swap_size, int swap_len, int stride, T* A_d,
                                T* temp_d, int blocksPerGrid,
                                int threadsPerBlock) {
    if (swap_len < SWAP_LEN_PANEL) {
        void* args[] = {reinterpret_cast<void*>(&source_raw_ptr),
                        reinterpret_cast<void*>(&target_raw_ptr),
                        reinterpret_cast<void*>(&swap_size),
                        reinterpret_cast<void*>(&swap_len),
                        reinterpret_cast<void*>(&stride),
                        reinterpret_cast<void*>(&A_d),
                        reinterpret_cast<void*>(&temp_d)};
        using KernelType = decltype(&swapByPivotingKernel<T, 16>);
        KernelType kernel_ptr = &swapByPivotingKernel<T, 16>;
        CUDA_CHECK(
            cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel_ptr),
                                        blocksPerGrid, threadsPerBlock, args));
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        for (int i = 0; i < swap_len;
             i += (swap_len + threadsPerBlock - 1) / threadsPerBlock) {
            auto each_swap_len =
                (swap_len + threadsPerBlock - 1) / threadsPerBlock;
            auto swap_len_panel = std::min(swap_len - i, each_swap_len);
            auto A_d_panel = A_d + i * stride;
            void* args[] = {reinterpret_cast<void*>(&source_raw_ptr),
                            reinterpret_cast<void*>(&target_raw_ptr),
                            reinterpret_cast<void*>(&swap_size),
                            reinterpret_cast<void*>(&swap_len_panel),
                            reinterpret_cast<void*>(&stride),
                            reinterpret_cast<void*>(&A_d_panel),
                            reinterpret_cast<void*>(&temp_d)};
            using KernelType = decltype(&swapByPivotingKernel<T, 16>);
            KernelType kernel_ptr = &swapByPivotingKernel<T, 16>;
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(kernel_ptr), blocksPerGrid,
                threadsPerBlock, args));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}
