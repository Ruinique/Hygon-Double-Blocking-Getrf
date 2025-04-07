#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdio>
#include <cstdlib>

#define MAGIC_SEED 261825ULL

// cuda API error checking
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "UNKNOWN CUBLAS STATUS";
    }
}

// cuBLAS API error checking
#define CUBLAS_CHECK(call)                                                      \
    {                                                                           \
        cublasStatus_t err = call;                                              \
        if (err != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cublasGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// 在 utils.h 中添加 cusolver 错误字符串函数
const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "UNKNOWN CUSOLVER STATUS";
    }
}

// 定义 CUSOLVER_CHECK 宏
#define CUSOLVER_CHECK(call)                                                      \
    {                                                                             \
        cusolverStatus_t err = call;                                              \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuSolver error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cusolverGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    }

cudaEvent_t default_start, default_end;

void startTimer(cudaEvent_t& begin = default_start, cudaEvent_t& end = default_end) {
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer(cudaEvent_t& begin = default_start, cudaEvent_t& end = default_end) {
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

template <typename T>
void generateNormalMatrix(thrust::device_vector<T>& A_device_vector, size_t m,
                          size_t n, size_t seed_offset = 0) {
    T* dA = thrust::raw_pointer_cast(A_device_vector.data());
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    unsigned long long seed = MAGIC_SEED;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    if constexpr (std::is_same_v<T, float>)
        curandGenerateNormal(gen, dA, m * n, 0.0f, 1.0f);
    else if constexpr (std::is_same_v<T, double>)
        curandGenerateNormalDouble(gen, dA, m * n, 0.0, 1.0);
    curandDestroyGenerator(gen);
}

template <typename T>
void printMatrixHost(thrust::host_vector<T>& A_host_vector, size_t m, size_t n) {
    if (m <= 32 && n <= 32)
        for (size_t i = 0; i < m; i++) {
            printf("%2zu: ", i);
            for (size_t j = 0; j < n; j++) {
                printf("%4.2f ", static_cast<double>(A_host_vector[j * m + i]));
            }
            printf("\n");
        }
    else
        printf("\nToo large to print\n");
}

template <typename T>
void printMatrixDevice(thrust::device_vector<T>& A_device_vector, size_t m, size_t n,
                       const char* name) {
    // TODO: 记得删掉
    bool donothing = false;
    if (donothing) return;
    printf("\n========== matrix %s ==========\n", name);
    if (m > 32 || n > 32) {
        printf("\nToo large to print\n");
    } else {
        thrust::host_vector<T> A_host_vector = A_device_vector;
        printMatrixHost(A_host_vector, m, n);
    }
    printf("\n========== matrix %s ==========\n", name);
}

template <typename T, class Func>
__global__ void cleanMatrix(T* A_d, size_t m, size_t n, size_t lda, Func func) {
    // gird stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += blockDim.x * gridDim.x) {
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n;
             j += blockDim.y * gridDim.y) {
            func(i, j, A_d, m);
        }
    }
}

template <typename T>
T computeFrobeniusNorm(size_t m, size_t n,
                       thrust::device_vector<T>& A_device_vector) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    T res;
    auto dA = thrust::raw_pointer_cast(A_device_vector.data());
    int incx = 1;
    if constexpr (std::is_same_v<T, float>)
        cublasSnrm2_v2(handle, m * n, dA, incx, &res);
    else if constexpr (std::is_same_v<T, double>)
        cublasDnrm2_v2(handle, m * n, dA, incx, &res);
    cublasDestroy(handle);
    return res;
}
