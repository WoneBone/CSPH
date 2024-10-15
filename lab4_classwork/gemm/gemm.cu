/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans)                \
  {                                        \
    cudaAssert((ans), __FILE__, __LINE__); \
  }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n",
            cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)
#else
#define cudaCheckError(ans) ans
#define CUBLAS_CHECK(ans) ans
#endif

// define col major access
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}

/**
 * @brief Performs matrix multiplication using cuBLAS.
 *
 * This function multiplies two matrices A and B, and stores the result in matrix C.
 * The dimensions of the matrices are specified by m, n, and k.
 *
 * @param A Pointer to the first input matrix (m x k).
 * @param B Pointer to the second input matrix (k x n).
 * @param C Pointer to the output matrix (m x n).
 * @param m M dimension.
 * @param n N dimension.
 * @param k K dimension.
 * @param computeType The compute type to be used by cuBLAS (see slides for more information).
 * @param mode String to help identify the type of computation (use "FP32", "FP16" or "TF32" in the appropriate functions).
 * @param warm_up If true, performs a warm-up run before the actual computation where the timings are not considered.
 */

void cublas_gemm(float *A, float *B, float *C, int m, int n, int k, int lda, cublasComputeType_t computeType, const char *mode, bool warm_up = false) {
  // TODO: TASK 10
  // Complete this base function
  int *devA, *devB, *devC;
  
  cudaMalloc(&devA, (m*k) * sizeof(int));
  cudaMalloc(&devB, (n*k) * sizeof(int));
  cudaMalloc(&devC, (m*n) * sizeof(int));
  // Here you should copy host input matrices to the device

  cudaMemcpy(devA, A,  (m*k) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, B,  (n*k) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(devC, C,  (m*n) * sizeof(int), cudaMemcpyHostToDevice);

  // Here you should create a handle for cuBLAS and initialize it with cublasCreate()
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Here you should define whether the matrices are transposed or not
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // Scale factors are initialized: alpha = 1, beta = 0 for C = A * B
  // DO NOT MODIFY THESE VALUES
  const float alpha = 1.f;
  const float beta = 0.f;

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * The kernel is executed between the start and stop events.
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Here you should call the cuBLAS function that performs the matrix multiplication cublasGemmEx()
  // Do not forget to wrap it with the CUBLAS_CHECK() macro to detect errors
  CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, m, n, k, &alpha, devA, CUDA_R_32F, m, devB, CUDA_R_32F, k, &beta, devC,CUDA_R_32F, m, computeType, CUBLAS_GEMM_DEFAULT));

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * Here the time is recorded and printed.
   * The performance is calculated in GFLOPS.
   */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!warm_up) {
    printf("Kernel Time (%s): %f ms\n", mode, elapsedTime);
    printf("GFLOPS (%s): %f\n", mode, 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
  }

  // Here you should copy the result matrix back to the host
  cudaMemcpy(C, devC,  (m*n) * sizeof(int), cudaMemcpyDeviceToHost);
  // Do not forget to free the memory on the device here
  cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
  // Do not forget to destroy the cublas handle here with the cublasDestroy() function
  cublasDestroy(handle);
  return;
}

void cublas_gemm_fp32(float *A, float *B, float *C, int m, int n, int k, int lda, bool warm_up = false) {
  // Call the cublas gemm function for FP32 using cuda cores
  cublas_gemm(A, B, C, m, n, k, lda, CUBLAS_COMPUTE_32F, "FP32", warm_up);
  return;
}

void cublas_gemm_fp16(float *A, float *B, float *C, int m, int n, int k, int lda, bool warm_up = false) {
  // TODO: TASK 10
  // Complete function so that GEMM executes with Tensor Cores using FP16
  cublas_gemm(A, B, C, m, n, k, lda,  CUBLAS_COMPUTE_32F_FAST_16F, "FP16", warm_up);

  return;
}

void cublas_gemm_4xfp32(float *A, float *B, float *C, int m, int n, int k, int num_partitions, bool warm_up = false) {
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

  // Here you should allocate the memory on the device
  float *d_A, *d_B, *d_C;

  cudaCheckError(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

  // Here you should copy host input matrices to the device
  cudaCheckError(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  // cublasSetMatrix(m, k, sizeof(float), A, lda, d_A, m);
  cudaCheckError(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

  // Here you should create a handle for cuBLAS and initialize it with cublasCreate()
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Here you should define whether the matrices are transposed or not
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // Scale factors are initialized: alpha = 1, beta = 0 for C = A * B
  // DO NOT MODIFY THESE VALUES
  const float alpha = 1.f;
  const float beta = 0.f;

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * The kernel is executed between the start and stop events.
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int partition_size = m / num_partitions;

  for (int i = 0; i < num_partitions; i++) {
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, partition_size, n,
                              k, &alpha, d_A + i * partition_size, CUDA_R_32F, m, d_B,
                              CUDA_R_32F, k, &beta, d_C + i * partition_size, CUDA_R_32F,
                              m, computeType, CUBLAS_GEMM_DEFAULT));
  }

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * Here the time is recorded and printed.
   * The performance is calculated in GFLOPS.
   */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!warm_up) {
    printf("Kernel Time (4xFP32): %f ms\n", elapsedTime);
    printf("GFLOPS (4xFP32): %f\n", 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
  }

  // Here you should copy the result matrix back to the host
  cudaCheckError(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  // Do not forget to free the memory on the device here
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));

  // Do not forget to destroy the cublas handle here with the cublasDestroy() function
  CUBLAS_CHECK(cublasDestroy(handle));

  return;
}

void cublas_gemm_mixed(float *A, float *B, float *C, int m, int n, int k, int num_partitions, bool warm_up = false) {
  // TODO: TASK 12
  // Correct function so that the error is zero and the performance is maximized

  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_16F;

  // Here you should allocate the memory on the device
  float *d_A, *d_B, *d_C;

  cudaCheckError(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

  // Here you should copy host input matrices to the device
  cudaCheckError(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  // cublasSetMatrix(m, k, sizeof(float), A, lda, d_A, m);
  cudaCheckError(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

  // Here you should create a handle for cuBLAS and initialize it with cublasCreate()
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // Here you should define whether the matrices are transposed or not
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // Scale factors are initialized: alpha = 1, beta = 0 for C = A * B
  // DO NOT MODIFY THESE VALUES
  const float alpha = 1.f;
  const float beta = 0.f;

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * The kernel is executed between the start and stop events.
   */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int partition_size = m / num_partitions;

  for (int i = 0; i < num_partitions; i++) {
    CUBLAS_CHECK(cublasGemmEx(handle, transa, transb, partition_size, n,
                              k, &alpha, d_A + i * partition_size, CUDA_R_32F, m, d_B,
                              CUDA_R_32F, k, &beta, d_C + i * partition_size, CUDA_R_32F,
                              m, computeType, CUBLAS_GEMM_DEFAULT));
  }

  /* DO NOT MODIFY THIS PART
   * This part of the code is responsible for accurately measuring the time taken by the kernel.
   * Here the time is recorded and printed.
   * The performance is calculated in GFLOPS.
   */
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (!warm_up) {
    printf("Kernel Time (Mixed): %f ms\n", elapsedTime);
    printf("GFLOPS (Mixed): %f\n", 2.0 * m * n * k / (elapsedTime * 1e-3) / 1e9);
  }

  // Here you should copy the result matrix back to the host
  cudaCheckError(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

  // Do not forget to free the memory on the device here
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));

  // Do not forget to destroy the cublas handle here with the cublasDestroy() function
  CUBLAS_CHECK(cublasDestroy(handle));

  return;
}
