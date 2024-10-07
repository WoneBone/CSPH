#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <cstring>
#include <string>

#include "../utils/CycleTimer.h"

// define col major access
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

void printCudaInfo();

// Function declarations
void cublas_gemm_fp32(float *A, float *B, float *C, int m, int n, int k, bool warmup = false);

void cublas_gemm_fp16(float *A, float *B, float *C, int m, int n, int k, bool warmup = false);

void cublas_gemm_tf32(float *A, float *B, float *C, int m, int n, int k, bool warmup = false);

void gemm_cpu(float *A, float *B, double *C, int m, int n, int k);

double euclidean_distance(double *A, float *B, int n);

int main(int argc, char **argv) {
  int m = 128;
  int n = 128;
  int k = 128;

  printCudaInfo();

  // parse command line arguments
  int o;
  while ((o = getopt(argc, argv, "m:n:k:a:h")) != -1) switch (o) {
      case 'a':
        m = n = k = atoi(optarg);
        break;
      case 'm':
        m = atoi(optarg);
        break;
      case 'n':
        n = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 'h':
        fprintf(stdout,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=128]\n\t-n \t N "
                "dimension [int] [default=128]\n\t-k \t K dimension [int] "
                "[default=128]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr,
                "Usage: %s [OPTION]...\n\n\t-m \t M dimension [int] "
                "[default=128]\n\t-n \t N "
                "dimension [int] [default=128]\n\t-k \t K dimension [int] "
                "[default=128]\n\t-a \t All "
                "dimensions [int]\n\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

  printf("Performing GEMM with dimensions: %d x %d x %d\n\n", m, n, k);

  // allocate memory for matrices
  float *A = new float[m * k];
  float *B = new float[k * n];
  double *C = new double[m * n];
  float *C_fp32 = new float[m * n];
  float *C_fp16 = new float[m * n];
  float *C_tf32 = new float[m * n];

  // initialize A and B with random values
  for (int i = 0; i < m * k; i++) A[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10));
  for (int i = 0; i < k * n; i++) B[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 10));

  // CPU GEMM
  gemm_cpu(A, B, C, m, n, k);

  // warm up GPU to get more accurate timings
  cublas_gemm_fp32(A, B, C_fp32, m, n, k, true);

  // CUBLAS GEMM for FP32, FP16, and TF32
  double startTime = CycleTimer::currentSeconds();
  cublas_gemm_fp32(A, B, C_fp32, m, n, k);
  double endTime = CycleTimer::currentSeconds();
  printf("GPU time (FP32): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // warm up kernel for FP16
  cublas_gemm_fp16(A, B, C_fp16, m, n, k, true);

  // Measure performance of FP16
  startTime = CycleTimer::currentSeconds();
  cublas_gemm_fp16(A, B, C_fp16, m, n, k);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (FP16): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // warm up kernel for TF32
  cublas_gemm_tf32(A, B, C_tf32, m, n, k, true);

  // Measure performance of TF32
  startTime = CycleTimer::currentSeconds();
  cublas_gemm_tf32(A, B, C_tf32, m, n, k);
  endTime = CycleTimer::currentSeconds();
  printf("GPU time (TF32): %.3f ms\n\n", 1000.f * (endTime - startTime));

  // compare results with CPU GEMM
  printf("Euclidean distance between CPU and FP32: %.9f\n", euclidean_distance(C, C_fp32, m * n));
  printf("Euclidean distance between CPU and FP16: %.9f\n", euclidean_distance(C, C_fp16, m * n));
  printf("Euclidean distance between CPU and TF32: %.9f\n", euclidean_distance(C, C_tf32, m * n));

  // free memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_fp32;
  delete[] C_fp16;
  delete[] C_tf32;

  return 0;
}

/**
 * @brief Performs General Matrix Multiply (GEMM) on CPU.
 *
 * This function computes the product of two matrices A and B, and stores the result in matrix C.
 * The matrices are stored in column-major order.
 *
 * @param A Pointer to the first input matrix (m x k).
 * @param B Pointer to the second input matrix (k x n).
 * @param C Pointer to the output matrix (m x n).
 * @param m M dimension.
 * @param n N dimension.
 * @param k K dimension.
 */
void gemm_cpu(float *A, float *B, double *C, int m, int n, int k) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int l = 0; l < k; l++) sum += A[IDX2C(i, l, m)] * B[IDX2C(l, j, k)];
      C[IDX2C(i, j, m)] = sum;
    }
}

/**
 * @brief Computes the Euclidean distance between two matrices.
 *
 * This function calculates the Euclidean distance between two matrices A and B
 * of length n=w*h. The Euclidean distance is defined as the square root of the sum
 * of the squared differences between corresponding elements of the matrices.
 *
 * @param A Pointer to the first matrix (array of doubles) - CPU result.
 * @param B Pointer to the second matrix (array of floats) - corresponding GPU result.
 * @param n The number of elements in each matrix.
 * @return The Euclidean distance between the two matrices.
 */
double euclidean_distance(double *A, float *B, int n) {
  double sum = 0;
  for (int i = 0; i < n; i++) sum += pow((A[i] - B[i]), 2);
  return sqrt(sum);
}