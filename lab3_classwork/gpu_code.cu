#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#else
#define cudaCheckError(ans) ans
#endif

static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void exclusive_scan(int* input, int N, int* result, int threadsPerBlock);
double cudaScanThrust(int* inarray, int* end, int* resultarray);

///////////////////////////////////////
//// WRITE YOUR CUDA KERNELS HERE /////
///////////////////////////////////////

__global__ void  gpu_initX(int N, int* devX)
{
    // TODO
}

__global__ void gpu_makeZ(int N, int* devX, int* devY, int* devZ)
{
    // TODO
}

__device__ int gpu_condition(int i, int *A)
{
    // TODO
    return 0;
}

__global__ void gpu_makeW(int N, int* devZ, int* devW)
{
    // TODO
}

__global__ void gpu_find_pattern(int N, int* devW, int* devZ, int* output) 
{
    // TODO
}


////////////////////////////////////////////////////////////////////////////////
// runGPU 
/////////////////////////////////////////
// Timing wrapper around your complete GPU code. You should not modify this function.
int runGPU(int threadsPerBlock, int N, int* hostY, int *gpuX, int *gpuY, int *gpuZ, int *gpuW, int *gpuResult)
{
    // here we declare the arrays that we will need 
    int *devX, *devY;
    int *devZ, *devW;
    int *devResult;
    int gpuCount = 0;

    int numBlocks = updiv(N,threadsPerBlock);

    ////////////////////////////
    /// GPU DATA ALLOCATION ////
    ////////////////////////////
    /// TODO :: you should malloc devX, devY, devZ, devW, and devResult
    // all arrays are integer arrays of size of N 


    //////////////////////
    /// H2D TRANSFERS ////
    //////////////////////
    /// TODO: complete H2D for hostY to devY

    //////////////////////////////////
    /// YOUR CUDA KERNEL LAUNCHES ////
    //////////////////////////////////
    gpu_initX<<<numBlocks, threadsPerBlock>>>(N, devX);
    gpu_makeZ<<<numBlocks, threadsPerBlock>>>(N, devX, devY, devZ);
    gpu_makeW<<<numBlocks, threadsPerBlock>>>(N, devZ, devW);
    exclusive_scan(devW, N, devW, threadsPerBlock); //if you want to use it
    gpu_find_pattern<<<numBlocks, threadsPerBlock>>>(N, devW, devZ, devResult);
   
    //////////////////////
    /// H2D TRANSFERS ////
    //////////////////////
    // TODO :: You should copy back devX, devY, devZ, devW, devResults
    // into arrays gpuX, gpuX, gpuZ, gpuW, gpuResults, respectively 
 
    //////////////////
    /// CUDA FREE ////
    //////////////////
    // TODO:: You should free all arrays that you allocated before!

    return gpuCount;
}

////////////////////////////////////////////////////////////////
//// EXCLUSIVE SCAN AND HELP FUNCTIONS FROM LAB3 TUTORIALS /////
////////////////////////////////////////////////////////////////

void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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



// exclusive_scan --
//
__global__ void upsweep_kernel(int N, int* output, int two_d, int two_dplus1) 
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    size_t idx = two_dplus1*i;

    if(idx<N)
    {
        output[idx+two_dplus1-1]+=output[idx+two_d-1];
    }
}

__global__ void downsweep_kernel(int N, int* output, int two_d, int two_dplus1)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    size_t idx = two_dplus1*i;

    if(idx<N)
    {
        int t=output[idx+two_d-1];
        output[idx+two_d-1]=output[idx+two_dplus1-1];
        output[idx+two_dplus1-1]+=t;
    }
}


void exclusive_scan(int* input, int N, int* result, int threadsPerBlock)
{
    int numThreadBlocks;

    for(int two_d=1; two_d<nextPow2(N)/2; two_d*=2)
    {
        int two_dplus1=2*two_d;
        numThreadBlocks = updiv(nextPow2(N)/two_dplus1, threadsPerBlock); 
        upsweep_kernel<<<numThreadBlocks, threadsPerBlock>>>(nextPow2(N), result, two_d, two_dplus1);
        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaMemset(result+nextPow2(N)-1, 0, sizeof(int)));

    for(int two_d=nextPow2(N)/2; two_d>=1; two_d/=2)
    {
        int two_dplus1=2*two_d;
        numThreadBlocks = updiv(nextPow2(N)/two_dplus1, threadsPerBlock);        
        downsweep_kernel<<<numThreadBlocks, threadsPerBlock>>>(nextPow2(N), result, two_d, two_dplus1);
        cudaCheckError(cudaDeviceSynchronize());
    }
}

// cudaScanThrust --
//
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}