#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

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

static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// we can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  
//
// Also, as per the comments in cudaScan(), we can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

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


//More kernel more gooder
//Alters mask so that it is set everytime A[i] = A[i+1]
__global__ void songo_cu(int* input, int *mask)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(input[i] == input[i+1]){
		mask[i] = 1;
	}
	else {
		mask[i] = 0;
	}
    printf("Set value %d and %d and %d\n",input[i], input[i+1],mask[i]);
}

//More kernel more gooder
__global__ void sango_cu(int* input, int *mask, int *output)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(mask[i] == 1){
		output[input[i]] = i;
	}
    //printf("Set value %d and %d and %d\n",input[i], output[input[i]],mask[i]);
}


void exclusive_scan(int* input, int N, int* result)
{

    // Exclusive scan implementation is provided here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  This implementation makes multiple calls
    // to CUDA kernel functions to execute the scan

    int numThreadBlocks;

    for(int two_d=1; two_d<nextPow2(N)/2; two_d*=2)
    {
        int two_dplus1=2*two_d;
        numThreadBlocks = updiv(nextPow2(N)/two_dplus1, THREADS_PER_BLOCK); 
        upsweep_kernel<<<numThreadBlocks, THREADS_PER_BLOCK>>>(nextPow2(N), result, two_d, two_dplus1);
        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaMemset(result+nextPow2(N)-1, 0, sizeof(int)));

    for(int two_d=nextPow2(N)/2; two_d>=1; two_d/=2)
    {
        int two_dplus1=2*two_d;
        numThreadBlocks = updiv(nextPow2(N)/two_dplus1, THREADS_PER_BLOCK);        
        downsweep_kernel<<<numThreadBlocks, THREADS_PER_BLOCK>>>(nextPow2(N), result, two_d, two_dplus1);
        cudaCheckError(cudaDeviceSynchronize());
    }
}



//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_result);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Note that the provided implementation is not expected to achieve
// performance that is competition to the Thrust version, but it is fun to try.
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


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // STUDENTS TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size.
	int *mask;
	int *indexH, *maskH;
	int *index;
    int rounded_length = nextPow2(length);
	int final_size = 0;

	indexH = (int *) malloc(rounded_length * sizeof(int));
	maskH = (int *) malloc(rounded_length * sizeof(int));
	cudaMalloc(&mask, rounded_length * sizeof(int));
	cudaMalloc(&index, rounded_length * sizeof(int));
	printf("allocation done\n");
	int numBlocks = rounded_length/THREADS_PER_BLOCK;
	songo_cu<<<numBlocks, THREADS_PER_BLOCK>>>(device_input, mask);
    cudaCheckError(cudaDeviceSynchronize());
	printf("Songo feito\n");
    cudaMemcpy(index, mask,  rounded_length * sizeof(int), cudaMemcpyDeviceToDevice);
	for(int i =0; i < length; i++){
		printf("| %d \t|", maskH[i]);
	}


    exclusive_scan(mask, length, index);
    cudaCheckError(cudaDeviceSynchronize());
	printf("scan feito\n");

	sango_cu<<<numBlocks, THREADS_PER_BLOCK>>>(index, mask, device_output);
    cudaCheckError(cudaDeviceSynchronize());
	printf("sango feito\n");

    cudaMemcpy(indexH, index, rounded_length * sizeof(int), cudaMemcpyDeviceToHost);
	final_size = indexH[length];
	for(int i =0; i < length; i++){
		printf("| %d \t|", indexH[i]);
	}

	cudaFree(mask);
	cudaFree(index);
	free(indexH);

	printf("final size: %d\n", final_size);
	printf("length: %d\n", length);


    return final_size; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



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
