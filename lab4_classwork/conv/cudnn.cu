#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <string>

#include "../utils/CycleTimer.h"
#include "../utils/stb_image.h"
#include "../utils/stb_image_write.h"

enum Mode { FP32 = 1,
            FP16_32 = 2 };

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
#else
#define cudaCheckError(ans) ans
#endif

// Macro for accessing filter array
#define id_filter(f, c, j, i, num_filters, num_channels, filter_size) ((f) * (num_channels) * (filter_size) * (filter_size) + (c) * (filter_size) * (filter_size) + (j) * (filter_size) + (i))

// Function definitions to reduce the output of the convolution to a single channel
void reduce_channels_cudnn(float *output_r, float *output, int out_w, int out_h, int k);
void reduce_channels_cudnn(float *output_r, __half *output, int out_w, int out_h, int k);
void reduce_channels_cudnn(float *output_r, void *output_, Mode mode, int out_w, int out_h, int k);

/**
 * @brief Builds a convolutional graph using cuDNN frontend API.
 *
 * This function constructs a convolutional graph with the specified parameters
 * and returns a tuple containing the graph and its associated tensors.
 *
 * @param n Number of images in the batch.
 * @param c Number of input feature maps.
 * @param h Height of the input feature map.
 * @param w Width of the input feature map.
 * @param k Number of output feature maps.
 * @param r Height of the convolution filter.
 * @param s Width of the convolution filter.
 * @param padding Padding size for the convolution.
 * @param stride Stride size for the convolution.
 * @param dilation Dilation size for the convolution.
 * @param handle cuDNN handle for managing the cuDNN library context.
 * @param mode Mode specifying the data type and computation type (FP32, FP16_32, TF32).
 *
 * @return A tuple containing:
 *         - A shared pointer to the constructed graph.
 *         - The input tensor.
 *         - The filter tensor.
 *         - The output tensor.
 */
auto build_graph(int n, int c, int h, int w, int k, int r, int s, int padding,
                 int stride, int dilation, cudnnHandle_t handle, Mode mode) {
  // create a graph object
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  // If we are executing with FP32 or TF32, set the input and output data type to FLOAT
  if (mode & FP32) {
    graph->set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  }  // If we are executing with FP16_32, set the input and output data type to HALF and the accumulation data type to FLOAT
  else if (mode & FP16_32) {
    graph->set_io_data_type(cudnn_frontend::DataType_t::HALF)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  }

  // Declare input tensor with its characteristics
  auto input = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                 .set_name("input")
                                 .set_dim({n, c, h, w})
                                 .set_stride({c * h * w, 1, c * w, c}));

  // Declare filter tensor with its characteristics
  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("filter")
                             .set_dim({k, c, r, s})
                             .set_stride({c * r * s, r * s, s, 1}));

  // Declare convolution operation with its characteristics
  auto conv_options = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_padding({padding, padding})
                          .set_stride({stride, stride})
                          .set_dilation({dilation, dilation});

  // Add a forward convolution operation to the graph with the input and filter tensors and the convolution options we just defined
  auto Y = graph->conv_fprop(input, W, conv_options);

  // Declare the output tensor
  Y->set_output(true);

  // Validate the graph
  graph->validate().is_good();

  graph->build_operation_graph(handle).is_good();

  graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}).is_good();

  // If we are executing with FP16_32 or TF32, force the use of Tensor Cores
  if (mode & FP16_32) {
    graph->select_numeric_notes(std::vector<cudnn_frontend::NumericalNote_t>(
        1, cudnn_frontend::NumericalNote_t::TENSOR_CORE));
  }

  graph->check_support(handle).is_good();

  // Finish building the computation graph
  graph->build_plans(handle).is_good();

  // Return the graph and its associated tensors
  return std::make_tuple(graph, input, W, Y);
}

void cudnn_convolution(float *data_f, float *filters, int k, int c, int filter_size, int h, int w, int out_h, int out_w, Mode mode) {
  int n = 1;            // number of images
  int r = filter_size;  // filter size
  int s = filter_size;  // filter size
  int padding = 0;      // padding
  int stride = 1;       // stride
  int dilation = 1;     // dilation

  // Allocate the output array in the heap
  float *output = new float[out_h * out_w * k];

  // convert input to half precision in case we are working with FP16_32
  __half *data_h = new __half[n * c * h * w];
  for (int i = 0; i < n * c * h * w; i++) {
    data_h[i] = __float2half(data_f[i]);
  }

  // convert filters to half precision
  __half *filters_h = new __half[k * c * r * s];
  for (int i = 0; i < k * c * r * s; i++) {
    filters_h[i] = __float2half(filters[i]);
  }

  __half *output_h = new __half[n * k * out_h * out_w];

  // Pointer manipulation to allow for the different data types with no additional code
  int size_io = sizeof(float);
  void *data_ = data_f;
  void *filters_ = filters;
  void *output_ = output;
  if (mode & FP16_32) {
    size_io = sizeof(__half);
    data_ = data_h;
    filters_ = filters_h;
    output_ = output_h;
  }

  // start measuring the time of the actual execution
  double start_GPU = CycleTimer::currentSeconds();

  // allocate memory for input, filter, and output tensors on device
  float *deviceInput, *deviceFilter, *deviceOutput;
  cudaCheckError(cudaMalloc((void **)&deviceInput, n * c * h * w * size_io));
  cudaCheckError(cudaMalloc((void **)&deviceFilter, k * c * r * s * size_io));
  cudaCheckError(cudaMalloc((void **)&deviceOutput, n * k * out_h * out_w * sizeof(float)));

  // copy input and filter tensors to device
  cudaCheckError(cudaMemcpy(deviceInput, data_, n * c * h * w * size_io, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(deviceFilter, filters_, k * c * r * s * size_io, cudaMemcpyHostToDevice));
  cudaCheckError(cudaDeviceSynchronize());

  // create a cuDNN handle
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  // build the graph
  double build_start = CycleTimer::currentSeconds();
  auto [graph, input, W, Y] =
      build_graph(n, c, h, w, k, r, s, padding, stride, dilation, handle, mode);
  double build_end = CycleTimer::currentSeconds();
  int8_t *workspace_ptr;
  cudaCheckError(
      cudaMalloc((void **)&workspace_ptr, graph->get_workspace_size()));

  // pack the input, filter, and output tensors into a map
  // Basically, this is a dictionary that maps the tensor in the graph to its corresponding device memory pointer
  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void *>
      variant_pack = {{input, deviceInput}, {W, deviceFilter}, {Y, deviceOutput}};

  // Measure the execution time of the graph
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // execute the graph
  auto status = graph->execute(handle, variant_pack, workspace_ptr);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // print the execution time and performance
  printf("Kernel Time (cuDNN): %f ms\n", elapsedTime);
  printf("GFLOPS (cuDNN): %f\n", 2.0 * r * s * c * k * out_w * out_h / (elapsedTime * 1e-3) / 1e9);

  // copy the output tensor back to the host
  cudaCheckError(cudaMemcpy(output_, deviceOutput, n * k * out_h * out_w * size_io, cudaMemcpyDeviceToHost));

  // measure the time of the whole process
  double end_GPU = CycleTimer::currentSeconds();
  printf("Build time: %.3f ms\n", (build_end - build_start) * 1000);
  printf("Total time: %.3f ms\n", (end_GPU - start_GPU) * 1000);

  // reduce the output to a single channel
  float *output_r = new float[out_w * out_h];

  reduce_channels_cudnn(output_r, output_, mode, out_w, out_h, k);

  unsigned char *data_int = new unsigned char[out_w * out_h];

  //  convert float to unsigned char
  for (int i = 0; i < out_w * out_h; i++) {
    data_int[i] = (unsigned char)round(output_r[i] * 255.f);
  }

  stbi_write_png("output_cudnn.png", out_w, out_h, 1, data_int, out_w);

  // free memory
  cudaCheckError(cudaFree(deviceInput));
  cudaCheckError(cudaFree(deviceFilter));
  cudaCheckError(cudaFree(deviceOutput));

  delete[] output;
  delete[] data_h;
  delete[] filters_h;
  delete[] output_h;
  delete[] output_r;
  delete[] data_int;

  return;
}

void reduce_channels_cudnn(float *output_r, float *output, int out_w, int out_h, int k) {
  // reduce the output of the convolution to a single channel by summing the activations of each channel
  // with some edge case fixing
  for (int j = 0; j < out_h; j++)
    for (int i = 0; i < out_w; i++) {
      output_r[j * out_w + i] = 0;
      for (int f = 0; f < k; f++) {
        output[(j * out_w + i) * k + f] = output[(j * out_w + i) * k + f] > 0 ? output[(j * out_w + i) * k + f] : -output[(j * out_w + i) * k + f];
        if (output_r[j * out_w + i] + output[(j * out_w + i) * k + f] > 1.f) {
          output_r[j * out_w + i] = 1.f;
          continue;
        } else
          output_r[j * out_w + i] += output[(j * out_w + i) * k + f];
      }
    }
  return;
}

void reduce_channels_cudnn(float *output_r, __half *output, int out_w, int out_h, int k) {
  // reduce the output of the convolution to a single channel by summing the activations of each channel
  // with some edge case fixing
  for (int j = 0; j < out_h; j++)
    for (int i = 0; i < out_w; i++) {
      output_r[j * out_w + i] = 0;
      for (int f = 0; f < k; f++) {
        output[(j * out_w + i) * k + f] = output[(j * out_w + i) * k + f] > __float2half(0.f) ? output[(j * out_w + i) * k + f] : -output[(j * out_w + i) * k + f];
        if (output_r[j * out_w + i] + __half2float(output[(j * out_w + i) * k + f]) > 1.f) {
          output_r[j * out_w + i] = 1.f;
          continue;
        } else
          output_r[j * out_w + i] += __half2float(output[(j * out_w + i) * k + f]);
      }
    }
  return;
}

// Wrapper function to allow for the different data types with no additional code through parameter overloading
void reduce_channels_cudnn(float *output_r, void *output_, Mode mode, int out_w, int out_h, int k) {
  if (mode & FP32) {
    reduce_channels_cudnn(output_r, (float *)output_, out_w, out_h, k);
  } else if (mode & FP16_32) {
    reduce_channels_cudnn(output_r, (__half *)output_, out_w, out_h, k);
  }
  return;
}