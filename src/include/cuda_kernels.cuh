#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace SOb {

cudaError_t setOutputToOnes_launcher(float* d_input, float* d_output, int size);

cudaError_t preprocess_depth_image(
    float* depth_image,
    int width,
    int height
);

void cleanup_depth_preprocessor();

cudaError_t testCudaKernel(float* d_input, float* d_output, int size) {


}