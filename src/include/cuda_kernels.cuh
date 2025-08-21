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

void convert_uint8_img_to_float_img(
    const uint8_t* input_data,
    float* output_data,
    int batch_size,
    int input_channels,
    int height,
    int width,
    cudaStream_t stream = nullptr
);

}