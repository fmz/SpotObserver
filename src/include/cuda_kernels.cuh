#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace SOb {

cudaError_t setOutputToOnes_launcher(float* d_input, float* d_output, int size);

cudaError_t preprocess_depth_image(
    float* depth_image,
    int width,
    int height
);

size_t depth_preprocessor_get_workspace_size(int width, int height);
cudaError_t preprocess_depth_image2(
    float* depth_image,
    int width,
    int height,
    uint8_t* workspace
);

void cleanup_depth_preprocessor();

void convert_uint8_img_to_float_img(
    const uint8_t* d_in,  // [N,H,W,4] or [N,H,W,3]
    float* d_out,         // [N,3,H,W]
    int N, int H, int W, int C,
    bool normalize = true,
    cudaStream_t stream = 0
);

void loadImageToCudaFloatRGB(const std::string& path, int& outW, int& outH, float* d_image);


}