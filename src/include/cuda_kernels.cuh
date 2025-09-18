#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace SOb {

cudaError_t setOutputToOnes_launcher(float* d_input, float* d_output, int size);

// size_t depth_preprocessor_get_workspace_size(int width, int height);
// cudaError_t preprocess_depth_image(
//     float* depth_image,
//     int width,
//     int height,
//     int downscale_factor,
//     uint8_t* workspace,
//     bool rotate_90_cw = false
// );

size_t depth_preprocessor2_get_workspace_size(int width, int height);
cudaError_t preprocess_depth_image2(
    float* depth_image_in,
    float* depth_image_out,
    int width,
    int height,
    bool do_nearest_neighbor_fill,
    int downscale_factor,
    uint8_t* workspace,
    bool rotate_90_cw = false,
    cudaStream_t stream = 0
);

cudaError_t postprocess_depth_image(
    float* depth_image,
    int width,
    int height,
    float* workspace,
    bool rotate_90_ccw = false,
    cudaStream_t stream = 0
);

void convert_uint8_img_to_float_img(
    const uint8_t* d_in,  // [N,H,W,4] or [N,H,W,3]
    float* d_out,         // [N,3,H,W] or [N,3,W,H] if rotated
    int N, int H, int W, int C,
    bool normalize = true,
    cudaStream_t stream = 0,
    bool rotate_90_cw = false
);

void loadImageToCudaFloatRGB(const std::string& path, int& outW, int& outH, float* d_image);

// Running average depth maintenance
cudaError_t prefill_invalid_depth(
    float* d_depth_data,
    float* d_depth_out,
    float* d_depth_cache,
    int width,
    int height,
    float min_valid_depth = 0.01f,
    float max_valid_depth = 100.0f,
    cudaStream_t stream = 0
);
cudaError_t update_depth_cache(
    const float* generated_depth,
    const float* sparse_depth,
    float* cached_depth, // Input/output: running average buffer
    int width,
    int height,
    float min_valid_depth = 0.01f,
    float max_valid_depth = 100.0f,
    cudaStream_t stream = 0
);

}