#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace SOb {

cudaError_t setOutputToOnes_launcher(float* d_input, float* d_output, int size);

// Parameters for reprojecting a raw depth image into an RGB camera's image plane
// (client-side equivalent of Spot's *_depth_in_visual_frame sources).
// Folded projection, built on the CPU from intrinsics + extrinsics:
//   p = z * M * [u, v, 1]^T + t,  with M = K_rgb * R * K_depth^-1 and t = K_rgb * translation
// where (u, v, z) is a depth pixel and (p.x/p.z, p.y/p.z, p.z) is its RGB-frame pixel + depth.
struct DepthRegistrationParams {
    float M[9]; // Row-major 3x3
    float t[3];
    float inv_depth_scale{0.f}; // Meters per raw uint16 unit (1 / ImageSource.depth_scale)
    int src_width{0};
    int src_height{0};
    int dst_width{0};
    int dst_height{0};
};

// Reprojects d_depth_in (src-sized, raw uint16 as served by Spot, converted to
// meters via inv_depth_scale) into d_depth_out (dst-sized, meters, fully
// overwritten; pixels with no depth sample are 0).
cudaError_t register_depth_to_rgb(
    const uint16_t* d_depth_in,
    float* d_depth_out,
    const DepthRegistrationParams& params,
    cudaStream_t stream = 0
);

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
// Fused output postprocess + EMA cache update: optionally rotates the
// generated depth back into the sensor orientation (via workspace), folds it
// into the running cache, and writes the blended result back over
// generated_depth as the published output — one pass, no extra copies.
cudaError_t postprocess_depth_image(
    float* generated_depth, // Input: model output; overwritten with the blended result
    const float* sparse_depth,
    float* cached_depth,    // Input/output: running average buffer
    float alpha_valid,      // If old and new depth are valid, use this alpha
    float alpha_invalid,    // If old depth is valid but new depth is invalid, use this alpha (between old depth and generated depth)
    int width,
    int height,
    float* workspace,       // >= width*height floats; only used when rotating
    bool rotate_90_ccw = false,
    float min_valid_depth = 0.01f,
    float max_valid_depth = 100.0f,
    cudaStream_t stream = 0
);

}