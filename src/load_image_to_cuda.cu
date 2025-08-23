//
// Created by brown on 8/21/2025.
//

#pragma once

#include "cuda_kernels.cuh"
#include "logger.h"
#include "dumper.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace SOb {
#define CHECK_CUDA(call)                                                          \
do {                                                                          \
cudaError_t err__ = (call);                                               \
if (err__ != cudaSuccess) {                                               \
std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << " -> "   \
<< cudaGetErrorString(err__) << std::endl;                  \
std::exit(EXIT_FAILURE);                                              \
}                                                                         \
} while (0)

// Convert uint8 -> float (0..1) kernel & hwc to chw conversion
__global__ void u8_to_float_rgb(const unsigned char* __restrict__ src,
                                float* __restrict__ dst,
                                int width, int height, int src_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    int s = idx * src_channels;
    int d = idx;
    dst[d]             = src[s + 0] / 255.0f;
    dst[d + total]     = src[s + 1] / 255.0f;
    dst[d + 2 * total] = src[s + 2] / 255.0f;
}

// Loads image, returns device pointer with float RGB (H * W * 3), plus width & height.
void loadImageToCudaFloatRGB(
    const std::string& path,
    int& outW,
    int& outH,
    float* d_rgb
) {
    int w, h, comp;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 0);
    std::cerr << "image " << path << " loaded: "
              << w << "x" << h << " with " << comp << " channels\n";
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    if (comp < 3) {
        stbi_image_free(data);
        throw std::runtime_error("Image does not have at least 3 channels.");
    }

    // Allocate device destination
    size_t pixelCount = static_cast<size_t>(w) * h;

    // Option A (GPU conversion): copy raw u8 then launch kernel
    unsigned char* d_src = nullptr;
    CHECK_CUDA(cudaMalloc(&d_src, pixelCount * comp));
    CHECK_CUDA(cudaMemcpy(d_src, data, pixelCount * comp, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int((pixelCount) + threads - 1) / threads);
    u8_to_float_rgb<<<blocks, threads>>>(d_src, d_rgb, w, h, comp);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_src));

    // Cleanup host
    stbi_image_free(data);

    outW = w;
    outH = h;

    DumpRGBImageFromCudaCHW(
        d_rgb,
        w,
        h,
        "rgb",
        91
    );
}

// int main(int argc, char** argv)
// {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <image_path>\n";
//         return 1;
//     }
//     try {
//         int w = 0, h = 0;
//         float* d_image = loadImageToCudaFloatRGB(argv[1], w, h);
//         std::cout << "Loaded image " << argv[1] << " size: " << w << "x" << h
//                   << " -> device float RGB buffer\n";
//
//         // (Example) Verify by copying first few floats back
//         std::vector<float> sample(9);
//         CHECK_CUDA(cudaMemcpy(sample.data(), d_image, sample.size() * sizeof(float), cudaMemcpyDeviceToHost));
//         std::cout << "First 3 pixels (R,G,B each): ";
//         for (size_t i = 0; i < sample.size(); ++i) {
//             std::cout << sample[i] << (i + 1 == sample.size() ? '\n' : ' ');
//         }
//
//         cudaFree(d_image);
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << "\n";
//         return 2;
//     }
//     return 0;
//}

}