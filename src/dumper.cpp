#include "dumper.h"
#include "logger.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>
#include <torch/torch.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace SOb {

static bool m_dumps_enabled = false;
static std::string m_dump_path = "";

namespace fs = std::filesystem;

bool create_directory_if_not_exists(const std::string& path) {
    try {
        bool exists = fs::exists(path);
        if (exists) {
            if (fs::is_directory(path)) {
                return true; // Directory already exists
            } else {
                LogMessage("Path exists but is not a directory: {}", path);
                return false; // Path exists but is not a directory
            }
        }
        fs::create_directories(path);
    } catch (const fs::filesystem_error& e) {
        LogMessage("Failed to create directory: {}. Error: {}", path, e.what());
        return false; // Failed to create directory
    }
    
    return true; // Directory created successfully
}

static std::string get_file_path(
    const std::string& path,
    const std::string& subdir,
    const std::string& base_name,
    int32_t dump_id,
    const std::string& extension
) {
    return path + "/" + subdir + "/" + base_name + std::to_string(dump_id) + extension;
}

bool ToggleDumping(const std::string& dump_path) {
    if (dump_path.empty()) {
        m_dumps_enabled = false;
        m_dump_path = "";
        LogMessage("Debug dumps disabled.");
        return false;
    }
    
    m_dumps_enabled = true;
    m_dump_path = dump_path;

    bool dirs_created = create_directory_if_not_exists(m_dump_path);

    if (!dirs_created) {
        LogMessage("Failed to create debug dump directories at: {}", m_dump_path);
        m_dumps_enabled = false;
        m_dump_path = "";
        return false;
    }
    
    LogMessage("Debug dumps enabled for path: {}", m_dump_path);
    return true;
}

static void _dump_RGB_image(
    const std::vector<uint8_t>& hwc_image,
    int32_t width,
    int32_t height,
    int32_t channels,
    const std::string& subdir,
    int32_t dump_id
) {
    // Generate filename
    std::string subdir_path = m_dump_path + "/" + subdir;
    if (!create_directory_if_not_exists(subdir_path)) {
        LogMessage("_dump_RGB_image: Failed to create directory: {}", subdir_path);
        return;
    }
    std::string file_path = get_file_path(
        m_dump_path,
        subdir,
        "",
        dump_id,
        ".png"
    );

    // Write image to file
    const int stride_in_bytes = width * channels;
    if (!stbi_write_png(file_path.c_str(), width, height, channels, hwc_image.data(), stride_in_bytes)) {
        LogMessage("Failed to write RGB image to: {}", file_path);
    }
}


void DumpRGBImageFromCudaHWC(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id) {
    if (!m_dumps_enabled) {
        return;
    }

    const size_t num_pixels = width * height;
    const size_t float_size = num_pixels * 3 * sizeof(float);

    // Allocate host memory for float image
    std::vector<float> h_image(num_pixels * 3);

    // Copy image from device to host
    cudaError_t err = cudaMemcpy(h_image.data(), image, float_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LogMessage("Failed to copy RGB image from device to host: {}", cudaGetErrorString(err));
        return;
    }

    std::vector<uint8_t> h_image_u8(num_pixels * 3);

    // Convert float [0,1] to uint8 [0,255]
    for (size_t i = 0; i < num_pixels; ++i) {
        h_image_u8[i * 3 + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i * 3 + 0])) * 255.0f);
        h_image_u8[i * 3 + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i * 3 + 1])) * 255.0f);
        h_image_u8[i * 3 + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i * 3 + 2])) * 255.0f);
    }

   return _dump_RGB_image(h_image_u8, width, height, 3, subdir, dump_id);
}

void DumpRGBImageFromCudaCHW(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id) {
    if (!m_dumps_enabled) {
        return;
    }

    const size_t num_pixels = width * height;
    const size_t float_size = num_pixels * 3 * sizeof(float);

    // Allocate host memory for float image
    std::vector<float> h_image(num_pixels * 3);

    // Copy image from device to host
    cudaError_t err = cudaMemcpy(h_image.data(), image, float_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LogMessage("Failed to copy RGB image from device to host: {}", cudaGetErrorString(err));
        return;
    }

    std::vector<uint8_t> h_image_u8(num_pixels * 3);

    // Convert float [0,1] to uint8 [0,255], and convert the format to HWC instead of CHW
    for (size_t i = 0; i < num_pixels; ++i) {
        h_image_u8[i * 3 + 0] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i])) * 255.0f);
        h_image_u8[i * 3 + 1] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i + num_pixels])) * 255.0f);
        h_image_u8[i * 3 + 2] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, h_image[i + 2 * num_pixels])) * 255.0f);
    }

    LogMessage("Dumping float RGB image of size {}x{} to {}/{}.png", width, height, subdir, dump_id);
    return _dump_RGB_image(h_image_u8, width, height, 3, subdir, dump_id);
}

void DumpRGBImageFromCuda(
    const uint8_t* image,
    int32_t width,
    int32_t height,
    int32_t num_channels,
    const std::string& subdir,
    int32_t dump_id
) {
    if (!m_dumps_enabled) {
        return;
    }

    const size_t num_pixels = width * height;
    const size_t num_elements = num_pixels * num_channels;
    const size_t byte_size = num_elements;

    // Allocate host memory
    std::vector<uint8_t> h_image(byte_size);

    // Copy image from device to host
    cudaError_t err = cudaMemcpy(h_image.data(), image, byte_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LogMessage("Failed to copy RGB image from device to host: {}", cudaGetErrorString(err));
        return;
    }
    LogMessage("Dumping uint8 RGB image of size {}x{} to {}/{}.png", width, height, subdir, dump_id);

    return _dump_RGB_image(h_image, width, height, num_channels, subdir, dump_id);
}

void DumpDepthImageFromCuda(
    const float* depth,
    int32_t width,
    int32_t height,
    const std::string& subdir,
    int32_t dump_id,
    bool png_mode
) {
    if (!m_dumps_enabled) {
        return;
    }

    const size_t num_pixels = width * height;
    const size_t float_size = num_pixels * sizeof(float);

    // Allocate host memory for float depth image
    std::vector<float> h_depth(num_pixels);

    // Copy depth image from device to host
    cudaError_t err = cudaMemcpy(h_depth.data(), depth, float_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LogMessage("Failed to copy depth image from device to host: {}", cudaGetErrorString(err));
        return;
    }

    // Ensure the output subdir exists (shared by both modes).
    std::string subdir_path = m_dump_path + "/" + subdir;
    if (!create_directory_if_not_exists(subdir_path)) {
        LogMessage("Failed to create directory: {}", subdir_path);
        return;
    }

    if (png_mode) {
        // PNG mode: min/max-normalize to 8-bit grayscale for visualization.
        // This is lossy and must not be used when the depth needs to be reloaded.
        float min_val = h_depth[0];
        float max_val = h_depth[0];
        for (size_t i = 1; i < num_pixels; ++i) {
            if (h_depth[i] < min_val) min_val = h_depth[i];
            if (h_depth[i] > max_val) max_val = h_depth[i];
        }

        // Allocate host memory for 8-bit grayscale image
        std::vector<uint8_t> h_depth_u8(num_pixels);
        const float range = max_val - min_val;

        // Normalize and conveert to uint8
        if (range > 0.0f) {
            for (size_t i = 0; i < num_pixels; ++i) {
                h_depth_u8[i] = static_cast<uint8_t>(((h_depth[i] - min_val) / range) * 255.0f);
            }
        } else {
            // If range is zero, the image is constant. Set to mid-gray.
            std::fill(h_depth_u8.begin(), h_depth_u8.end(), 128);
        }

        std::string file_path = get_file_path(m_dump_path, subdir, "", dump_id, ".png");
        const int channels = 1; // Grayscale
        const int stride_in_bytes = width * channels;
        if (!stbi_write_png(file_path.c_str(), width, height, channels, h_depth_u8.data(), stride_in_bytes)) {
            LogMessage("Failed to write depth image to: {}", file_path);
        }
        return;
    }

    // Raw mode (default): write the full-precision float32 buffer, prefixed with
    // the element count, so it can be reloaded losslessly.
    std::string file_path = get_file_path(
        m_dump_path,
        subdir,
        "",
        dump_id,
        ""
    );

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        LogMessage("Failed to write depth image to: {}", file_path);
        return;
    }

    const uint32_t count = static_cast<uint32_t>(h_depth.size());
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(h_depth.data()), static_cast<std::streamsize>(h_depth.size() * sizeof(float)));

    if (!file) {
        LogMessage("Failed while writing depth image to: {}", file_path);
    }
}

void DumpCameraTransform(
    const std::string& name,
    const std::string& subdir,
    const float* transform,
    int32_t dump_id
) {
    if (!m_dumps_enabled) {
        return;
    }
    if (transform == nullptr) {
        LogMessage("DumpCameraTransform: Cannot dump null transform {}", name);
        return;
    }

    std::string subdir_path = m_dump_path + "/" + subdir;
    if (!create_directory_if_not_exists(subdir_path)) {
        LogMessage("DumpCameraTransforms: Failed to create directory: {}", subdir_path);
        return;
    }

    // File name includes frame_idx only if id > 0; if passing in -1, (body_T_cam), write file without idx.
    const std::string file_name = dump_id >= 0
        ? name + "_" + std::to_string(dump_id) + ".pt"
        : name + ".pt";
    std::string file_path = m_dump_path + "/" + subdir + "/" + file_name;

    std::array<float, 16> transform_copy;
    std::copy(transform, transform + transform_copy.size(), transform_copy.begin());

    torch::Tensor transform_tensor = torch::from_blob(
        transform_copy.data(),
        {4, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    try {
        torch::save(transform_tensor, file_path);
    } catch (const std::exception& e) {
        LogMessage("DumpCameraTransform: Failed to write transform to {}: {}", file_path, e.what());
        return;
    }
}

// void DumpDepthImageFromCuda(const uint16_t* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id) {
//     if (!m_dumps_enabled) {
//         return;
//     }
//
//     const size_t num_pixels = width * height;
//     const size_t byte_size = num_pixels * sizeof(uint16_t);
//
//     // Allocate host memory for float depth image
//     std::vector<uint16_t> h_depth(num_pixels);
//
//     // Copy depth image from device to host
//     cudaError_t err = cudaMemcpy(h_depth.data(), depth, byte_size, cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess) {
//         LogMessage("Failed to copy depth image from device to host: {}", cudaGetErrorString(err));
//         return;
//     }
//
//     // Find min and max depth values for normalization
//     uint16_t min_val = h_depth[0];
//     uint16_t max_val = h_depth[0];
//     for (size_t i = 1; i < num_pixels; ++i) {
//         if (h_depth[i] < min_val) min_val = h_depth[i];
//         if (h_depth[i] > max_val) max_val = h_depth[i];
//     }
//
//     // Allocate host memory for 8-bit grayscale image
//     std::vector<uint8_t> h_depth_u8(num_pixels);
//     uint16_t range = max_val - min_val;
//
//     // Normalize and convert to uint8
//     if (range > 0) {
//         for (size_t i = 0; i < num_pixels; ++i) {
//             h_depth_u8[i] = static_cast<uint8_t>(((h_depth[i] - min_val) / range) * 255.0f);
//         }
//     } else {
//         // If range is zero, the image is constant. Set to mid-gray.
//         std::fill(h_depth_u8.begin(), h_depth_u8.end(), 128);
//     }
//
//     // Generate filename
//     std::string file_path = m_dump_path + "/" + subdir + "/depth_" + std::to_string(dump_id) + ".png";
//
//     // Write image to file
//     const int channels = 1; // Grayscale
//     const int stride_in_bytes = width * channels;
//     if (!stbi_write_png(file_path.c_str(), width, height, channels, h_depth_u8.data(), stride_in_bytes)) {
//         LogMessage("Failed to write depth image to: {}", file_path);
//     }
// }

} // namespace STb
