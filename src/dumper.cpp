#include "dumper.h"
#include "logger.h"

#include <filesystem>
#include <cuda_runtime.h>

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

bool ToggleDumping(const std::string& dump_path) {
    if (dump_path.empty()) {
        m_dumps_enabled = false;
        m_dump_path = "";
        LogMessage("Debug dumps disabled.");
        return false;
    }
    
    m_dumps_enabled = true;
    m_dump_path = dump_path;

    // Create necessary directories
    bool dirs_created = create_directory_if_not_exists(m_dump_path);

    // FIXME: REWORK THIS
    dirs_created &= create_directory_if_not_exists(m_dump_path + "/rgb");
    dirs_created &= create_directory_if_not_exists(m_dump_path + "/depth");
    dirs_created &= create_directory_if_not_exists(m_dump_path + "/output_depth");

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
    std::string file_path = m_dump_path + "/" + subdir + "/rgb_" + std::to_string(dump_id) + ".png";

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

    LogMessage("Dumping float RGB image of size {}x{} to {}/rgb_{}.png", width, height, subdir, dump_id);
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
    LogMessage("Dumping uint8 RGB image of size {}x{} to {}/rgb_{}.png", width, height, subdir, dump_id);

    return _dump_RGB_image(h_image, width, height, num_channels, subdir, dump_id);

}

void DumpDepthImageFromCuda(const float* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id) {
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

    // Find min and max depth values for normalization
    float min_val = h_depth[0];
    float max_val = h_depth[0];
    for (size_t i = 1; i < num_pixels; ++i) {
        if (h_depth[i] < min_val) min_val = h_depth[i];
        if (h_depth[i] > max_val) max_val = h_depth[i];
    }

    // Allocate host memory for 8-bit grayscale image
    std::vector<uint8_t> h_depth_u8(num_pixels);
    float range = max_val - min_val;

    // Normalize and convert to uint8
    if (range > 0) {
        for (size_t i = 0; i < num_pixels; ++i) {
            h_depth_u8[i] = static_cast<uint8_t>(((h_depth[i] - min_val) / range) * 255.0f);
        }
    } else {
        // If range is zero, the image is constant. Set to mid-gray.
        std::fill(h_depth_u8.begin(), h_depth_u8.end(), 128);
    }

    // Generate filename
    std::string subdir_path = m_dump_path + "/" + subdir;
    if (!create_directory_if_not_exists(subdir_path)) {
        LogMessage("Failed to create directory: {}", subdir_path);
        return;
    }
    std::string file_path = m_dump_path + "/" + subdir + "/depth_" + std::to_string(dump_id) + ".png";

    // Write image to file
    const int channels = 1; // Grayscale
    const int stride_in_bytes = width * channels;
    if (!stbi_write_png(file_path.c_str(), width, height, channels, h_depth_u8.data(), stride_in_bytes)) {
        LogMessage("Failed to write depth image to: {}", file_path);
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