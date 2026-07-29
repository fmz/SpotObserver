#pragma once

#include <string>

namespace SOb {

bool ToggleDumping(const std::string& dump_path);
void DumpRGBImageFromCudaCHW(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);
void DumpRGBImageFromCudaHWC(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);
void DumpRGBImageFromCuda(const uint8_t* image, int32_t width, int32_t height, int32_t num_channels, const std::string& subdir, int32_t dump_id);
// png_mode=false writes full-precision float32 binary (count-prefixed, reloadable);
// png_mode=true writes a min/max-normalized 8-bit grayscale PNG for visualization.
void DumpDepthImageFromCuda(const float* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id, bool png_mode = false);
// scale converts raw u16 units to meters (e.g. 1 / ImageSource.depth_scale).
void DumpDepthImageFromCuda(const uint16_t* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id, float scale = 1.f);
void DumpCameraTransform(const std::string& name, const std::string& subdir, const float* transform, int32_t dump_id);


}
