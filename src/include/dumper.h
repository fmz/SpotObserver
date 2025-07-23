#pragma once

#include <string>

namespace SOb {

bool ToggleDumping(const std::string& dump_path);
void DumpRGBImageFromCuda(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);
void DumpRGBImageFromCuda(const uint8_t* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);
void DumpDepthImageFromCuda(const float* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);
void DumpDepthImageFromCuda(const uint16_t* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id);


}