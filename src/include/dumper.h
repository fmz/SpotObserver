#pragma once

#include <string>

namespace SOb {

bool ToggleDumping(const std::string& dump_path);
void DumpRGBImageFromCudaCHW(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id_major, int32_t dump_id_minor = 0);
void DumpRGBImageFromCudaHWC(const float* image, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id_major, int32_t dump_id_minor = 0);
void DumpRGBImageFromCuda(const uint8_t* image, int32_t width, int32_t height, int32_t num_channels, const std::string& subdir, int32_t dump_id_major, int32_t dump_id_minor = 0);
void DumpDepthImageFromCuda(const float* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id_major, int32_t dump_id_minor = 0);
void DumpDepthImageFromCuda(const uint16_t* depth, int32_t width, int32_t height, const std::string& subdir, int32_t dump_id_major, int32_t dump_id_minor = 0);


}