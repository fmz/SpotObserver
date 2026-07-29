//
// Created by fmz on 8/13/2025.
//

#pragma once

#include <cstdint>
#include "IUnityInterface.h"

namespace SOb {

void initUnityInterop(IUnityInterfaces* unity);
void shutdownUnityInterop();

bool registerOutputTextures(
    int32_t robot_id,
    int32_t cam_stream_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
);

bool uploadNextImageSetToUnity(int32_t robot_id, int32_t cam_stream_id);
bool uploadNextVisionPipelineImageSetToUnity(int32_t robot_id, int32_t cam_stream_id);
bool clearOutputTextures(int32_t robot_id);

}