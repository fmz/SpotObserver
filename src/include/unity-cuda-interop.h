//
// Created by fmz on 8/13/2025.
//

#pragma once

#include "logger.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cuda.h>

#ifdef _WIN32
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <libloaderapi.h>

#else
#error "Only Windows is supported for D3D12"
#endif

// This ordering is required unfortunately
#include "spot-observer.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"

namespace SOb {

bool registerOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
);

bool uploadNextImageSetToUnity(int32_t robot_id);
bool uploadNextVisionPipelineImageSetToUnity(int32_t robot_id);

}