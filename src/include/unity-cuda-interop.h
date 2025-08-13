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

// Unity interface
static IUnityInterfaces*      s_Unity  = nullptr;
static IUnityGraphics*        s_Gfx    = nullptr;
static IUnityGraphicsD3D12v8* s_Gfx12  = nullptr;
static ID3D12Device*          s_Device = nullptr;
static ID3D12CommandQueue*  s_CmdQueue = nullptr;  // added global command queue

// DX12 state
static ID3D12CommandAllocator*    s_CmdAlloc   = nullptr;
static ID3D12GraphicsCommandList* s_CmdList    = nullptr;
static ID3D12Fence*               s_Fence      = nullptr;
static HANDLE                     s_FenceEvent = nullptr;
static UINT64                     s_FenceValue = 1;

// Cache entry holds our shared D3D12 buffer + CUDA import info
struct DX12InteropCacheEntry {
    ID3D12Resource*        sharedBuf = nullptr;
    cudaExternalMemory_t   extMem    = {};
    CUdeviceptr            cudaPtr   = 0;
};

bool registerOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
);

bool uploadNextImageSetToUnity(int32_t robot_id);

}