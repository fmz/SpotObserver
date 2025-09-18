//
// Created by fmz on 8/13/2025.
//

#include "unity-cuda-interop.h"

#include "d3dx12.h"
#include <unordered_map>
#include <string>

namespace SOb {

// Unity interface
static IUnityInterfaces*      s_Unity  = nullptr;
static IUnityGraphics*        s_Gfx    = nullptr;
static IUnityGraphicsD3D12v8* s_Gfx12  = nullptr;
static ID3D12Device*          s_Device = nullptr;
static ID3D12CommandQueue*    s_CmdQueue = nullptr;  // Our own command queue

// DX12 state
static ID3D12CommandAllocator*    s_CmdAlloc   = nullptr;
static ID3D12GraphicsCommandList* s_CmdList    = nullptr;
static ID3D12Fence*               s_Fence      = nullptr;
static HANDLE                     s_FenceEvent = nullptr;
static UINT64                     s_FenceValue = 1000;

cudaStream_t s_cudaStream = 0;

// Cache entry holds our shared D3D12 buffer + CUDA import info
struct DX12InteropCacheEntry {
    ID3D12Resource*        sharedBuf = nullptr;
    cudaExternalMemory_t   extMem    = {};
    CUdeviceptr            cudaPtr   = 0;
    size_t                 bufSize   = 0;
};

static std::unordered_map<ID3D12Resource*, DX12InteropCacheEntry> s_InteropCache = {};

// robot ID -> <D3D12 textures>
static std::unordered_map<int32_t, std::vector<ID3D12Resource*>> s_OutputRGBTextures = {};
static std::unordered_map<int32_t, std::vector<ID3D12Resource*>> s_OutputDepthTextures = {};

// Store texture footprints for proper copying
struct TextureFootprintInfo {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT numRows;
    UINT64 rowSizeInBytes;
    UINT64 totalBytes;
};
static std::unordered_map<ID3D12Resource*, TextureFootprintInfo> s_TextureFootprints = {};

static DX12InteropCacheEntry* _getOrCreateInteropEntry(ID3D12Resource* d3d12_resource, size_t buf_size_in_bytes) {
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot create interop entry.");
        return nullptr;
    }

    DX12InteropCacheEntry& cache_entry = s_InteropCache[d3d12_resource];

    // On first use, create shared D3D12 buffer + import into CUDA
    if (!cache_entry.sharedBuf) {
        LogMessage("Creating new interop entry for resource: {}. Size: {} bytes", (void*)d3d12_resource, buf_size_in_bytes);

        // Create a default-heap, shared buffer
        D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width              = buf_size_in_bytes;
        rd.Height             = 1;
        rd.DepthOrArraySize   = 1;
        rd.MipLevels          = 1;
        rd.Format             = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count   = 1;
        rd.SampleDesc.Quality = 0;
        rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = s_Device->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_SHARED,
            &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&cache_entry.sharedBuf)
        );
        if (!checkHR(hr, "CreateCommittedResource for shared buffer")) return nullptr;

        // Share it and import into CUDA
        HANDLE sharedHandle = nullptr;
        hr = s_Device->CreateSharedHandle(
            cache_entry.sharedBuf,
            nullptr,
            GENERIC_ALL,
            nullptr,
            &sharedHandle
        );
        if (!checkHR(hr, "CreateSharedHandle")) return nullptr;

        cudaExternalMemoryHandleDesc hDesc = {};
        hDesc.type                = cudaExternalMemoryHandleTypeD3D12Resource;
        hDesc.handle.win32.handle = sharedHandle;
        hDesc.size                = buf_size_in_bytes;
        hDesc.flags               = cudaExternalMemoryDedicated;

        cudaError_t cerr = cudaImportExternalMemory(&cache_entry.extMem, &hDesc);
        if (!checkCUDA(cerr, "cudaImportExternalMemory")) {
            CloseHandle(sharedHandle);
            return nullptr;
        }

        LogMessage("Created shared D3D12 buffer: {} (size: {})", (void*)cache_entry.sharedBuf, buf_size_in_bytes);

        cudaExternalMemoryBufferDesc bDesc = {};
        bDesc.offset = 0;
        bDesc.size   = buf_size_in_bytes;
        cerr = cudaExternalMemoryGetMappedBuffer(
            (void**)&cache_entry.cudaPtr,
            cache_entry.extMem,
            &bDesc
        );
        CloseHandle(sharedHandle);
        if (!checkCUDA(cerr, "cudaExternalMemoryGetMappedBuffer")) return nullptr;

        cache_entry.bufSize = buf_size_in_bytes;
        LogMessage("Mapped CUDA buffer: {} (size: {})", (void*)cache_entry.cudaPtr, buf_size_in_bytes);
    }
    return &cache_entry;
}

// Helper function to add a resource barrier
static void __add_barrier(
    ID3D12GraphicsCommandList* cl,
    ID3D12Resource*            res,
    D3D12_RESOURCE_STATES      before,
    D3D12_RESOURCE_STATES      after)
{
    if (before == after) return; // no-op
    D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
        res,
        before,
        after
    );
    cl->ResourceBarrier(1, &b);
}

static void __on_graphics_device_event(UnityGfxDeviceEventType type) {
    switch (type) {
    case kUnityGfxDeviceEventInitialize:
        s_Gfx12  = s_Unity->Get<IUnityGraphicsD3D12v8>();
        s_Gfx    = s_Unity->Get<IUnityGraphics>();
        s_Device = s_Gfx12 ? s_Gfx12->GetDevice() : nullptr;
        break;

    case kUnityGfxDeviceEventShutdown:
        s_Device = nullptr;
        s_Gfx12  = nullptr;
        break;

    default:
        break;
    }
}

void initUnityInterop(IUnityInterfaces* unity) {
    s_Unity  = unity;
    s_Gfx    = unity->Get<IUnityGraphics>();
    s_Gfx->RegisterDeviceEventCallback(__on_graphics_device_event);

    // Call once in case the device already exists (Unity tells you what renderer is active)
    __on_graphics_device_event(kUnityGfxDeviceEventInitialize);

    // once s_Device is valid, create our shared D3D12 objects:
    if (s_Device && !s_CmdAlloc) {
        HRESULT hr = S_OK;

        // 1) Allocator + command‐list
        hr = s_Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&s_CmdAlloc)
        );
        if (!checkHR(hr, "CreateCommandAllocator")) return;

        hr = s_Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            s_CmdAlloc,
            nullptr,
            IID_PPV_ARGS(&s_CmdList)
        );
        if (!checkHR(hr, "CreateCommandList")) return;
        s_CmdList->Close();

        // 2) Fence + event
        hr = s_Device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&s_Fence)
        );
        if (!checkHR(hr, "CreateFence")) return;
        s_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!s_FenceEvent) {
            LogMessage("Failed to create fence event.");
            return;
        }
        s_FenceValue = 1;

        // 3) Command queue from Unity
        if (s_Gfx12) {
            s_CmdQueue = s_Gfx12->GetCommandQueue();
        } else {
            D3D12_COMMAND_QUEUE_DESC qd = {};
            hr = s_Device->CreateCommandQueue(&qd, IID_PPV_ARGS(&s_CmdQueue));
            if (!checkHR(hr, "CreateCommandQueue")) return;
        }

        // 4) Create CUDA stream
        cudaError_t cerr = cudaStreamCreateWithFlags(&s_cudaStream, cudaStreamNonBlocking);
        if (!checkCUDA(cerr, "cudaStreamCreateWithFlags")) return;
    } else {
        LogMessage("Failed to initialize D3D12 objects: device {} allocator {}",
            (void*)s_Device, (void*)s_CmdAlloc);
    }
}

void shutdownUnityInterop() {
    if (s_Gfx)
        s_Gfx->UnregisterDeviceEventCallback(__on_graphics_device_event);

    if (s_FenceEvent)    CloseHandle(s_FenceEvent);
    if (s_Fence)         s_Fence->Release();
    if (s_CmdList)       s_CmdList->Release();
    if (s_CmdAlloc)      s_CmdAlloc->Release();
    if (s_CmdQueue)      s_CmdQueue->Release(); // release queue

    s_FenceEvent = nullptr;
    s_Fence      = nullptr;
    s_CmdList    = nullptr;
    s_CmdAlloc   = nullptr;
    s_CmdQueue   = nullptr;

    for (auto& entry : s_InteropCache) {
        cudaFree(reinterpret_cast<void*>(entry.second.cudaPtr));
        cudaDestroyExternalMemory(entry.second.extMem);
        entry.second.sharedBuf->Release();
    }
}

bool registerOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (texture or buffer)
    void* out_depth_tex,      // ID3D12Resource* (texture or buffer)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
) {
    // Sanity checks
    if (!out_img_tex || !out_depth_tex) {
        LogMessage("Invalid output textures: out_img_tex = {}, out_depth_tex = {}", (void*)out_img_tex, (void*)out_depth_tex);
        return false;
    }
    if (robot_id < 0 || cam_bit == 0 || cam_bit >= NUM_CAMERAS) {
        LogMessage("Invalid robot ID or camera bitmask: robot_id = {}, cam_bit = {}", robot_id, cam_bit);
        return false;
    }
    if (img_buffer_size <= 0 || depth_buffer_size <= 0) {
        LogMessage("Invalid buffer sizes: img_buffer_size = {}, depth_buffer_size = {}", img_buffer_size, depth_buffer_size);
        return false;
    }
    if (__num_set_bits(cam_bit) != 1) {
        LogMessage("Invalid camera bitmask: cam_bit = {:#x}. Must be a single bit set.", cam_bit);
        return false;
    }

    // Grab Unity's D3D12 resource
    ID3D12Resource* rgb_resource   = reinterpret_cast<ID3D12Resource*>(out_img_tex);
    ID3D12Resource* depth_resource = reinterpret_cast<ID3D12Resource*>(out_depth_tex);

    // Print the full resource descriptions
    auto rgb_desc = rgb_resource->GetDesc();
    LogMessage(
        "RGB resource {} description: Dimension={}, {}x{}x{} (format: {}). MipLevels = {}, Layout = {}, Flags = {}",
        out_img_tex,
        uint32_t(rgb_desc.Dimension),
        rgb_desc.Width, rgb_desc.Height, rgb_desc.DepthOrArraySize,
        uint32_t(rgb_desc.Format), rgb_desc.MipLevels, uint32_t(rgb_desc.Layout), uint64_t(rgb_desc.Flags)
    );

    auto depth_desc = depth_resource->GetDesc();
    LogMessage(
        "Depth resource {} description: Dimension={}, {}x{}x{} (format: {}). MipLevels = {}, Layout = {}, Flags = {}",
        out_depth_tex,
        uint32_t(depth_desc.Dimension),
        depth_desc.Width, depth_desc.Height, depth_desc.DepthOrArraySize,
        uint32_t(depth_desc.Format), depth_desc.MipLevels, uint32_t(depth_desc.Layout), uint64_t(depth_desc.Flags)
    );

    // Ensure we have a valid device
    if (!s_Device) {
        LogMessage("Getting device from resource...");
        HRESULT hr = rgb_resource->GetDevice(IID_PPV_ARGS(&s_Device));
        if (!checkHR(hr, "GetDevice from resource")) return false;
    }

    // If resources are textures, calculate and store footprints
    if (rgb_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        TextureFootprintInfo& rgb_info = s_TextureFootprints[rgb_resource];
        s_Device->GetCopyableFootprints(&rgb_desc, 0, 1, 0,
            &rgb_info.footprint, &rgb_info.numRows, &rgb_info.rowSizeInBytes, &rgb_info.totalBytes);
        LogMessage("RGB texture footprint: totalBytes={}, numRows={}, rowSizeInBytes={}, depth={}, rowPitch={}",
            rgb_info.totalBytes,
            rgb_info.numRows,
            rgb_info.rowSizeInBytes,
            rgb_info.footprint.Footprint.Depth,
            rgb_info.footprint.Footprint.RowPitch);

        // Use the actual required buffer size for texture data
        img_buffer_size = static_cast<int32_t>(rgb_info.totalBytes);
    }

    if (depth_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        TextureFootprintInfo& depth_info = s_TextureFootprints[depth_resource];
        s_Device->GetCopyableFootprints(&depth_desc, 0, 1, 0,
            &depth_info.footprint, &depth_info.numRows, &depth_info.rowSizeInBytes, &depth_info.totalBytes);
        LogMessage("Depth texture footprint: totalBytes={}, rowPitch={}",
            depth_info.totalBytes, depth_info.footprint.Footprint.RowPitch);

        // Use the actual required buffer size for texture data
        depth_buffer_size = static_cast<int32_t>(depth_info.totalBytes);
    }

    // Create interop entries with proper sizes
    DX12InteropCacheEntry* input_entry_ptr = _getOrCreateInteropEntry(rgb_resource, img_buffer_size);
    if (!input_entry_ptr) {
        LogMessage("Failed to get or create input interop entry.");
        return false;
    }

    DX12InteropCacheEntry* depth_entry_ptr = _getOrCreateInteropEntry(depth_resource, depth_buffer_size);
    if (!depth_entry_ptr) {
        LogMessage("Failed to get or create depth interop entry.");
        return false;
    }

    // Register the output textures in the global maps
    s_OutputRGBTextures[robot_id].push_back(rgb_resource);
    s_OutputDepthTextures[robot_id].push_back(depth_resource);

    LogMessage("Registered output textures for robot ID {}: RGB = {}, Depth = {}", robot_id, (void*)rgb_resource, (void*)depth_resource);

    return true;
}

bool clearOutputTextures(int32_t robot_id) {
    if (robot_id < 0) {
        LogMessage("Invalid robot ID: {}", robot_id);
        return false;
    }

    auto erase_interop_entries = [&](auto& tex_map) {
        for (auto& tex : tex_map) {
            if (s_InteropCache.contains(tex)) {
                DX12InteropCacheEntry& entry = s_InteropCache[tex];
                if (entry.extMem) {
                    cudaDestroyExternalMemory(entry.extMem);
                    entry.extMem = nullptr;
                }
                entry.cudaPtr = 0;
                if (entry.sharedBuf) {
                    entry.sharedBuf->Release();
                    entry.sharedBuf = nullptr;
                }
                entry.bufSize = 0;
                s_InteropCache.erase(tex);
            }
            if (s_TextureFootprints.contains(tex)) {
                s_TextureFootprints.erase(tex);
            }
        }
    };

    // Remove and release all interop entries for this robot ID
    erase_interop_entries(s_OutputRGBTextures[robot_id]);
    erase_interop_entries(s_OutputDepthTextures[robot_id]);

    s_OutputRGBTextures.erase(robot_id);
    s_OutputDepthTextures.erase(robot_id);

    LogMessage("Cleared output textures for robot ID {}", robot_id);
    return true;
}

// Helper function type for getting images
typedef bool (*GetImageSetFunc)(int32_t robot_id, int32_t n_images_requested, uint8_t** images, float** depths);

// Common implementation for uploading image sets to Unity buffers
static bool uploadImageSetToUnityCommon(int32_t robot_id, GetImageSetFunc getImageSetFunc, const char* operation_name) {
    // Sanity checks
    if (robot_id < 0) {
        LogMessage("Invalid robot ID: {}", robot_id);
        return false;
    }

    // Ensure we have our own command queue (not Unity's)
    if (!s_CmdQueue) {
        if (!s_Device) {
            LogMessage("D3D12 device is not initialized. Cannot create command queue.");
            return false;
        }

        D3D12_COMMAND_QUEUE_DESC qDesc = {};
        qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        qDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        qDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        HRESULT hr = s_Device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&s_CmdQueue));
        if (!checkHR(hr, "CreateCommandQueue")) return false;
        LogMessage("Created our own command queue: {}", (void*)s_CmdQueue);
    }

    // Ensure the command allocator is initialized
    if (!s_CmdAlloc) {
        HRESULT hr = s_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_CmdAlloc));
        if (!checkHR(hr, "CreateCommandAllocator")) return false;
        hr = s_Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_CmdAlloc, nullptr, IID_PPV_ARGS(&s_CmdList));
        if (!checkHR(hr, "CreateCommandList")) return false;
        s_CmdList->Close();
        hr = s_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s_Fence));
        if (!checkHR(hr, "CreateFence")) return false;
        s_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!s_FenceEvent) {
            LogMessage("Failed to create fence event.");
            return false;
        }
    }

    // Check if we have registered textures for this robot ID
    auto rgb_it = s_OutputRGBTextures.find(robot_id);
    auto depth_it = s_OutputDepthTextures.find(robot_id);
    if (rgb_it == s_OutputRGBTextures.end() || depth_it == s_OutputDepthTextures.end()) {
        LogMessage("No registered output textures for robot ID: {}", robot_id);
        return false;
    }

    std::vector<ID3D12Resource*>& rgb_textures = rgb_it->second;
    std::vector<ID3D12Resource*>& depth_textures = depth_it->second;

    size_t num_textures = rgb_textures.size();
    if (num_textures != depth_textures.size()) {
        LogMessage("Mismatch in number of RGB and depth textures for robot ID: {}", robot_id);
        return false;
    }

    // Grab the latest image set using the provided function
    uint8_t* rgb_images[NUM_CAMERAS];
    float* depth_images[NUM_CAMERAS];
    bool ret = getImageSetFunc(robot_id, num_textures, rgb_images, depth_images);
    if (!ret) {
        LogMessage("No new images ready for {}: {}", operation_name, robot_id);
        return false;
    }

    // Copy data to the shared buffers
    for (int32_t i = 0; i < rgb_textures.size(); i++) {
        ID3D12Resource* rgb_resource = rgb_textures[i];
        ID3D12Resource* depth_resource = depth_textures[i];

        if (!s_InteropCache.contains(rgb_resource) || !s_InteropCache.contains(depth_resource)) {
            LogMessage("No interop entry found for resources. Did you register the textures?");
            return false;
        }

        DX12InteropCacheEntry& rgb_entry   = s_InteropCache[rgb_resource];
        DX12InteropCacheEntry& depth_entry = s_InteropCache[depth_resource];

        // For textures, we need to handle the data layout properly
        auto rgb_desc = rgb_resource->GetDesc();
        if (rgb_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && s_TextureFootprints.contains(rgb_resource)) {
            // Handle texture data with proper row pitch
            TextureFootprintInfo& info = s_TextureFootprints[rgb_resource];

            // TODO: Your CUDA data might need reformatting to match the texture layout
            // For now, assuming the CUDA data is tightly packed RGB
            size_t src_size = 4 * 640 * 480; // Assuming RGB data

            checkCudaError(
                cudaMemcpyAsync(
                    reinterpret_cast<uint8_t*>(rgb_entry.cudaPtr),
                    rgb_images[i],
                    src_size,
                    cudaMemcpyDeviceToDevice,
                    s_cudaStream
                ),
                "cudaMemcpyAsync for RGB texture data"
            );
        } else {
            // Buffer copy (original behavior)
            size_t rgb_buf_size = 4 * 640 * 480 * sizeof(uint8_t);
            checkCudaError(
                cudaMemcpyAsync(
                    reinterpret_cast<uint8_t*>(rgb_entry.cudaPtr),
                    rgb_images[i],
                    rgb_buf_size,
                    cudaMemcpyDeviceToDevice,
                    s_cudaStream
                ),
                "cudaMemcpyAsync for RGB buffer"
            );
        }

        // Similar for depth
        auto depth_desc = depth_resource->GetDesc();
        if (depth_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && s_TextureFootprints.contains(depth_resource)) {
            TextureFootprintInfo& info = s_TextureFootprints[depth_resource];
            size_t src_size = 640 * 480 * sizeof(float);

            checkCudaError(
                cudaMemcpyAsync(
                    reinterpret_cast<float*>(depth_entry.cudaPtr),
                    depth_images[i],
                    src_size,
                    cudaMemcpyDeviceToDevice,
                    s_cudaStream
                ),
                "cudaMemcpyAsync for Depth texture data"
            );
        } else {
            size_t depth_buf_size = 640 * 480 * sizeof(float);
            checkCudaError(
                cudaMemcpyAsync(
                    reinterpret_cast<float*>(depth_entry.cudaPtr),
                    depth_images[i],
                    depth_buf_size,
                    cudaMemcpyDeviceToDevice,
                    s_cudaStream
                ),
                "cudaMemcpyAsync for Depth buffer"
            );
        }
    }

    LogMessage("Copied images to shared buffers for {} robot ID: {}", operation_name, robot_id);

    // Copy to Unity resources
    s_CmdAlloc->Reset();
    s_CmdList->Reset(s_CmdAlloc, nullptr);

    for (int32_t i = 0; i < rgb_textures.size(); i++) {
        ID3D12Resource* rgb_resource = rgb_textures[i];
        ID3D12Resource* depth_resource = depth_textures[i];

        DX12InteropCacheEntry& rgb_entry   = s_InteropCache[rgb_resource];
        DX12InteropCacheEntry& depth_entry = s_InteropCache[depth_resource];

        auto rgb_desc = rgb_resource->GetDesc();
        auto depth_desc = depth_resource->GetDesc();

        // Handle based on resource dimension
        if (rgb_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            // Texture copy
            __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

            TextureFootprintInfo& info = s_TextureFootprints[rgb_resource];

            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = rgb_entry.sharedBuf;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = info.footprint;

            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = rgb_resource;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = 0;

            s_CmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

            __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        } else {
            // Buffer copy (original code)
            __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

            s_CmdList->CopyBufferRegion(
                rgb_resource, 0,
                rgb_entry.sharedBuf, 0,
                rgb_entry.bufSize
            );

            __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }

        // Similar for depth
        if (depth_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

            TextureFootprintInfo& info = s_TextureFootprints[depth_resource];

            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = depth_entry.sharedBuf;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = info.footprint;

            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = depth_resource;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = 0;

            s_CmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

            __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        } else {
            __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

            s_CmdList->CopyBufferRegion(
                depth_resource, 0,
                depth_entry.sharedBuf, 0,
                depth_entry.bufSize
            );

            __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
    }

    s_CmdList->Close();
    ID3D12CommandList* lists[] = { s_CmdList };
    s_CmdQueue->ExecuteCommandLists(1, lists);
    s_CmdQueue->Signal(s_Fence, ++s_FenceValue);

    // Wait for the GPU to finish
    if (s_Fence->GetCompletedValue() < s_FenceValue) {
        s_Fence->SetEventOnCompletion(s_FenceValue, s_FenceEvent);
        WaitForSingleObject(s_FenceEvent, INFINITE);
    }
    s_FenceValue++;

    LogMessage("Copied images to d3d12 resources for {} robot: {}", operation_name, robot_id);

    return true;
}

bool uploadNextImageSetToUnity(int32_t robot_id) {
    return uploadImageSetToUnityCommon(robot_id, SOb_GetNextImageSet, "robot");
}

bool uploadNextVisionPipelineImageSetToUnity(int32_t robot_id) {
    return uploadImageSetToUnityCommon(robot_id, SOb_GetNextVisionPipelineImageSet, "vision pipeline");
}

}

// Rest of the code remains the same (extern "C" functions)...