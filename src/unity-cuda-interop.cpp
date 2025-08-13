//
// Created by fmz on 8/13/2025.
//

#include "unity-cuda-interop.h"

#include "d3dx12.h"
#include <unordered_map>
#include <string>

namespace SOb {
static std::unordered_map<ID3D12Resource*, DX12InteropCacheEntry> s_InteropCache = {};

// robot ID -> <D3D12 textures>
static std::unordered_map<int32_t, std::vector<ID3D12Resource*>> s_OutputRGBTextures = {};
static std::unordered_map<int32_t, std::vector<ID3D12Resource*>> s_OutputDepthTextures = {};

// Unity-specific initialization. Sets up the D3D12 device and command queue.
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType type) {
    using namespace SOb;

    switch (type) {
    case kUnityGfxDeviceEventInitialize:
        s_Gfx12  = s_Unity->Get<IUnityGraphicsD3D12v8>();
        s_Gfx    = s_Unity->Get<IUnityGraphics>();
        s_Device = s_Gfx12 ? s_Gfx12->GetDevice() : nullptr;
        break;

    case kUnityGfxDeviceEventShutdown:
        s_Device = nullptr;
        s_Gfx12  = nullptr;
        // Tear down interop cache
        for (auto& entry : s_InteropCache) {
            cudaFree(reinterpret_cast<void*>(entry.second.cudaPtr));
            cudaDestroyExternalMemory(entry.second.extMem);
            entry.second.sharedBuf->Release();
        }
        s_InteropCache.clear();
        s_OutputRGBTextures.clear();
        s_OutputDepthTextures.clear();
        break;

    default:
        break;
    }
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload() {
    using namespace SOb;

    if (s_Gfx)
        s_Gfx->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);

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

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unity) {
    using namespace SOb;

    s_Unity  = unity;
    s_Gfx    = unity->Get<IUnityGraphics>();
    s_Gfx->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    // Call once in case the device already exists (Unity tells you what renderer is active)
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);

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
        s_CmdList->Close(); // start closed

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
    } else {
        LogMessage("Failed to initialize D3D12 objects: device {} allocator {}",
            (void*)s_Device, (void*)s_CmdAlloc);
    }
}

static DX12InteropCacheEntry* _getOrCreateInteropEntry(ID3D12Resource* d3d12_resource, size_t buf_size_in_bytes) {
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot create interop entry.");
        return nullptr;
    }

    DX12InteropCacheEntry& cache_entry = s_InteropCache[d3d12_resource];

    // On first use, create shared D3D12 buffer + import into CUDA
    if (!cache_entry.sharedBuf) {
        LogMessage("Failed to find interop entry for resource: {}. Creating new entry.", (void*)d3d12_resource);
        // 4a) create a default‐heap, shared buffer
        {
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
            if (!checkHR(hr, "CreateCommittedResource")) return nullptr;
        }

        // 4b) share it and import into CUDA
        {
            HANDLE sharedHandle = nullptr;
            HRESULT hr = s_Device->CreateSharedHandle(
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
            if (!checkCUDA(cerr, "cudaImportExternalMemory")) return nullptr;
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
            LogMessage("Mapped CUDA buffer: {} (size: {})", (void*)cache_entry.cudaPtr, buf_size_in_bytes);
        }
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
    if (before == after) return; // no‑op
    D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
        res,
        before,
        after
    );
    cl->ResourceBarrier(1, &b);
}

bool registerOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
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

    // Grab Unity’s D3D12 resource & device/queue
    ID3D12Resource* rgb_resource   = reinterpret_cast<ID3D12Resource*>(out_img_tex);
    ID3D12Resource* depth_resource = reinterpret_cast<ID3D12Resource*>(out_depth_tex);

    ID3D12Device*   device         = s_Device;

    // Print the full resource descriptions
    auto rgb_desc = rgb_resource->GetDesc();
    LogMessage(
        "RGB resource {} description: {}x{}x{} (format: {}). Alignment = {}, MipLevels = {}, Layout = {}, Flags = {}",
        out_img_tex,
        rgb_desc.Width, rgb_desc.Height, rgb_desc.DepthOrArraySize,
        uint32_t(rgb_desc.Format), rgb_desc.Alignment, rgb_desc.MipLevels, uint32_t(rgb_desc.Layout), uint64_t(rgb_desc.Flags)
    );

    auto depth_desc = depth_resource->GetDesc();
    LogMessage(
        "Depth resource {} description: {}x{}x{} (format: {}). Alignment = {}, MipLevels = {}, Layout = {}, Flags = {}",
        out_depth_tex,
        depth_desc.Width, depth_desc.Height, depth_desc.DepthOrArraySize,
        uint32_t(depth_desc.Format), depth_desc.Alignment, depth_desc.MipLevels, uint32_t(depth_desc.Layout), uint64_t(depth_desc.Flags)
    );

    // Ensure we have a valid device, if not, try to get it from the Unity texture
    if (!device) {
        bool get_unity_device = true;

        LogMessage("s_Device is null.");
        if (!s_Unity)
        {
            LogMessage("s_Unity is null.");
            get_unity_device = false;
        } else
        {
            LogMessage("s_Unity = {}, s_Unity->Get<IUnityGraphicsD3D12>() = {}, s_Unity->Get<IUnityGraphics>() = {}",
                (void*)s_Unity, (void*)s_Unity->Get<IUnityGraphicsD3D12>(), (void*)s_Unity->Get<IUnityGraphics>());

            auto test1  = s_Unity->Get<IUnityGraphicsD3D12>();
            auto test2 = s_Unity->Get<IUnityGraphicsD3D12v2>();
            auto test3 = s_Unity->Get<IUnityGraphicsD3D12v3>();
            auto test4 = s_Unity->Get<IUnityGraphicsD3D12v4>();
            auto test5 = s_Unity->Get<IUnityGraphicsD3D12v5>();
            auto test6 = s_Unity->Get<IUnityGraphicsD3D12v6>();
            auto test7 = s_Unity->Get<IUnityGraphicsD3D12v7>();
            auto test8 = s_Unity->Get<IUnityGraphicsD3D12v8>();

            LogMessage("d3d12 {}, d3d12_v2 {}, d3d12_v3 {}, d3d12_v4 {}, d3d12_v5 {}, d3d12_v6 {}, d3d12_v7 {}, d3d12_v8 {}",
                (void*)test1, (void*)test2, (void*)test3, (void*)test4, (void*)test5, (void*)test6, (void*)test7, (void*)test8);


            // s_Device = s_Unity->Get<IUnityGraphicsD3D12>()->GetDevice();
            // LogMessage("Got device from unity s_Device: {}", (void*)s_Device);
            get_unity_device = false;
        }

        if (!get_unity_device)
        {
            LogMessage("Device is null. Trying to get it from RGB resource.");
            HRESULT hr = rgb_resource->GetDevice(IID_PPV_ARGS(&device));
            if (!checkHR(hr, "getDevice")) return false;

            // Sanity check
            ID3D12Device* depth_device = nullptr;
            hr = depth_resource->GetDevice(IID_PPV_ARGS(&depth_device));
            if (!checkHR(hr, "getDevice for depth resource")) return false;
            if (device != depth_device) {
                LogMessage("Output and depth resources are not from the same device!");
                return false;
            }

            s_Device = device;
        }
    }

    // TODO: consider preparing the resources ahead of time.
    DX12InteropCacheEntry* input_entry_ptr = _getOrCreateInteropEntry(rgb_resource, img_buffer_size);
    if (!input_entry_ptr) {
        LogMessage("Failed to get or create input interop entry.");
        return false;
    }

    DX12InteropCacheEntry* depth_entry_ptr = nullptr;
    depth_entry_ptr = _getOrCreateInteropEntry(depth_resource, depth_buffer_size);
    if (!depth_entry_ptr) {
        LogMessage("Failed to get or create depth interop entry.");
        return false;
    }

    // Register the output textures in the global maps
    s_OutputRGBTextures[robot_id].push_back(rgb_resource);
    s_OutputDepthTextures[robot_id].push_back(depth_resource);

    return true;
}

bool uploadNextImageSetToUnity(int32_t robot_id) {
    // Sanity checks
    if (robot_id < 0) {
        LogMessage("Invalid robot ID: {}", robot_id);
        return false;
    }
    // if (!s_CmdAlloc || !s_CmdList || !s_CmdQueue || !s_Fence || !s_FenceEvent) {
    //     LogMessage("D3D12 command infrastructure is not initialized.");
    //     return false;
    // }

    // Ensure command queue
    ID3D12CommandQueue* queue = s_CmdQueue;
    if (!queue) {
        if (s_Gfx12)
            queue = s_Gfx12->GetCommandQueue();
        else {
            D3D12_COMMAND_QUEUE_DESC qd = {};
            s_Device->CreateCommandQueue(&qd, IID_PPV_ARGS(&s_CmdQueue));
            queue = s_CmdQueue;
        }
        s_CmdQueue = queue;
    }

    // Ensure the command allocator is initialized
    if (!s_CmdAlloc) {
        if (!s_Device) {
            LogMessage("D3D12 device is not initialized. Cannot create command allocator.");
            LogMessage("You probably forgot to register textures!");
            return false;
        }
        // lazy init for test harness
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
        s_FenceValue = 1;
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

    // Grab the latest image set from the robot
    float* rgb_images[NUM_CAMERAS];
    float* depth_images[NUM_CAMERAS];
    bool ret = SOb_GetNextImageSet(robot_id, num_textures, rgb_images, depth_images);
    if (!ret) {
        LogMessage("Failed to get next image set for robot ID: {}", robot_id);
        return false;
    }

    // TODO: We really gotta work on the hardcoded sizes here...
    size_t rgb_buf_size_in_bytes   = 3 * 640 * 480 * sizeof(float); // RGB
    size_t depth_buf_size_in_bytes = 640 * 480 * sizeof(float);   // Depth

    // For each texture, copy the data from the shared buffers to Unity textures
    // Note: we assume that the textures are in the same order as the robot's cameras
    // TODO: The following code does a double copy. Please figure out a way to avoid this.
    for (int32_t i = 0; i < rgb_textures.size(); i++) {
        ID3D12Resource* rgb_resource = rgb_textures[i];
        if (!rgb_resource) {
            LogMessage("RGB texture for robot ID {} at index {} is null", robot_id, i);
            return false;
        }
        ID3D12Resource* depth_resource = depth_textures[i];
        if (!depth_resource) {
            LogMessage("Depth texture for robot ID {} at index {} is null", robot_id, i);
            return false;
        }

        // Get interop entries
        if (!s_InteropCache.contains(rgb_resource)) {
            LogMessage("No interop entry found for RGB resource: {}. Did you register the textures?", (void*)rgb_resource);
            return false;
        }
        if (!s_InteropCache.contains(depth_resource)) {
            LogMessage("No interop entry found for Depth resource: {}. Did you register the textures?", (void*)depth_resource);
            return false;
        }
        DX12InteropCacheEntry& rgb_entry   = s_InteropCache[rgb_resource];
        DX12InteropCacheEntry& depth_entry = s_InteropCache[depth_resource];

        // Copy data to the shared buffers
        checkCudaError(
            cudaMemcpyAsync(
                reinterpret_cast<float*>(rgb_entry.cudaPtr),
                rgb_images[i],
                rgb_buf_size_in_bytes,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpyAsync for RGB texture"
        );
        checkCudaError(
            cudaMemcpyAsync(
                reinterpret_cast<float*>(depth_entry.cudaPtr),
                depth_images[i],
                depth_buf_size_in_bytes,
                cudaMemcpyDeviceToDevice
            ),
            "cudaMemcpyAsync for Depth texture"
        );
    }

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");

    LogMessage("Copied images to shared buffers for robot ID: {}", robot_id);

    // Copy to Unity resources
    s_CmdAlloc->Reset();
    s_CmdList->Reset(s_CmdAlloc, nullptr);
    for (int32_t i = 0; i < num_textures; i++) {
        LogMessage("About to copy images to d3d12 for robot ID: {}, texture index: {}", robot_id, i);
        ID3D12Resource* rgb_resource = rgb_textures[i];
        ID3D12Resource* depth_resource = depth_textures[i];

        DX12InteropCacheEntry& rgb_entry   = s_InteropCache[rgb_resource];
        DX12InteropCacheEntry& depth_entry = s_InteropCache[depth_resource];

        // Prepare inputs for copy
        __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

        __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

        s_CmdList->CopyBufferRegion(
            rgb_resource,         0,      // dest + offset
            rgb_entry.sharedBuf,  0,      // src  + offset
            rgb_buf_size_in_bytes         // copy size
        );

        s_CmdList->CopyBufferRegion(
            depth_resource,         0,      // dest + offset
            depth_entry.sharedBuf,  0,      // src  + offset
            depth_buf_size_in_bytes         // copy size
        );

        // transition back to COMMON
        __add_barrier(s_CmdList, rgb_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        __add_barrier(s_CmdList, rgb_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

        __add_barrier(s_CmdList, depth_resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        __add_barrier(s_CmdList, depth_entry.sharedBuf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    s_CmdList->Close();
    ID3D12CommandList* lists[] = { s_CmdList };
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(s_Fence, ++s_FenceValue);

    // Wait for the GPU to finish copying the input resource
    if (s_Fence->GetCompletedValue() < s_FenceValue) {
        s_Fence->SetEventOnCompletion(s_FenceValue, s_FenceEvent);
        WaitForSingleObject(s_FenceEvent, INFINITE);
    }
    s_FenceValue++;

    LogMessage("Copied images to d3d12 resource: {}", robot_id);

    return true;
}

}