//
// Created by fmz on 8/13/2025.
//

#include "unity-cuda-interop.h"

#include "d3dx12.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <wrl/client.h>

namespace SOb {

using Microsoft::WRL::ComPtr;

static constexpr size_t kStagingRingSize = 3;

// Unity interface
static IUnityInterfaces*      s_Unity = nullptr;
static IUnityGraphics*        s_Gfx   = nullptr;
static IUnityGraphicsD3D12v8* s_Gfx12 = nullptr;

// DX12 state
static ComPtr<ID3D12Device>       s_Device;
static ComPtr<ID3D12CommandQueue> s_CmdQueue;
static ComPtr<ID3D12Fence>        s_SharedFence;
static HANDLE                     s_SharedFenceEvent = nullptr;
static UINT64                     s_NextFenceValue   = 1;

static cudaStream_t              s_CudaStream = nullptr;
static cudaExternalSemaphore_t   s_CudaFenceSemaphore = nullptr;
static bool                      s_CudaDeviceMatched = false;
static int                       s_CudaInteropDevice = -1;

struct CommandContext {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    UINT64 fenceValue = 0;
};

static std::array<CommandContext, kStagingRingSize> s_CommandContexts;
static size_t s_NextCommandContext = 0;

struct TextureFootprintInfo {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT numRows = 0;
    UINT64 rowSizeInBytes = 0;
    UINT64 totalBytes = 0;
};

struct RegisteredResource {
    ID3D12Resource* resource = nullptr; // Unity/test owns this resource.
    D3D12_RESOURCE_DESC desc = {};
    TextureFootprintInfo footprint = {};
    bool isTexture = false;
    size_t stagingBytes = 0;
    size_t sourceBytes = 0;
    size_t sourceRowBytes = 0;
    UINT sourceRows = 1;
};

struct StagingSlot {
    ComPtr<ID3D12Resource> buffer;
    cudaExternalMemory_t externalMemory = nullptr;
    void* cudaPtr = nullptr;
    size_t sizeBytes = 0;
    UINT64 fenceValue = 0;
    bool reserved = false;

    StagingSlot() = default;
    StagingSlot(const StagingSlot&) = delete;
    StagingSlot& operator=(const StagingSlot&) = delete;

    ~StagingSlot() {
        destroy();
    }

    void destroy() {
        if (cudaPtr) {
            cudaFree(cudaPtr);
            cudaPtr = nullptr;
        }
        if (externalMemory) {
            cudaDestroyExternalMemory(externalMemory);
            externalMemory = nullptr;
        }
        buffer.Reset();
        sizeBytes = 0;
        fenceValue = 0;
        reserved = false;
    }
};

struct ResourceInteropEntry {
    ID3D12Resource* unityResource = nullptr;
    size_t stagingBytes = 0;
    std::array<StagingSlot, kStagingRingSize> slots;

    ResourceInteropEntry() = default;
    ResourceInteropEntry(const ResourceInteropEntry&) = delete;
    ResourceInteropEntry& operator=(const ResourceInteropEntry&) = delete;

    ~ResourceInteropEntry() {
        destroy();
    }

    void destroy() {
        for (auto& slot : slots) {
            slot.destroy();
        }
        unityResource = nullptr;
        stagingBytes = 0;
    }
};

struct UploadRequest {
    int32_t robotId = -1;
    int32_t camStreamId = -1;
    int32_t sourceKind = SOb_UnityUploadSource_RawCameraFrames;
};

struct UploadResourcePair {
    RegisteredResource* rgb = nullptr;
    RegisteredResource* depth = nullptr;
    StagingSlot* rgbSlot = nullptr;
    StagingSlot* depthSlot = nullptr;
};

struct OutputResourcePair {
    ID3D12Resource* rgb = nullptr;
    ID3D12Resource* depth = nullptr;
};

static std::unordered_map<ID3D12Resource*, RegisteredResource> s_RegisteredResources;
static std::unordered_map<ID3D12Resource*, std::unique_ptr<ResourceInteropEntry>> s_InteropCache;
static std::unordered_map<const void*, int> s_CudaPointerDeviceCache;

// robot ID -> cam_stream_id -> camera bit -> D3D12 output resources.
static std::unordered_map<int32_t, std::unordered_map<int32_t, std::map<uint32_t, OutputResourcePair>>> s_OutputResources;

static std::mutex s_ResourceMutex;
static std::mutex s_UploadQueueMutex;
static std::deque<UploadRequest> s_UploadQueue;

// Helper function type for getting images
typedef bool (*GetImageSetFunc)(int32_t robot_id, int32_t cam_stream_id, int32_t n_images_requested, uint8_t** images, float** depths);

#ifdef SOB_ENABLE_TEST_HOOKS
static SOb_TestImageProvider s_TestImageProvider = nullptr;
#endif

static ID3D12Fence* copyCompletionFence() {
    if (s_Gfx12) {
        return s_Gfx12->GetFrameFence();
    }
    return s_SharedFence.Get();
}

static bool isFenceComplete(UINT64 value) {
    ID3D12Fence* fence = copyCompletionFence();
    return value == 0 || (fence && fence->GetCompletedValue() >= value);
}

static UINT64 nextFenceValue() {
    return s_NextFenceValue++;
}

static size_t bytesPerPixelForFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
        return 4;
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM:
        return 2;
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_SNORM:
        return 1;
    default:
        return 0;
    }
}

static bool checkCU(CUresult result, const std::string& operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }

    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    LogMessage("{} failed: ({}) {}", operation, name ? name : "unknown", message ? message : "unknown");
    return false;
}

static bool waitForFence(UINT64 value) {
    ID3D12Fence* fence = copyCompletionFence();
    if (!fence || value == 0 || fence->GetCompletedValue() >= value) {
        return true;
    }

    if (!s_SharedFenceEvent) {
        s_SharedFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!s_SharedFenceEvent) {
            LogMessage("Failed to create D3D12 fence event.");
            return false;
        }
    }

    HRESULT hr = fence->SetEventOnCompletion(value, s_SharedFenceEvent);
    if (!checkHR(hr, "SetEventOnCompletion")) {
        return false;
    }
    WaitForSingleObject(s_SharedFenceEvent, INFINITE);
    return true;
}

static bool ensureCudaDeviceMatchesD3D12Adapter() {
    if (s_CudaDeviceMatched) {
        return true;
    }
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot match CUDA device.");
        return false;
    }

    LUID adapterLuid = s_Device->GetAdapterLuid();
    char adapterLuidBytes[sizeof(LUID)] = {};
    std::memcpy(adapterLuidBytes, &adapterLuid, sizeof(LUID));

    if (!checkCU(cuInit(0), "cuInit")) {
        return false;
    }

    int deviceCount = 0;
    if (!checkCU(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount")) {
        return false;
    }

    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        CUdevice cuDevice = 0;
        if (!checkCU(cuDeviceGet(&cuDevice, deviceIndex), "cuDeviceGet")) {
            continue;
        }

        char cudaLuid[sizeof(LUID)] = {};
        unsigned int deviceNodeMask = 0;
        CUresult luidResult = cuDeviceGetLuid(cudaLuid, &deviceNodeMask, cuDevice);
        if (luidResult != CUDA_SUCCESS) {
            continue;
        }

        if (std::memcmp(adapterLuidBytes, cudaLuid, sizeof(LUID)) == 0) {
            cudaError_t cerr = cudaSetDevice(deviceIndex);
            if (!checkCUDA(cerr, "cudaSetDevice for D3D12 adapter LUID")) {
                return false;
            }
            s_CudaDeviceMatched = true;
            s_CudaInteropDevice = deviceIndex;
            LogMessage("Matched D3D12 adapter LUID to CUDA device {} (node mask {}).", deviceIndex, deviceNodeMask);
            return true;
        }
    }

    LogMessage("No CUDA device matches the D3D12 adapter LUID.");
    return false;
}

bool ensureCudaDeviceForD3D12Interop() {
    if (!s_Device) {
        return true;
    }
    return ensureCudaDeviceMatchesD3D12Adapter();
}

int getD3D12InteropCudaDevice() {
    return s_CudaInteropDevice;
}

static bool ensureCudaStream() {
    if (s_CudaStream) {
        return true;
    }
    if (!ensureCudaDeviceMatchesD3D12Adapter()) {
        return false;
    }

    cudaError_t cerr = cudaStreamCreateWithFlags(&s_CudaStream, cudaStreamNonBlocking);
    return checkCUDA(cerr, "cudaStreamCreateWithFlags");
}

static bool ensureSharedFenceInterop() {
    if (s_SharedFence && s_CudaFenceSemaphore) {
        return true;
    }
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot create shared fence.");
        return false;
    }
    if (!ensureCudaStream()) {
        return false;
    }

    HRESULT hr = s_Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&s_SharedFence));
    if (!checkHR(hr, "CreateFence for CUDA/D3D12 interop")) {
        return false;
    }

    HANDLE sharedHandle = nullptr;
    hr = s_Device->CreateSharedHandle(s_SharedFence.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);
    if (!checkHR(hr, "CreateSharedHandle for CUDA/D3D12 fence")) {
        s_SharedFence.Reset();
        return false;
    }

    cudaExternalSemaphoreHandleDesc desc = {};
    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = sharedHandle;

    cudaError_t cerr = cudaImportExternalSemaphore(&s_CudaFenceSemaphore, &desc);
    CloseHandle(sharedHandle);
    if (!checkCUDA(cerr, "cudaImportExternalSemaphore for D3D12 fence")) {
        s_SharedFence.Reset();
        s_CudaFenceSemaphore = nullptr;
        return false;
    }

    s_NextFenceValue = std::max<UINT64>(s_NextFenceValue, 1);
    return true;
}

static bool ensureCommandObjects() {
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot create command objects.");
        return false;
    }

    if (!s_CmdQueue) {
        if (s_Gfx12) {
            s_CmdQueue = s_Gfx12->GetCommandQueue();
        } else {
            D3D12_COMMAND_QUEUE_DESC qd = {};
            qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            HRESULT hr = s_Device->CreateCommandQueue(&qd, IID_PPV_ARGS(&s_CmdQueue));
            if (!checkHR(hr, "CreateCommandQueue")) {
                return false;
            }
        }
    }

    for (auto& context : s_CommandContexts) {
        if (context.allocator && context.list) {
            continue;
        }

        HRESULT hr = s_Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&context.allocator)
        );
        if (!checkHR(hr, "CreateCommandAllocator")) {
            return false;
        }

        hr = s_Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            context.allocator.Get(),
            nullptr,
            IID_PPV_ARGS(&context.list)
        );
        if (!checkHR(hr, "CreateCommandList")) {
            return false;
        }
        context.list->Close();
    }

    return ensureSharedFenceInterop();
}

static CommandContext* acquireCommandContext() {
    if (!ensureCommandObjects()) {
        return nullptr;
    }

    for (size_t attempt = 0; attempt < s_CommandContexts.size(); ++attempt) {
        size_t index = (s_NextCommandContext + attempt) % s_CommandContexts.size();
        CommandContext& context = s_CommandContexts[index];
        if (!isFenceComplete(context.fenceValue)) {
            continue;
        }

        s_NextCommandContext = (index + 1) % s_CommandContexts.size();
        HRESULT hr = context.allocator->Reset();
        if (!checkHR(hr, "ID3D12CommandAllocator::Reset")) {
            return nullptr;
        }
        hr = context.list->Reset(context.allocator.Get(), nullptr);
        if (!checkHR(hr, "ID3D12GraphicsCommandList::Reset")) {
            return nullptr;
        }
        context.fenceValue = 0;
        return &context;
    }

    LogMessage("No free D3D12 command context is available for Unity upload.");
    return nullptr;
}

static void addBarrier(
    ID3D12GraphicsCommandList* cl,
    ID3D12Resource* res,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after
) {
    if (before == after) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(res, before, after);
    cl->ResourceBarrier(1, &barrier);
}

static bool createStagingSlot(StagingSlot& slot, size_t sizeBytes) {
    if (!s_Device) {
        LogMessage("D3D12 device is not initialized. Cannot create staging slot.");
        return false;
    }
    if (!ensureCudaStream()) {
        return false;
    }

    slot.destroy();

    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
        sizeBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    HRESULT hr = s_Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&slot.buffer)
    );
    if (!checkHR(hr, "CreateCommittedResource for shared CUDA/D3D12 staging buffer")) {
        slot.destroy();
        return false;
    }

    HANDLE sharedHandle = nullptr;
    hr = s_Device->CreateSharedHandle(slot.buffer.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);
    if (!checkHR(hr, "CreateSharedHandle for CUDA/D3D12 staging buffer")) {
        slot.destroy();
        return false;
    }

    cudaExternalMemoryHandleDesc memoryDesc = {};
    memoryDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memoryDesc.handle.win32.handle = sharedHandle;
    memoryDesc.size = sizeBytes;
    memoryDesc.flags = cudaExternalMemoryDedicated;

    cudaError_t cerr = cudaImportExternalMemory(&slot.externalMemory, &memoryDesc);
    CloseHandle(sharedHandle);
    if (!checkCUDA(cerr, "cudaImportExternalMemory for staging buffer")) {
        slot.destroy();
        return false;
    }

    cudaExternalMemoryBufferDesc bufferMapDesc = {};
    bufferMapDesc.offset = 0;
    bufferMapDesc.size = sizeBytes;
    cerr = cudaExternalMemoryGetMappedBuffer(&slot.cudaPtr, slot.externalMemory, &bufferMapDesc);
    if (!checkCUDA(cerr, "cudaExternalMemoryGetMappedBuffer for staging buffer")) {
        slot.destroy();
        return false;
    }

    slot.sizeBytes = sizeBytes;
    return true;
}

static ResourceInteropEntry* getOrCreateInteropEntry(RegisteredResource& resource) {
    if (!ensureCommandObjects()) {
        return nullptr;
    }

    auto& entryPtr = s_InteropCache[resource.resource];
    if (!entryPtr) {
        entryPtr = std::make_unique<ResourceInteropEntry>();
        entryPtr->unityResource = resource.resource;
        entryPtr->stagingBytes = resource.stagingBytes;
    }

    if (entryPtr->stagingBytes != resource.stagingBytes) {
        entryPtr->destroy();
        entryPtr->unityResource = resource.resource;
        entryPtr->stagingBytes = resource.stagingBytes;
    }

    for (auto& slot : entryPtr->slots) {
        if (slot.buffer && slot.sizeBytes == resource.stagingBytes) {
            continue;
        }
        if (!createStagingSlot(slot, resource.stagingBytes)) {
            entryPtr->destroy();
            s_InteropCache.erase(resource.resource);
            return nullptr;
        }
    }

    return entryPtr.get();
}

static StagingSlot* acquireStagingSlot(RegisteredResource& resource) {
    ResourceInteropEntry* entry = getOrCreateInteropEntry(resource);
    if (!entry) {
        return nullptr;
    }

    for (auto& slot : entry->slots) {
        if (!slot.reserved && isFenceComplete(slot.fenceValue)) {
            slot.reserved = true;
            return &slot;
        }
    }

    LogMessage("No free staging slot is available for resource {}.", (void*)resource.resource);
    return nullptr;
}

static bool fillRegisteredResource(
    RegisteredResource& out,
    ID3D12Resource* resource,
    int32_t registeredBufferSize,
    const char* label
) {
    out = {};
    out.resource = resource;
    out.desc = resource->GetDesc();
    out.isTexture = out.desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    if (out.isTexture) {
        size_t bytesPerPixel = bytesPerPixelForFormat(out.desc.Format);
        if (bytesPerPixel == 0) {
            LogMessage("{} texture format {} is not supported for Unity upload.", label, uint32_t(out.desc.Format));
            return false;
        }

        s_Device->GetCopyableFootprints(
            &out.desc,
            0,
            1,
            0,
            &out.footprint.footprint,
            &out.footprint.numRows,
            &out.footprint.rowSizeInBytes,
            &out.footprint.totalBytes
        );

        out.sourceRows = out.footprint.numRows;
        out.sourceRowBytes = static_cast<size_t>(out.desc.Width) * bytesPerPixel;
        out.sourceBytes = out.sourceRowBytes * out.sourceRows;
        out.stagingBytes = static_cast<size_t>(out.footprint.totalBytes);
        if (registeredBufferSize <= 0 || out.sourceBytes != static_cast<size_t>(registeredBufferSize)) {
            LogMessage("{} texture source size mismatch: texture requires {} source bytes, registered size is {}.",
                label, out.sourceBytes, registeredBufferSize);
            return false;
        }

        LogMessage("{} texture footprint: {}x{}, format={}, totalBytes={}, rowSizeInBytes={}, rowPitch={}",
            label,
            out.desc.Width,
            out.desc.Height,
            uint32_t(out.desc.Format),
            out.footprint.totalBytes,
            out.footprint.rowSizeInBytes,
            out.footprint.footprint.Footprint.RowPitch);
    } else if (out.desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        out.sourceBytes = registeredBufferSize > 0 ? static_cast<size_t>(registeredBufferSize) : static_cast<size_t>(out.desc.Width);
        if (out.sourceBytes > static_cast<size_t>(out.desc.Width)) {
            LogMessage("{} buffer is too small: resource has {} bytes, registered size is {}.",
                label, out.desc.Width, out.sourceBytes);
            return false;
        }
        out.stagingBytes = out.sourceBytes;
        out.sourceRowBytes = out.sourceBytes;
        out.sourceRows = 1;
    } else {
        LogMessage("{} resource dimension {} is not supported for Unity upload.", label, uint32_t(out.desc.Dimension));
        return false;
    }

    return out.stagingBytes > 0 && out.sourceBytes > 0;
}

static bool resourceBelongsToCurrentDevice(ID3D12Resource* resource, const char* label) {
    ComPtr<ID3D12Device> resourceDevice;
    HRESULT hr = resource->GetDevice(IID_PPV_ARGS(&resourceDevice));
    if (!checkHR(hr, std::format("GetDevice for {}", label))) {
        return false;
    }
    if (s_Device && resourceDevice.Get() != s_Device.Get()) {
        LogMessage("{} resource belongs to a different D3D12 device.", label);
        return false;
    }
    return true;
}

static void releaseInteropEntry(ID3D12Resource* resource) {
    auto interopIt = s_InteropCache.find(resource);
    if (interopIt != s_InteropCache.end()) {
        if (s_SharedFence) {
            for (auto& slot : interopIt->second->slots) {
                waitForFence(slot.fenceValue);
            }
        }
        interopIt->second->destroy();
        s_InteropCache.erase(interopIt);
    }
    s_RegisteredResources.erase(resource);
}

static bool signalCudaFence(UINT64 value) {
    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = value;

    cudaExternalSemaphore_t semaphores[] = { s_CudaFenceSemaphore };
    cudaError_t cerr = cudaSignalExternalSemaphoresAsync(semaphores, &params, 1, s_CudaStream);
    return checkCUDA(cerr, "cudaSignalExternalSemaphoresAsync for staging upload");
}

static bool validateCudaSourcePointer(const void* source, const char* label) {
    auto cachedIt = s_CudaPointerDeviceCache.find(source);
    if (cachedIt != s_CudaPointerDeviceCache.end()) {
        if (s_CudaInteropDevice >= 0 && cachedIt->second != s_CudaInteropDevice) {
            LogMessage("{} source pointer is on CUDA device {}, but D3D12 interop uses device {}.",
                label, cachedIt->second, s_CudaInteropDevice);
            return false;
        }
        return true;
    }

    cudaPointerAttributes attrs = {};
    cudaError_t cerr = cudaPointerGetAttributes(&attrs, source);
    if (!checkCUDA(cerr, std::format("cudaPointerGetAttributes for {}", label))) {
        return false;
    }

#if CUDART_VERSION >= 10000
    if (attrs.type != cudaMemoryTypeDevice && attrs.type != cudaMemoryTypeManaged) {
#else
    if (attrs.memoryType != cudaMemoryTypeDevice) {
#endif
        LogMessage("{} source pointer is not CUDA device memory.", label);
        return false;
    }

    s_CudaPointerDeviceCache[source] = attrs.device;
    if (s_CudaInteropDevice >= 0 && attrs.device != s_CudaInteropDevice) {
        LogMessage("{} source pointer is on CUDA device {}, but D3D12 interop uses device {}.",
            label, attrs.device, s_CudaInteropDevice);
        return false;
    }

    return true;
}

static bool copyCudaImageToStaging(
    StagingSlot& slot,
    const RegisteredResource& resource,
    const void* source,
    const char* label
) {
    if (!source) {
        LogMessage("{} source image pointer is null.", label);
        return false;
    }
    if (!validateCudaSourcePointer(source, label)) {
        return false;
    }

    if (resource.isTexture) {
        auto* dst = reinterpret_cast<uint8_t*>(slot.cudaPtr) + resource.footprint.footprint.Offset;
        const size_t dstPitch = resource.footprint.footprint.Footprint.RowPitch;
        cudaError_t cerr = cudaMemcpy2DAsync(
            dst,
            dstPitch,
            source,
            resource.sourceRowBytes,
            resource.sourceRowBytes,
            resource.sourceRows,
            cudaMemcpyDeviceToDevice,
            s_CudaStream
        );
        return checkCUDA(cerr, std::format("cudaMemcpy2DAsync for {}", label));
    }

    cudaError_t cerr = cudaMemcpyAsync(
        slot.cudaPtr,
        source,
        resource.sourceBytes,
        cudaMemcpyDeviceToDevice,
        s_CudaStream
    );
    return checkCUDA(cerr, std::format("cudaMemcpyAsync for {}", label));
}

static void recordCopyToUnityResource(
    ID3D12GraphicsCommandList* commandList,
    const RegisteredResource& resource,
    StagingSlot& slot
) {
    addBarrier(commandList, resource.resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    addBarrier(commandList, slot.buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);

    if (resource.isTexture) {
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = slot.buffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = resource.footprint.footprint;

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = resource.resource;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    } else {
        commandList->CopyBufferRegion(
            resource.resource,
            0,
            slot.buffer.Get(),
            0,
            resource.sourceBytes
        );
    }

    addBarrier(commandList, slot.buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    addBarrier(commandList, resource.resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
}

static bool submitCommandList(
    CommandContext& context,
    const std::vector<UploadResourcePair>& uploadPairs,
    UINT64 cudaReadyFenceValue
) {
    HRESULT hr = context.list->Close();
    if (!checkHR(hr, "ID3D12GraphicsCommandList::Close")) {
        return false;
    }

    hr = s_CmdQueue->Wait(s_SharedFence.Get(), cudaReadyFenceValue);
    if (!checkHR(hr, "ID3D12CommandQueue::Wait for CUDA staging upload")) {
        return false;
    }

    UINT64 completionFenceValue = 0;
    if (s_Gfx12) {
        std::vector<UnityGraphicsD3D12ResourceState> states;
        states.reserve(uploadPairs.size() * 2);
        for (const auto& pair : uploadPairs) {
            states.push_back({ pair.rgb->resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON });
            states.push_back({ pair.depth->resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON });
        }

        completionFenceValue = s_Gfx12->ExecuteCommandList(
            context.list.Get(),
            static_cast<int>(states.size()),
            states.empty() ? nullptr : states.data()
        );
        if (completionFenceValue == 0 || !s_Gfx12->GetFrameFence()) {
            LogMessage("Unity ExecuteCommandList did not return a usable frame fence value.");
            return false;
        }
    } else {
        ID3D12CommandList* lists[] = { context.list.Get() };
        s_CmdQueue->ExecuteCommandLists(1, lists);

        completionFenceValue = nextFenceValue();
        hr = s_CmdQueue->Signal(s_SharedFence.Get(), completionFenceValue);
        if (!checkHR(hr, "ID3D12CommandQueue::Signal upload completion fence")) {
            return false;
        }
    }

    context.fenceValue = completionFenceValue;
    for (const auto& pair : uploadPairs) {
        pair.rgbSlot->reserved = false;
        pair.rgbSlot->fenceValue = completionFenceValue;
        pair.depthSlot->reserved = false;
        pair.depthSlot->fenceValue = completionFenceValue;
    }

    // The standalone integration test does not have Unity's frame fence/state tracking,
    // so make its callback deterministic. Unity-loaded uploads stay asynchronous.
    if (!s_Gfx12) {
        return waitForFence(completionFenceValue);
    }

    return true;
}

static bool collectRegisteredStreamResources(
    int32_t robotId,
    int32_t camStreamId,
    std::vector<OutputResourcePair>& resources
) {
    resources.clear();

    auto robotIt = s_OutputResources.find(robotId);
    if (robotIt == s_OutputResources.end()) {
        LogMessage("No registered output resources for robot ID {}.", robotId);
        return false;
    }

    auto streamIt = robotIt->second.find(camStreamId);
    if (streamIt == robotIt->second.end()) {
        LogMessage("No registered output resources for cam-stream ID {}.", camStreamId);
        return false;
    }

    for (const auto& [camBit, pair] : streamIt->second) {
        if (!pair.rgb || !pair.depth) {
            LogMessage("Incomplete output resource pair for camera bit {:#x}.", camBit);
            return false;
        }
        resources.push_back(pair);
    }

    if (resources.empty()) {
        LogMessage("No registered output resources for robot ID {} @ cam-stream ID {}.",
            robotId, camStreamId);
        return false;
    }

    return true;
}

static GetImageSetFunc imageSetFuncForSourceKind(int32_t sourceKind, const char** operationName) {
    switch (sourceKind) {
    case SOb_UnityUploadSource_RawCameraFrames:
        if (operationName) {
            *operationName = "raw camera";
        }
        return SOb_GetNextImageSet;
    case SOb_UnityUploadSource_VisionPipelineFrames:
        if (operationName) {
            *operationName = "vision pipeline";
        }
        return SOb_GetNextVisionPipelineImageSet;
#ifdef SOB_ENABLE_TEST_HOOKS
    case SOb_UnityUploadSource_TestFrames:
        if (operationName) {
            *operationName = "test";
        }
        return s_TestImageProvider;
#endif
    default:
        if (operationName) {
            *operationName = "unknown";
        }
        return nullptr;
    }
}

static bool processUploadRequest(const UploadRequest& request) {
    if (request.robotId < 0 || request.camStreamId < 0) {
        LogMessage("Invalid Unity upload request: robot ID {}, cam-stream ID {}.",
            request.robotId, request.camStreamId);
        return false;
    }

    const char* operationName = nullptr;
    GetImageSetFunc getImageSetFunc = imageSetFuncForSourceKind(request.sourceKind, &operationName);
    if (!getImageSetFunc) {
        LogMessage("Invalid Unity upload source kind: {}.", request.sourceKind);
        return false;
    }

    std::lock_guard resourceLock(s_ResourceMutex);

    std::vector<OutputResourcePair> registeredOutputs;
    if (!collectRegisteredStreamResources(request.robotId, request.camStreamId, registeredOutputs)) {
        return false;
    }

    const size_t numResources = registeredOutputs.size();
    if (numResources == 0 || numResources > NUM_CAMERAS) {
        LogMessage("Invalid number of registered resources for Unity upload: {}.", numResources);
        return false;
    }

    uint8_t* rgbImages[NUM_CAMERAS] = {};
    float* depthImages[NUM_CAMERAS] = {};
    if (!getImageSetFunc(
        request.robotId,
        request.camStreamId,
        static_cast<int32_t>(numResources),
        rgbImages,
        depthImages
    )) {
        return false;
    }

    std::vector<UploadResourcePair> uploadPairs;
    uploadPairs.reserve(numResources);
    auto releaseReservedSlots = [&]() {
        for (auto& pair : uploadPairs) {
            if (pair.rgbSlot) {
                pair.rgbSlot->reserved = false;
            }
            if (pair.depthSlot) {
                pair.depthSlot->reserved = false;
            }
        }
    };

    for (size_t i = 0; i < numResources; ++i) {
        auto rgbIt = s_RegisteredResources.find(registeredOutputs[i].rgb);
        auto depthIt = s_RegisteredResources.find(registeredOutputs[i].depth);
        if (rgbIt == s_RegisteredResources.end() || depthIt == s_RegisteredResources.end()) {
            LogMessage("Registered stream resource missing interop metadata.");
            releaseReservedSlots();
            return false;
        }

        UploadResourcePair pair = {};
        pair.rgb = &rgbIt->second;
        pair.depth = &depthIt->second;
        pair.rgbSlot = acquireStagingSlot(*pair.rgb);
        pair.depthSlot = acquireStagingSlot(*pair.depth);
        if (!pair.rgbSlot || !pair.depthSlot) {
            if (pair.rgbSlot) {
                pair.rgbSlot->reserved = false;
            }
            if (pair.depthSlot) {
                pair.depthSlot->reserved = false;
            }
            releaseReservedSlots();
            return false;
        }

        if (!copyCudaImageToStaging(*pair.rgbSlot, *pair.rgb, rgbImages[i], "RGB image") ||
            !copyCudaImageToStaging(*pair.depthSlot, *pair.depth, depthImages[i], "depth image")) {
            uploadPairs.push_back(pair);
            releaseReservedSlots();
            return false;
        }

        uploadPairs.push_back(pair);
    }

    CommandContext* context = acquireCommandContext();
    if (!context) {
        releaseReservedSlots();
        return false;
    }

    for (const auto& pair : uploadPairs) {
        recordCopyToUnityResource(context->list.Get(), *pair.rgb, *pair.rgbSlot);
        recordCopyToUnityResource(context->list.Get(), *pair.depth, *pair.depthSlot);
    }

    UINT64 cudaReadyFenceValue = nextFenceValue();
    if (!signalCudaFence(cudaReadyFenceValue)) {
        releaseReservedSlots();
        return false;
    }

    if (!submitCommandList(*context, uploadPairs, cudaReadyFenceValue)) {
        releaseReservedSlots();
        return false;
    }

    LogMessage("Submitted {} Unity upload for robot ID {} @ cam-stream ID {}.",
        operationName, request.robotId, request.camStreamId);
    return true;
}

static void UNITY_INTERFACE_API onRenderEvent(int eventId) {
    if (eventId != SOb_UnityUploadEventId) {
        return;
    }

    std::deque<UploadRequest> requests;
    {
        std::lock_guard lock(s_UploadQueueMutex);
        requests.swap(s_UploadQueue);
    }

    while (!requests.empty()) {
        UploadRequest request = requests.front();
        requests.pop_front();
        processUploadRequest(request);
    }
}

static void releaseD3DInteropStateLocked() {
    for (auto& context : s_CommandContexts) {
        waitForFence(context.fenceValue);
        context.list.Reset();
        context.allocator.Reset();
        context.fenceValue = 0;
    }

    s_InteropCache.clear();
    s_RegisteredResources.clear();
    s_CudaPointerDeviceCache.clear();
    s_OutputResources.clear();

    if (s_CudaFenceSemaphore) {
        cudaDestroyExternalSemaphore(s_CudaFenceSemaphore);
        s_CudaFenceSemaphore = nullptr;
    }
    if (s_CudaStream) {
        cudaStreamDestroy(s_CudaStream);
        s_CudaStream = nullptr;
    }

    if (s_SharedFenceEvent) {
        CloseHandle(s_SharedFenceEvent);
        s_SharedFenceEvent = nullptr;
    }

    s_SharedFence.Reset();
    s_CmdQueue.Reset();
    s_Device.Reset();
    s_CudaDeviceMatched = false;
    s_CudaInteropDevice = -1;
    s_NextFenceValue = 1;
}

static void onGraphicsDeviceEvent(UnityGfxDeviceEventType type) {
    switch (type) {
    case kUnityGfxDeviceEventInitialize:
        s_Gfx12 = s_Unity ? s_Unity->Get<IUnityGraphicsD3D12v8>() : nullptr;
        s_Gfx = s_Unity ? s_Unity->Get<IUnityGraphics>() : nullptr;
        s_Device = s_Gfx12 ? s_Gfx12->GetDevice() : nullptr;
        if (s_Gfx12) {
            s_CmdQueue = s_Gfx12->GetCommandQueue();
        }
        break;

    case kUnityGfxDeviceEventShutdown:
        {
            std::scoped_lock lock(s_ResourceMutex, s_UploadQueueMutex);
            s_UploadQueue.clear();
            releaseD3DInteropStateLocked();
        }
        s_Gfx12 = nullptr;
        break;

    default:
        break;
    }
}

void initUnityInterop(IUnityInterfaces* unity) {
    s_Unity = unity;
    s_Gfx = unity ? unity->Get<IUnityGraphics>() : nullptr;
    if (s_Gfx) {
        s_Gfx->RegisterDeviceEventCallback(onGraphicsDeviceEvent);
        onGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
    }
}

void shutdownUnityInterop() {
    {
        std::scoped_lock lock(s_ResourceMutex, s_UploadQueueMutex);
        s_UploadQueue.clear();
        releaseD3DInteropStateLocked();
    }

    if (s_Gfx) {
        s_Gfx->UnregisterDeviceEventCallback(onGraphicsDeviceEvent);
    }

    s_Gfx12 = nullptr;
    s_Gfx = nullptr;
    s_Unity = nullptr;
}

bool registerOutputTextures(
    int32_t robot_id,
    int32_t cam_stream_id,
    uint32_t cam_bit,
    void* out_img_tex,
    void* out_depth_tex,
    int32_t img_buffer_size,
    int32_t depth_buffer_size
) {
    if (!out_img_tex || !out_depth_tex) {
        LogMessage("Invalid output resources: out_img_tex = {}, out_depth_tex = {}",
            out_img_tex, out_depth_tex);
        return false;
    }
    if (robot_id < 0) {
        LogMessage("registerOutputTextures: Invalid robot ID {}", robot_id);
        return false;
    }
    if (cam_stream_id < 0) {
        LogMessage("registerOutputTextures: Invalid cam stream ID {}", cam_stream_id);
        return false;
    }
    if (cam_bit == 0 || cam_bit >= NUM_CAMERAS || __num_set_bits(cam_bit) != 1) {
        LogMessage("registerOutputTextures: Invalid camera bitmask {:#x}. Must be a single bit.", cam_bit);
        return false;
    }
    if (img_buffer_size <= 0 || depth_buffer_size <= 0) {
        LogMessage("Invalid buffer sizes: img_buffer_size = {}, depth_buffer_size = {}",
            img_buffer_size, depth_buffer_size);
        return false;
    }

    std::lock_guard resourceLock(s_ResourceMutex);

    auto* rgbResource = reinterpret_cast<ID3D12Resource*>(out_img_tex);
    auto* depthResource = reinterpret_cast<ID3D12Resource*>(out_depth_tex);

    if (!s_Device) {
        HRESULT hr = rgbResource->GetDevice(IID_PPV_ARGS(&s_Device));
        if (!checkHR(hr, "GetDevice from registered D3D12 resource")) {
            return false;
        }
    }
    if (!resourceBelongsToCurrentDevice(rgbResource, "RGB") ||
        !resourceBelongsToCurrentDevice(depthResource, "Depth")) {
        return false;
    }

    RegisteredResource rgbMetadata;
    RegisteredResource depthMetadata;
    if (!fillRegisteredResource(rgbMetadata, rgbResource, img_buffer_size, "RGB") ||
        !fillRegisteredResource(depthMetadata, depthResource, depth_buffer_size, "Depth")) {
        return false;
    }

    auto& streamResources = s_OutputResources[robot_id][cam_stream_id];
    auto oldIt = streamResources.find(cam_bit);
    if (oldIt != streamResources.end()) {
        if (oldIt->second.rgb && oldIt->second.rgb != rgbResource) {
            releaseInteropEntry(oldIt->second.rgb);
        }
        if (oldIt->second.depth && oldIt->second.depth != depthResource) {
            releaseInteropEntry(oldIt->second.depth);
        }
    }

    s_RegisteredResources[rgbResource] = rgbMetadata;
    s_RegisteredResources[depthResource] = depthMetadata;

    if (!getOrCreateInteropEntry(s_RegisteredResources[rgbResource]) ||
        !getOrCreateInteropEntry(s_RegisteredResources[depthResource])) {
        releaseInteropEntry(rgbResource);
        releaseInteropEntry(depthResource);
        streamResources.erase(cam_bit);
        return false;
    }

    streamResources[cam_bit] = { rgbResource, depthResource };

    LogMessage("Registered output resources for robot ID {} @ cam-stream ID {}, camera bit {:#x}: RGB = {}, Depth = {}",
        robot_id, cam_stream_id, cam_bit, (void*)rgbResource, (void*)depthResource);
    return true;
}

bool clearOutputTextures(int32_t robot_id) {
    if (robot_id < 0) {
        LogMessage("Invalid robot ID: {}", robot_id);
        return false;
    }

    {
        std::scoped_lock lock(s_ResourceMutex, s_UploadQueueMutex);

        auto robotIt = s_OutputResources.find(robot_id);
        if (robotIt != s_OutputResources.end()) {
            for (auto& streamEntry : robotIt->second) {
                for (auto& cameraEntry : streamEntry.second) {
                    releaseInteropEntry(cameraEntry.second.rgb);
                    releaseInteropEntry(cameraEntry.second.depth);
                }
            }
            s_OutputResources.erase(robotIt);
        }

        std::erase_if(s_UploadQueue, [robot_id](const UploadRequest& request) {
            return request.robotId == robot_id;
        });
    }

    LogMessage("Cleared output textures for robot ID {}", robot_id);
    return true;
}

void* getRenderEventFunc() {
    return reinterpret_cast<void*>(onRenderEvent);
}

bool enqueueUnityUpload(int32_t robot_id, int32_t cam_stream_id, int32_t source_kind) {
    if (robot_id < 0 || cam_stream_id < 0) {
        LogMessage("enqueueUnityUpload: Invalid robot ID {} or cam-stream ID {}.", robot_id, cam_stream_id);
        return false;
    }
    if (!imageSetFuncForSourceKind(source_kind, nullptr)) {
        LogMessage("enqueueUnityUpload: Invalid source kind {}.", source_kind);
        return false;
    }

    std::scoped_lock lock(s_ResourceMutex, s_UploadQueueMutex);

    std::vector<OutputResourcePair> registeredOutputs;
    if (!collectRegisteredStreamResources(robot_id, cam_stream_id, registeredOutputs)) {
        return false;
    }

    auto duplicateIt = std::find_if(s_UploadQueue.begin(), s_UploadQueue.end(),
        [robot_id, cam_stream_id, source_kind](const UploadRequest& request) {
            return request.robotId == robot_id &&
                   request.camStreamId == cam_stream_id &&
                   request.sourceKind == source_kind;
        });
    if (duplicateIt != s_UploadQueue.end()) {
        return true;
    }

    s_UploadQueue.push_back({ robot_id, cam_stream_id, source_kind });
    return true;
}

bool uploadNextImageSetToUnity(int32_t robot_id, int32_t cam_stream_id) {
    return enqueueUnityUpload(robot_id, cam_stream_id, SOb_UnityUploadSource_RawCameraFrames);
}

bool uploadNextVisionPipelineImageSetToUnity(int32_t robot_id, int32_t cam_stream_id) {
    return enqueueUnityUpload(robot_id, cam_stream_id, SOb_UnityUploadSource_VisionPipelineFrames);
}

#ifdef SOB_ENABLE_TEST_HOOKS
bool setTestUnityUploadImageProvider(SOb_TestImageProvider provider) {
    s_TestImageProvider = provider;
    return true;
}
#endif

} // namespace SOb
