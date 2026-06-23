#ifdef WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "d3dx12.h"
#include "spot-observer.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr int32_t kRobotId = 101;
constexpr int32_t kStreamId = 202;
constexpr size_t kWidth = 16;
constexpr size_t kHeight = 8;
constexpr size_t kRgbBytes = kWidth * kHeight * 4;
constexpr size_t kDepthElements = kWidth * kHeight;
constexpr size_t kDepthBytes = kDepthElements * sizeof(float);

struct SyntheticSources {
    std::array<uint8_t*, 2> rgb = {};
    std::array<float*, 2> depth = {};
    int calls = 0;
};

SyntheticSources g_sources;

void ThrowIfFailed(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(operation) + " failed with HRESULT " + std::to_string(hr));
    }
}

void CheckCuda(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(err));
    }
}

bool UNITY_INTERFACE_API ProvideSyntheticImages(
    int32_t robot_id,
    int32_t cam_stream_id,
    int32_t n_images_requested,
    uint8_t** images,
    float** depths
) {
    if (robot_id != kRobotId || cam_stream_id != kStreamId || n_images_requested != 2) {
        return false;
    }

    ++g_sources.calls;
    images[0] = g_sources.rgb[0];   // BACK, lowest SpotCamera bit.
    depths[0] = g_sources.depth[0];
    images[1] = g_sources.rgb[1];   // FRONTLEFT.
    depths[1] = g_sources.depth[1];
    return true;
}

struct AdapterSelection {
    ComPtr<IDXGIAdapter1> adapter;
    int cudaDevice = -1;
};

AdapterSelection FindCudaBackedAdapter() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return {};
    }

    int cudaDeviceCount = 0;
    if (cuDeviceGetCount(&cudaDeviceCount) != CUDA_SUCCESS || cudaDeviceCount <= 0) {
        return {};
    }

    ComPtr<IDXGIFactory6> factory;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    for (UINT adapterIndex = 0;; ++adapterIndex) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(adapterIndex, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        for (int cudaIndex = 0; cudaIndex < cudaDeviceCount; ++cudaIndex) {
            CUdevice cuDevice = 0;
            if (cuDeviceGet(&cuDevice, cudaIndex) != CUDA_SUCCESS) {
                continue;
            }

            char cudaLuid[sizeof(LUID)] = {};
            unsigned int nodeMask = 0;
            if (cuDeviceGetLuid(cudaLuid, &nodeMask, cuDevice) != CUDA_SUCCESS) {
                continue;
            }

            if (std::memcmp(&desc.AdapterLuid, cudaLuid, sizeof(LUID)) == 0) {
                return { adapter, cudaIndex };
            }
        }
    }

    return {};
}

ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device* device,
    UINT64 size,
    D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_STATES initialState
) {
    D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(heapType);
    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(size);

    ComPtr<ID3D12Resource> resource;
    ThrowIfFailed(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&resource)
    ), "CreateCommittedResource");

    return resource;
}

std::vector<uint8_t> ReadBackBuffer(ID3D12Device* device, ID3D12Resource* source, UINT64 size) {
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    ComPtr<ID3D12CommandAllocator> allocator;
    ThrowIfFailed(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&allocator)
    ), "CreateCommandAllocator");

    ComPtr<ID3D12GraphicsCommandList> list;
    ThrowIfFailed(device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&list)
    ), "CreateCommandList");

    ComPtr<ID3D12Resource> readback = CreateBuffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_RESOURCE_BARRIER toCopySource =
        CD3DX12_RESOURCE_BARRIER::Transition(source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &toCopySource);
    list->CopyBufferRegion(readback.Get(), 0, source, 0, size);
    D3D12_RESOURCE_BARRIER toCommon =
        CD3DX12_RESOURCE_BARRIER::Transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list->ResourceBarrier(1, &toCommon);
    ThrowIfFailed(list->Close(), "ID3D12GraphicsCommandList::Close");

    ID3D12CommandList* lists[] = { list.Get() };
    queue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> fence;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!eventHandle) {
        throw std::runtime_error("CreateEvent failed");
    }

    ThrowIfFailed(queue->Signal(fence.Get(), 1), "ID3D12CommandQueue::Signal");
    ThrowIfFailed(fence->SetEventOnCompletion(1, eventHandle), "SetEventOnCompletion");
    WaitForSingleObject(eventHandle, INFINITE);
    CloseHandle(eventHandle);

    std::vector<uint8_t> out(size);
    void* mapped = nullptr;
    D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(size) };
    ThrowIfFailed(readback->Map(0, &readRange, &mapped), "Map readback");
    std::memcpy(out.data(), mapped, out.size());
    D3D12_RANGE noWrite = { 0, 0 };
    readback->Unmap(0, &noWrite);
    return out;
}

void ExpectBytesEqual(const std::vector<uint8_t>& actual, const void* expected, size_t size, const char* label) {
    if (actual.size() != size || std::memcmp(actual.data(), expected, size) != 0) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

} // namespace

int main() {
    try {
        AdapterSelection selection = FindCudaBackedAdapter();
        if (!selection.adapter || selection.cudaDevice < 0) {
            std::cout << "SKIP: no CUDA-backed D3D12 adapter found.\n";
            return 0;
        }

        CheckCuda(cudaSetDevice(selection.cudaDevice), "cudaSetDevice");

        ComPtr<ID3D12Device> device;
        ThrowIfFailed(D3D12CreateDevice(
            selection.adapter.Get(),
            D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(&device)
        ), "D3D12CreateDevice");

        std::array<std::vector<uint8_t>, 2> expectedRgb;
        std::array<std::vector<float>, 2> expectedDepth;
        for (size_t cam = 0; cam < expectedRgb.size(); ++cam) {
            expectedRgb[cam].resize(kRgbBytes);
            expectedDepth[cam].resize(kDepthElements);
            for (size_t i = 0; i < kRgbBytes; ++i) {
                expectedRgb[cam][i] = static_cast<uint8_t>((i * (cam + 3) + 17 + cam) & 0xff);
            }
            for (size_t i = 0; i < kDepthElements; ++i) {
                expectedDepth[cam][i] = static_cast<float>(cam * 1000 + i) * 0.125f;
            }

            CheckCuda(cudaMalloc(&g_sources.rgb[cam], kRgbBytes), "cudaMalloc RGB");
            CheckCuda(cudaMalloc(&g_sources.depth[cam], kDepthBytes), "cudaMalloc depth");
            CheckCuda(cudaMemcpy(g_sources.rgb[cam], expectedRgb[cam].data(), kRgbBytes, cudaMemcpyHostToDevice), "cudaMemcpy RGB H2D");
            CheckCuda(cudaMemcpy(g_sources.depth[cam], expectedDepth[cam].data(), kDepthBytes, cudaMemcpyHostToDevice), "cudaMemcpy depth H2D");
        }

        struct OutputPair {
            ComPtr<ID3D12Resource> rgb;
            ComPtr<ID3D12Resource> depth;
        };
        OutputPair back {
            CreateBuffer(device.Get(), kRgbBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON),
            CreateBuffer(device.Get(), kDepthBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)
        };
        OutputPair frontLeft {
            CreateBuffer(device.Get(), kRgbBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON),
            CreateBuffer(device.Get(), kDepthBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)
        };

        if (!SOb_Test_SetUnityUploadImageProvider(ProvideSyntheticImages)) {
            throw std::runtime_error("SOb_Test_SetUnityUploadImageProvider failed");
        }

        // Register out of camera-bit order and re-register FRONTLEFT once. The upload must still
        // map provider image 0 to BACK and image 1 to FRONTLEFT, with no duplicate resources.
        if (!SOb_RegisterUnityReadbackBuffers(kRobotId, kStreamId, FRONTLEFT, frontLeft.rgb.Get(), frontLeft.depth.Get(), kRgbBytes, kDepthBytes) ||
            !SOb_RegisterUnityReadbackBuffers(kRobotId, kStreamId, BACK, back.rgb.Get(), back.depth.Get(), kRgbBytes, kDepthBytes) ||
            !SOb_RegisterUnityReadbackBuffers(kRobotId, kStreamId, FRONTLEFT, frontLeft.rgb.Get(), frontLeft.depth.Get(), kRgbBytes, kDepthBytes)) {
            throw std::runtime_error("SOb_RegisterUnityReadbackBuffers failed");
        }

        if (!SOb_EnqueueUnityUpload(kRobotId, kStreamId, SOb_UnityUploadSource_TestFrames) ||
            !SOb_EnqueueUnityUpload(kRobotId, kStreamId, SOb_UnityUploadSource_TestFrames)) {
            throw std::runtime_error("SOb_EnqueueUnityUpload failed");
        }

        using RenderEventFunc = void (UNITY_INTERFACE_API *)(int);
        auto renderEvent = reinterpret_cast<RenderEventFunc>(SOb_GetRenderEventFunc());
        if (!renderEvent) {
            throw std::runtime_error("SOb_GetRenderEventFunc returned null");
        }

        renderEvent(SOb_UnityUploadEventId);
        if (g_sources.calls != 1) {
            throw std::runtime_error("Upload queue did not coalesce duplicate requests");
        }

        auto backRgb = ReadBackBuffer(device.Get(), back.rgb.Get(), kRgbBytes);
        auto backDepth = ReadBackBuffer(device.Get(), back.depth.Get(), kDepthBytes);
        auto frontRgb = ReadBackBuffer(device.Get(), frontLeft.rgb.Get(), kRgbBytes);
        auto frontDepth = ReadBackBuffer(device.Get(), frontLeft.depth.Get(), kDepthBytes);

        ExpectBytesEqual(backRgb, expectedRgb[0].data(), kRgbBytes, "BACK RGB");
        ExpectBytesEqual(backDepth, expectedDepth[0].data(), kDepthBytes, "BACK depth");
        ExpectBytesEqual(frontRgb, expectedRgb[1].data(), kRgbBytes, "FRONTLEFT RGB");
        ExpectBytesEqual(frontDepth, expectedDepth[1].data(), kDepthBytes, "FRONTLEFT depth");

        SOb_ClearUnityReadbackBuffers(kRobotId);
        SOb_Test_SetUnityUploadImageProvider(nullptr);
        for (size_t cam = 0; cam < g_sources.rgb.size(); ++cam) {
            cudaFree(g_sources.rgb[cam]);
            cudaFree(g_sources.depth[cam]);
        }

        std::cout << "interop-upload-test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "interop-upload-test failed: " << e.what() << "\n";
        SOb_ClearUnityReadbackBuffers(kRobotId);
        SOb_Test_SetUnityUploadImageProvider(nullptr);
        for (size_t cam = 0; cam < g_sources.rgb.size(); ++cam) {
            if (g_sources.rgb[cam]) cudaFree(g_sources.rgb[cam]);
            if (g_sources.depth[cam]) cudaFree(g_sources.depth[cam]);
        }
        return 1;
    }
}
