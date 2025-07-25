//
// Created by fmz on 7/23/2025.
//

#include "logger.h"
#include "spot-connection.h"
#include "utils.h"
#include "dumper.h"

#include <stdexcept>
#include <unordered_map>

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

#include "d3dx12.h"


namespace SOb {

// Global state

// Function pointer for Unity logging callback
LogCallback unityLogCallback = nullptr;
bool logging_enabled = true;

// Map to hold robot connections by ID
static int32_t __next_robot_id = 0; // Incremental ID for each robot connection
static std::unordered_map<int32_t, SpotConnection> __robot_connections;

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

static std::unordered_map<ID3D12Resource*, DX12InteropCacheEntry> s_InteropCache = {};

DX12InteropCacheEntry* getOrCreateInteropEntry(ID3D12Device* device, ID3D12Resource* d3d12_resource, size_t buf_size_in_bytes) {
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
            HRESULT hr = device->CreateCommittedResource(
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
            HRESULT hr = device->CreateSharedHandle(
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

// Needed to get Unity to load our DLL dependencies from the same directory as the plugin
static bool SetDLLDirectory() {
    // Get the directory where our plugin is located
    HMODULE hModule = NULL;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCWSTR)&SetDLLDirectory, &hModule)) {
        wchar_t path[MAX_PATH];
        if (GetModuleFileNameW(hModule, path, MAX_PATH)) {
            // Remove the filename to get just the directory
            wchar_t* lastSlash = wcsrchr(path, L'\\');
            if (lastSlash) {
                *lastSlash = L'\0';
                // Add this directory to the DLL search path
                if (SetDllDirectoryW(path)) {
                    // Convert wide string to narrow string for logging
                    char narrowPath[MAX_PATH];
                    WideCharToMultiByte(CP_UTF8, 0, path, -1, narrowPath, MAX_PATH, NULL, NULL);
                    LogMessage("Set DLL directory to: {}", std::string(narrowPath));
                    return true;
                } else {
                    LogMessage("Failed to set DLL directory, error: {}", GetLastError());
                }
            }
        } else {
            LogMessage("Failed to get module filename, error: {}", GetLastError());
        }
    } else {
        LogMessage("Failed to get module handle, error: {}", GetLastError());
    }
    return false;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        LogMessage("DLL_PROCESS_ATTACH");
        SetDLLDirectory();

        break;
    case DLL_THREAD_ATTACH:
        LogMessage("DLL_THREAD_ATTACH");
        break;
    case DLL_THREAD_DETACH:
        LogMessage("DLL_THREAD_DETACH");
        break;
    case DLL_PROCESS_DETACH:
        LogMessage("DLL_PROCESS_DETACH");
        break;
    }
    return TRUE;
}



static int32_t ConnectToSpot(const std::string& robot_ip, const std::string& username, const std::string& password) {
    try {
        int32_t robot_id = __next_robot_id++;
        auto [it, inserted] = __robot_connections.try_emplace(robot_id);
        if (inserted) {
            bool success = it->second.connect(robot_ip, username, password);
            if (!success) {
                __robot_connections.erase(it);
                LogMessage("SOb::ConnectToSpot: Failed to connect to robot {}", robot_ip);
                return -1;
            }
        }
        return robot_id;

    } catch (const std::exception& e) {
        LogMessage("SOb_ConnectToSpot: Exception while connecting to robot {}: {}", robot_ip, e.what());
        return -1;
    }
}

}

extern "C" {

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

// Connect to a spot robot. Returns a newly assigned ID, or -1 in case of an error.
UNITY_INTERFACE_EXPORT
int32_t UNITY_INTERFACE_API SOb_ConnectToSpot(const char* robot_ip, const char* username, const char* password) {
    try {
        const std::string robot_ip_str = robot_ip;
        const std::string username_str = username;
        const std::string password_str = password;

        return SOb::ConnectToSpot(robot_ip_str, username_str, password_str);
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_ConnectToSpot: Exception while connecting to robot {}: {}", robot_ip, e.what());
        return -1; // Return -1 on error
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_DisconnectFromSpot(int32_t robot_id) {
    auto it = SOb::__robot_connections.find(robot_id);
    if (it == SOb::__robot_connections.end()) {
        SOb::LogMessage("SOb_DisconnectFromSpot: Robot ID {} not found", robot_id);
        return false; // Robot ID not found
    }
    try {
        SOb::__robot_connections.erase(it); // Remove from the map
        SOb::LogMessage("SOb_DisconnectFromSpot: Successfully disconnected robot ID {}", robot_id);
        return true; // Success
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_DisconnectFromSpot: Exception while disconnecting robot ID {}: {}", robot_id, e.what());
        return false; // Error during disconnection
    }
}

// Start reading spot camera feeds. Runs in a separate thread.
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ReadCameraFeeds(int32_t robot_id, uint32_t cam_bitmask) {
    auto it = SOb::__robot_connections.find(robot_id);
    if (it == SOb::__robot_connections.end()) {
        SOb::LogMessage("SOb_ReadCameraFeeds: Robot ID {} not found", robot_id);
        return false; // Robot ID not found
    }

    try {
        bool success = it->second.streamCameras(cam_bitmask);
        if (!success) {
            SOb::LogMessage("SOb_ReadCameraFeeds: Failed to start reading cameras for robot ID {}", robot_id);
            return false;
        }
        SOb::LogMessage("SOb_ReadCameraFeeds: Successfully started reading cameras for robot ID {}", robot_id);
        return true; // Success
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_ReadCameraFeeds: Exception while reading cameras for robot ID {}: {}", robot_id, e.what());
        return false; // Error during camera reading
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_LaunchVisionPipeline(int32_t robot_id, uint32_t cam_bitmask);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_GetNextImageSet(
    int32_t robot_id,
    int32_t cam_bitmask,
    float** images,
    float** depths
) {
    return true;
}



// Config calls
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthCompletion(bool enable) {
    SOb::LogMessage("SOb_ToggleDepthCompletion called with enable: {}", enable);
    return true;
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthAveraging(bool enable) {
    SOb::LogMessage("SOb_ToggleDepthAveraging called with enable: {}", enable);
    return true;
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthAveragingWithOpticalFlow(bool enable) {
    SOb::LogMessage("SOb_ToggleDepthAveraging called with enable: {}", enable);
    return true;
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleLogging(bool enable) {
    SOb::ToggleLogging(enable);
    return true;
}

// Terminal outputs aren't logged to Unity by default. We need to set up a callback
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_SetUnityLogCallback(const SOb::LogCallback callback) {
    SOb::unityLogCallback = callback;
    return true;
}
UNITY_INTERFACE_EXPORT
void UNITY_INTERFACE_API SOb_ToggleDebugDumps(const char* dump_path) {
    if (!dump_path) {
        SOb::LogMessage("UB_ToggleDebugDumps called with empty path.");
        return;
    }

    if (!SOb::ToggleDumping(dump_path)) {
        SOb::LogMessage("Failed to enable debug dumps for path: {}", dump_path);
        return;
    }

    SOb::LogMessage("Debug dumps enabled successfully");
}

} // extern "C"

