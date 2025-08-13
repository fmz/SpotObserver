//
// Created by fmz on 7/23/2025.
//

#include "logger.h"
#include "spot-connection.h"
#include "utils.h"
#include "dumper.h"

#include <stdexcept>
#include <unordered_map>

#include "unity-cuda-interop.h"

namespace SOb {

// Global state

// Function pointer for Unity logging callback
LogCallback unityLogCallback = nullptr;
bool logging_enabled = true;

// Map to hold robot connections by ID
static int32_t __next_robot_id = 0; // Incremental ID for each robot connection
static std::unordered_map<int32_t, SpotConnection> __robot_connections;

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
        int32_t robot_id = __next_robot_id;
        __next_robot_id++;
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

static bool GetNextImageSet(
    int32_t robot_id,
    int32_t n_images_requested,
    float** images,
    float** depths
) {
    auto it = __robot_connections.find(robot_id);
    if (it == __robot_connections.end()) {
        LogMessage("SOb_GetNextImageSet: Robot ID {} not found", robot_id);
        return false;
    }
    SpotConnection& robot = it->second;
    if (!robot.isConnected()) {
        LogMessage("SOb_GetNextImageSet: Robot ID {} is not connected", robot_id);
        return false;
    }
    if (!robot.isStreaming()) {
        LogMessage("SOb_GetNextImageSet: Robot ID {} is not streaming", robot_id);
        return false;
    }
    if (n_images_requested <= 0 || n_images_requested > robot.getCurrentNumCams()) {
        LogMessage("SOb_GetNextImageSet: Invalid number of images requested: {}", n_images_requested);
        return false;
    }

    // Pop images from the circular buffer
    robot.getCurrentImages(n_images_requested, images, depths);

    return true;
}

}

extern "C" {


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
bool UNITY_INTERFACE_API SOb_RegisterOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
) {
    try {
        bool ret = SOb::registerOutputTextures(
            robot_id,
            cam_bit,
            out_img_tex,
            out_depth_tex,
            img_buffer_size,
            depth_buffer_size
        );
        if (!ret) {
            SOb::LogMessage("SOb_RegisterOutputTextures: Failed to register output textures for robot ID {}", robot_id);
            return false;
        }
        SOb::LogMessage("SOb_RegisterOutputTextures: Successfully registered output textures for robot ID {}", robot_id);
        return true;
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_RegisterOutputTextures: Exception while registering output textures for robot ID {}: {}", robot_id, e.what());
        return false;
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_GetNextImageSet(
    int32_t robot_id,
    int32_t n_images_requested,
    float** images,
    float** depths
) {
    try {
        bool ret = SOb::GetNextImageSet(robot_id, n_images_requested, images, depths);
        if (!ret) {
            SOb::LogMessage("SOb_GetNextImageSet: Failed to get next image set for robot ID {}", robot_id);
            return false; // Failed to get images
        }
        return ret;
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb::GetNextImageSet: Exception while getting next image set for robot ID {}: {}", robot_id, e.what());
        return false;
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_UploadNextImageSetToUnity(int32_t robot_id) {
    try {
        bool ret = SOb::uploadNextImageSetToUnity(robot_id);
        if (!ret) {
            SOb::LogMessage("SOb_UploadNextImageSetToUnity: Failed to get next image set for robot ID {}", robot_id);
            return false; // Failed to get images
        }
        return ret;
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_UploadNextImageSetToUnity: Exception while getting next image set for robot ID {}: {}", robot_id, e.what());
        return false;
    }
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

