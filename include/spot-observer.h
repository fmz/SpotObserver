//
// Created by fmz on 7/23/2025.
//

#ifndef SPOT_OBSERVER_H
#define SPOT_OBSERVER_H

#include <cstdint>

#include "IUnityInterface.h"

namespace SOb {
// Define a function pointer type for logging
typedef void (*LogCallback)(const char* message);

}

extern "C" {

enum SpotCamera {
    BACK        = 0x1,
    FRONTLEFT   = 0x2,
    FRONTRIGHT  = 0x4,
    LEFT        = 0x8,
    RIGHT       = 0x10,
    HAND        = 0x20,
    NUM_CAMERAS = 0x40,
};

// Unity Stuff
UNITY_INTERFACE_EXPORT
void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unity);
UNITY_INTERFACE_EXPORT
void UNITY_INTERFACE_API UnityPluginUnload();

// Connect to a spot robot. Returns a newly assigned ID, or -1 in case of an error.
UNITY_INTERFACE_EXPORT
int32_t UNITY_INTERFACE_API SOb_ConnectToSpot(const char* robot_ip, const char* username, const char* password);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_DisconnectFromSpot(int32_t robot_id);

// Start reading spot camera feeds. Runs in a separate threads.
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ReadCameraFeeds(int32_t robot_id, uint32_t cam_bitmask);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_LaunchVisionPipeline(int32_t robot_id, uint32_t cam_bitmask);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_GetNextImageSet(
    int32_t robot_id,
    int32_t n_images_requested,
    float** images,
    float** depths
);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_RegisterOutputTextures(
    int32_t robot_id,
    uint32_t cam_bit,         // Single bit only
    void* out_img_tex,        // ID3D12Resource* (aka texture)
    void* out_depth_tex,      // ID3D12Resource* (aka texture)
    int32_t img_buffer_size,  // In bytes
    int32_t depth_buffer_size // In bytes
);

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_UploadNextImageSetToUnity(int32_t robot_id);

// Config calls
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthCompletion(bool enable);
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthAveraging(bool enable);
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleDepthAveragingWithOpticalFlow(bool enable);
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_ToggleLogging(bool enable);

// Terminal outputs aren't logged to Unity by default. We need to set up a callback
UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_SetUnityLogCallback(const SOb::LogCallback callback);

UNITY_INTERFACE_EXPORT
void UNITY_INTERFACE_API SOb_ToggleDebugDumps(const char* dump_path);

}

#endif //SPOT_OBSERVER_H
