//
// Created by fmz on 7/23/2025.
//

#include "logger.h"
#include "spot-connection.h"
#include "utils.h"
#include "dumper.h"
#include "model.h"
#include "vision-pipeline.h"

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

// Keeping track of loaded models
std::unordered_map<std::string, SObModel> s_path_to_model_map;
std::unordered_map<SObModel, std::string> s_model_to_path_map;

// Vision pipeline management
static std::unordered_map<int32_t, std::unique_ptr<VisionPipeline>> __vision_pipelines;

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
        // Clean up all vision pipelines before DLL unload
        try {
            for (auto& [robot_id, pipeline] : __vision_pipelines) {
                LogMessage("DLL_PROCESS_DETACH: Stopping vision pipeline for robot ID {}", robot_id);
                pipeline->stop();
            }
            __vision_pipelines.clear();
            LogMessage("DLL_PROCESS_DETACH: All vision pipelines stopped and cleared");
        } catch (const std::exception& e) {
            LogMessage("DLL_PROCESS_DETACH: Exception while cleaning up vision pipelines: {}", e.what());
        }
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

static bool getNextImageSet(
    int32_t robot_id,
    int32_t n_images_requested,
    uint8_t** images,
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
    if (robot.getCurrentImages(n_images_requested, images, depths)) {
        LogMessage("SOb_GetNextImageSet: Successfully retrieved {} images for robot ID {}", n_images_requested, robot_id);
        return true;
    } else {
        // LogMessage("SOb_GetNextImageSet: Failed to retrieve images for robot ID {}", robot_id);
        return false;
    }

    return true;
}

static bool getNextImageSetFromVisionPipeline(
    int32_t robot_id,
    int32_t n_images_requested,
    uint8_t** images,
    float** depths
) {
    auto pipeline_it = __vision_pipelines.find(robot_id);
    if (pipeline_it == __vision_pipelines.end()) {
        LogMessage("SOb_GetNextVisionPipelineImageSet: Vision pipeline not found for robot ID {}", robot_id);
        return false;
    }
    
    auto& pipeline = pipeline_it->second;
    if (!pipeline->isRunning()) {
        LogMessage("SOb_GetNextVisionPipelineImageSet: Vision pipeline is not running for robot ID {}", robot_id);
        return false;
    }
    
    if (n_images_requested <= 0) {
        LogMessage("SOb_GetNextVisionPipelineImageSet: Invalid number of images requested: {}", n_images_requested);
        return false;
    }
    
    // Get the maximum number of images from pipeline configuration
    int32_t max_images = static_cast<int32_t>(pipeline->getInputShape().N);
    if (n_images_requested > max_images) {
        LogMessage("SOb_GetNextVisionPipelineImageSet: Requested {} images but pipeline only supports {}", n_images_requested, max_images);
        return false;
    }

    // Get images from the vision pipeline
    if (pipeline->getCurrentImages(n_images_requested, images, depths)) {
        LogMessage("SOb_GetNextVisionPipelineImageSet: Successfully retrieved {} images from vision pipeline for robot ID {}", n_images_requested, robot_id);
        return true;
    } else {
        //LogMessage("SOb_GetNextVisionPipelineImageSet: Failed to retrieve images from vision pipeline for robot ID {}", robot_id);
        return false;
    }
}

static SObModel loadTorchModel(const std::string& modelPath, const std::string& backend) {
    LogMessage("Loading Torch model: {}", modelPath);
    LogMessage("Using backend: {}", backend);

    if (s_path_to_model_map.contains(modelPath)) {
        // Reload the model
        LogMessage("Model already loaded, reloading: {}", modelPath);
        SObModel existing_model = s_path_to_model_map[modelPath];
        delete reinterpret_cast<TorchModel*>(existing_model);
        s_path_to_model_map.erase(modelPath);
    }

    SObModel ret = nullptr;
    try {
        auto* model = new TorchModel(modelPath, backend);
        ret = reinterpret_cast<SObModel>(model);
    } catch (const std::exception& e) {
        LogMessage("Exception while loading model: {}", e.what());
        return nullptr;
    }

    if (ret) {
        s_path_to_model_map[modelPath] = ret;
        s_model_to_path_map[ret] = modelPath;
        LogMessage("Successfully loaded Torch model: {}", modelPath);
    } else {
        LogMessage("Failed to load Torch model: {}", modelPath);
    }
    return ret;
}

static SObModel loadONNXModel(const std::string& modelPath, const std::string& backend) {
    LogMessage("Loading ONNX model: {}", modelPath);
    LogMessage("Using Provider: {}", backend);

    if (s_path_to_model_map.contains(modelPath)) {
        LogMessage("Bug in ONNX model destructor. New model will NOT be loaded. Please restart Unity to load a new model.");
        return s_path_to_model_map[modelPath];
    }

    SObModel ret = nullptr;
    try {
        auto* model = new ONNXModel(modelPath, backend);
        ret = reinterpret_cast<SObModel>(model);
    } catch (const std::exception& e) {
        LogMessage("Exception while loading ONNX model: {}", e.what());
        return nullptr;
    }

    if (ret) {
        s_path_to_model_map[modelPath] = ret;
        s_model_to_path_map[ret] = modelPath;
        LogMessage("Successfully loaded ONNX model: {}", modelPath);
    } else {
        LogMessage("Failed to load ONNX model: {}", modelPath);
    }
    return ret;
}

static void unloadModel(SObModel model) {
    if (!model) {
        LogMessage("SOb::unloadModel: Model is null, nothing to unload");
        return;
    }

    auto it = s_model_to_path_map.find(model);
    if (it == s_model_to_path_map.end()) {
        LogMessage("SOb::unloadModel: Model not found in map");
        return;
    }

    std::string modelPath = it->second;
    s_path_to_model_map.erase(modelPath);
    s_model_to_path_map.erase(it);

    delete reinterpret_cast<MLModel*>(model);
    LogMessage("SOb::unloadModel: Successfully unloaded model from path: {}", modelPath);
}

} // namespace SOb

extern "C" {


// Connect to a spot robot. Returns a newly assigned ID, or -1 in case of an error.
UNITY_INTERFACE_EXPORT
int32_t UNITY_INTERFACE_API SOb_ConnectToSpot(const char* robot_ip, const char* username, const char* password) {
    try {
        if (!robot_ip || !username || !password) {
            SOb::LogMessage("SOb_ConnectToSpot: Invalid null pointer parameters");
            return -1;
        }
        
        const std::string robot_ip_str = robot_ip;
        const std::string username_str = username;
        const std::string password_str = password;

        return SOb::ConnectToSpot(robot_ip_str, username_str, password_str);
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_ConnectToSpot: Exception while connecting to robot {}: {}", robot_ip ? robot_ip : "null", e.what());
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
        // First, clean up any associated vision pipeline
        auto pipeline_it = SOb::__vision_pipelines.find(robot_id);
        if (pipeline_it != SOb::__vision_pipelines.end()) {
            SOb::LogMessage("SOb_DisconnectFromSpot: Stopping vision pipeline for robot ID {}", robot_id);
            
            // Take ownership of the pipeline to prevent access during cleanup
            auto pipeline = std::move(pipeline_it->second);
            SOb::__vision_pipelines.erase(pipeline_it);
            
            // Now safely stop the pipeline
            pipeline->stop();
            SOb::LogMessage("SOb_DisconnectFromSpot: Vision pipeline stopped and removed for robot ID {}", robot_id);
        }
        
        // Then remove the robot connection
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
SObModel UNITY_INTERFACE_API SOb_LoadModel(const char* modelPath, const char* backend) {
    if (!modelPath || !backend) {
        SOb::LogMessage("SOb_LoadModel: Invalid null pointer parameters");
        return nullptr;
    }
    
    // If model filename ends with .onnx, use ONNX model loader
    std::string model_path_str(modelPath);

    SObModel ret = nullptr;
    if (model_path_str.ends_with(".onnx")) {
        ret = SOb::loadONNXModel(modelPath, backend);
    } else {
        ret = SOb::loadTorchModel(modelPath, backend);
    }
    if (!ret) {
        SOb::LogMessage("Failed to load model: {} with backend: {}", modelPath, backend);
    }

    return ret;
}

UNITY_INTERFACE_EXPORT
void UNITY_INTERFACE_API SOb_UnloadModel(SObModel model) {
    SOb::unloadModel(model);
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_LaunchVisionPipeline(int32_t robot_id, SObModel model) {
    using namespace SOb;
    try {
        auto robot_it = __robot_connections.find(robot_id);
        if (robot_it == __robot_connections.end()) {
            LogMessage("SOb_LaunchVisionPipeline: Robot ID {} not found", robot_id);
            return false;
        }

        if (!robot_it->second.isConnected() || !robot_it->second.isStreaming()) {
            LogMessage("SOb_LaunchVisionPipeline: Robot ID {} must be connected and streaming", robot_id);
            return false;
        }

        if (!model) {
            LogMessage("SOb_LaunchVisionPipeline: Invalid model provided");
            return false;
        }

        // Check if pipeline already exists for this robot
        if (__vision_pipelines.find(robot_id) != __vision_pipelines.end()) {
            LogMessage("SOb_LaunchVisionPipeline: Vision pipeline already exists for robot ID {}", robot_id);
            return false;
        }

        // Cast the model to the proper type
        auto ml_model = reinterpret_cast<MLModel*>(model);
        
        // Create shared pointer to robot connection
        const SpotConnection& spot_connection = robot_it->second;
        
        // Get batch size from the number of cameras being processed
        int32_t batch_size = spot_connection.getCurrentNumCams();
        LogMessage("SOb_LaunchVisionPipeline: Using batch size {} based on number of cameras", batch_size);
        
        // Define tensor shapes using the actual batch size from spot connection
        TensorShape input_shape(batch_size, 4, 480, 640);   // NCHW format with dynamic batch size
        TensorShape depth_shape(batch_size, 1, 480, 640);   // Single channel depth with dynamic batch size
        TensorShape output_shape(batch_size, 1, 480, 640);  // Output shape with dynamic batch size
        
        // Create vision pipeline
        auto pipeline = std::make_unique<VisionPipeline>(
            *ml_model,
            spot_connection,
            input_shape,
            depth_shape,
            output_shape,
            25
        );
        
        // Start the pipeline
        if (!pipeline->start()) {
            LogMessage("SOb_LaunchVisionPipeline: Failed to start vision pipeline for robot ID {}", robot_id);
            return false;
        }
        
        // Store the pipeline
        __vision_pipelines[robot_id] = std::move(pipeline);
        
        LogMessage("SOb_LaunchVisionPipeline: Successfully launched vision pipeline for robot ID {}", robot_id);
        return true;
        
    } catch (const std::exception& e) {
        LogMessage("SOb_LaunchVisionPipeline: Exception while launching vision pipeline for robot ID {}: {}", robot_id, e.what());
        return false;
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_StopVisionPipeline(int32_t robot_id) {
    using namespace SOb;
    try {
        auto pipeline_it = __vision_pipelines.find(robot_id);
        if (pipeline_it == __vision_pipelines.end()) {
            LogMessage("SOb_StopVisionPipeline: Vision pipeline not found for robot ID {}", robot_id);
            return false;
        }

        auto& pipeline = pipeline_it->second;
        pipeline->stop();
        __vision_pipelines.erase(pipeline_it);
        
        LogMessage("SOb_StopVisionPipeline: Successfully stopped vision pipeline for robot ID {}", robot_id);
        return true;
        
    } catch (const std::exception& e) {
        LogMessage("SOb_StopVisionPipeline: Exception while stopping vision pipeline for robot ID {}: {}", robot_id, e.what());
        return false;
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_RegisterUnityReadbackBuffers(
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
    uint8_t** images,
    float** depths
) {
    try {
        if (!images || !depths) {
            SOb::LogMessage("SOb_GetNextImageSet: Invalid null pointer parameters (images or depths)");
            return false;
        }
        
        bool ret = SOb::getNextImageSet(robot_id, n_images_requested, images, depths);
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
bool UNITY_INTERFACE_API SOb_GetNextVisionPipelineImageSet(
    int32_t robot_id,
    int32_t n_images_requested,
    uint8_t** images,
    float** depths
) {
    try {
        if (!images || !depths) {
            SOb::LogMessage("SOb_GetNextVisionPipelineImageSet: Invalid null pointer parameters (images or depths)");
            return false;
        }
        
        bool ret = SOb::getNextImageSetFromVisionPipeline(robot_id, n_images_requested, images, depths);
        if (!ret) {
            SOb::LogMessage("SOb_GetNextVisionPipelineImageSet: Failed to get next image set from vision pipeline for robot ID {}", robot_id);
            return false;
        }
        return ret;
    } catch (const std::exception& e) {
        SOb::LogMessage("SOb_GetNextVisionPipelineImageSet: Exception while getting next image set from vision pipeline for robot ID {}: {}", robot_id, e.what());
        return false;
    }
}

UNITY_INTERFACE_EXPORT
bool UNITY_INTERFACE_API SOb_PushNextImageSetToUnityBuffers(int32_t robot_id) {
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

