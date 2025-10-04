//
// Created by fmz on 7/24/2025.
//

#ifdef WIN32_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <string>
#include <chrono>
#include <format>
#include <unordered_map>

#include "spot-observer.h"
#include "d3dx12.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using Microsoft::WRL::ComPtr;
using namespace std::chrono;

// ---- constants ----------------------------------------------------------------------------
static constexpr int WIDTH = 640;
static constexpr int HEIGHT = 480;
static constexpr int CHANNELS = 4;
static constexpr UINT64 IMAGE_BUFSIZE = WIDTH * HEIGHT * CHANNELS * sizeof(uint8_t);
static constexpr UINT64 DEPTH_BUFSIZE = WIDTH * HEIGHT * sizeof(float);

// ---- Test Logger --------------------------------------------------------------------------
void TestLogCallback(const char* message) {
    std::cout << "[SpotObserver Log] " << message << std::endl;
}

// ---- helpers ------------------------------------------------------------------------------
void ThrowIfFailed(HRESULT hr, const std::string& context = "") {
    if(FAILED(hr)) {
        std::string errorMsg = std::system_category().message(hr);
        std::cerr << "Error";
        if (!context.empty()) {
            std::cerr << " in " << context;
        }
        std::cerr << ": HRESULT=0x" << std::hex << hr << " (" << errorMsg << ")\n";
        std::exit(hr);
    }
}

static uint32_t __num_set_bits(uint32_t bitmask) {
    return __popcnt(bitmask);
}


static int32_t connect_to_spot(
    const std::string& robot_ip,
    const std::string& username,
    const std::string& password
) {
    std::cout << "Connecting to Spot robot at " << robot_ip << " with user " << username << std::endl;
    int32_t spot_id = SOb_ConnectToSpot(robot_ip.c_str(), username.c_str(), password.c_str());
    if (spot_id < 0) {
        std::cerr << "Failed to connect to Spot robot" << std::endl;
        return -1;
    }
    std::cout << "Connected to Spot robot with ID: " << spot_id << std::endl;
    return spot_id;
}

static int32_t start_cam_stream(int32_t spot_id, uint32_t cam_bitmask) {
    int32_t cam_stream_id = SOb_CreateCameraStream(spot_id, cam_bitmask);
    if (cam_stream_id < 0) {
        std::cerr << "Failed to start reading camera feeds" << std::endl;
        SOb_DisconnectFromSpot(spot_id);
        return -1;
    }
    std::cout << "Started camera stream with ID: " << cam_stream_id << std::endl;
    return cam_stream_id;
}

static int32_t disconnect_from_spots(const int32_t spot_ids[], size_t num_spots) {
    for (size_t i = 0; i < num_spots; i++) {
        if (spot_ids[i] >= 0) {
            if (SOb_DisconnectFromSpot(spot_ids[i])) {
                std::cout << "Disconnected from Spot robot with ID: " << spot_ids[i] << std::endl;
            } else {
                std::cerr << "Failed to disconnect from Spot robot with ID: " << spot_ids[i] << std::endl;
            }
        }
    }
    return 0;
}

// ===========================================================================================
//  main
// ===========================================================================================
int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT1_IP> <ROBOT2_IP> <username> <password> [model_path]" << std::endl;
        return 1;
    }

    std::vector<std::string> robot_ips = {argv[1], argv[2]};
    std::string username = argv[3];
    std::string password = argv[4];

    // ---------------------------------------------------------------------------------------
    // 1. Register Log Callback
    // ---------------------------------------------------------------------------------------
    std::cout << "Registering test log callback...\n";
    SOb_SetUnityLogCallback(TestLogCallback);
    SOb_ToggleLogging(true);

    //SOb_ToggleDebugDumps("./spot_dump_dx12");

    // ---------------------------------------------------------------------------------------
    // 2. Initialize D3D12
    // ---------------------------------------------------------------------------------------
    std::cout << "Initializing D3D12...\n";
    ComPtr<IDXGIFactory6> factory;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter1> hwAdapter;
    for(UINT idx=0; factory->EnumAdapters1(idx, &hwAdapter)==S_OK; ++idx) {
        DXGI_ADAPTER_DESC1 desc;
        hwAdapter->GetDesc1(&desc);
        if(!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            std::wcout << L"Using adapter: " << desc.Description << std::endl;
            break;
        }
        hwAdapter.Reset();
    }
    if(!hwAdapter) {
        std::cerr << "No suitable hardware adapter found.\n";
        return 1;
    }

    ComPtr<ID3D12Device> device;
    ThrowIfFailed(D3D12CreateDevice(
        hwAdapter.Get(),
        D3D_FEATURE_LEVEL_12_0,
        IID_PPV_ARGS(&device)
    ));

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    ComPtr<ID3D12CommandQueue> cmdQueue;
    ThrowIfFailed(device->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&cmdQueue)));

    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    ThrowIfFailed(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&cmdAlloc)
    ));

    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ThrowIfFailed(device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        cmdAlloc.Get(),
        nullptr,
        IID_PPV_ARGS(&cmdList)
    ));

    ComPtr<ID3D12Fence> fence;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HANDLE fenceEvt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvt) {
        std::cerr << "Failed to create fence event.\n";
        return 1;
    }
    UINT64 fenceVal = 1;

    // ---------------------------------------------------------------------------------------
    // 3. Connect to Spot robots
    // ---------------------------------------------------------------------------------------
    int32_t spot_ids[2] = {-1, -1};
    std::unordered_map<int32_t, std::vector<int32_t>> cam_stream_ids;

    std::vector<uint32_t> cam_bitmasks = {FRONTRIGHT | FRONTLEFT, HAND};
    for (size_t i = 0; i < 2; i++) {
        if (robot_ips[i] == "0") {
            spot_ids[i] = -1;
        } else {
            spot_ids[i] = connect_to_spot(robot_ips[i], username, password);
            if (spot_ids[i] < 0) {
                disconnect_from_spots(spot_ids, 2);
                return -1;
            }
        }
    }
    std::cout << "Connected to Spot robots with IDs: " << spot_ids[0] << ", " << spot_ids[1] << std::endl;

    for (size_t i = 0; i < 2; i++) {
        if (spot_ids[i] < 0) continue;
        for (uint32_t cam_bitmask : cam_bitmasks) {
            int32_t cam_stream_id = start_cam_stream(spot_ids[i], cam_bitmask);

            if (cam_stream_id < 0) {
                disconnect_from_spots(spot_ids, 2);
                return -1;
            }
            cam_stream_ids[spot_ids[i]].push_back(cam_stream_id);
        }
    }

    // Calculate total cameras across all streams
    uint32_t total_cameras = 0;
    for (uint32_t cam_bitmask : cam_bitmasks) {
        total_cameras += __num_set_bits(cam_bitmask);
    }
    uint32_t num_cameras = total_cameras;

    // ---------------------------------------------------------------------------------------
    // 4. Create D3D12 resources for each camera stream
    // ---------------------------------------------------------------------------------------
    std::cout << "Creating D3D12 textures for camera streams...\n";

    struct CameraResources {
        ComPtr<ID3D12Resource> image_texture;
        ComPtr<ID3D12Resource> depth_texture;
    };

    // Structure: robot -> stream -> camera resources
    std::unordered_map<int32_t, std::vector<std::vector<CameraResources>>> robot_resources;

    D3D12_HEAP_PROPERTIES heap_default  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC rgb_desc = CD3DX12_RESOURCE_DESC::Buffer(IMAGE_BUFSIZE);
    D3D12_RESOURCE_DESC depth_desc = CD3DX12_RESOURCE_DESC::Buffer(DEPTH_BUFSIZE);

    for (size_t robot = 0; robot < 2; robot++) {
        if (spot_ids[robot] < 0) continue;

        int32_t spot_id = spot_ids[robot];
        robot_resources[spot_id].resize(cam_bitmasks.size());

        for (size_t stream = 0; stream < cam_bitmasks.size(); stream++) {
            uint32_t num_cams_in_stream = __num_set_bits(cam_bitmasks[stream]);
            robot_resources[spot_id][stream].resize(num_cams_in_stream);

            for (uint32_t cam = 0; cam < num_cams_in_stream; cam++) {
                // Create image texture (RGB)
                ThrowIfFailed(device->CreateCommittedResource(
                    &heap_default,
                    D3D12_HEAP_FLAG_NONE,
                    &rgb_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&robot_resources[spot_id][stream][cam].image_texture)
                ), "CreateCommittedResource for rgb texture");

                // Create depth texture
                ThrowIfFailed(device->CreateCommittedResource(
                    &heap_default,
                    D3D12_HEAP_FLAG_NONE,
                    &depth_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&robot_resources[spot_id][stream][cam].depth_texture)
                ), "CreateCommittedResource for depth texture");

                std::wstring img_name = L"Robot" + std::to_wstring(robot) + L"_Stream" + std::to_wstring(stream) + L"_Cam" + std::to_wstring(cam) + L"_Image";
                std::wstring depth_name = L"Robot" + std::to_wstring(robot) + L"_Stream" + std::to_wstring(stream) + L"_Cam" + std::to_wstring(cam) + L"_Depth";
                robot_resources[spot_id][stream][cam].image_texture->SetName(img_name.c_str());
                robot_resources[spot_id][stream][cam].depth_texture->SetName(depth_name.c_str());
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // 5. Register textures with SpotObserver plugin
    // ---------------------------------------------------------------------------------------
    std::cout << "Registering D3D12 resources with SpotObserver...\n";

    int32_t total_streams_active = 0;
    for (size_t robot = 0; robot < 2; robot++) {
        if (spot_ids[robot] < 0) continue;

        int32_t spot_id = spot_ids[robot];
        for (size_t stream = 0; stream < cam_bitmasks.size(); stream++) {
            uint32_t num_cams_in_stream = __num_set_bits(cam_bitmasks[stream]);
            int32_t cam_stream_id = cam_stream_ids[spot_id][stream];

            for (uint32_t cam = 0; cam < num_cams_in_stream; cam++) {
                bool reg_result = SOb_RegisterUnityReadbackBuffers(
                    spot_id,
                    cam_stream_id,
                    1 << cam, // Single bit for camera within the stream
                    robot_resources[spot_id][stream][cam].image_texture.Get(),
                    robot_resources[spot_id][stream][cam].depth_texture.Get(),
                    static_cast<int32_t>(IMAGE_BUFSIZE),
                    static_cast<int32_t>(DEPTH_BUFSIZE)
                );

                if (!reg_result) {
                    std::cerr << "Failed to register textures for robot " << robot << " stream " << stream << " camera " << cam << std::endl;
                    disconnect_from_spots(spot_ids, 2);
                    return -1;
                }
            }
            total_streams_active++;
        }
    }

    bool using_vision_pipeline = (argc == 6);
    SObModel model = nullptr;
    if (using_vision_pipeline) {
        const char* model_path = argv[5];
        std::cout << "Loading model from: " << model_path << std::endl;
        model = SOb_LoadModel(model_path, "cuda");
        if (!model) {
            std::cerr << "Failed to load model from: " << argv[5] << std::endl;
            for (int32_t spot = 0; spot < 2; spot++) {
                cv::destroyAllWindows();
                SOb_DisconnectFromSpot(spot_ids[spot]);
            }
            return -1;
        }
        std::cout << "Model loaded successfully!" << std::endl;

        // Launch vision pipeline on both robots (first stream only)
        for (size_t i = 0; i < 2; i++) {
            if (spot_ids[i] < 0) continue;
            int32_t spot_id = spot_ids[i];
            int32_t cam_stream_id = cam_stream_ids[spot_id][0]; // Use first stream
            bool ret = SOb_LaunchVisionPipeline(spot_id, cam_stream_id, model);
            if (!ret) {
                std::cerr << "Failed to launch vision pipeline on robot " << i << std::endl;
                disconnect_from_spots(spot_ids, 2);
                SOb_UnloadModel(model);
                cv::destroyAllWindows();
                return -1;
            }
            std::cout << "Vision pipeline launched on robot " << i << std::endl;
        }
    }

    // ---------------------------------------------------------------------------------------
    // 6. Main camera reading loop with OpenCV display
    // ---------------------------------------------------------------------------------------
    std::cout << "Starting camera feed reading loop with OpenCV display...\n";
    std::cout << "Press 'q' in any OpenCV window to quit...\n";

    // Initialized output pointers for GPU device pointers
    std::vector<uint32_t> num_images_requested_per_stream;
    std::vector<uint8_t**> images_gpu;
    std::vector<float**> depths_gpu;

    for (uint32_t cam_bitmask : cam_bitmasks) {
        uint32_t num_bits_set = __num_set_bits(cam_bitmask);
        num_images_requested_per_stream.push_back(num_bits_set);
        images_gpu.push_back(new uint8_t*[num_bits_set]);
        depths_gpu.push_back(new float*[num_bits_set]);
    }

    std::vector<uint8_t> image_cpu_buffer(640 * 480 * 4);
    std::vector<float> depth_cpu_buffer(640 * 480);

    bool new_images = false;
    auto start_time = high_resolution_clock::now();
    bool exit_requested = false;

    while (!exit_requested) {
        if (new_images) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            double latency_ms = double(duration.count()) / 1000.0;

            std::cout << std::format("integ-test-dx12: total latency: {:.4f} ms\n", latency_ms);
            start_time = end_time; // Reset start time for next push
            new_images = false;
        }

        for (int32_t spot = 0; spot < 2; spot++) {
            int32_t spot_id = spot_ids[spot];
            if (spot_id < 0) {
                continue;
            }

            for (int32_t stream = 0; stream < cam_stream_ids[spot_id].size(); stream++) {
                uint32_t num_images_requested = num_images_requested_per_stream[stream];
                uint8_t** images_set = images_gpu[stream];
                float** depths_set = depths_gpu[stream];
                int32_t cam_stream_id = cam_stream_ids[spot_id][stream];

                // Get GPU pointers from SpotObserver
                if (using_vision_pipeline && stream == 0) {
                    if (!SOb_GetNextVisionPipelineImageSet(spot_id, cam_stream_id, int32_t(num_images_requested), images_set, depths_set)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        continue;
                    }
                } else {
                    if (!SOb_GetNextImageSet(spot_id, cam_stream_id, int32_t(num_images_requested), images_set, depths_set)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        continue;
                    }
                }
                new_images = true;

                // Upload GPU data to D3D12 textures, then read back and display
                for (uint32_t i = 0; i < num_images_requested; i++) {
                    // Copy from GPU to CPU buffers
                    cudaMemcpyAsync(
                        image_cpu_buffer.data(),
                        images_set[i],
                        4 * 640 * 480,
                        cudaMemcpyDeviceToHost
                    );
                    cudaMemcpyAsync(
                        depth_cpu_buffer.data(),
                        depths_set[i],
                        640 * 480 * sizeof(float),
                        cudaMemcpyDeviceToHost
                    );
                    cudaStreamSynchronize(0); // Wait for the copy to complete

                    // Now write to D3D12 readback buffers
                    uint8_t* image_data = nullptr;
                    float*   depth_data = nullptr;

                    D3D12_RANGE image_read_range = {0, IMAGE_BUFSIZE};
                    D3D12_RANGE depth_read_range = {0, DEPTH_BUFSIZE};
                    D3D12_RANGE no_write = {0, 0};

                    ThrowIfFailed(robot_resources[spot_id][stream][i].image_texture->Map(
                        0, &image_read_range, reinterpret_cast<void**>(&image_data)
                    ), "Map image readback");

                    ThrowIfFailed(robot_resources[spot_id][stream][i].depth_texture->Map(
                        0, &depth_read_range, reinterpret_cast<void**>(&depth_data)
                    ), "Map depth readback");

                    // Copy from CPU buffers to D3D12 mapped memory
                    std::memcpy(image_data, image_cpu_buffer.data(), IMAGE_BUFSIZE);
                    std::memcpy(depth_data, depth_cpu_buffer.data(), DEPTH_BUFSIZE);

                    // Create OpenCV matrices from D3D12 readback data
                    cv::Mat image(480, 640, CV_8UC4, image_data);
                    cv::Mat depth(480, 640, CV_32FC1, depth_data);

                    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
                    cv::normalize(depth, depth, 0, 1, cv::NORM_MINMAX);

                    cv::imshow("SPOT " + std::to_string(spot) + " Stream " + std::to_string(stream) + " RGB" + std::to_string(i), image);
                    cv::imshow("SPOT " + std::to_string(spot) + " Stream " + std::to_string(stream) + " Depth" + std::to_string(i), depth);

                    // Unmap the readback buffers
                    robot_resources[spot_id][stream][i].image_texture->Unmap(0, &no_write);
                    robot_resources[spot_id][stream][i].depth_texture->Unmap(0, &no_write);
                }

                if (cv::waitKey(1) == 'q') {
                    exit_requested = true;
                    break;
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // 7. Cleanup
    // ---------------------------------------------------------------------------------------
    std::cout << "Cleaning up...\n";

    disconnect_from_spots(spot_ids, 2);
    cv::destroyAllWindows();

    if (model) SOb_UnloadModel(model);
    for (auto image_set : images_gpu) delete[] image_set;
    for (auto depth_set : depths_gpu) delete[] depth_set;

    CloseHandle(fenceEvt);

    std::cout << "TEST COMPLETED: Successfully displayed camera feeds via D3D12 Unity API\n";
    return 0;
}
