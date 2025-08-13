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

#include "spot-observer.h"
#include "d3dx12.h"
#include <opencv2/opencv.hpp>

using Microsoft::WRL::ComPtr;
using namespace std::chrono;

// ---- constants ----------------------------------------------------------------------------
static constexpr int WIDTH = 640;
static constexpr int HEIGHT = 480;
static constexpr int CHANNELS = 3;
static constexpr UINT64 IMAGE_BUFSIZE = WIDTH * HEIGHT * CHANNELS * sizeof(float);
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

static int32_t connect_to_spot_and_start_cam_feed(
    const std::string& robot_ip,
    const std::string& username,
    const std::string& password,
    uint32_t cam_bitmask
) {
    std::cout << "Connecting to Spot robot at " << robot_ip << " with user " << username << std::endl;
    int32_t spot_id = SOb_ConnectToSpot(robot_ip.c_str(), username.c_str(), password.c_str());
    if (spot_id < 0) {
        std::cerr << "Failed to connect to Spot robot" << std::endl;
        return -1;
    }

    bool ret = SOb_ReadCameraFeeds(spot_id, cam_bitmask);
    if (!ret) {
        std::cerr << "Failed to start reading camera feeds" << std::endl;
        SOb_DisconnectFromSpot(spot_id);
        return -1;
    }

    return spot_id;
}

// ===========================================================================================
//  main
// ===========================================================================================
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT1_IP> <ROBOT2_IP> <username> <password>" << std::endl;
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

    //SOb_ToggleDebugDumps("./spot_dump");

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
    int32_t spot_ids[2];
    uint32_t cam_bitmask = FRONTLEFT | FRONTRIGHT | HAND;

    for (size_t i = 0; i < 2; i++) {
        if (robot_ips[i] == "0") {
            spot_ids[i] = -1;
        } else {
            spot_ids[i] = connect_to_spot_and_start_cam_feed(robot_ips[i], username, password, cam_bitmask);
            if (spot_ids[i] < 0) {
                std::cerr << "Failed to connect to Spot robot " << i << std::endl;
                return -1;
            }
        }
    }
    std::cout << "Connected to Spot robots with IDs: " << spot_ids[0] << ", " << spot_ids[1] << std::endl;

    uint32_t num_cameras = __num_set_bits(cam_bitmask);

    // ---------------------------------------------------------------------------------------
    // 4. Create D3D12 resources for each camera
    // ---------------------------------------------------------------------------------------
    std::cout << "Creating D3D12 textures for " << num_cameras << " cameras per robot...\n";

    struct CameraResources {
        ComPtr<ID3D12Resource> image_texture;
        ComPtr<ID3D12Resource> depth_texture;
    };

    std::vector<std::vector<CameraResources>> robot_resources(2, std::vector<CameraResources>(num_cameras));

    D3D12_HEAP_PROPERTIES heap_default  = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC rgb_desc = CD3DX12_RESOURCE_DESC::Buffer(IMAGE_BUFSIZE);
    D3D12_RESOURCE_DESC depth_desc = CD3DX12_RESOURCE_DESC::Buffer(DEPTH_BUFSIZE);

    for (size_t robot = 0; robot < 2; robot++) {
        if (spot_ids[robot] < 0) continue;

        for (uint32_t cam = 0; cam < num_cameras; cam++) {
            // Create image texture (RGB)

            ThrowIfFailed(device->CreateCommittedResource(
                &heap_default,
                D3D12_HEAP_FLAG_NONE,
                &rgb_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&robot_resources[robot][cam].image_texture)
            ), "CreateCommittedResource for rgb texture");

            // Create depth texture
            ThrowIfFailed(device->CreateCommittedResource(
                &heap_default,
                D3D12_HEAP_FLAG_NONE,
                &depth_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&robot_resources[robot][cam].depth_texture)
            ), "CreateCommittedResource for depth texture");

            std::wstring img_name = L"Robot" + std::to_wstring(robot) + L"_Cam" + std::to_wstring(cam) + L"_Image";
            std::wstring depth_name = L"Robot" + std::to_wstring(robot) + L"_Cam" + std::to_wstring(cam) + L"_Depth";
            robot_resources[robot][cam].image_texture->SetName(img_name.c_str());
            robot_resources[robot][cam].depth_texture->SetName(depth_name.c_str());
        }
    }

    // ---------------------------------------------------------------------------------------
    // 5. Register textures with SpotObserver plugin
    // ---------------------------------------------------------------------------------------
    std::cout << "Registering D3D12 resources with SpotObserver...\n";

    for (size_t robot = 0; robot < 2; robot++) {
        if (spot_ids[robot] < 0) continue;

        uint32_t cam_bits[] = {FRONTLEFT, FRONTRIGHT, HAND};
        for (uint32_t cam = 0; cam < num_cameras; cam++) {
            bool reg_result = SOb_RegisterOutputTextures(
                spot_ids[robot],
                cam_bits[cam],
                robot_resources[robot][cam].image_texture.Get(),
                robot_resources[robot][cam].depth_texture.Get(),
                static_cast<int32_t>(IMAGE_BUFSIZE),
                static_cast<int32_t>(DEPTH_BUFSIZE)
            );

            if (!reg_result) {
                std::cerr << "Failed to register textures for robot " << robot << " camera " << cam << std::endl;
                return -1;
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // 6. Main camera reading loop with OpenCV display
    // ---------------------------------------------------------------------------------------
    std::cout << "Starting camera feed reading loop with OpenCV display...\n";
    std::cout << "Press 'q' in any OpenCV window to quit...\n";

    int32_t frame_count = 0;
    bool should_quit = false;

    auto start_time = high_resolution_clock::now();

    while (!should_quit) {
        auto frame_start = high_resolution_clock::now();

        for (size_t robot = 0; robot < 2; robot++) {
            if (spot_ids[robot] < 0) continue;

            // Upload next image batch to Unity textures
            bool upload_result = SOb_UploadNextImageSetToUnity(spot_ids[robot]);
            if (!upload_result) {
                std::cerr << "Failed to upload image batch for robot " << robot << std::endl;
                continue;
            }
            std::cout << "Successfully uploaded image batch for robot " << robot << " at frame " << frame_count << std::endl;

            // Read back and display all camera data
            for (uint32_t cam = 0; cam < num_cameras; cam++) {
                float* image_data = nullptr;
                float* depth_data = nullptr;

                D3D12_RANGE image_read_range = {0, IMAGE_BUFSIZE};
                D3D12_RANGE depth_read_range = {0, DEPTH_BUFSIZE};
                D3D12_RANGE no_write = {0, 0};

                ThrowIfFailed(robot_resources[robot][cam].image_texture->Map(
                    0, &image_read_range, reinterpret_cast<void**>(&image_data)
                ), "Map image readback");

                ThrowIfFailed(robot_resources[robot][cam].depth_texture->Map(
                    0, &depth_read_range, reinterpret_cast<void**>(&depth_data)
                ), "Map depth readback");

                // Create OpenCV matrices from D3D12 readback data
                cv::Mat image(HEIGHT, WIDTH, CV_32FC3, image_data);
                cv::Mat depth(HEIGHT, WIDTH, CV_32FC1, depth_data);

                // Convert RGB to BGR for OpenCV display
                cv::Mat image_bgr;
                cv::cvtColor(image, image_bgr, cv::COLOR_RGB2BGR);

                // Create window names
                std::string image_window = "SPOT " + std::to_string(robot) + " RGB" + std::to_string(cam);
                std::string depth_window = "SPOT " + std::to_string(robot) + " Depth" + std::to_string(cam);

                // Display the images
                cv::imshow(image_window, image_bgr);
                cv::imshow(depth_window, depth);

                // Unmap the readback buffers
                robot_resources[robot][cam].image_texture->Unmap(0, &no_write);
                robot_resources[robot][cam].depth_texture->Unmap(0, &no_write);

                // Analyze data for stats (only for first camera for brevity)
                if (cam == 0 && frame_count % 10 == 0) {
                    float img_sum = 0.0f, depth_sum = 0.0f;
                    int img_nonzero = 0, depth_nonzero = 0;

                    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++) {
                        img_sum += image_data[i];
                        if (std::abs(image_data[i]) > 1e-6) img_nonzero++;
                    }

                    for (int i = 0; i < WIDTH * HEIGHT; i++) {
                        depth_sum += depth_data[i];
                        if (std::abs(depth_data[i]) > 1e-6) depth_nonzero++;
                    }

                    std::cout << std::format("Robot {}, Frame {}: Image avg={:.3f}, nonzero={}, Depth avg={:.3f}, nonzero={}\n",
                        robot, frame_count, img_sum / (WIDTH * HEIGHT * CHANNELS), img_nonzero,
                        depth_sum / (WIDTH * HEIGHT), depth_nonzero);
                }
            }

            // Check for quit key
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                should_quit = true;
                break;
            }
        }

        auto frame_end = high_resolution_clock::now();
        auto frame_duration = duration_cast<microseconds>(frame_end - frame_start);

        if (frame_count % 10 == 0) {
            std::cout << std::format("Frame {} processing time: {:.2f} ms\n",
                frame_count, double(frame_duration.count()) / 1000.0);
        }

        frame_count++;
    }

    auto end_time = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end_time - start_time);

    std::cout << "Processed " << frame_count << " frames in " << total_duration.count() << " ms\n";
    std::cout << "Average frame time: " << (float(total_duration.count()) / float(frame_count)) << " ms\n";

    // ---------------------------------------------------------------------------------------
    // 7. Cleanup
    // ---------------------------------------------------------------------------------------
    std::cout << "Cleaning up...\n";

    cv::destroyAllWindows();

    for (int robot = 0; robot < 2; robot++) {
        if (spot_ids[robot] >= 0) {
            SOb_DisconnectFromSpot(spot_ids[robot]);
        }
    }

    CloseHandle(fenceEvt);

    std::cout << "TEST COMPLETED: Successfully displayed camera feeds via D3D12 Unity API\n";
    return 0;
}
