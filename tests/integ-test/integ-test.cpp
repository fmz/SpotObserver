//
// Created by fmz on 7/24/2025.
//

#include "spot-observer.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


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

int main(int argc, char* argv[]) {
    using namespace std::chrono;

    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT1_IP> <ROBOT2_IP> <username> <password> [model_path]" << std::endl;
        return 1;
    }

    std::vector<std::string> robot_ips = {argv[1], argv[2]};
    std::string username  = argv[3];
    std::string password  = argv[4];

    //SOb_ToggleDebugDumps("./spot_dump");

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

    // TODO: Setup a listener for ctrl-c to gracefully stop the connection
    // std::cout << "Press Ctrl-C to stop reading camera feeds..." << std::endl;

    bool using_vision_pipeline = (argc == 6);
    SObModel model = nullptr;
    if (using_vision_pipeline) {
        const char* model_path = argv[5];
        std::cout << "Loading model from: " << model_path << std::endl;
        model = SOb_LoadModel(model_path, "cuda");
        if (!model) {
            std::cerr << "Failed to load model from: " << argv[5] << std::endl;
            disconnect_from_spots(spot_ids, 2);
            cv::destroyAllWindows();

            return -1;
        }
        std::cout << "Model loaded successfully!" << std::endl;

        // Launch vision pipeline on both robots
        for (size_t i = 0; i < 2; i++) {
            if (spot_ids[i] < 0) continue;
            // Launch vision pipeline only on the first camera stream
            bool ret = SOb_LaunchVisionPipeline(spot_ids[i], cam_stream_ids[spot_ids[i]][0], model);
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

    // Initialized output pointers
    std::vector<uint32_t> num_images_requested_per_stream;
    std::vector<uint8_t**> images;
    std::vector<float**> depths;

    for (uint32_t cam_bitmask : cam_bitmasks) {
        uint32_t num_bits_set = __num_set_bits(cam_bitmask);
        num_images_requested_per_stream.push_back(num_bits_set);
        images.push_back(new uint8_t*[num_bits_set]);
        depths.push_back(new float*[num_bits_set]);
    }

    std::vector<uint8_t> image_cpu_buffer(640 * 480 * 4);
    std::vector<float> depth_cpu_buffer(640 * 480);

    bool new_images = false;
    time_point<high_resolution_clock> start_time = high_resolution_clock::now();
    bool exit_requested = false;
    while (!exit_requested) {
        if (new_images) {
            time_point<high_resolution_clock> end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            double latency_ms = double(duration.count()) / 1000.0;

            std::cout << std::format("integ-test: total latency: {:.4f} ms\n", latency_ms);
            start_time = end_time; // Reset start time for next push
            new_images = false;
        }

        for (int32_t spot = 0; spot < 2; spot++) {
            int32_t spot_id = spot_ids[spot];
            if (spot_id < 0) {
                std::cout << "Skipping Spot " << spot << " as it is not connected." << std::endl;
                continue;
            }
            for (int32_t stream = 0; stream < cam_stream_ids[spot_id].size(); stream++) {
                uint32_t num_images_requested = num_images_requested_per_stream[stream];
                uint8_t** images_set = images[stream];
                float** depths_set = depths[stream];
                int32_t cam_stream_id = cam_stream_ids[spot_id][stream];

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
                for (uint32_t i = 0; i < num_images_requested; i++) {
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

                    cv::Mat image(480, 640, CV_8UC4, image_cpu_buffer.data());
                    cv::Mat depth(480, 640, CV_32FC1, depth_cpu_buffer.data());

                    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
                    cv::normalize(depth, depth, 0, 1, cv::NORM_MINMAX);

                    cv::imshow("SPOT " + std::to_string(spot) + " Stream " + std::to_string(stream) + " RGB" + std::to_string(i), image);
                    cv::imshow("SPOT " + std::to_string(spot) + " Stream " + std::to_string(stream) + " Depth" + std::to_string(i), depth);
                }
                if (cv::waitKey(1) == 'q') {
                    exit_requested = true;
                    break;
                }
            }
        }
    }

    disconnect_from_spots(spot_ids, 2);
    cv::destroyAllWindows();

    if (model) SOb_UnloadModel(model);
    for (auto image_set : images) delete[] image_set;
    for (auto depth_set : depths) delete[] depth_set;

    return 0;
}
