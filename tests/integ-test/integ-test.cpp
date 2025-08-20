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

int main(int argc, char* argv[]) {
    using namespace std::chrono;

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT1_IP> <ROBOT2_IP> <username> <password>" << std::endl;
        return 1;
    }

    std::vector<std::string> robot_ips = {argv[1], argv[2]};
    std::string username  = argv[3];
    std::string password  = argv[4];

    //SOb_ToggleDebugDumps("./spot_dump");

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

    // TODO: Setup a listener for ctrl-c to gracefully stop the connection
    // std::cout << "Press Ctrl-C to stop reading camera feeds..." << std::endl;

    uint8_t** images;
    float** depths;

    uint32_t num_images_requested = __num_set_bits(cam_bitmask);
    images = new uint8_t*[num_images_requested];
    depths = new float*[num_images_requested];

    std::vector<float> image_cpu_buffer(640 * 480 * 3);
    std::vector<float> depth_cpu_buffer(640 * 480);

    int32_t img_batch_id = 0;
    bool new_images = false;
    while (true) {
        static time_point<high_resolution_clock> start_time = high_resolution_clock::now();

        if (new_images) {
            time_point<high_resolution_clock> end_time = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end_time - start_time);
            double latency_ms = double(duration.count()) / 1000.0;

            std::cout << std::format("integ-test: {}-rgbd read latency: {:.4f} ms\n", num_images_requested, latency_ms);
            start_time = end_time; // Reset start time for next push
            new_images = false;
        }

        for (int32_t spot = 0; spot < 2; spot++) {
            if (spot_ids[spot] < 0) {
                std::cout << "Skipping Spot " << spot << " as it is not connected." << std::endl;
                continue;
            }

            if (!SOb_GetNextImageSet(spot_ids[spot], int32_t(num_images_requested), images, depths)) {
                std::cout << "No new images from Spot " << spot << ", skipping." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            new_images = true;
            for (uint32_t i = 0; i < num_images_requested; i++) {
                // std::cout << "SPOT " << spot << " Image batch " << img_batch_id <<" Image " << i << ": ";
                // std::cout << images[i] << " ";
                // std::cout << depths[i] << std::endl;
                cudaMemcpyAsync(
                    image_cpu_buffer.data(),
                    images[i],
                    4 * 640 * 480,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyAsync(
                    depth_cpu_buffer.data(),
                    depths[i],
                    640 * 480 * sizeof(float),
                    cudaMemcpyDeviceToHost
                );
                cudaStreamSynchronize(0); // Wait for the copy to complete

                cv::Mat image(480, 640, CV_8UC4, image_cpu_buffer.data());
                cv::Mat depth(480, 640, CV_32FC1, depth_cpu_buffer.data());

                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

                cv::imshow("SPOT " + std::to_string(spot) + " RGB" + std::to_string(i), image);
                cv::imshow("SPOT " + std::to_string(spot) + " Depth" + std::to_string(i), depth);
            }
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        img_batch_id++;
    }

    for (int32_t spot = 0; spot < 2; spot++) {
        cv::destroyAllWindows();
        SOb_DisconnectFromSpot(spot_ids[0]);
    }

    return 0;
}
