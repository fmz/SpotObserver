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

int main(int argc, char* argv[]) {
    using namespace std::chrono;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT_IP> <username> <password>" << std::endl;
        return 1;
    }

    std::string robot_ip = argv[1];
    std::string username = argv[2];
    std::string password = argv[3];

    std::cout << "Connecting to Spot robot at " << robot_ip << " with user " << username << std::endl;

    //SOb_ToggleDebugDumps("./spot_dump");
    int32_t spot_id = SOb_ConnectToSpot(robot_ip.c_str(), username.c_str(), password.c_str());

    uint32_t cam_bitmask = HAND | FRONTLEFT | FRONTRIGHT;
    bool ret = SOb_ReadCameraFeeds(spot_id, cam_bitmask);
    if (!ret) {
        std::cerr << "Failed to start reading camera feeds" << std::endl;
        SOb_DisconnectFromSpot(spot_id);
        return 1;
    }

    // TODO: Setup a listener for ctrl-c to gracefully stop the connection
    std::cout << "Press Ctrl-C to stop reading camera feeds..." << std::endl;
    float** images, **depths;
    uint32_t num_images_requested = __num_set_bits(cam_bitmask);
    images = new float*[num_images_requested];
    depths = new float*[num_images_requested];

    std::vector<float> image_cpu_buffer(640 * 480 * 3);
    std::vector<float> depth_cpu_buffer(640 * 480);

    while (true) {
        static time_point<high_resolution_clock> start_time = high_resolution_clock::now();
        time_point<high_resolution_clock> end_time = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end_time - start_time);
        double latency_ms = duration.count() / 1000.0;

        std::cout << std::format("integ-test: {}-rgbd read latency: {:.4f} ms\n", num_images_requested, latency_ms);
        start_time = end_time; // Reset start time for next push

        SOb_GetNextImageSet(spot_id, num_images_requested, images, depths);
        for (uint32_t i = 0; i < num_images_requested; i++) {
            cudaMemcpy(
                image_cpu_buffer.data(),
                images[i],
                3 * 640 * 480 * sizeof(float),
                cudaMemcpyDeviceToHost
            );
            cudaMemcpy(
                depth_cpu_buffer.data(),
                depths[i],
                640 * 480 * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            cv::Mat image(480, 640, CV_32FC3, image_cpu_buffer.data());
            cv::Mat depth(480, 640, CV_32FC1, depth_cpu_buffer.data());

            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

            cv::imshow("RGB" + std::to_string(i), image);
            cv::imshow("Depth" + std::to_string(i), depth);
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    SOb_DisconnectFromSpot(spot_id);

    return 0;
}
