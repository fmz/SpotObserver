//
// Created by brown on 7/23/2025.
//

#pragma once

#include <bosdyn/client/sdk/client_sdk.h>
#include <bosdyn/client/robot/robot.h>
#include <bosdyn/client/image/image_client.h>

#include <stop_token>

namespace SOb {
/**
 * Thread-safe image queue for passing data between producer and consumer threads
 */
class ReaderWriterCBuf {
private:
    std::atomic<int> read_idx_{0}; // Head index for circular buffer
    std::atomic<int> write_idx_{0}; // Tail index for circular buffer
    std::atomic<bool> new_data_{false}; // Flag to indicate new data is available

    size_t n_elems_per_rgb_{0}; // Bytes per RGB image
    size_t n_elems_per_depth_{0}; // Bytes per depth image
    size_t n_images_per_response_{0}; // Number of images (rgb and depth should be equal) per response

    // Circular buffer data. CUDA memory
    float* rgb_data_{nullptr};
    float* depth_data_{nullptr};

    const size_t max_size_; // Maximum size of the queue

public:
    explicit ReaderWriterCBuf(size_t max_size = 30) : max_size_(max_size) {}

    ReaderWriterCBuf(const ReaderWriterCBuf&) = delete;
    ReaderWriterCBuf& operator=(const ReaderWriterCBuf&) = delete;

    ~ReaderWriterCBuf();

    bool initialize(
        size_t n_bytes_per_rgb,
        size_t n_bytes_per_depth,
        size_t n_images_per_response
    );

    /**
     * Push image data to queue (non-blocking, drops oldest if full)
     */
    void push(const google::protobuf::RepeatedPtrField<bosdyn::api::ImageResponse>& responses);

    /**
     * Consume image and depth data
     */
    std::pair<float*, float*>  pop(int32_t count);

    friend class SpotConnection;
};

///////////////////////////////////////////////////////////////////////////////

class SpotConnection {
private:
    std::unique_ptr<bosdyn::client::Robot> robot_;
    std::unique_ptr<bosdyn::client::ClientSdk> sdk_;

    bosdyn::client::ImageClient* image_client_;

    // Thread data
    ReaderWriterCBuf image_lifo_;
    std::atomic<bool> quit_requested_{false};
    std::atomic<int> num_samples_{0};
    bool connected_;
    bool streaming_;
    // Camera feed params
    uint32_t current_cam_mask_;
    int32_t current_num_cams_;
    bosdyn::api::GetImageRequest current_request_;
    std::unique_ptr<std::jthread> image_streamer_thread_ = nullptr;

private:
    bosdyn::api::GetImageRequest _createImageRequest(
        const std::vector<std::string>& rgb_sources,
        const std::vector<std::string>& depth_sources
    );

    // Producer thread that requests images from the robot
    void _spotCamReaderThread(std::stop_token stop_token);
    // Launches the streaming thread
    void _startStreamingThread();
    // Stops the streaming thread
    void _joinStreamingThread();

public:

    SpotConnection();
    ~SpotConnection();

    bool connect(
        const std::string& robot_ip,
        const std::string& username,
        const std::string& password
    );

    bool streamCameras(uint32_t cam_mask);
    bool getCurrentImages(
        int32_t n_images_requested,
        float** images,
        float** depths
    );

    bool     isConnected()       const { return connected_; }
    bool     isStreaming()       const { return streaming_; }
    uint32_t getCurrentCamMask() const { return current_cam_mask_; }
    int32_t  getCurrentNumCams() const { return current_num_cams_; }

};

} // namespace SOb