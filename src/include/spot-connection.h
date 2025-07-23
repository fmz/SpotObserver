//
// Created by brown on 7/23/2025.
//

#pragma once

#include <bosdyn/client/sdk/client_sdk.h>
#include <bosdyn/client/robot/robot.h>
#include <bosdyn/client/image/image_client.h>

namespace SOb {
/**
 * Thread-safe image queue for passing data between producer and consumer threads
 */
class ReaderWriterCBuf {
private:
    std::atomic<int> read_idx_{0}; // Head index for circular buffer
    std::atomic<int> write_idx_{0}; // Tail index for circular buffer
    std::atomic<size_t> size_{0}; // Current size of the queue

    size_t n_bytes_per_rgb_{0}; // Bytes per RGB image
    size_t n_bytes_per_depth_{0}; // Bytes per depth image
    size_t n_images_per_response_{0}; // Number of images (rgb and depth should be equal) per response

    // Circular buffer data. CUDA memory
    uint8_t* rgb_data_;
    uint16_t* depth_data_;

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
    std::pair<uint8_t*, uint16_t*> pop(int32_t count);
};

///////////////////////////////////////////////////////////////////////////////

class SpotConnection {
private:
    std::unique_ptr<bosdyn::client::Robot> robot_;
    std::unique_ptr<bosdyn::client::ClientSdk> sdk_;

    bosdyn::client::ImageClient* image_client_;

    ReaderWriterCBuf rgb_cbuf_;

    bool connected_;

    // Camera feed params
    uint32_t current_cam_mask_;
    bosdyn::api::GetImageRequest current_request_;

private:
    bosdyn::api::GetImageRequest createImageRequest(
        const std::vector<std::string>& rgb_sources,
        const std::vector<std::string>& depth_sources
    );

public:

    SpotConnection();
    ~SpotConnection();

    bool connect(
        const std::string& robot_ip,
        const std::string& username,
        const std::string& password
    );

    bool streamCameras(uint32_t cam_mask);

};
} // namespace SOb