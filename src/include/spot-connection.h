//
// Created by fmz on 7/23/2025.
//

#pragma once

#include "spot-observer.h"
#include "model.h"

#include <bosdyn/client/sdk/client_sdk.h>
#include <bosdyn/client/robot/robot.h>
#include <bosdyn/client/image/image_client.h>

#include <stop_token>
#include <cuda_runtime.h>

namespace SOb {
/**
 * Thread-safe image queue for passing data between producer and consumer threads
 */
class ReaderWriterCBuf {
private:
    std::atomic<int> read_idx_{0}; // Head index for circular buffer
    std::atomic<int> write_idx_{0}; // Tail index for circular buffer
    mutable std::atomic<bool> new_data_{false}; // Flag to indicate new data is available

    size_t n_elems_per_rgb_{0}; // Bytes per RGB image
    size_t n_elems_per_depth_{0}; // Bytes per depth image

    std::vector<SpotCamera> cameras_;
    size_t n_images_per_response_{0}; // Number of images per response

    // Circular buffer data. CUDA memory
    uint8_t* rgb_data_{nullptr};
    float*   depth_data_{nullptr};
    float*   cached_depth_{nullptr};

    bool first_run_{true};
    const size_t max_size_; // Maximum size of the queue

    // Owned by SpotConnection
    cudaStream_t cuda_stream_{nullptr};

public:
    explicit ReaderWriterCBuf(size_t max_size);

    ReaderWriterCBuf(const ReaderWriterCBuf&) = delete;
    ReaderWriterCBuf& operator=(const ReaderWriterCBuf&) = delete;

    ~ReaderWriterCBuf();

    bool initialize(
        size_t n_bytes_per_rgb,
        size_t n_bytes_per_depth,
        const std::vector<SpotCamera>& cameras
    );

    /**
     * Push image data to queue (non-blocking, drops oldest if full)
     */
    void push(const google::protobuf::RepeatedPtrField<bosdyn::api::ImageResponse>& responses);

    /**
     * Consume image and depth data
     */
    std::pair<uint8_t*, float*> pop(int32_t count) const;

    // Attach a stream created/owned by SpotConnection.
    inline void attachCudaStream(cudaStream_t stream) { cuda_stream_ = stream; }

    friend class SpotCamStream;
};

///////////////////////////////////////////////////////////////////////////////

class SpotConnection;
class VisionPipeline;

class SpotCamStream {
    SpotCamStream(const SpotCamStream&) = delete;
    SpotCamStream& operator=(const SpotCamStream&) = delete;
    SpotCamStream() = delete;

    bosdyn::client::ImageClient* image_client_;

    // Thread data
    ReaderWriterCBuf image_lifo_;
    std::atomic<bool> quit_requested_{false};
    std::atomic<int> num_samples_{0};
    bool streaming_;
    // Camera feed params
    uint32_t current_cam_mask_;
    int32_t current_num_cams_;
    bosdyn::api::GetImageRequest current_request_;
    std::vector<SpotCamera> camera_order_;

    std::unique_ptr<std::jthread> image_streamer_thread_ = nullptr;

    // One CUDA stream per connection (owned here).
    cudaStream_t cuda_stream_{nullptr};

    SpotConnection& robot_;

    TensorShape current_rgb_shape_{0, 0, 0, 0};
    TensorShape current_depth_shape_{0, 0, 0, 0};

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
    SpotCamStream(
        SpotConnection& robot,
        bosdyn::client::ImageClient* image_client,
        int32_t image_lifo_max_size
    );
    ~SpotCamStream();

    bool streamCameras(uint32_t cam_mask);
    bool getCurrentImages(
        int32_t n_images_requested,
        uint8_t** images,
        float** depths
    ) const;

    bool     isStreaming()       const { return streaming_; }
    uint32_t getCurrentCamMask() const { return current_cam_mask_; }
    int32_t  getCurrentNumCams() const { return current_num_cams_; }

    TensorShape getCurrentRGBTensorShape() const { return current_rgb_shape_; }
    TensorShape getCurrentDepthTensorShape() const { return current_depth_shape_; }

    // Return the connection's CUDA stream.
    const cudaStream_t getCudaStream() const { return cuda_stream_; }
};

class SpotConnection {
    std::unique_ptr<bosdyn::client::Robot> robot_;
    std::unique_ptr<bosdyn::client::ClientSdk> sdk_;
    bosdyn::client::ImageClient* image_client_;

    int32_t image_lifo_max_size_;

    std::unordered_map<int32_t, std::unique_ptr<SpotCamStream>> cam_streams_;
    std::unordered_map<int32_t, std::unique_ptr<VisionPipeline>> vision_pipelines_;

    bool connected_;

    int32_t next_stream_id_{0xee1};

public:

    SpotConnection(
        const std::string& robot_ip,
        const std::string& username,
        const std::string& password
    );
    ~SpotConnection();

    int32_t createCamStream(uint32_t cam_mask);
    bool removeCamStream(int32_t stream_id);
    SpotCamStream* getCamStream(int32_t stream_id);

    bool createVisionPipeline(MLModel& model, int32_t stream_id);
    bool removeVisionPipeline(int32_t stream_id);
    VisionPipeline* getVisionPipeline(int32_t stream_id);

    bool isConnected() const { return connected_; }

    friend class SpotCamStream;
};

} // namespace SOb