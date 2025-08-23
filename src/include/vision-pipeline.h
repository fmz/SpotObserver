//
// Created by brown on 8/20/2025.
//

#pragma once

#include "model.h"
#include "spot-connection.h"

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <stop_token>
#include <cuda_runtime.h>

namespace SOb {

class VisionPipeline {
private:
    struct CudaWorkspace {
        float*   d_depth_data_{nullptr};
        float*   d_rgb_float_data_{nullptr};
        uint8_t* d_depth_preprocessor_workspace_{nullptr};
    };

    MLModel& model_;
    const SpotConnection& spot_connection_;
    
    // Threading
    std::unique_ptr<std::jthread> pipeline_thread_;
    std::atomic<bool> running_{false};
    
    // Result buffer (thread-safe circular buffer)
    size_t write_idx_{0};
    std::atomic<size_t> read_idx_{0};
    mutable std::atomic<bool> new_data_{0};
    const size_t max_size_;

    // Configuration
    TensorShape input_shape_;
    TensorShape depth_shape_;
    TensorShape output_shape_;

    // CUDA resources
    cudaStream_t cuda_stream_;
    CudaWorkspace cuda_ws_;
    float* d_output_buffer_{nullptr};
    uint8_t* d_rgb_data_{nullptr};

    void pipelineWorker(std::stop_token stop_token);
    bool allocateCudaBuffers();
    void deallocateCudaBuffers();
    
public:
    VisionPipeline(
        MLModel& model,
        const SpotConnection& spot_connection,
        const TensorShape& input_shape,
        const TensorShape& depth_shape,
        const TensorShape& output_shape,
        size_t max_results = 10
    );

    ~VisionPipeline();
    
    // Control methods
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }

    bool getCurrentImages(
        int32_t n_images_requested,
        uint8_t** images,
        float** depths
    ) const;

    // Configuration
    const TensorShape& getInputShape() const { return input_shape_; }
    const TensorShape& getDepthShape() const { return depth_shape_; }
    const TensorShape& getOutputShape() const { return output_shape_; }
};

} // namespace SOb