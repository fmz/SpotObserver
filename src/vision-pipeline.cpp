//
// Created by brown on 8/20/2025.
//

#include "include/vision-pipeline.h"
#include <chrono>
#include <algorithm>
#include <iostream>

#include "cuda_kernels.cuh"
#include "utils.h"

namespace SOb {

VisionPipeline::VisionPipeline(
    MLModel& model,
    const SpotConnection& spot_connection,
    const TensorShape& input_shape,
    const TensorShape& depth_shape,
    const TensorShape& output_shape,
    size_t max_results
  ) : model_(model)
    , spot_connection_(std::move(spot_connection))
    , input_shape_(input_shape)
    , depth_shape_(depth_shape)
    , output_shape_(output_shape)
    , max_size_(max_results)
    , cuda_stream_(spot_connection_.getCudaStream())
{ }

VisionPipeline::~VisionPipeline() {
    stop();
    deallocateCudaBuffers();
}

bool VisionPipeline::start() {
    if (running_.load()) {
        return false; // Already running
    }
    
    if (!spot_connection_.isConnected() || !spot_connection_.isStreaming()) {
        LogMessage("SpotConnection must be connected and streaming before starting pipeline");
        return false;
    }

    if (input_shape_.N != depth_shape_.N) {
        LogMessage("Incompatible shapes: input_shape_.N != depth_shape_.N");
    }
    if (input_shape_.N != output_shape_.N) {
        LogMessage("Incompatible shapes: input_shape_.N != output_shape_.N");
    }
    size_t num_cams = spot_connection_.getCurrentNumCams();
    if (input_shape_.N != num_cams) {
        LogMessage("Incompatible shapes: input_shape_.N != spot_connection_.getCurrentNumCams()");
    }

    if (!allocateCudaBuffers()) {
        LogMessage("Failed to allocate CUDA buffers");
        return false;
    }

    read_idx_.store(0);
    write_idx_ = 0;
    new_data_.store(false);

    running_.store(true);
    pipeline_thread_ = std::make_unique<std::jthread>([this](std::stop_token token) {
        pipelineWorker(token);
    });
    
    return true;
}

void VisionPipeline::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (pipeline_thread_ && pipeline_thread_->joinable()) {
        pipeline_thread_->request_stop();
        pipeline_thread_->join();
    }
}

bool VisionPipeline::allocateCudaBuffers() {
    deallocateCudaBuffers();

    size_t rgb_size = max_size_ * input_shape_.N * input_shape_.C * input_shape_.H * input_shape_.W * sizeof(uint8_t);
    checkCudaError(cudaMalloc(&d_rgb_data_, rgb_size), "cudaMalloc for vision pipeline RGB data");

    size_t output_size = max_size_ * output_shape_.N * output_shape_.C * output_shape_.H * output_shape_.W * sizeof(float);
    checkCudaError(cudaMalloc(&d_output_buffer_, output_size),"cudaMalloc for vision pipeline output");

    LogMessage("VisionPipeline: Allocated CUDA buffers: "
               "d_rgb_data_ = {}, d_output_buffer_ = {}, with sizes: {} bytes and {} bytes",
                (void*)d_rgb_data_, (void*)d_output_buffer_,
                rgb_size, output_size);
    return true;
}

void VisionPipeline::deallocateCudaBuffers() {
    if (d_rgb_data_) {
        cudaFree(d_rgb_data_);
        d_rgb_data_ = nullptr;
    }
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
}

void VisionPipeline::pipelineWorker(std::stop_token stop_token) {
    size_t num_images_per_iter = output_shape_.N;

    uint8_t** rgb_images = new uint8_t*[num_images_per_iter];
    float** depth_images = new float*[num_images_per_iter];

    size_t num_rgb_elemenets   = input_shape_.N * input_shape_.C * input_shape_.H * input_shape_.W;
    size_t num_output_elements = output_shape_.N * output_shape_.C * output_shape_.H * output_shape_.W;

    while (!stop_token.stop_requested() && running_.load()) {
        try {
            // Get current images from SpotConnection
            if (!spot_connection_.getCurrentImages(
                    spot_connection_.getCurrentNumCams(),
                    rgb_images,
                    depth_images)
            ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // Prepare device pointers
            uint8_t* d_rgb_ptr = d_rgb_data_ + write_idx_ * num_rgb_elemenets;
            float* depth_write_ptr = d_output_buffer_ + write_idx_ * num_output_elements;

            // Copy RGB images to device (in order to ensure synchronization between processed depth and RGB images)
            checkCudaError(cudaMemcpyAsync(
                d_rgb_ptr,
                rgb_images[0],
                num_rgb_elemenets * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice,
                cuda_stream_
            ), "cudaMemcpyAsync for RGB images");

            // Preprocess depth images
            // for (size_t i = 0; i < num_images_per_iter; i++) {
            //     checkCudaError(preprocess_depth_image(
            //         depth_images[i],
            //         depth_shape_.W,
            //         depth_shape_.H
            //     ), "preprocess_depth_image");
            // }

            // Run inference on all images at ONCE ^^
            bool inference_success = model_.runInference(
                d_rgb_ptr,
                depth_images[0],
                depth_write_ptr,
                input_shape_,
                depth_shape_,
                output_shape_
            );
            
            if (!inference_success) {
                LogMessage("Inference failed");
                continue;
            }

            // Update read index
            read_idx_.store(write_idx_, std::memory_order_release);
            new_data_.store(true, std::memory_order_release);

            // Update write index (circular)
            LogMessage("Updating write index from {} to {}",
                       write_idx_, (write_idx_ + 1) % max_size_);
            write_idx_ = (write_idx_ + 1) % max_size_;

        } catch (const std::exception& e) {
            std::cerr << "Exception in pipeline worker: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    // Cleanup
    delete[] rgb_images;
    delete[] depth_images;
}

bool VisionPipeline::getCurrentImages(
    int32_t n_images_requested,
    uint8_t** images,
    float** depths
) const {
    bool expected_new_data = true;
    bool desired_new_data = false;
    if (!new_data_.compare_exchange_weak(expected_new_data, desired_new_data)) {
        return false;
    }

    size_t n_elems_per_rgb      = input_shape_.C * input_shape_.H * input_shape_.W;
    size_t n_elems_per_depth    = output_shape_.C * output_shape_.H * output_shape_.W;
    size_t n_elems_rgb_total    = input_shape_.N * n_elems_per_rgb;
    size_t n_elems_depth_total  = output_shape_.N * n_elems_per_depth;

    int read_idx = read_idx_.load(std::memory_order_relaxed);

    uint8_t* rgb_data_out = d_rgb_data_      + read_idx * n_elems_rgb_total;
    float* depth_data_out = d_output_buffer_ + read_idx * n_elems_depth_total;

    LogMessage("VisionPipeline::getCurrentImages, popping {} images from index {}",
               input_shape_.N, read_idx);

    for (int32_t i = 0; i < n_images_requested; i++) {
        images[i] = rgb_data_out   + i * n_elems_per_rgb;
        depths[i] = depth_data_out + i * n_elems_per_depth;
    }

    return true;
}

} // namespace SOb