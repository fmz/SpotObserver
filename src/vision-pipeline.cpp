//
// Created by brown on 8/20/2025.
//

#include "include/vision-pipeline.h"
#include "cuda_kernels.cuh"
#include "utils.h"
#include "dumper.h"

#include <chrono>
#include <algorithm>

namespace SOb {

static int32_t dump_id = 900;

VisionPipeline::VisionPipeline(
    MLModel& model,
    const SpotCamStream& spot_cam_stream_,
    const TensorShape& input_shape,
    const TensorShape& depth_shape,
    const TensorShape& output_shape,
    size_t max_results
  ) : model_(model)
    , spot_cam_stream_(std::move(spot_cam_stream_))
    , input_shape_(input_shape)
    , depth_shape_(depth_shape)
    , output_shape_(output_shape)
    , max_size_(max_results)
    , cuda_stream_(spot_cam_stream_.getCudaStream())
{ }

VisionPipeline::~VisionPipeline() {
    stop();
    deallocateCudaBuffers();

}

bool VisionPipeline::start() {
    if (running_.load()) {
        return false; // Already running
    }
    
    if (!spot_cam_stream_.isStreaming()) {
        LogMessage("SpotCamStream must be connected and streaming before starting pipeline");
        return false;
    }

    TensorShape stream_rgb_shape = spot_cam_stream_.getCurrentRGBTensorShape();
        if (input_shape_ != stream_rgb_shape) {
        LogMessage("Incompatible shapes: input_shape {} != stream_rgb_shape {}",
            input_shape_.to_string(),
            stream_rgb_shape.to_string()
        );
        return false;
    }
    TensorShape stream_depth_shape = spot_cam_stream_.getCurrentDepthTensorShape();
    if (depth_shape_ != stream_depth_shape) {
        LogMessage("Incompatible shapes: depth_shape_ {} != stream_depth_shape {}",
            depth_shape_.to_string(),
            stream_depth_shape.to_string()
        );
        return false;
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
    LogMessage("Vision pipeline stopping...");

    if (pipeline_thread_ && pipeline_thread_->joinable()) {
        pipeline_thread_->request_stop();
        pipeline_thread_->join();
        LogMessage("Vision pipeline thread joined");
    }
}

bool VisionPipeline::allocateCudaBuffers() {
    deallocateCudaBuffers();

    size_t rgb_image_size_batch = input_shape_.N * input_shape_.C * input_shape_.H * input_shape_.W * sizeof(uint8_t);
    size_t rgb_buffer_size = max_size_ * rgb_image_size_batch;

    checkCudaError(cudaMalloc(&d_rgb_data_, rgb_buffer_size), "cudaMalloc for vision pipeline RGB data");
    checkCudaError(cudaMalloc(&cuda_ws_.d_rgb_float_data_, rgb_image_size_batch * sizeof(float)), "cudaMalloc for vision pipeline RGB float data");

    size_t depth_size = depth_shape_.N * depth_shape_.C * depth_shape_.H * depth_shape_.W * sizeof(float);
    checkCudaError(cudaMalloc(&cuda_ws_.d_depth_data_, depth_size), "cudaMalloc for vision pipeline input depth data");
    checkCudaError(cudaMalloc(&cuda_ws_.d_preprocessed_depth_data_, depth_size), "cudaMalloc for vision pipeline input depth data preprocessed");
    checkCudaError(cudaMalloc(&cuda_ws_.d_depth_cached_, depth_size), "cudaMalloc for vision pipeline input depth cached");

    size_t depth_workspace_size = depth_preprocessor2_get_workspace_size(depth_shape_.W, depth_shape_.H);
    checkCudaError(cudaMalloc(&cuda_ws_.d_depth_preprocessor_workspace_, depth_workspace_size), "cudaMalloc for vision pipeline depth preprocessor workspace");

    size_t output_size = max_size_ * output_shape_.N * output_shape_.C * output_shape_.H * output_shape_.W * sizeof(float);
    checkCudaError(cudaMalloc(&d_output_buffer_, output_size),"cudaMalloc for vision pipeline output");

    LogMessage("Allocated CUDA buffers for vision pipeline:"
        "\n  RGB data: {} bytes"
        "\n  RGB float data: {} bytes"
        "\n  Depth data: {} bytes"
        "\n  Depth preprocessor workspace: {} bytes"
        "\n  Output buffer: {} bytes",
        rgb_buffer_size,
        rgb_image_size_batch * sizeof(float),
        depth_size,
        depth_workspace_size,
        output_size
    );

    return true;
}

void VisionPipeline::deallocateCudaBuffers() {
    if (d_rgb_data_) {
        cudaFree(d_rgb_data_);
        d_rgb_data_ = nullptr;
    }
    if (cuda_ws_.d_rgb_float_data_) {
        cudaFree(cuda_ws_.d_rgb_float_data_);
        cuda_ws_.d_rgb_float_data_ = nullptr;
    }
    if (cuda_ws_.d_depth_data_) {
        cudaFree(cuda_ws_.d_depth_data_);
        cuda_ws_.d_depth_data_ = nullptr;
    }
    if (cuda_ws_.d_preprocessed_depth_data_) {
        cudaFree(cuda_ws_.d_preprocessed_depth_data_);
        cuda_ws_.d_preprocessed_depth_data_ = nullptr;
    }
    if (cuda_ws_.d_depth_preprocessor_workspace_) {
        cudaFree(cuda_ws_.d_depth_preprocessor_workspace_);
        cuda_ws_.d_depth_preprocessor_workspace_ = nullptr;
    }
    if (cuda_ws_.d_depth_cached_) {
        cudaFree(cuda_ws_.d_depth_cached_);
        cuda_ws_.d_depth_cached_ = nullptr;
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
    size_t num_depth_elements  = depth_shape_.N * depth_shape_.C * depth_shape_.H * depth_shape_.W;
    size_t num_output_elements = output_shape_.N * output_shape_.C * output_shape_.H * output_shape_.W;

    while (!stop_token.stop_requested() && running_.load()) {
        auto start_time = std::chrono::high_resolution_clock::now();
        try {
            // Get current images from SpotConnection
            if (!spot_cam_stream_.getCurrentImages(
                    input_shape_.N,
                    rgb_images,
                    depth_images)
            ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // Prepare device pointers
            uint8_t* d_rgb_ptr = d_rgb_data_ + write_idx_ * num_rgb_elemenets;
            float*   d_depth_output_ptr = d_output_buffer_ + write_idx_ * num_output_elements;

            constexpr int32_t depth_scale_factor = 4;
            constexpr float   inv_scale_factor = 1.0f / depth_scale_factor;

            // Copy inputs on the per-connection stream
            checkCudaError(cudaMemcpyAsync(
                d_rgb_ptr,
                rgb_images[0],
                num_rgb_elemenets * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice,
                cuda_stream_ // NEW
            ), "cudaMemcpyAsync for RGB images");

            checkCudaError(cudaMemcpyAsync(
                cuda_ws_.d_depth_data_,
                depth_images[0],
                num_depth_elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                cuda_stream_ // NEW
            ), "cudaMemcpyAsync for depth images");

            // Convert RGB images to float and normalize to [0,1] on our stream
            bool do_rotate_90_cw = false;
            convert_uint8_img_to_float_img(
                d_rgb_ptr,
                cuda_ws_.d_rgb_float_data_,
                input_shape_.N,
                input_shape_.H,
                input_shape_.W,
                input_shape_.C,
                true,
                cuda_stream_,
                do_rotate_90_cw
            );

            // Preprocess depth images
            TensorShape input_shape_float = input_shape_;
            input_shape_float.C = 3;
            if (do_rotate_90_cw) {
                input_shape_float.H = input_shape_.W;
                input_shape_float.W = input_shape_.H;
            }

            TensorShape depth_shape = depth_shape_;
            if (do_rotate_90_cw) {
                depth_shape.H = depth_shape_.W / depth_scale_factor;
                depth_shape.W = depth_shape_.H / depth_scale_factor;
            } else {
                depth_shape.H = depth_shape_.H / depth_scale_factor;
                depth_shape.W = depth_shape_.W / depth_scale_factor;
            }

            TensorShape output_shape = output_shape_;
            if (do_rotate_90_cw) {
                output_shape.H = output_shape_.W;
                output_shape.W = output_shape_.H;
            } else {
                output_shape.H = output_shape_.H;
                output_shape.W = output_shape_.W;
            }
            LogMessage("num_images_per_iter = {}", num_images_per_iter);

            for (size_t i = 0; i < num_images_per_iter; i++) {
                float* cur_rgb_input_ptr    = cuda_ws_.d_rgb_float_data_ + i * 3 * input_shape_.H * input_shape_.W;
                float* cur_depth_input_ptr  = cuda_ws_.d_depth_data_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;
                float* cur_preprocessed_depth_ptr = cuda_ws_.d_preprocessed_depth_data_ + i * depth_shape.C * depth_shape.H * depth_shape.W;
                float* cur_depth_output_ptr = d_depth_output_ptr + i * output_shape_.C * output_shape_.H * output_shape_.W;
                float* depth_cache_ptr      = cuda_ws_.d_depth_cached_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;

                LogMessage("Starting pipeline for image {}. cur_rgb_ptr = {:#x}, cur_depth_ptr = {:#x}, cur_depth_output_ptr = {:#x}",
                           i, size_t(cur_rgb_input_ptr), size_t(cur_depth_input_ptr), size_t(cur_depth_output_ptr));
                if (!first_run_) {
                    checkCudaError(prefill_invalid_depth(
                        cur_depth_input_ptr,
                        cur_preprocessed_depth_ptr,
                        depth_cache_ptr,
                        depth_shape_.W,
                        depth_shape_.H,
                        0.01f,
                        100.0f,
                        cuda_stream_
                    ), "prefill_invalid_depth");

                    // Downscale
                    checkCudaError(preprocess_depth_image2(
                        cur_preprocessed_depth_ptr,
                        cur_preprocessed_depth_ptr,
                        depth_shape_.W,
                        depth_shape_.H,
                        false,
                        depth_scale_factor,
                        cuda_ws_.d_depth_preprocessor_workspace_,
                        do_rotate_90_cw,
                        cuda_stream_
                    ), "preprocess_depth_image");

                } else {
                    checkCudaError(preprocess_depth_image2(
                        cur_depth_input_ptr,
                        cur_preprocessed_depth_ptr,
                        depth_shape_.W,
                        depth_shape_.H,
                        true,
                        depth_scale_factor,
                        cuda_ws_.d_depth_preprocessor_workspace_,
                        do_rotate_90_cw,
                        cuda_stream_
                    ), "preprocess_depth_image");
                }
            }

            auto preprocess_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_time - start_time);
            LogMessage("VisionPipeline preprocess time: {} ms", duration.count());

            // Stream-scoped sync before inference (keeps other connections running)
            checkCudaError(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize before running inference");

            // TODO: Support running models on cuda_stream_
            bool inference_success = model_.runInference(
                cuda_ws_.d_rgb_float_data_,
                cuda_ws_.d_preprocessed_depth_data_,
                d_depth_output_ptr,
                input_shape_float,
                depth_shape,
                output_shape
            );

            if (!inference_success) {
                LogMessage("Inference failed");
                continue;
            }


            auto inference_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - preprocess_time);
            LogMessage("VisionPipeline inference time: {} ms", duration.count());

            for (size_t i = 0; i < num_images_per_iter; i++) {
                float* cur_rgb_input_ptr    = cuda_ws_.d_rgb_float_data_ + i * 3 * input_shape_.H * input_shape_.W;
                float* cur_depth_input_ptr  = cuda_ws_.d_depth_data_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;
                float* cur_preprocessed_depth_ptr = cuda_ws_.d_preprocessed_depth_data_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;
                float* cur_depth_output_ptr = d_depth_output_ptr + i * output_shape_.C * output_shape_.H * output_shape_.W;
                float* depth_cache_ptr      = cuda_ws_.d_depth_cached_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;

                // Postprocess output: rotate back if input was rotated
                float* temp_output_ptr = reinterpret_cast<float*>(cuda_ws_.d_depth_preprocessor_workspace_);
                checkCudaError(postprocess_depth_image(
                    cur_depth_output_ptr,
                    output_shape.W,
                    output_shape.H,
                    temp_output_ptr,
                    do_rotate_90_cw,
                    cuda_stream_
                ), "postprocess_depth_image");

                checkCudaError(update_depth_cache(
                    cur_depth_output_ptr,
                    cur_depth_input_ptr,
                    depth_cache_ptr,
                    output_shape.W,
                    output_shape.H,
                    0.01f,
                    100.0f,
                    cuda_stream_
                ), "update_depth_cache");

                // Ensure dumps see completed work (dumpers likely use default stream)
                checkCudaError(cudaStreamSynchronize(cuda_stream_), "sync before dumps");

                DumpRGBImageFromCudaCHW(
                    cur_rgb_input_ptr,
                    input_shape_.W,
                    input_shape_.H,
                    "input-rgb",
                    dump_id
                );
                DumpDepthImageFromCuda(
                    cur_depth_input_ptr,
                    depth_shape_.W,
                    depth_shape_.H,
                    "input-depth",
                    dump_id
                );
                DumpDepthImageFromCuda(
                    d_depth_output_ptr,
                    depth_shape_.W,
                    depth_shape_.H,
                    "output-depth",
                    dump_id
                );
                dump_id++;
            }

            checkCudaError(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize after postprocess");
            auto postprocess_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_time - inference_time);
            LogMessage("VisionPipeline postprocess time: {} ms", duration.count());

            // Publish only after stream work completes
            read_idx_.store(write_idx_, std::memory_order_release);
            new_data_.store(true, std::memory_order_release);

            LogMessage("VisionPipeline: Updating write index from {} to {}",
                       write_idx_, (write_idx_ + 1) % max_size_);
            write_idx_ = (write_idx_ + 1) % max_size_;

        } catch (const std::exception& e) {
            LogMessage("Exception in pipeline worker: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LogMessage("VisionPipeline iteration time: {} ms", duration.count());
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

    for (int32_t i = 0; i < n_images_requested; i++) {
        images[i] = rgb_data_out   + i * n_elems_per_rgb;
        depths[i] = depth_data_out + i * n_elems_per_depth;
    }

    return true;
}

} // namespace SOb