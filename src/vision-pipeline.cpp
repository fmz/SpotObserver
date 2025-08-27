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
    if (cuda_ws_.d_depth_preprocessor_workspace_) {
        cudaFree(cuda_ws_.d_depth_preprocessor_workspace_);
        cuda_ws_.d_depth_preprocessor_workspace_ = nullptr;
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
            float*   d_depth_output_ptr = d_output_buffer_ + write_idx_ * num_output_elements;

            // Copy inputs to our local locations (in order to ensure synchronization between processed depth and RGB images)
            checkCudaError(cudaMemcpyAsync(
                d_rgb_ptr,
                rgb_images[0],
                num_rgb_elemenets * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice
                ), "cudaMemcpyAsync for RGB images");

            checkCudaError(cudaMemcpyAsync(
                cuda_ws_.d_depth_data_,
                depth_images[0],
                num_depth_elements * sizeof(float),
                cudaMemcpyDeviceToDevice
                ), "cudaMemcpyAsync for RGB images");

            // Convert RGB images to float and normalize to [0,1]
            bool do_rotate_90_cw = false;
            convert_uint8_img_to_float_img(
                d_rgb_ptr,
                cuda_ws_.d_rgb_float_data_,
                input_shape_.N,
                input_shape_.H,
                input_shape_.W,
                input_shape_.C,
                true,
                0,
                do_rotate_90_cw
            );

            // Preprocess depth images
            LogMessage("num_images_per_iter = {}", num_images_per_iter);
            for (size_t i = 0; i < num_images_per_iter; i++) {
                float* cur_rgb_input_ptr    = cuda_ws_.d_rgb_float_data_ + i * 3 * input_shape_.H * input_shape_.W;
                float* cur_depth_input_ptr  = cuda_ws_.d_depth_data_ + i * depth_shape_.C * depth_shape_.H * depth_shape_.W;
                float* cur_depth_output_ptr = d_depth_output_ptr + i * output_shape_.C * output_shape_.H * output_shape_.W;

                LogMessage("Starting pipeline for image {}. cur_rgb_ptr = {:#x}, cur_depth_ptr = {:#x}, cur_depth_output_ptr = {:#x}",
                           i, size_t(cur_rgb_input_ptr), size_t(cur_depth_input_ptr), size_t(cur_depth_output_ptr));


                int32_t depth_scale_factor = 4;
                checkCudaError(preprocess_depth_image2(
                    cur_depth_input_ptr,
                    depth_shape_.W,
                    depth_shape_.H,
                    depth_scale_factor,
                    cuda_ws_.d_depth_preprocessor_workspace_,
                    do_rotate_90_cw
                ), "preprocess_depth_image");


                TensorShape input_shape_float = input_shape_;
                input_shape_float.N = 1; // Process one image at a time
                input_shape_float.C = 3; // float RGB has 3 channels
                if (do_rotate_90_cw) {
                    input_shape_float.H = input_shape_.W;
                    input_shape_float.W = input_shape_.H;
                }

                TensorShape depth_shape_single = depth_shape_;
                depth_shape_single.N = 1; // Process one image at a time
                if (do_rotate_90_cw) {
                    depth_shape_single.H = depth_shape_.W / depth_scale_factor;
                    depth_shape_single.W = depth_shape_.H / depth_scale_factor;
                } else {
                    depth_shape_single.H = depth_shape_.H / depth_scale_factor;
                    depth_shape_single.W = depth_shape_.W / depth_scale_factor;
                }

                TensorShape output_shape_single = output_shape_;
                output_shape_single.N = 1; // Process one image at a time
                if (do_rotate_90_cw) {
                    output_shape_single.H = output_shape_.W;
                    output_shape_single.W = output_shape_.H;
                } else {
                    output_shape_single.H = output_shape_.H;
                    output_shape_single.W = output_shape_.W;
                }

                // Run inference
                // Need to synchronize here to ensure that the depth preprocessor is done before we run inference
                // TODO: Make this more efficient by using CUDA events and/or streams
                checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize before running inference");

                bool inference_success = model_.runInference(
                    cur_rgb_input_ptr,
                    cur_depth_input_ptr,
                    cur_depth_output_ptr,
                    input_shape_float,
                    depth_shape_single,
                    output_shape_single
                );

                if (!inference_success) {
                    LogMessage("Inference failed");
                    continue;
                }

                // Postprocess output: rotate back if input was rotated
                LogMessage("About to postprocess depth image");
                float* temp_output_ptr = reinterpret_cast<float*>(cuda_ws_.d_depth_preprocessor_workspace_);
                checkCudaError(postprocess_depth_image(
                    cur_depth_output_ptr,
                    output_shape_single.W,
                    output_shape_single.H,
                    temp_output_ptr,
                    do_rotate_90_cw
                ), "postprocess_depth_image");

                LogMessage("Done postprocessing depth image");

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

            // Update read index
            read_idx_.store(write_idx_, std::memory_order_release);
            new_data_.store(true, std::memory_order_release);

            // Update write index (circular)
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

    LogMessage("VisionPipeline::getCurrentImages, popping {} images from index {}",
               input_shape_.N, read_idx);

    for (int32_t i = 0; i < n_images_requested; i++) {
        images[i] = rgb_data_out   + i * n_elems_per_rgb;
        depths[i] = depth_data_out + i * n_elems_per_depth;
    }

    return true;
}

} // namespace SOb