//
// Created by fmz on 7/23/2025.
//
#include "spot-observer.h"
#include "spot-connection.h"
#include "logger.h"
#include "utils.h"
#include "dumper.h"
#include "vision-pipeline.h"

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include <thread>

namespace SOb {

static cv::Mat convert_image_to_cv_mat(const bosdyn::api::Image& img, double depth_scale = 1.0) {
    if (img.format() == bosdyn::api::Image::FORMAT_JPEG) {
        // Decode JPEG data
        std::vector<uchar> data(img.data().begin(), img.data().end());
        cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);
        // Convert to float rgb
        cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
        // image.convertTo(image, CV_32F, 1.0 / 255.0); // Convert to float in range [0, 1]
        return image;
    } else {
        // Handle raw pixel data based on pixel format
        cv::Mat image;

        switch (img.pixel_format()) {
        case bosdyn::api::Image::PIXEL_FORMAT_RGB_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC3,
                          const_cast<char*>(img.data().data()));
            image.convertTo(image, CV_32F, 1.0 / 255.0); // Convert to float in range [0, 1]CV_32F
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC4,
                          const_cast<char*>(img.data().data()));
            image.convertTo(image, CV_32F, 1.0 / 255.0); // Convert to float in range [0, 1]
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC1,
                          const_cast<char*>(img.data().data())  );
            image.convertTo(image, CV_32F, 1.0 / 255.0); // Convert to float in range [0, 1]
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U16:
            image = cv::Mat(img.rows(), img.cols(), CV_16UC1,
                          const_cast<char*>(img.data().data()));
            image.convertTo(image, CV_32F, 1.0/depth_scale);
            break;

        default:
            std::cerr << "Unsupported pixel format: " << img.pixel_format() << std::endl;
            return cv::Mat();
        }

        return image.clone();
    }
}

ReaderWriterCBuf::ReaderWriterCBuf(size_t max_size)
    : max_size_(max_size)
    , rgb_data_(nullptr)
    , depth_data_(nullptr)
{
    // Stream is owned by SpotConnection and attached via attachCudaStream().
    LogMessage("Created ReaderWriterCBuf with max size {}", max_size_);
}

ReaderWriterCBuf::~ReaderWriterCBuf() {
    if (rgb_data_) {
        cudaFree(rgb_data_);
        rgb_data_ = nullptr;
    }
    if (depth_data_) {
        cudaFree(depth_data_);
        depth_data_ = nullptr;
    }
    if (cached_depth_) {
        cudaFree(cached_depth_);
        cached_depth_ = nullptr;
    }
    // Do not destroy cuda_stream_ here; SpotConnection owns it.
    LogMessage("Destroyed ReaderWriterCBuf");
}

bool ReaderWriterCBuf::initialize(
    size_t n_elems_per_rgb,
    size_t n_elems_per_depth,
    const std::vector<SpotCamera>& cameras
) {
    n_elems_per_rgb_ = n_elems_per_rgb;
    n_elems_per_depth_ = n_elems_per_depth;
    cameras_ = cameras;
    n_images_per_response_ = cameras.size();

    if (rgb_data_ != nullptr || depth_data_ != nullptr) {
        if (!rgb_data_)   checkCudaError(cudaFree(rgb_data_), "cudaFree for RGB data");
        if (!depth_data_) checkCudaError(cudaFree(depth_data_), "cudaFree for Depth data");

        LogMessage("ReaderWriterCBuf::initialize: Re-initializing, freed existing buffers");
    }

    // Allocate CUDA memory for circular buffer
    size_t total_size_rgb   = max_size_ * n_elems_per_rgb * n_images_per_response_ * sizeof(uint8_t);
    size_t size_depth_per_response = n_elems_per_depth * n_images_per_response_ * sizeof(float);
    size_t total_size_depth = max_size_ * size_depth_per_response;

    checkCudaError(cudaMalloc(&rgb_data_, total_size_rgb), "cudaMalloc for RGB data");
    checkCudaError(cudaMalloc(&depth_data_, total_size_depth), "cudaMalloc for Depth data");
    checkCudaError(cudaMalloc(&cached_depth_, size_depth_per_response), "cudaMalloc for Cached Depth data");

    LogMessage("Allocated {} bytes for RGB data and {} bytes for Depth data in ReaderWriterCBuf"
                "({} bytes per RGB, {} bytes per Depth, {} images per response, max size {})",
               total_size_rgb, total_size_depth, n_elems_per_rgb * sizeof(uint8_t), n_elems_per_depth * sizeof(float), n_images_per_response_, max_size_);
    LogMessage("Min RGB address = {:#x}, Max RGB address = {:#x}",
               size_t(rgb_data_), size_t(rgb_data_) + total_size_rgb);
    LogMessage("Min depth address = {:#x}, Max depth address = {:#x}",
               size_t(depth_data_), size_t(depth_data_) + total_size_depth);

    write_idx_ = 0;
    read_idx_ = 0;
    new_data_ = false;

    return true;
}

/**
 * Push image data to queue (non-blocking, drops oldest if full)
 */
void ReaderWriterCBuf::push(const google::protobuf::RepeatedPtrField<bosdyn::api::ImageResponse>& responses) {
    using namespace std::chrono;

    if (n_elems_per_rgb_ == 0 || n_elems_per_depth_ == 0) {
        throw std::runtime_error("ReaderWriterCBuf::push: n_elems_per_rgb_ == 0");
    }

    // static time_point<high_resolution_clock> start_time = high_resolution_clock::now();
    // time_point<high_resolution_clock> end_time = high_resolution_clock::now();
    //
    // auto duration = duration_cast<microseconds>(end_time - start_time);
    // double latency_ms = duration.count() / 1000.0;
    //
    // LogMessage("ReaderWriterCBuf::push: latency: {:.4f} ms", latency_ms);
    // start_time = end_time; // Reset start time for next push

    // Compute write pointer
    int write_idx = write_idx_.load(std::memory_order_relaxed);
    uint8_t* rgb_write_ptr = rgb_data_   + write_idx * n_elems_per_rgb_   * n_images_per_response_;
    float* depth_write_ptr = depth_data_ + write_idx * n_elems_per_depth_ * n_images_per_response_;
    float* depth_cache_ptr = cached_depth_;

    int32_t n_rgbs_written = 0;
    int32_t n_depths_written = 0;

    for (const auto& response : responses) {
        const auto& img = response.shot().image();

        switch (img.pixel_format()) {
        case bosdyn::api::Image::PIXEL_FORMAT_RGB_U8:
        case bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U8:
        {
            cv::Mat cv_img = convert_image_to_cv_mat(img);

            size_t image_size = cv_img.cols * cv_img.rows * 4;//(img.pixel_format() == bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8 ? 4 : 3);
            if (image_size != n_elems_per_rgb_) {
                LogMessage("Image size mismatch: expected {}, got {}", n_elems_per_rgb_, image_size);
                throw std::runtime_error("ReaderWriterCBuf::push: Image size mismatch");
            }

            // See if we need to do any cam-specific preprocessing:
            switch (cameras_[n_rgbs_written]) {
            case SpotCamera::FRONTLEFT:
            case SpotCamera::FRONTRIGHT:
                // Mirror image
                //cv::flip(cv_img, cv_img, 0); // Flip vertically
                break;
            }

            LogMessage("Copying RGB image of size {} bytes to write pointer at index {}, {:#x} rgb_write_ptr",
                       image_size, write_idx, size_t(rgb_write_ptr));

            checkCudaError(
                cudaMemcpyAsync(
                    rgb_write_ptr,
                    cv_img.data,
                    image_size * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    cuda_stream_ /* use per-connection stream */
                ),
                "cudaMemcpyAsync RGB"
            );

            DumpRGBImageFromCuda(
                rgb_write_ptr,
                cv_img.cols,
                cv_img.rows,
                cv_img.channels(),
                "rgb",
                n_rgbs_written + write_idx * n_images_per_response_
            );

            rgb_write_ptr += n_elems_per_rgb_;
            n_rgbs_written++;
        }
        break;

        case bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U16:
        {
            cv::Mat cv_img = convert_image_to_cv_mat(img, response.source().depth_scale());
            int depth_width = cv_img.cols;
            int depth_height = cv_img.rows;

            size_t depth_size = depth_width * depth_height;
            if (depth_size != n_elems_per_depth_) {
                LogMessage("Depth size mismatch: expected {}, got {}", n_elems_per_depth_, depth_size);
                throw std::runtime_error("ReaderWriterCBuf::push: Depth size mismatch");
            }

            LogMessage("Copying depth image of size {} bytes to write pointer at index {}, {:#x} rgb_write_ptr",
                       depth_size, write_idx, size_t(depth_write_ptr));

            checkCudaError(
                cudaMemcpyAsync(
                    depth_write_ptr,
                    cv_img.data,
                    depth_size * sizeof(float),
                    cudaMemcpyHostToDevice,
                    cuda_stream_
                ),
                "cudaMemcpyAsync DEPTH"
            );

            DumpDepthImageFromCuda(
                depth_write_ptr,
                cv_img.cols,
                cv_img.rows,
                "depth",
                n_depths_written + write_idx * n_images_per_response_
            );

            depth_write_ptr += n_elems_per_depth_;
            depth_cache_ptr += n_elems_per_depth_;
            n_depths_written++;
        }
        break;

        default:
            LogMessage("Unsupported pixel format: {img.pixel_format()}");
            throw std::runtime_error("ReaderWriterCBuf::push: Got an unsupported pixel format");
        }
    }

    // Make this batch visible only after the stream's async work completes.
    checkCudaError(cudaStreamSynchronize(cuda_stream_), "cudaStreamSynchronize after push");

    // Update indices/flags
    assert(n_rgbs_written == n_depths_written);
    // Update the read index to the write index we just wrote to.
    read_idx_.store(write_idx, std::memory_order_release);
    new_data_.store(true, std::memory_order_release);
    LogMessage("ReaderWriterCBuf::push: updating write index from {} to {}",
               write_idx, (write_idx + 1) % max_size_);
    write_idx = (write_idx + 1) % max_size_;
    write_idx_.store(write_idx, std::memory_order_release);
    first_run_ = false;
}

/**
 * Consume image and depth data
 * Using more of a LIFO approach here, so that we can pop the most recent data first (see push function)
 */
std::pair<uint8_t*, float*> ReaderWriterCBuf::pop(int32_t count) const {
    bool expected_new_data = true;
    bool desired_new_data = false;
    if (!new_data_.compare_exchange_weak(expected_new_data, desired_new_data)) {
        return std::make_pair(nullptr, nullptr);
    }
    new_data_.store(false, std::memory_order_release);

    int read_idx = read_idx_.load(std::memory_order_relaxed);

    uint8_t* rgb_data_out = rgb_data_   + read_idx * n_elems_per_rgb_   * n_images_per_response_;
    float* depth_data_out = depth_data_ + read_idx * n_elems_per_depth_ * n_images_per_response_;

    // read_idx += 1;
    // read_idx = read_idx % max_size_;

    LogMessage("ReaderWriterCBuf::pop, popping {} images from index {}",
               count, read_idx);
    //read_idx_.store(read_idx, std::memory_order_release);

    return std::make_pair(rgb_data_out, depth_data_out);
}

///////////////////////////////////////////////////////////////////////////////

static std::vector<std::string> get_rgb_cam_names_from_bit_mask(uint32_t bitmask) {
    std::vector<std::string> cam_names;
    cam_names.reserve(__num_set_bits(bitmask));

    if (bitmask & SpotCamera::BACK)       cam_names.emplace_back("back_fisheye_image");
    if (bitmask & SpotCamera::FRONTLEFT)  cam_names.emplace_back("frontleft_fisheye_image");
    if (bitmask & SpotCamera::FRONTRIGHT) cam_names.emplace_back("frontright_fisheye_image");
    if (bitmask & SpotCamera::LEFT)       cam_names.emplace_back("left_fisheye_image");
    if (bitmask & SpotCamera::RIGHT)      cam_names.emplace_back("right_fisheye_image");
    if (bitmask & SpotCamera::HAND)       cam_names.emplace_back("hand_color_image");

    return cam_names;
}

static std::vector<std::string> get_depth_cam_names_from_bit_mask(uint32_t bitmask) {
    std::vector<std::string> cam_names;
    cam_names.reserve(__num_set_bits(bitmask));

    if (bitmask & SpotCamera::BACK)       cam_names.emplace_back("back_depth_in_visual_frame");
    if (bitmask & SpotCamera::FRONTLEFT)  cam_names.emplace_back("frontleft_depth_in_visual_frame");
    if (bitmask & SpotCamera::FRONTRIGHT) cam_names.emplace_back("frontright_depth_in_visual_frame");
    if (bitmask & SpotCamera::LEFT)       cam_names.emplace_back("left_depth_in_visual_frame");
    if (bitmask & SpotCamera::RIGHT)      cam_names.emplace_back("right_depth_in_visual_frame");
    if (bitmask & SpotCamera::HAND)       cam_names.emplace_back("hand_depth_in_hand_color_frame");

    return cam_names;
}

static std::vector<SpotCamera> convert_bitmask_to_spot_cam_vector(uint32_t bitmask) {
    std::vector<SpotCamera> cams;

    cams.reserve(__num_set_bits(bitmask));

    if (bitmask & SpotCamera::BACK)       cams.emplace_back(SpotCamera::BACK);
    if (bitmask & SpotCamera::FRONTLEFT)  cams.emplace_back(SpotCamera::FRONTLEFT);
    if (bitmask & SpotCamera::FRONTRIGHT) cams.emplace_back(SpotCamera::FRONTRIGHT);
    if (bitmask & SpotCamera::LEFT)       cams.emplace_back(SpotCamera::LEFT);
    if (bitmask & SpotCamera::RIGHT)      cams.emplace_back(SpotCamera::RIGHT);
    if (bitmask & SpotCamera::HAND)       cams.emplace_back(SpotCamera::HAND);

    return cams;
}

///////////////////////////////////////////////////////////////////////////////

SpotCamStream::SpotCamStream(
    SpotConnection& robot,
    bosdyn::client::ImageClient* image_client,
    int32_t image_lifo_max_size
)
    : robot_(robot)
    , image_client_(image_client)
    , image_lifo_(image_lifo_max_size)
    , streaming_(false)
{
    // Create one CUDA stream per SpotConnection and attach it to the buffer.
    checkCudaError(
        cudaStreamCreate(&cuda_stream_),
        "cudaStreamCreate for SpotConnection"
    );
    image_lifo_.attachCudaStream(cuda_stream_);
    LogMessage("SpotCamStream::connect: Created CUDA stream {:#x} and attached to buffer",
               size_t(cuda_stream_));

}

SpotCamStream::~SpotCamStream() {
    quit_requested_.store(true);
    _joinStreamingThread();

    if (cuda_stream_) {
        checkCudaError(cudaStreamDestroy(cuda_stream_), "cudaStreamDestroy for SpotConnection");
        cuda_stream_ = nullptr;
        LogMessage("SpotCamStream::~SpotConnection: Destroyed CUDA stream");
    }
    // TODO: figure out how to cleanup image_client_
}

bosdyn::api::GetImageRequest SpotCamStream::_createImageRequest(
    const std::vector<std::string>& rgb_sources,
    const std::vector<std::string>& depth_sources
) {
    bosdyn::api::GetImageRequest request;

    // Add RGB image requests
    for (const std::string& source : rgb_sources) {
        bosdyn::api::ImageRequest* image_request = request.add_image_requests();
        image_request->set_image_source_name(source);
        image_request->set_quality_percent(100.0);
        image_request->set_image_format(bosdyn::api::Image_Format_FORMAT_JPEG);
        image_request->set_pixel_format(bosdyn::api::Image::PIXEL_FORMAT_RGB_U8);
    }

    // Add depth image requests
    for (const std::string& source : depth_sources) {
        bosdyn::api::ImageRequest* image_request = request.add_image_requests();
        image_request->set_image_source_name(source);
        image_request->set_quality_percent(100.0);
        // TODO: Use FORMAT_RLE format for depth (compresses 0s)
        image_request->set_image_format(bosdyn::api::Image_Format_FORMAT_RAW);
        image_request->set_pixel_format(bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16);
    }

    return request;
}

void SpotCamStream::_startStreamingThread() {
    // Create and start thread
    image_streamer_thread_ = std::make_unique<std::jthread>([this](std::stop_token stop_token) {
        _spotCamReaderThread(stop_token);
    });
}

// Image producer thread that requests images from the robot
void SpotCamStream::_spotCamReaderThread(std::stop_token stop_token) {
    LogMessage("Producer thread started");

    while (!stop_token.stop_requested() && !quit_requested_.load()) {
        try {
            auto start = std::chrono::high_resolution_clock::now();

            // Request images from all cameras
            bosdyn::client::GetImageResultType response = image_client_->GetImage(current_request_);
            if (!response.status) {
                LogMessage("Failed to get images: {}", response.status.message());
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Update statistics atomically
            num_samples_.fetch_add(1);

            // Process and queue images
            image_lifo_.push(response.response.image_responses());
            auto end = std::chrono::high_resolution_clock::now();
            LogMessage("Total time for GetImage and push: {:.4f} ms",
                       std::chrono::duration<double, std::milli>(end - start).count());
        } catch (const std::exception& e) {
            LogMessage("Error in producer thread: {}", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    LogMessage("Exiting producer thread.");
}

void SpotCamStream::_joinStreamingThread() {
    if (image_streamer_thread_) {
        image_streamer_thread_->request_stop();
        if (image_streamer_thread_->joinable()) {
            image_streamer_thread_->join();
            LogMessage("Joined streaming thread");
        }
        image_streamer_thread_.reset();
        LogMessage("Streaming thread stopped");
    } else {
        LogMessage("SpotCamStream::_joinStreamingThread: No streaming thread to join");
        return;
    }
}

bool SpotCamStream::streamCameras(uint32_t cam_mask) {
    if (!robot_.connected_) {
        LogMessage("SpotCamStream::streamCameras: Not connected to robot");
        return false;
    }

    if (cam_mask == 0 || cam_mask >= SpotCamera::NUM_CAMERAS) {
        LogMessage("SpotCamStream::streamCameras: Invalid camera mask: {:#x}", cam_mask);
        return false;
    }

    int32_t num_cams_requested = 0;
    if (cam_mask != current_cam_mask_) {
        LogMessage("Creating a new Spot image request with mask: {:#x}", cam_mask);
        std::vector<std::string> rgb_sources = get_rgb_cam_names_from_bit_mask(cam_mask);
        std::vector<std::string> depth_sources = get_depth_cam_names_from_bit_mask(cam_mask);
        camera_order_ = convert_bitmask_to_spot_cam_vector(cam_mask);

        LogMessage("Creating a request for {} RGB cameras and {} depth cameras",
                   rgb_sources.size(), depth_sources.size());

        for (int32_t i = 0; i < rgb_sources.size(); i++) {
            LogMessage("RGB Camera {}: {}", i, rgb_sources[i]);
        }
        for (int32_t i = 0; i < depth_sources.size(); i++) {
            LogMessage("Depth Camera {}: {}", i, depth_sources[i]);
        }

        num_cams_requested = rgb_sources.size();

        current_request_ = _createImageRequest(rgb_sources, depth_sources);
        if (image_streamer_thread_ != nullptr) {
            _joinStreamingThread();
        }
    }

    try {
        constexpr int32_t max_connection_retries = 3;
        // Query the robot for images and fill in image metadata
        // Max 3 retries
        for (int32_t i = 0; i < max_connection_retries; i++) {
            bosdyn::client::GetImageResultType response = image_client_->GetImage(current_request_);
            if (!response.status) {
                LogMessage("SpotCamStream::streamCameras: Failed to get images: {}",
                           response.status.message());
                LogMessage("SpotCamStream::streamCameras: Retrying... ({}/{})",
                           i + 1, max_connection_retries);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            LogMessage("SpotCamStream::streamCameras: Successfully retrieved images");
            const auto& image_responses = response.response.image_responses();
            if (image_responses.empty()) {
                LogMessage("SpotCamStream::streamCameras: No images received in response");
                return false;
            }
            // Read image sizes
            current_rgb_shape_ = TensorShape{
                size_t(num_cams_requested),
                4, //(image_responses[0].shot().image().pixel_format() == bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8 ? 4 : 3)
                size_t(image_responses[0].shot().image().rows()),
                size_t(image_responses[0].shot().image().cols())
            };

            size_t rgb_ref_size = current_rgb_shape_.H * current_rgb_shape_.W * 4;
                                  //(image_responses[0].shot().image().pixel_format() == bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8 ? 4 : 3);;
            // For debugging purposes, ensure that all RGB images have the same size
            for (int32_t j = 1; j < num_cams_requested; j++) {
                const auto& img_response = image_responses[j];
                size_t rgb_size = img_response.shot().image().rows() * img_response.shot().image().cols() * 4;
                                  //(img_response.shot().image().pixel_format() == bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8 ? 4 : 3);
                if (rgb_ref_size != rgb_size) {
                    LogMessage("SpotCamStream::streamCameras: Inconsistent RGB image sizes"
                               "(expected {}, got {})", rgb_ref_size, rgb_size);
                    return false;
                }
            }
            // Same thing for depth images
            current_depth_shape_ = TensorShape{
                size_t(num_cams_requested),
                1,
                size_t(image_responses[num_cams_requested].shot().image().rows()),
                size_t(image_responses[num_cams_requested].shot().image().cols())
            };

            size_t depth_ref_size = current_depth_shape_.H * current_depth_shape_.W;
            for (int32_t j = num_cams_requested + 1; j < image_responses.size(); j++) {
                const auto& img_response = image_responses[j];
                size_t depth_size = img_response.shot().image().rows() * img_response.shot().image().cols();
                if (depth_ref_size != depth_size) {
                    LogMessage("SpotCamStream::streamCameras: Inconsistent depth image sizes"
                               "(expected {}, got {})", depth_ref_size, depth_size);
                    return false;
                }
            }

            // (Re)initialize circular buffer
            image_lifo_.initialize(
                rgb_ref_size,
                depth_ref_size,
                camera_order_
            );

            break;
        }

        _startStreamingThread();
        streaming_ = true;

    } catch (const std::exception& e) {
        LogMessage("SpotCamStream::streamCameras: Exception while getting images: {}", e.what());
        streaming_ = false;
        return false;
    }

    current_cam_mask_ = cam_mask;
    current_num_cams_ = num_cams_requested;
    return true;
}

bool SpotCamStream::getCurrentImages(
    int32_t n_images_requested,
    uint8_t** images,
    float** depths
) const {
    auto [ret_images, ret_depths] = image_lifo_.pop(n_images_requested);
    if (ret_images == nullptr || ret_depths == nullptr) {
        return false;
    }

    for (int32_t i = 0; i < n_images_requested; i++) {
        images[i] = ret_images + i * image_lifo_.n_elems_per_rgb_;
        depths[i] = ret_depths + i * image_lifo_.n_elems_per_depth_;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////

SpotConnection::SpotConnection(
    const std::string& robot_ip,
    const std::string& username,
    const std::string& password
)
    : robot_(nullptr)
    , image_client_(nullptr)
    , image_lifo_max_size_(5)
    , connected_(false)
{
    // Create SDK instance
    sdk_ = bosdyn::client::CreateStandardSDK("SpotObserverConnection");
    if (!sdk_) {
        LogMessage("SpotConnection::SpotConnection: Failed to create SDK instance");
        throw std::runtime_error("Failed to create Spot SDK instance");
    }

    try {
        // Create robot using ClientSDK
        bosdyn::client::Result<std::unique_ptr<bosdyn::client::Robot>> robot_result = sdk_->CreateRobot(robot_ip);
        if (!robot_result.status) {
            throw std::runtime_error(std::format("SpotConnection::connect: Failed to connect to robot: {}",
                       robot_result.status.message()));
        }

        robot_ = std::move(robot_result.response);

        // Authenticate
        bosdyn::common::Status auth_status = robot_->Authenticate(username, password);
        if (!auth_status) {
            throw std::runtime_error(std::format("SpotConnection::connect: Failed to authenticate: {}",
                auth_status.message()));
        }

        // Create image client
        bosdyn::client::Result<bosdyn::client::ImageClient*> image_client_result =
            robot_->EnsureServiceClient<bosdyn::client::ImageClient>();

        if (!image_client_result.status) {
            throw std::runtime_error(std::format("SpotConnection::connect: Failed to create image client: {}",
                       image_client_result.status.message()));
        }

        image_client_ = image_client_result.response;

        LogMessage("SpotConnection::connect: Connected to Spot robot at {}", robot_ip);

        connected_ = true;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::format("SpotConnection::connect: Exception while connecting to robot {}: {}",
            robot_ip, e.what()));
    }
}

SpotConnection::~SpotConnection() {
    LogMessage("SpotConnection::~SpotConnection: Disconnecting from robot");

    vision_pipelines_.clear();
    cam_streams_.clear();

    if (connected_) {
        robot_.reset();
        sdk_.reset();
        // TODO: figure out how to cleanup image_client_
        connected_ = false;
    }
    LogMessage("SpotConnection::~SpotConnection: Robot state fully cleaned up");
}

int32_t SpotConnection::createCamStream(uint32_t cam_mask) {
    if (!connected_) {
        LogMessage("SpotConnection::createCamStream: Not connected to robot");
        return -1;
    }

    try {
        int32_t stream_id = next_stream_id_++;
        auto [it, inserted] = cam_streams_.try_emplace(
            stream_id,
            std::make_unique<SpotCamStream>(*this, image_client_, image_lifo_max_size_)
        );

        LogMessage("SpotConnection::createCamStream: Created camera stream with mask {:#x}",
                   cam_mask);

        if (!it->second->streamCameras(cam_mask)) {
            LogMessage("SpotConnection::createCamStream: Failed to start streaming cameras with mask {:#x}",
                       cam_mask);
            cam_streams_.erase(it);
            return -1;
        }

        return stream_id;

    } catch (const std::exception& e) {
        LogMessage("SpotConnection::createCamStream: Exception while creating camera stream: {}",
                   e.what());
        return -1;
    }
}

bool SpotConnection::removeCamStream(int32_t stream_id) {
    auto cam_stream_it = cam_streams_.find(stream_id);
    if (cam_stream_it == cam_streams_.end()) {
        LogMessage("SpotConnection::removeCamStream: Camera stream {} doesn't exist",
                   stream_id);
        return false;
    }
    try {
        // Remove the associated vision pipeline if any
        removeVisionPipeline(stream_id);

        cam_streams_.erase(cam_stream_it);
        LogMessage("SpotConnection::removeCamStream: Removed camera stream {}", stream_id);
        return true;
    } catch (const std::exception& e) {
        LogMessage("SpotConnection::removeCamStream: Exception while removing camera: {}",  e.what());
        return false;
    }
}

SpotCamStream* SpotConnection::getCamStream(int32_t stream_id) {
    auto cam_stream_it = cam_streams_.find(stream_id);
    if (cam_stream_it == cam_streams_.end()) {
        LogMessage("SpotConnection::getCamStream: Camera stream {} doesn't exist",
                   stream_id);
        return nullptr;
    }
    return cam_stream_it->second.get();
}

bool SpotConnection::createVisionPipeline(MLModel& model, int32_t stream_id) {
    SpotCamStream* cam_stream = getCamStream(stream_id);
    if (cam_stream == nullptr || !cam_stream->isStreaming()) {
        LogMessage("SpotConnection::createVisionPipeline: Camera stream {} doesn't exist. "
            "Please create and start streaming before creating a vision pipeline.",
            stream_id);
        return false;
    }
    if (getVisionPipeline(stream_id) != nullptr) {
        LogMessage("SpotConnection::createVisionPipeline: Vision pipeline for stream {} already exists",
                   stream_id);
        return false;
    }
    // Get the expected tensor shapes from the camera stream
    TensorShape rgb_shape = cam_stream->getCurrentRGBTensorShape();
    TensorShape depth_shape = cam_stream->getCurrentDepthTensorShape();

    try {
        auto [it, inserted] = vision_pipelines_.try_emplace(
            stream_id,
            std::make_unique<VisionPipeline>(
                model,
                *cam_stream,
                rgb_shape,
                depth_shape,
                depth_shape,
                image_lifo_max_size_
            )
        );
        if (!inserted) {
            LogMessage("SpotConnection::createVisionPipeline: Failed to insert vision pipeline for stream {}",
                       stream_id);
            return false;
        }
        LogMessage("SpotConnection::createVisionPipeline: Created vision pipeline for stream {}",
                   stream_id);

        // Start the vision pipeline processing thread
        if (!it->second->start()) {
            LogMessage("SOb_LaunchVisionPipeline: Failed to start vision pipeline for stream {}",
                       stream_id);
            return false;
        }

        LogMessage("SOb_LaunchVisionPipeline: Successfully launched vision pipeline for stream {}",
                   stream_id);

        return true;
    } catch (const std::exception& e) {
        LogMessage("SpotConnection::createVisionPipeline: Exception while creating vision pipeline: {}",
                   e.what());
        return false;
    }
}
bool SpotConnection::removeVisionPipeline(int32_t stream_id) {
    auto vision_pipeline_it = vision_pipelines_.find(stream_id);
    if (vision_pipeline_it == vision_pipelines_.end()) {
        LogMessage("SpotConnection::removeVisionPipeline: Vision pipeline for stream {} doesn't exist",
                   stream_id);
        return false;
    }

    try {
        vision_pipelines_.erase(vision_pipeline_it);
        LogMessage("SpotConnection::removeVisionPipeline: Removed vision pipeline for stream {}",
                   stream_id);
        return true;
    } catch (const std::exception& e) {
        LogMessage("SpotConnection::removeVisionPipeline: Exception while removing vision pipeline: {}",
                   e.what());
        return false;
    }
}

VisionPipeline* SpotConnection::getVisionPipeline(int32_t stream_id) {
    auto vision_pipeline_it = vision_pipelines_.find(stream_id);
    if (vision_pipeline_it == vision_pipelines_.end()) {
        LogMessage("SpotConnection::getVisionPipeline: Vision pipeline for stream {} doesn't exist",
                   stream_id);
        return nullptr;
    }
    return vision_pipeline_it->second.get();
}


} // namespace SOb
