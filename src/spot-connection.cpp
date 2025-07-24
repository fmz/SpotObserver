//
// Created by brown on 7/23/2025.
//
#include "spot-observer.h"
#include "spot-connection.h"
#include "logger.h"
#include "utils.h"
#include "dumper.h"

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

namespace SOb {

static cv::Mat convert_image_to_cv_mat(const bosdyn::api::ImageResponse& image_response) {
    const auto& img = image_response.shot().image();

    if (img.format() == bosdyn::api::Image::FORMAT_JPEG) {
        // Decode JPEG data
        std::vector<uchar> data(img.data().begin(), img.data().end());
        return cv::imdecode(data, cv::IMREAD_COLOR);
    } else {
        // Handle raw pixel data based on pixel format
        cv::Mat image;

        switch (img.pixel_format()) {
        case bosdyn::api::Image::PIXEL_FORMAT_RGB_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC3,
                          const_cast<char*>(img.data().data()));
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC4,
                          const_cast<char*>(img.data().data()));
            cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U8:
            image = cv::Mat(img.rows(), img.cols(), CV_8UC1,
                          const_cast<char*>(img.data().data()));
            break;

        case bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U16:
            image = cv::Mat(img.rows(), img.cols(), CV_16UC1,
                          const_cast<char*>(img.data().data()));
            break;

        default:
            std::cerr << "Unsupported pixel format: " << img.pixel_format() << std::endl;
            return cv::Mat();
        }

        return image;
    }
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
    LogMessage("Destroyed ReaderWriterCBuf");
}

bool ReaderWriterCBuf::initialize(
    size_t n_bytes_per_rgb,
    size_t n_bytes_per_depth,
    size_t n_images_per_response
) {
    n_bytes_per_rgb_ = n_bytes_per_rgb;
    n_bytes_per_depth_ = n_bytes_per_depth;
    n_images_per_response_ = n_images_per_response;

    if (rgb_data_ != nullptr || depth_data_ != nullptr) {
        if (!rgb_data_)   checkCudaError(cudaFree(rgb_data_), "cudaFree for RGB data");
        if (!depth_data_) checkCudaError(cudaFree(depth_data_), "cudaFree for Depth data");

        LogMessage("ReaderWriterCBuf::initialize: Re-initializing, freed existing buffers");
    }

    // Allocate CUDA memory for circular buffer
    size_t total_size_rgb   = max_size_ * n_bytes_per_rgb;
    size_t total_size_depth = max_size_ + n_bytes_per_depth;

    checkCudaError(cudaMalloc(&rgb_data_, total_size_rgb), "cudaMalloc for RGB data");
    checkCudaError(cudaMalloc(&depth_data_, total_size_depth), "cudaMalloc for Depth data");

    write_idx_ = 0;
    read_idx_ = 0;
    size_ = 0;

    return true;
}

static int32_t dump_id = 0; // Global dump ID for debugging

/**
 * Push image data to queue (non-blocking, drops oldest if full)
 */
void ReaderWriterCBuf::push(const google::protobuf::RepeatedPtrField<bosdyn::api::ImageResponse>& responses) {
    if (n_bytes_per_rgb_ == 0 || n_bytes_per_depth_ == 0) {
        throw std::runtime_error("ReaderWriterCBuf::push: n_bytes_per_rgb_ == 0");
    }

    // Compute write pointer
    int write_idx = write_idx_.load(std::memory_order_relaxed);
    uint8_t* rgb_write_ptr    = rgb_data_ + write_idx * n_bytes_per_rgb_;
    uint16_t* depth_write_ptr = depth_data_ + write_idx * n_bytes_per_depth_;

    int32_t n_rgbs_written = 0;
    int32_t n_depths_written = 0;

    for (const auto& response : responses) {
        cv::Mat cv_img = convert_image_to_cv_mat(response);

        const auto& img_response = response.shot().image();

        switch (img_response.pixel_format()) {
        case bosdyn::api::Image::PIXEL_FORMAT_RGB_U8:
        case bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U8:
        {
            size_t image_size = cv_img.cols * cv_img.rows * (img_response.pixel_format() == bosdyn::api::Image::PIXEL_FORMAT_RGBA_U8 ? 4 : 3);
            if (image_size != n_bytes_per_rgb_) {
                LogMessage("Image size mismatch: expected {}, got {}", n_bytes_per_rgb_, image_size);
                throw std::runtime_error("ReaderWriterCBuf::push: Image size mismatch");
            }
            checkCudaError(
                cudaMemcpyAsync(
                    rgb_write_ptr,
                    cv_img.data,
                    image_size,
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpyAsync"
            );

            DumpRGBImageFromCuda(
                rgb_write_ptr,
                cv_img.cols,
                cv_img.rows,
                "rgb",
                dump_id
            );

            rgb_write_ptr += n_bytes_per_rgb_;
            n_rgbs_written++;
        }
        break;

        case bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16:
        case bosdyn::api::Image::PIXEL_FORMAT_GREYSCALE_U16:
        {
            size_t depth_size = cv_img.cols * cv_img.rows * sizeof(uint16_t);
            if (depth_size != n_bytes_per_depth_) {
                LogMessage("Depth size mismatch: expected {}, got {}", n_bytes_per_depth_, depth_size);
                throw std::runtime_error("ReaderWriterCBuf::push: Depth size mismatch");
            }
            checkCudaError(
                cudaMemcpyAsync(
                    depth_write_ptr,
                    cv_img.data,
                    depth_size,
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpyAsync"
            );

            DumpDepthImageFromCuda(
                depth_write_ptr,
                cv_img.cols,
                cv_img.rows,
                "depth",
                dump_id++
            );

            depth_write_ptr += n_bytes_per_depth_;
            n_depths_written++;
        }
        break;

        default:
            LogMessage("Unsupported pixel format: {img.pixel_format()}");
            throw std::runtime_error("ReaderWriterCBuf::push: Got an unsupported pixel format");
        }
    }
    // Update write index
    assert(n_rgbs_written == n_rgbs_written);
    write_idx = (write_idx + n_rgbs_written) % max_size_;
    write_idx_.store(write_idx, std::memory_order_release);

    size_ = size_ + n_rgbs_written;
}

/**
 * Consume image and depth data
 * TODO: Use more of a LIFO approach here, so that we can pop the most recent data first.
 */
std::pair<uint8_t*, uint16_t*> ReaderWriterCBuf::pop(int32_t count) {
    while (size_ == 0) {
        // Wait until there is data to consume
        LogMessage("ReaderWriterCBuf::pop: Buffer is empty, waiting for data...");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    int read_idx = read_idx_.load(std::memory_order_relaxed);

    uint8_t* rgb_data_out = rgb_data_ + read_idx * n_bytes_per_rgb_;
    uint16_t* depth_data_out = depth_data_ + read_idx * n_bytes_per_depth_;

    read_idx += count;

    read_idx_.store(read_idx, std::memory_order_release);

    return std::make_pair(rgb_data_out, depth_data_out);
}

///////////////////////////////////////////////////////////////////////////////

static std::vector<std::string> get_rgb_cam_names_from_bit_mask(uint32_t bitmask) {
    std::vector<std::string> cam_names;
    cam_names.reserve(6);

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
    cam_names.reserve(6);

    if (bitmask & SpotCamera::BACK)       cam_names.emplace_back("back_depth_in_visual_frame");
    if (bitmask & SpotCamera::FRONTLEFT)  cam_names.emplace_back("frontleft_depth_in_visual_frame");
    if (bitmask & SpotCamera::FRONTRIGHT) cam_names.emplace_back("frontright_depth_in_visual_frame");
    if (bitmask & SpotCamera::LEFT)       cam_names.emplace_back("left_depth_in_visual_frame");
    if (bitmask & SpotCamera::RIGHT)      cam_names.emplace_back("right_depth_in_visual_frame");
    if (bitmask & SpotCamera::HAND)       cam_names.emplace_back("hand_depth_in_hand_color_frame");

    return cam_names;
}

///////////////////////////////////////////////////////////////////////////////

SpotConnection::SpotConnection()
    : robot_(nullptr)
    , image_client_(nullptr)
    , rgb_cbuf_(30)
    , connected_(false)
{
    // Create SDK instance
    sdk_ = bosdyn::client::CreateStandardSDK("SpotObserverConnection");
    if (!sdk_) {
        LogMessage("SpotConnection::SpotConnection: Failed to create SDK instance");
        throw std::runtime_error("Failed to create Spot SDK instance");
    }
}

SpotConnection::~SpotConnection() {
    // TODO: figure out how to cleanup image_client_
}

bosdyn::api::GetImageRequest SpotConnection::createImageRequest(
    const std::vector<std::string>& rgb_sources,
    const std::vector<std::string>& depth_sources
) {
    bosdyn::api::GetImageRequest request;

    // Add RGB image requests
    for (const std::string& source : rgb_sources) {
        ::bosdyn::api::ImageRequest* image_request = request.add_image_requests();
        image_request->set_image_source_name(source);
        image_request->set_quality_percent(100.0);
        image_request->set_pixel_format(::bosdyn::api::Image::PIXEL_FORMAT_RGB_U8);
    }

    // Add depth image requests
    for (const std::string& source : depth_sources) {
        ::bosdyn::api::ImageRequest* image_request = request.add_image_requests();
        image_request->set_image_source_name(source);
        image_request->set_quality_percent(100.0);
        image_request->set_pixel_format(::bosdyn::api::Image::PIXEL_FORMAT_DEPTH_U16);
    }

    return request;
}

bool SpotConnection::connect(
    const std::string& robot_ip,
    const std::string& username,
    const std::string& password
) {
    try {
        // Create robot using ClientSDK
        bosdyn::client::Result<std::unique_ptr<bosdyn::client::Robot>> robot_result = sdk_->CreateRobot(robot_ip);
        if (!robot_result.status) {
            LogMessage("SpotConnection::connect: Failed to connect to robot: {}",
                       robot_result.status.message());
            return false;
        }

        robot_ = std::move(robot_result.response);

        // Authenticate
        bosdyn::common::Status auth_status = robot_->Authenticate(username, password);
        if (!auth_status) {
            LogMessage("SpotConnection::connect: Failed to authenticate: {}", auth_status.message());
            return false;
        }

        // Create image client
        bosdyn::client::Result<bosdyn::client::ImageClient*> image_client_result =
            robot_->EnsureServiceClient<bosdyn::client::ImageClient>();

        if (!image_client_result.status) {
            LogMessage("SpotConnection::connect: Failed to create image client: {}",
                       image_client_result.status.message());
            return false;
        }

        image_client_ = image_client_result.response;

        LogMessage("SpotConnection::connect: Connected to Spot robot at {}", robot_ip);

        connected_ = true;

        return true;

    } catch (const std::exception& e) {
        LogMessage("SpotConnection::connect: Exception while connecting to robot {}: {}",
            robot_ip, e.what());
        return false;
    }
}

bool SpotConnection::streamCameras(uint32_t cam_mask) {
    if (!connected_) {
        LogMessage("SpotConnection::streamCameras: Not connected to robot");
        return false;
    }

    if (cam_mask == 0 || cam_mask >= SpotCamera::NUM_CAMERAS) {
        LogMessage("SpotConnection::streamCameras: Invalid camera mask: {:#x}", cam_mask);
        return false;
    }

    int32_t num_cams_requested = 0;
    if (cam_mask != current_cam_mask_) {
        LogMessage("Creating a new Spot image request with mask: {:#x}", cam_mask);
        std::vector<std::string> rgb_sources = get_rgb_cam_names_from_bit_mask(cam_mask);
        std::vector<std::string> depth_sources = get_depth_cam_names_from_bit_mask(cam_mask);

        LogMessage("Creating a request for {} RGB cameras and {} depth cameras",
                   rgb_sources.size(), depth_sources.size());

        for (int32_t i = 0; i < rgb_sources.size(); i++) {
            LogMessage("RGB Camera {}: {}", i, rgb_sources[i]);
        }
        for (int32_t i = 0; i < depth_sources.size(); i++) {
            LogMessage("Depth Camera {}: {}", i, depth_sources[i]);
        }

        num_cams_requested = rgb_sources.size();

        current_request_ = createImageRequest(rgb_sources, depth_sources);
    }

    try {
        constexpr int32_t max_connection_retries = 3;
        // Query the robot for images and fill in image metadata
        // Max 3 retries
        for (int32_t i = 0; i < max_connection_retries; i++) {
            bosdyn::client::GetImageResultType response = image_client_->GetImage(current_request_);
            if (!response.status) {
                LogMessage("SpotConnection::streamCameras: Failed to get images: {}",
                           response.status.message());
                LogMessage("SpotConnection::streamCameras: Retrying... ({}/{})",
                           i + 1, max_connection_retries);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            LogMessage("SpotConnection::streamCameras: Successfully retrieved images");
            const auto& image_responses = response.response.image_responses();
            if (image_responses.empty()) {
                LogMessage("SpotConnection::streamCameras: No images received in response");
                return false;
            }
            // Read image sizes
            size_t rgb_size = image_responses[0].shot().image().rows()*image_responses[0].shot().image().cols();
            // For debugging purposes, ensure that all RGB images have the same size
            for (int32_t j = 1; j < num_cams_requested; j++) {
                const auto& img_response = image_responses[j];
                if (img_response.shot().image().rows()*img_response.shot().image().cols() != rgb_size) {
                    LogMessage("SpotConnection::streamCameras: Inconsistent RGB image sizes");
                    return false;
                }
            }
            // Same thing for depth images
            size_t depth_size = image_responses[num_cams_requested].shot().image().rows() *
                                image_responses[num_cams_requested].shot().image().cols();
            for (int32_t j = num_cams_requested + 1; j < image_responses.size(); j++) {
                const auto& img_response = image_responses[j];
                if (img_response.shot().image().rows() * img_response.shot().image().cols() != depth_size) {
                    LogMessage("SpotConnection::streamCameras: Inconsistent depth image sizes");
                    return false;
                }
            }

            // (Re)initialize circular buffer
            // TODO: Un-hardcode the type sizes
            if (!rgb_cbuf_.initialize(
                    rgb_size * sizeof(uint8_t),
                    depth_size * sizeof(uint16_t),
                    num_cams_requested
                )) {
                LogMessage("SpotConnection::streamCameras: Failed to initialize circular buffer");
                return false;
            }

            break;
        }
    } catch (const std::exception& e) {
        LogMessage("SpotConnection::streamCameras: Exception while getting images: {}", e.what());
        return false;
    }

    current_cam_mask_ = cam_mask;
    return true;
}

} // namespace SOb
