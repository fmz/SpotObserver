/**
 * Spot Camera Streaming Application using Boston Dynamics C++ SDK
 */

#include <bosdyn/client/sdk/client_sdk.h>
#include <bosdyn/client/robot/robot.h>
#include <bosdyn/client/image/image_client.h>
#include <bosdyn/api/image.pb.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <iomanip>
#include <thread>

using namespace std::chrono;

class SpotCameraStreamer {
private:
    std::unique_ptr<bosdyn::client::Robot> robot_;
    bosdyn::client::ImageClient* image_client_;  // Raw pointer owned by Robot
    double total_latency_;
    int num_samples_;

public:
    SpotCameraStreamer() : image_client_(nullptr), total_latency_(0.0), num_samples_(0) {}

    /**
     * Convert Spot image response to OpenCV Mat
     */
    cv::Mat convertImageToMat(const bosdyn::api::ImageResponse& image_response) {
        const auto& img = image_response.shot().image();
        
        std::cout << "Image dims = " << img.rows() << "x" << img.cols() << std::endl;
        
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
            
            return image.clone(); // Return a copy to avoid data corruption
        }
    }

    /**
     * Connect to Spot robot
     */
    bool connect(const std::string& robot_ip, const std::string& username, 
                 const std::string& password) {
        try {
            // Create SDK instance
            std::unique_ptr<bosdyn::client::ClientSdk> sdk = bosdyn::client::CreateStandardSDK("SpotCameraStreamer");
            
            // Create robot using ClientSDK
            bosdyn::client::Result<std::unique_ptr<bosdyn::client::Robot>> robot_result = sdk->CreateRobot(robot_ip);
            if (!robot_result.status) {
                std::cerr << "Failed to create robot: " << robot_result.status.message() 
                         << std::endl;
                return false;
            }
            
            robot_ = std::move(robot_result.response);
            
            // Authenticate
            bosdyn::common::Status auth_status = robot_->Authenticate(username, password);
            if (!auth_status) {
                std::cerr << "Authentication failed: " << auth_status.message() << std::endl;
                return false;
            }
            
            // Create image client
            bosdyn::client::Result<bosdyn::client::ImageClient*> image_client_result = robot_->EnsureServiceClient<bosdyn::client::ImageClient>();
            if (!image_client_result.status) {
                std::cerr << "Failed to create image client: " 
                         << image_client_result.status.message() << std::endl;
                return false;
            }
            
            image_client_ = image_client_result.response;
            
            std::cout << "Connected to Spot. Streaming cameras... (press 'q' to quit)" 
                     << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Connection error: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * Create custom image request with specific format and quality
     */
    ::bosdyn::api::GetImageRequest createImageRequest() {
        ::bosdyn::api::GetImageRequest request;

        // RGB cameras with PIXEL_FORMAT_RGB_U8
        std::vector<std::string> rgb_sources = {
            "frontleft_fisheye_image",
            "frontright_fisheye_image"
        };

        // Depth cameras with PIXEL_FORMAT_DEPTH_U16
        std::vector<std::string> depth_sources = {
            "frontleft_depth",
            "frontright_depth"
        };

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

    /**
     * Main streaming loop
     */
    void streamCameras() {
        if (!image_client_) {
            std::cerr << "Image client not initialized" << std::endl;
            return;
        }

        // Create camera image source list
        ::bosdyn::api::GetImageRequest image_requests = createImageRequest();

        std::vector<std::string> window_names = {
            "Front Left RGB", 
            "Front Left Depth", 
            "Front Right RGB", 
            "Front Right Depth"
        };

        while (true) {
            try {
                std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
                
                // Request images from all cameras
                bosdyn::client::GetImageResultType response = image_client_->GetImage(image_requests);
                if (!response.status) {
                    std::cerr << "Failed to get images: " << response.status.message() 
                             << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                
                std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                double latency_ms = duration.count() / 1000.0;
                
                total_latency_ += latency_ms;
                num_samples_++;
                
                std::cout << "Frame Request Latency: " << std::fixed 
                         << std::setprecision(1) << latency_ms << " milliseconds" 
                         << std::endl;
                
                // Process and display images
                const auto& images = response.response.image_responses();
                for (size_t i = 0; i < images.size() && i < window_names.size(); ++i) {
                    cv::Mat frame = convertImageToMat(images[i]);
                    
                    if (frame.empty()) {
                        std::cout << "Empty frame received for camera " << i << std::endl;
                        continue;
                    }
                    
                    // Normalize depth images (odd indices)
                    if (i % 2 == 1) {
                        double max_val;
                        cv::minMaxLoc(frame, nullptr, &max_val);
                        if (max_val > 0) {
                            frame.convertTo(frame, CV_32F, 1.0 / max_val);
                        }
                    }
                    
                    cv::imshow(window_names[i], frame);
                }

                // Check for quit key
                if (cv::waitKey(1) == 'q') {
                    break;
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in streaming loop: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        // Cleanup
        cv::destroyAllWindows();
        
        // Print statistics
        if (num_samples_ > 0) {
            double avg_latency = total_latency_ / num_samples_;
            std::cout << "Average latency = " << avg_latency << " ms" << std::endl;
            std::cout << "Average FPS = " << (1000.0 / avg_latency) << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ROBOT_IP> <username> <password>" << std::endl;
        return 1;
    }
    
    std::string robot_ip = argv[1];
    std::string username = argv[2];
    std::string password = argv[3];
    
    SpotCameraStreamer streamer;
    
    if (!streamer.connect(robot_ip, username, password)) {
        std::cerr << "Failed to connect to robot" << std::endl;
        return 1;
    }
    
    streamer.streamCameras();
    
    return 0;
}