/**
 * Multi-threaded Spot Camera Streaming Application using Boston Dynamics C++ SDK
 * Uses C++20 features for modern thread-safe design
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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stop_token>

using namespace std::chrono;

/**
 * Thread-safe image queue for passing data between producer and consumer threads
 */
class ThreadSafeImageQueue {
public:
    struct ImageData {
        cv::Mat image;
        std::string window_name;
        size_t camera_index;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;

        ImageData() = default;
        ImageData(cv::Mat img, std::string name, size_t idx,
                 std::chrono::time_point<std::chrono::high_resolution_clock> ts)
            : image(std::move(img)), window_name(std::move(name)),
              camera_index(idx), timestamp(ts) {}
    };

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<ImageData> queue_;
    std::atomic<bool> shutdown_{false};
    const size_t max_size_;

public:
    explicit ThreadSafeImageQueue(size_t max_size = 30) : max_size_(max_size) {}

    /**
     * Push image data to queue (non-blocking, drops oldest if full)
     */
    void push(ImageData data) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Drop oldest frames if queue is full to prevent unbounded memory growth
        while (queue_.size() >= max_size_) {
            queue_.pop();
        }

        queue_.push(std::move(data));
        condition_.notify_one();
    }

    /**
     * Pop image data from queue (blocking with timeout)
     */
    bool pop(ImageData& data, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (condition_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_.load(); })) {
            if (!queue_.empty()) {
                data = std::move(queue_.front());
                queue_.pop();
                return true;
            }
        }
        return false;
    }

    /**
     * Signal shutdown to waiting threads
     */
    void shutdown() {
        shutdown_.store(true);
        condition_.notify_all();
    }

    /**
     * Get current queue size
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * Check if shutdown was requested
     */
    bool is_shutdown() const {
        return shutdown_.load();
    }
};

class SpotCameraStreamer {
private:
    std::unique_ptr<bosdyn::client::Robot> robot_;
    bosdyn::client::ImageClient* image_client_;  // Raw pointer owned by Robot

    // Threading components
    ThreadSafeImageQueue image_queue_;
    std::atomic<bool> quit_requested_{false};
    std::atomic<double> total_latency_{0.0};
    std::atomic<int> num_samples_{0};

    // Camera configuration
    ::bosdyn::api::GetImageRequest image_sources_;

    std::vector<std::string> window_names_ = {
        "Front Left RGB",
        "Front Left Depth",
        "Front Right RGB",
        "Front Right Depth"
    };

public:
    SpotCameraStreamer() : image_client_(nullptr), image_queue_(30) {
        image_sources_ = createImageRequest();
    }

    /**
     * Convert Spot image response to OpenCV Mat
     */
    cv::Mat convertImageToMat(const bosdyn::api::ImageResponse& image_response) {
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

            std::cout << "Connected to Spot. Starting multi-threaded streaming... (press 'q' to quit)"
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
     * Producer thread: Network communication with robot
     */
    void imageProducerThread(std::stop_token stop_token) {
        if (!image_client_) {
            std::cerr << "Image client not initialized" << std::endl;
            return;
        }

        std::cout << "Producer thread started" << std::endl;

        while (!stop_token.stop_requested() && !quit_requested_.load()) {
            try {
                auto start = std::chrono::high_resolution_clock::now();

                // Request images from all cameras
                bosdyn::client::GetImageResultType response = image_client_->GetImage(image_sources_);
                if (!response.status) {
                    std::cerr << "Failed to get images: " << response.status.message()
                             << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                double latency_ms = duration.count() / 1000.0;

                // Update statistics atomically
                total_latency_.store(total_latency_.load() + latency_ms);
                num_samples_.fetch_add(1);

                std::cout << "Frame Request Latency: " << std::fixed
                         << std::setprecision(1) << latency_ms << " ms, Queue size: "
                         << image_queue_.size() << std::endl;

                // Process and queue images
                const auto& images = response.response.image_responses();
                for (size_t i = 0; i < images.size() && i < window_names_.size(); ++i) {
                    cv::Mat frame = convertImageToMat(images[i]);

                    if (frame.empty()) {
                        std::cout << "Empty frame received for camera " << i << std::endl;
                        continue;
                    }

                    // Create image data and push to queue
                    ThreadSafeImageQueue::ImageData image_data{
                        std::move(frame),
                        window_names_[i],
                        i,
                        std::chrono::high_resolution_clock::now()
                    };

                    image_queue_.push(std::move(image_data));
                }

            } catch (const std::exception& e) {
                std::cerr << "Error in producer thread: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        std::cout << "Producer thread exiting" << std::endl;
    }

    /**
     * Consumer thread: Image processing and display
     */
    void imageConsumerThread(std::stop_token stop_token) {
        std::cout << "Consumer thread started" << std::endl;

        while (!stop_token.stop_requested() && !quit_requested_.load()) {
            ThreadSafeImageQueue::ImageData image_data;

            // Try to get image from queue with timeout
            if (image_queue_.pop(image_data, std::chrono::milliseconds(100))) {
                try {
                    cv::Mat frame = std::move(image_data.image);

                    // Normalize depth images (odd indices)
                    if (image_data.camera_index % 2 == 1) {
                        double max_val;
                        cv::minMaxLoc(frame, nullptr, &max_val);
                        if (max_val > 0) {
                            frame.convertTo(frame, CV_32F, 1.0 / max_val);
                        }
                    }

                    // Display the image
                    cv::imshow(image_data.window_name, frame);

                    // Check for quit key (non-blocking)
                    int key = cv::waitKey(1);
                    if (key == 'q' || key == 'Q') {
                        quit_requested_.store(true);
                        break;
                    }

                } catch (const std::exception& e) {
                    std::cerr << "Error in consumer thread: " << e.what() << std::endl;
                }
            }
        }

        std::cout << "Consumer thread exiting" << std::endl;
    }

    /**
     * Main streaming function that coordinates both threads
     */
    void streamCameras() {
        if (!image_client_) {
            std::cerr << "Image client not initialized" << std::endl;
            return;
        }

        // Create and start threads using C++20 jthread (automatically joinable)
        std::jthread producer_thread([this](std::stop_token stop_token) {
            imageProducerThread(stop_token);
        });

        std::jthread consumer_thread([this](std::stop_token stop_token) {
            imageConsumerThread(stop_token);
        });

        // Monitor for quit condition
        while (!quit_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Signal shutdown to queue and threads
        std::cout << "Shutdown requested, cleaning up..." << std::endl;
        image_queue_.shutdown();

        // Request stop for both threads (C++20 feature)
        producer_thread.request_stop();
        consumer_thread.request_stop();

        // jthread automatically joins on destruction, but we can be explicit
        if (producer_thread.joinable()) {
            producer_thread.join();
        }
        if (consumer_thread.joinable()) {
            consumer_thread.join();
        }

        // Cleanup
        cv::destroyAllWindows();

        // Print statistics
        int samples = num_samples_.load();
        if (samples > 0) {
            double avg_latency = total_latency_.load() / samples;
            std::cout << "Average latency = " << std::fixed << std::setprecision(2)
                     << avg_latency << " ms" << std::endl;
            std::cout << "Average FPS = " << std::fixed << std::setprecision(2)
                     << (1000.0 / avg_latency) << std::endl;
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