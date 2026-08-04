//
// Created by faisal on 5/12/2025.
//

#include "model.h"
#include "logger.h"
#include "utils.h"
#include "cuda_kernels.cuh"
#include "dumper.h"

#include <filesystem>
#include <cuda_runtime.h>

namespace SOb {

namespace fs = std::filesystem;

TorchModel::TorchModel(const std::string& model_path, const std::string& device_type)
    : m_module()
    , m_device(torch::kCPU)
{
    // Initialization check
    static bool torch_initialized = false;
    if (!torch_initialized) {
        try {
            // Initialize PyTorch properly
            torch::manual_seed(42);
            torch_initialized = true;
            LogMessage("PyTorch initialized successfully");
        } catch (const std::exception& e) {
            LogMessage("PyTorch initialization failed: " + std::string(e.what()));
            throw;
        }
    }

    // Check if model path is valid
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model " + model_path + " does not exist.");
    }

    // Load model
    try {
        m_module = torch::jit::load(model_path);
        LogMessage("Model loaded successfully from: " + model_path);

        // Memory overhead summary (gated by LogLevel::PERF). The on-disk
        // size of the serialized weights is a close proxy for the model's GPU footprint.
        LogPerf("[mem] TorchModel weights ({}): {:.2f} MB",
                model_path,
                fs::file_size(model_path) / (1024.0 * 1024.0));

        m_module.eval();
        LogMessage("Model set to evaluation mode.");

        // Set device
        setDevice(device_type);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + e.msg());
    }
}

TorchModel::~TorchModel() {
    // Destructor
    LogMessage("TorchModel destroyed.");
}

torch::Tensor TorchModel::_run_inference(
    const torch::Tensor& input_tensor,
    const std::optional<torch::Tensor>& depth_tensor
) {
    // Run inference
    auto start_inference = std::chrono::high_resolution_clock::now();
    torch::NoGradGuard no_grad;
    torch::Tensor transposed_input = input_tensor.transpose(2, 3);
    LogMessage("Input tensor shape: " + std::to_string(input_tensor.sizes()[0]) + ", " +
               std::to_string(input_tensor.sizes()[1]) + ", " +
               std::to_string(input_tensor.sizes()[2]) + ", " +
               std::to_string(input_tensor.sizes()[3]));

    std::vector<torch::jit::IValue> inputs{transposed_input};
    //std::vector<torch::jit::IValue> inputs{input_tensor};
    if (depth_tensor.has_value()) {
        LogMessage("Depth tensor shape: " + std::to_string(depth_tensor.value().sizes()[0]) + ", " +
            std::to_string(depth_tensor.value().sizes()[1]) + ", " +
            std::to_string(depth_tensor.value().sizes()[2]) + ", " +
            std::to_string(depth_tensor.value().sizes()[3]));
        // If depth tensor is provided, add it to the inputs
        torch::Tensor transposed_depth = depth_tensor.value().transpose(2, 3);
        inputs.push_back(transposed_depth);
        //inputs.push_back(depth_tensor.value());
    }

    // auto shape = input_tensor.sizes();
    // torch::Tensor output_tensor = torch::ones({1,1,shape[2],shape[3]}, torch::kFloat).to(m_device);
    torch::Tensor output_tensor = m_module.forward(inputs).toTensor();
    LogMessage("Output tensor shape: " + std::to_string(output_tensor.sizes()[0]) + ", " +
               std::to_string(output_tensor.sizes()[1]) + ", ");// +
               // std::to_string(output_tensor.sizes()[2]) + ", " +
               // std::to_string(output_tensor.sizes()[3]));
    size_t output_tensor_rank = output_tensor.dim();
    output_tensor = output_tensor.transpose(output_tensor_rank-2, output_tensor_rank-1).contiguous();
    auto end_inference = std::chrono::high_resolution_clock::now();

    LogMessage("Output image stats: min {}, max {}, mean {}, stddev {}",
        output_tensor.min().item<float>(),
        output_tensor.max().item<float>(),
        output_tensor.mean().item<float>(),
        output_tensor.std().item<float>()
    );
    LogMessage("Inference time: " + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count()) + " ms");
    
    return output_tensor;
}

bool TorchModel::runInference(
    const float* input_data,
    const float* depth_data,
    float*       output_data,
    TensorShape  input_shape,
    TensorShape  depth_shape,
    TensorShape  output_shape
) {
    try {
        static int32_t dump_id = 300;
        DumpRGBImageFromCudaCHW(
            input_data,
            input_shape.W,
            input_shape.H,
            "rgb",
            dump_id
        );
        // Create input tensor from a device pointer
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(input_data), 
            {long(input_shape.N), long(input_shape.C), long(input_shape.H), long(input_shape.W)}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(m_device)
        );

        torch::Tensor depth_tensor;
        if (depth_data) {
            depth_tensor = torch::from_blob(
                const_cast<float*>(depth_data),
                {long(depth_shape.N), long(depth_shape.C), long(depth_shape.H), long(depth_shape.W)},
                torch::TensorOptions().dtype(torch::kFloat32).device(m_device)
            );
        }

        // Create an output tensor from a device pointer
        torch::Tensor output_tensor = torch::from_blob(
            output_data, 
            {long(output_shape.N), long(output_shape.C), long(output_shape.H), long(output_shape.W)}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(m_device)
        );

        // Run inference
        LogMessage("About to run inference...");

        torch::Tensor model_output = _run_inference(input_tensor, depth_data ? std::optional<torch::Tensor>(depth_tensor) : std::nullopt);

        // Copy the output tensor to the output_tensor (GPU to GPU). Run some sanity checks first.
        // TODO: Figure out how to avoid copies
        size_t output_size = output_shape.N * output_shape.C * output_shape.H * output_shape.W;
/*
        if (output_tensor.numel() != output_size) {
            LogMessage("Output tensor size mismatch: expected " + std::to_string(output_size) + ", got " + std::to_string(output_tensor.numel()));
            throw std::runtime_error("Output size mismatch.");
        }
        if (model_output.numel() != output_tensor.numel()) {
            LogMessage("Model output size mismatch: expected " + std::to_string(model_output.numel()) + ", got " + std::to_string(output_tensor.numel()));
            throw std::runtime_error("Model output size mismatch.");
        }
        if (model_output.device() != output_tensor.device()) {
            LogMessage("Model output device mismatch: expected " + model_output.device().str() + ", got " + output_tensor.device().str());
            throw std::runtime_error("Model output device mismatch.");
        }
        if (model_output.dtype() != output_tensor.dtype()) {
            LogMessage("Model output dtype mismatch!");
            throw std::runtime_error("Model output dtype mismatch.");
        }
        */
        if (model_output.is_contiguous() && output_tensor.is_contiguous()) {
            auto output_shape = output_tensor.sizes();
            auto model_output_shape = model_output.sizes();
            //LogMessage("output_tensor = {}, {}, {}, {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
            //LogMessage("model_output = {}, {}, {}, {}", model_output_shape[0], model_output_shape[1], model_output_shape[2], model_output_shape[3]);
            // LogMessage("output_tensor device = {}", output_tensor.device().str());
            // LogMessage("model_output device = {}", model_output.device().str());
            // LogMessage("output_tensor pointer = {}", (void*)output_tensor.data_ptr<float>());
            // LogMessage("model_output pointer = {}",(void*) model_output.data_ptr<float>());
            LogMessage("About to copy model output to output tensor...");
            output_tensor.copy_(model_output);
            LogMessage("Done with copying model output to output tensor...");
        } else {
            LogMessage("Model output tensor is not contiguous");
            throw std::runtime_error("Model output tensor is not contiguous.");
        }
        DumpDepthImageFromCuda(
            depth_data,
            depth_shape.W,
            depth_shape.H,
            "preprocessed-depth",
            dump_id
        );

        DumpDepthImageFromCuda(
            output_data,
            output_shape.W,
            output_shape.H,
            "output",
            dump_id
        );
        dump_id++;
        //LogMessage("Inference completed successfully.");
    } catch (const std::exception& e) {
        LogMessage("Error during inference: " + std::string(e.what()));
        return false;        
    }

    return true;
}

bool TorchModel::runInference(
    const uint8_t* rgb_data,
    const float*   depth_data,
    float*         output_data,
    TensorShape    input_shape,
    TensorShape    depth_shape,
    TensorShape    output_shape
) {
    try {
        // Create input tensor from a device pointer
        torch::Tensor rgb_tensor = torch::from_blob(
            const_cast<uint8_t*>(rgb_data),
            {long(input_shape.N), long(input_shape.C), long(input_shape.H), long(input_shape.W)},
            torch::TensorOptions().dtype(torch::kUInt8).device(m_device)
        );

        torch::Tensor depth_tensor;
        if (depth_data) {
            depth_tensor = torch::from_blob(
                const_cast<float*>(depth_data),
                {long(depth_shape.N), long(depth_shape.C), long(depth_shape.H), long(depth_shape.W)},
                torch::TensorOptions().dtype(torch::kFloat32).device(m_device)
            );
        }

        // Create an output tensor from a device pointer
        torch::Tensor output_tensor = torch::from_blob(
            output_data,
            {long(output_shape.N), long(output_shape.C), long(output_shape.H), long(output_shape.W)},
            torch::TensorOptions().dtype(torch::kFloat32).device(m_device)
        );

        // Convert RGBA tensor RGB tensor
        if (input_shape.C == 4) {
            rgb_tensor = rgb_tensor.slice(1, 0, 3); // Remove alpha channel
        } else if (input_shape.C != 3) {
            LogMessage("Input tensor must have 3 or 4 channels (RGB or RGBA).");
            throw std::runtime_error("Invalid input tensor channels.");
        }

        // Convert to float32
        rgb_tensor = rgb_tensor.to(torch::kFloat32).div_(255.0);

        // Run inference
        LogMessage("About to run inference...");

        torch::Tensor model_output = _run_inference(rgb_tensor, depth_data ? std::optional<torch::Tensor>(depth_tensor) : std::nullopt);

        // Copy the output tensor to the output_tensor (GPU to GPU). Run some sanity checks first.
        // TODO: Figure out how to avoid copies
        size_t output_size = output_shape.N * output_shape.C * output_shape.H * output_shape.W;
/*
        if (output_tensor.numel() != output_size) {
            LogMessage("Output tensor size mismatch: expected " + std::to_string(output_size) + ", got " + std::to_string(output_tensor.numel()));
            throw std::runtime_error("Output size mismatch.");
        }
        if (model_output.numel() != output_tensor.numel()) {
            LogMessage("Model output size mismatch: expected " + std::to_string(model_output.numel()) + ", got " + std::to_string(output_tensor.numel()));
            throw std::runtime_error("Model output size mismatch.");
        }
        if (model_output.device() != output_tensor.device()) {
            LogMessage("Model output device mismatch: expected " + model_output.device().str() + ", got " + output_tensor.device().str());
            throw std::runtime_error("Model output device mismatch.");
        }
        if (model_output.dtype() != output_tensor.dtype()) {
            LogMessage("Model output dtype mismatch!");
            throw std::runtime_error("Model output dtype mismatch.");
        }
        */
        if (model_output.is_contiguous() && output_tensor.is_contiguous()) {
            auto output_shape = output_tensor.sizes();
            auto model_output_shape = model_output.sizes();
            //LogMessage("output_tensor = {}, {}, {}, {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
            //LogMessage("model_output = {}, {}, {}, {}", model_output_shape[0], model_output_shape[1], model_output_shape[2], model_output_shape[3]);
            // LogMessage("output_tensor device = {}", output_tensor.device().str());
            // LogMessage("model_output device = {}", model_output.device().str());
            // LogMessage("output_tensor pointer = {}", (void*)output_tensor.data_ptr<float>());
            // LogMessage("model_output pointer = {}",(void*) model_output.data_ptr<float>());
            LogMessage("About to copy model output to output tensor...");
            output_tensor.copy_(model_output);
            LogMessage("Done with copying model output to output tensor...");
        } else {
            LogMessage("Model output tensor is not contiguous");
            throw std::runtime_error("Model output tensor is not contiguous.");
        }

        //LogMessage("Inference completed successfully.");
    } catch (const std::exception& e) {
        LogMessage("Error during inference: " + std::string(e.what()));
        return false;
    }

    return true;
}
    
void TorchModel::setDevice(const std::string& device_type) {
    std::string device_type_lower = device_type;
    std::transform(device_type_lower.begin(), device_type_lower.end(), device_type_lower.begin(), ::tolower);
    
    try {
        if (device_type_lower == "cpu") {
            m_device = torch::kCPU;
        } else if (device_type_lower == "cuda" || device_type_lower == "gpu") {
            // Check CUDA availability
            if (!torch::cuda::is_available()) {
                LogMessage("CUDA not available, falling back to CPU");
                m_device = torch::kCPU;
                return;
            }
            m_device = torch::kCUDA;
        } else if (device_type_lower == "mps") {
            m_device = torch::kMPS;
            throw std::runtime_error("MPS is not supported yet.");
        } else {
            throw std::runtime_error("Unsupported device type: " + device_type);
        }

        LogMessage("Device set to: " + device_type);
    } catch (const std::exception& e) {
        LogMessage("Device setting failed: " + std::string(e.what()));
        LogMessage("Falling back to CPU");
        m_device = torch::kCPU;
    }
    // Move the model to the device
    m_module.to(m_device);

    // // Warm up the model with a dummy inference to avoid first-call overhead
    // if (m_device.is_cuda()) {
    //     try {
    //         torch::Tensor dummy_input = torch::randn({1, 3, 480, 640}, torch::TensorOptions().device(m_device));
    //         auto start_inference = std::chrono::high_resolution_clock::now();
    //         torch::NoGradGuard no_grad;
    //         m_module.forward({dummy_input});
    //         auto end_inference = std::chrono::high_resolution_clock::now();
    //         LogMessage("Model warmed up successfully. First inference time: " + std::to_string(
    //             std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count()) + " ms");
    //     } catch (const std::exception& e) {
    //         LogMessage("Model warmup failed: " + std::string(e.what()));
    //     }

    //     for (int i = 0; i < 100; ++i) {
    //         try {
    //             torch::Tensor dummy_input = torch::randn({1, 3, 480, 640}, torch::TensorOptions().device(m_device));
    //             auto start_inference = std::chrono::high_resolution_clock::now();
    //             torch::NoGradGuard no_grad;
    //             m_module.forward({dummy_input});
    //             auto end_inference = std::chrono::high_resolution_clock::now();
    //             LogMessage("Model warmed up successfully. Subsequent inference time: " + std::to_string(
    //                 std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count()) + " ms");
    //         } catch (const std::exception& e) {
    //             LogMessage("Model warmup failed: " + std::string(e.what()));
    //         }
    //     }
    // }
}

std::string TorchModel::getDevice() const {
    switch (m_device.type()) {
        case torch::kCPU:
            return "cpu";
        case torch::kCUDA:
            return "cuda";
        case torch::kMPS:
            return "mps";
        default:
            return "unknown";
    }
}

///// ONNXModel Implementation /////

ONNXModel::ONNXModel(const std::string& model_path, const std::string& device_type)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel")
    , m_sess_options()
    , m_memory_info(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
    , m_use_cuda(false)
{
    // Check if model path is valid
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model " + model_path + " does not exist.");
    }

    try {
        // Set device
        setDevice(device_type);

        // Load model
        std::wstring wide_model_path = std::wstring(model_path.begin(), model_path.end());
        m_session = std::make_unique<Ort::Session>(m_env, wide_model_path.c_str(), m_sess_options);

        LogMessage("ONNX model loaded successfully from: " + model_path);

        // Memory overhead summary (gated by LogLevel::PERF). The on-disk
        // size of the serialized weights is a close proxy for the model's GPU footprint.
        LogPerf("[mem] ONNXModel weights ({}): {:.2f} MB",
                model_path,
                fs::file_size(model_path) / (1024.0 * 1024.0));

        m_allocator = std::make_unique<Ort::Allocator>(*m_session, m_memory_info);

        // Print some allocator info
        auto info = m_allocator->GetInfo();
        std::cout << "Allocator info: " << info << std::endl;

        // I/O info
        size_t num_input_nodes = m_session->GetInputCount();
        // if (num_input_nodes != 1) {
        //     throw std::runtime_error("Model should have exactly 1 input, got " + std::to_string(num_input_nodes));
        // }
        size_t num_output_nodes = m_session->GetOutputCount();
        if (num_output_nodes != 1) {
            throw std::runtime_error("Model should have exactly 1 output, got " + std::to_string(num_output_nodes));
        }

        Ort::AllocatorWithDefaultOptions default_allocator;
        m_num_model_inputs = m_session->GetInputCount();
        size_t num_outputs = m_session->GetOutputCount();
        if (num_outputs != 1) {
            throw std::runtime_error("Model should have exactly 1 output, got " + std::to_string(num_outputs));
        }

        // Store input information
        for (size_t i = 0; i < m_num_model_inputs; ++i) {
            auto name = m_session->GetInputNameAllocated(i, default_allocator);
            m_input_names.push_back(std::string(name.get()));

            auto type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            m_input_shapes.push_back(tensor_info.GetShape());
            LogMessage("Model input #{} shape: [{}, {}, {}, {}]", i,
                m_input_shapes.back()[0], m_input_shapes.back()[1], m_input_shapes.back()[2], m_input_shapes.back()[3]
            );
        }

        // Store output information
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = m_session->GetOutputNameAllocated(i, default_allocator);
            m_output_names.push_back(std::string(name.get()));

            auto type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            m_output_shapes.push_back(tensor_info.GetShape());

            if (m_output_shapes.back().size() == 4) {
                LogMessage("Model output #{} shape: [{}, {}, {}, {}]", i,
                    m_output_shapes.back()[0], m_output_shapes.back()[1], m_output_shapes.back()[2], m_output_shapes.back()[3]
                );
            } else if (m_output_shapes.back().size() == 2) {
                LogMessage("Model output #{} shape: [{}, {}]", i, m_output_shapes.back()[0], m_output_shapes.back()[1]);
            } else {
                throw std::runtime_error("Unsupported output shape size: " + std::to_string(m_output_shapes.back().size()));
            }
        }

        LogMessage("Model has {} inputs and {} outputs", m_num_model_inputs, num_outputs);

        // Warm up the model if using CUDA
        // Warm up model if using CUDA
        if (m_use_cuda) {
            _warmupModel();
        }

    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Error loading ONNX model: " + std::string(e.what()));
    }
}

ONNXModel::~ONNXModel() {
    // if (m_input_buffer) {
    //     m_allocator->Free(m_input_buffer);
    // }
    // if (m_output_buffer) {
    //     m_allocator->Free(m_output_buffer);
    // }

    LogMessage("ONNXModel destroyed.");
}

// Ort::Value ONNXModel::_run_inference() {
//     auto start_inference = std::chrono::high_resolution_clock::now();
//
//     // Note: ONNX Runtime expects input in NCHW format by default
//     auto output_tensors = m_session->Run(Ort::RunOptions{nullptr},
//         m_input_names.data(),
//         &input_tensor,
//         1,
//         m_output_names.data(),
//         1);
//
//     auto end_inference = std::chrono::high_resolution_clock::now();
//     LogMessage("ONNX inference time: {} ms",
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count());
//
//     return std::move(output_tensors[0]);
// }

void ONNXModel::_warmupModel() {
    if (m_is_warmed_up || !m_use_cuda || m_input_shapes.empty()) {
        LogMessage("ONNX model is already warmed up or not using CUDA. Skipping warmup.");
        return;
    }

    try {
        LogMessage("Warming up ONNX model...");
        // Use first input/output for warmup

        // Handle inputs
        std::vector<void*> d_inputs(m_num_model_inputs, nullptr);
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(m_num_model_inputs);

        for (int32_t i = 0; i < m_num_model_inputs; ++i) {
            const auto& input_shape = m_input_shapes[i];

            size_t input_size = 1;
            for (auto dim : input_shape) {
                input_size *= static_cast<size_t>(dim > 0 ? dim : 1); // Handle dynamic dimensions
            }

            // Allocate temporary GPU memory
            checkCudaError(cudaMalloc(&d_inputs[i], input_size * sizeof(float)), "cudaMalloc input");

            // Create dummy data and copy to GPU
            std::vector<float> dummy_input(input_size, 0.5f);
            checkCudaError(cudaMemcpy(d_inputs[i], dummy_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");

            // Create Ort::Value for input tensor
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                static_cast<float*>(d_inputs[i]),
                input_size,
                input_shape.data(),
                input_shape.size()
            ));
        }

        // Setup dummy output
        const auto& output_shape = m_output_shapes[0];

        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= static_cast<size_t>(dim > 0 ? dim : 1);
        }
        void* d_output = nullptr;

        checkCudaError(cudaMalloc(&d_output, output_size * sizeof(float)), "cudaMalloc output");

        auto start = std::chrono::high_resolution_clock::now();

        // Create output tensors
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            static_cast<float*>(d_output),
            output_size,
            output_shape.data(),
            output_shape.size()
        );

        // Create IoBinding for reuse
        Ort::IoBinding binding(*m_session);
        for (int32_t i = 0; i < m_num_model_inputs; ++i) {
            binding.BindInput(m_input_names[i].c_str(), input_tensors[i]);
        }
        binding.BindOutput(m_output_names[0].c_str(), output_tensor);

        // Run warmup inference
        m_session->Run(Ort::RunOptions{nullptr}, binding);

        // Synchronize to ensure completion
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        auto end = std::chrono::high_resolution_clock::now();
        LogMessage("ONNX model warmed up successfully. Time: {} ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        // Cleanup
        for (int32_t i = 0; i < m_num_model_inputs; ++i) {
            checkCudaError(cudaFree(d_inputs[i]), "cudaFree input");
        }
        checkCudaError(cudaFree(d_output), "cudaFree output");

        m_is_warmed_up = true;

    } catch (const std::exception& e) {
        LogMessage("ONNX model warmup failed: {}", e.what());
        // Continue without warmup
    }
}

bool ONNXModel::runInference(
    const float* input_data,
    const float* depth_data,
    float* output_data,
    TensorShape input_shape,
    TensorShape depth_shape,
    TensorShape output_shape
) {
    // std::string img_path = "C:/Users/brown/Documents/fmz/test.jpg";
    // int32_t out_w, out_h;
    // static float* d_image = nullptr;
    // static float* d_depth = nullptr;
    // if (!d_image) {
    //     size_t image_size = input_shape.N * input_shape.C * input_shape.H * input_shape.W * sizeof(float);
    //     cudaMalloc(&d_image, image_size);
    //     cudaMalloc(&d_depth, image_size);
    // }
    // loadImageToCudaFloatRGB(img_path, out_w, out_h, d_image);
    //
    static int32_t dump_id = 500;
    DumpRGBImageFromCudaCHW(
        input_data,
        input_shape.W,
        input_shape.H,
        "rgb",
        dump_id
    );

    auto time_start = std::chrono::high_resolution_clock::now();

    try {
        // Create input tensor
        std::vector<int64_t> input_tensor_shape = {
            static_cast<int64_t>(input_shape.N),
            static_cast<int64_t>(input_shape.C),
            static_cast<int64_t>(input_shape.H),
            static_cast<int64_t>(input_shape.W)
        };

        std::vector<int64_t> depth_tensor_shape = {
            static_cast<int64_t>(depth_shape.N),
            static_cast<int64_t>(depth_shape.C),
            static_cast<int64_t>(depth_shape.H),
            static_cast<int64_t>(depth_shape.W)
        };

        std::vector<int64_t> output_tensor_shape = {
            static_cast<int64_t>(output_shape.N),
            static_cast<int64_t>(output_shape.C),
            static_cast<int64_t>(output_shape.H),
            static_cast<int64_t>(output_shape.W)
        };

        size_t input_size = input_shape.N * input_shape.C * input_shape.H * input_shape.W;
        size_t depth_size = depth_shape.N * depth_shape.C * depth_shape.H * depth_shape.W;
        size_t output_size = output_shape.N * output_shape.C * output_shape.H * output_shape.W;

        LogMessage("About to run ONNX inference...");

        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            const_cast<float*>(input_data),
            input_size,
            input_tensor_shape.data(),
            input_tensor_shape.size()
        );

        Ort::Value depth_tensor;
        if (depth_data) {
            depth_tensor = Ort::Value::CreateTensor<float>(
                m_memory_info,
                const_cast<float*>(depth_data),
                depth_size,
                depth_tensor_shape.data(),
                depth_tensor_shape.size()
            );
        }

        // Create output tensor
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info,
            const_cast<float*>(output_data),
            output_size,
            output_tensor_shape.data(),
            output_tensor_shape.size()
        );

        // Create IoBinding
        Ort::IoBinding binding(*m_session);
        binding.BindInput(m_input_names[0].c_str(), input_tensor);
        if (depth_data) {
            binding.BindInput(m_input_names[1].c_str(), depth_tensor);
        }
        binding.BindOutput(m_output_names[0].c_str(), output_tensor);

        // Run inference
        m_session->Run(Ort::RunOptions{nullptr}, binding);

        // Synchronize if using CUDA
        if (m_use_cuda) {
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }


        DumpDepthImageFromCuda(
            depth_data,
            depth_shape.W,
            depth_shape.H,
            "preprocessed-depth",
            dump_id
        );

        DumpDepthImageFromCuda(
            output_data,
            output_shape.W,
            output_shape.H,
            "output",
            dump_id
        );
        dump_id++;
        LogMessage("ONNX inference completed successfully.");
    } catch (const std::exception& e) {
        LogMessage("Error during ONNX inference: {}", e.what());
        return false;
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    LogMessage("Total ONNX inference time: {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count());

    return true;
}

bool ONNXModel::runInference(
    const uint8_t* input_data,
    const float*   depth_data,
    float*         output_data,
    TensorShape    input_shape,
    TensorShape    depth_shape,
    TensorShape    output_shape
) {
    // try {
    //     // Convert tensor to RGB float
    //     size_t total_output_size = input_shape.N * input_shape.H * input_shape.W * 3 * sizeof(float);
    //     if (!d_tmp_rgb || tmp_rgb_allocated_size < total_output_size) {
    //         if (d_tmp_rgb) {
    //             cudaFree(d_tmp_rgb);
    //         }
    //         checkCudaError(cudaMalloc(&d_tmp_rgb, total_output_size), "Failed to allocate memory for RGB conversion");
    //         tmp_rgb_allocated_size = total_output_size;
    //     }
    //     convert_uint8_img_to_float_img(
    //         input_data,
    //         d_tmp_rgb,
    //         input_shape.N,
    //         input_shape.H,
    //         input_shape.W,
    //         input_shape.C
    //     );
    //
    //     TensorShape input_shape_float = {
    //         input_shape.N,
    //         3, // RGB channels
    //         input_shape.H,
    //         input_shape.W
    //     };
    //
    //     // Hack:
    //     TensorShape hackinputshape = {
    //         1,
    //         3, // RGB channels
    //         input_shape.H,
    //         input_shape.W
    //     };
    //     TensorShape hackdepthshape = {
    //         1,
    //         depth_shape.C,
    //         depth_shape.H,
    //         depth_shape.W
    //     };
    //     TensorShape hackoutputshape = {
    //         1,
    //         output_shape.C,
    //         output_shape.H,
    //         output_shape.W
    //     };
    //
    //     // return runInference(
    //     //     reinterpret_cast<const float*>(d_tmp_rgb),
    //     //     depth_data,
    //     //     output_data,
    //     //     input_shape_float,
    //     //     depth_shape,
    //     //     output_shape
    //     // );
    //     return runInference(
    //         reinterpret_cast<const float*>(d_tmp_rgb),
    //         depth_data,
    //         output_data,
    //         hackinputshape,
    //         hackdepthshape,
    //         hackoutputshape
    //     );
    // } catch (const std::exception& e) {
    //     LogMessage("ONNXModel::runInference: Error during uint8 to float conversion: {}", e.what());
    //     return false;
    // }

    LogMessage("ONNXModel::runInference with uint8 input is not implemented yet!");
    throw std::runtime_error("Not implemented yet!");

    return true;
}

// look at this: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#samples (See the full cuda example. looks nothing like we got here)
void ONNXModel::setDevice(const std::string& device_type) {
    std::string device_type_lower = device_type;
    std::transform(device_type_lower.begin(), device_type_lower.end(), device_type_lower.begin(), ::tolower);

    try {
        if (device_type_lower == "cpu") {
            m_use_cuda = false;
            // CPU provider is added by default
            LogMessage("ONNX device set to: CPU");
            throw std::runtime_error("ONNX CPU provider is not supported yet. Use CUDA or GPU instead.");
        } else if (device_type_lower == "cuda" || device_type_lower == "gpu") {
            // Check CUDA availability
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0) {
                LogMessage("CUDA not available, falling back to CPU");
                m_use_cuda = false;
                return;
            }

            // Add CUDA provider
            const auto& ort_api = Ort::GetApi();

            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            ort_api.CreateCUDAProviderOptions(&cuda_options);
            std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(ort_api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, ort_api.ReleaseCUDAProviderOptions);

            std::vector<const char*> keys{"enable_cuda_graph"};
            std::vector<const char*> values{"1"}; // Enable CUDA graphs

            ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                static_cast<OrtSessionOptions*>(m_sess_options),
                rel_cuda_options.get()
            );

            m_memory_info = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

            m_use_cuda = true;
            LogMessage("ONNX device set to: CUDA");
        } else {
            throw std::runtime_error("Unsupported device type for ONNX: " + device_type);
        }
    } catch (const std::exception& e) {
        LogMessage("ONNX device setting failed: {}", e.what());
        LogMessage("Falling back to CPU");
        m_use_cuda = false;
    }
}

std::string ONNXModel::getDevice() const {
    return m_use_cuda ? "cuda" : "cpu";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Bytes per element for the types a KV cache is plausibly carried in.
static size_t onnxElementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return 1;
        default:                                     return 0; // unknown; reported as 0 MB
    }
}

StreamingONNXModel::StreamingONNXModel(const std::string& model_path, const std::string& device_type)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "StreamingONNXModel")
    , m_sess_options()
    , m_memory_info(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault))
{
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model " + model_path + " does not exist.");
    }

    try {
        _setDevice(device_type);
        if (!m_use_cuda) {
            throw std::runtime_error("StreamingONNXModel requires CUDA; the KV cache must stay device-resident.");
        }

        // Weights live in a sibling .onnx.data file, so this reads several GB.
        auto load_start = std::chrono::high_resolution_clock::now();
        std::wstring wide_model_path = std::wstring(model_path.begin(), model_path.end());
        m_session = std::make_unique<Ort::Session>(m_env, wide_model_path.c_str(), m_sess_options);
        auto load_end = std::chrono::high_resolution_clock::now();

        LogMessage("Streaming ONNX model loaded from {} in {} ms", model_path,
            std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count());

        _buildCacheNames();
        _readGeometry();
        _ensureScratch(m_cur_batch);

        m_binding = std::make_unique<Ort::IoBinding>(*m_session);
        resetState();

        const double cache_mb_per_frame =
            static_cast<double>(kNumCacheTensors) * m_num_heads * m_num_tokens * m_head_dim
            * onnxElementSize(m_cache_type) / (1024.0 * 1024.0);
        // The .onnx file is just the graph; the weights live in the external-data
        // sibling, so include it or the log understates the footprint ~400x.
        uintmax_t weight_bytes = fs::file_size(model_path);
        const fs::path external_data = fs::path(model_path).concat(".data");
        if (fs::exists(external_data)) {
            weight_bytes += fs::file_size(external_data);
        }
        LogPerf("[mem] StreamingONNXModel: graph + weights {:.2f} MB on disk, KV cache {:.2f} MB per retained frame "
                "(x2 live during Run)",
                weight_bytes / (1024.0 * 1024.0), cache_mb_per_frame);

    } catch (const Ort::Exception& e) {
        // The destructor does not run for a partially constructed object, so any
        // scratch already allocated has to be released here.
        _freeScratch();
        throw std::runtime_error("Error loading streaming ONNX model: " + std::string(e.what()));
    } catch (...) {
        _freeScratch();
        throw;
    }
}

StreamingONNXModel::~StreamingONNXModel() {
    // m_cache is declared after m_session, so the cache tensors are released
    // before the session and env they were allocated from.
    _freeScratch();
    LogMessage("StreamingONNXModel destroyed.");
}

void StreamingONNXModel::_freeScratch() {
    if (m_d_rgb)   { cudaFree(m_d_rgb);   m_d_rgb = nullptr; }
    if (m_d_depth) { cudaFree(m_d_depth); m_d_depth = nullptr; }
    if (m_d_empty) { cudaFree(m_d_empty); m_d_empty = nullptr; }
}

void StreamingONNXModel::_setDevice(const std::string& device_type) {
    std::string device_type_lower = device_type;
    std::transform(device_type_lower.begin(), device_type_lower.end(), device_type_lower.begin(), ::tolower);

    if (device_type_lower != "cuda" && device_type_lower != "gpu") {
        LogMessage("StreamingONNXModel: unsupported device '{}'", device_type);
        m_use_cuda = false;
        return;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        LogMessage("StreamingONNXModel: CUDA not available");
        m_use_cuda = false;
        return;
    }

    const auto& ort_api = Ort::GetApi();

    // Every OrtStatus* must be checked and released: silently ignoring them would
    // leave m_use_cuda true while the session actually ran on CPU, and device
    // pointers would then be bound with CUDA MemoryInfo.
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    if (OrtStatus* status = ort_api.CreateCUDAProviderOptions(&cuda_options)) {
        LogMessage("StreamingONNXModel: CreateCUDAProviderOptions failed: {}",
            ort_api.GetErrorMessage(status));
        ort_api.ReleaseStatus(status);
        m_use_cuda = false;
        return;
    }
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(ort_api.ReleaseCUDAProviderOptions)>
        rel_cuda_options(cuda_options, ort_api.ReleaseCUDAProviderOptions);

    // Arena strategy stays at the default (kNextPowerOfTwo) deliberately. The
    // cache tensors grow to a NEW size every frame until the window clamps, and
    // exact-size allocation (kSameAsRequested) can never reuse a freed chunk for
    // the next, larger request -- the arena extends every frame and VRAM climbs
    // ~quadratically until WDDM starts evicting (measured: 375 ms -> 14.8 s per
    // frame at the cliff). Power-of-two rounding puts consecutive sizes in the
    // same bucket, so freed generations actually get reused during growth.

    // Deliberately no enable_cuda_graph here: graph capture requires static
    // shapes, and the cache sequence axis grows for the first frames before the
    // retention window clamps it.
    if (OrtStatus* status = ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2(
            static_cast<OrtSessionOptions*>(m_sess_options),
            rel_cuda_options.get())) {
        LogMessage("StreamingONNXModel: registering the CUDA EP failed: {}",
            ort_api.GetErrorMessage(status));
        ort_api.ReleaseStatus(status);
        m_use_cuda = false;
        return;
    }

    m_memory_info = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    m_use_cuda = true;
    LogMessage("StreamingONNXModel device set to: CUDA");
}

void StreamingONNXModel::_buildCacheNames() {
    Ort::AllocatorWithDefaultOptions alloc;

    const size_t num_inputs  = m_session->GetInputCount();
    const size_t num_outputs = m_session->GetOutputCount();
    const size_t expected    = kNumFixedInputs + kNumCacheTensors;

    if (num_inputs != expected || num_outputs != expected) {
        throw std::runtime_error("Expected " + std::to_string(expected) + " inputs and outputs, got " +
            std::to_string(num_inputs) + " and " + std::to_string(num_outputs));
    }

    auto input_name  = [&](size_t i) { return std::string(m_session->GetInputNameAllocated(i, alloc).get()); };
    auto output_name = [&](size_t i) { return std::string(m_session->GetOutputNameAllocated(i, alloc).get()); };

    if (input_name(0) != "rgb" || input_name(1) != "sparse_depth") {
        throw std::runtime_error("Expected inputs [rgb, sparse_depth], got [" +
            input_name(0) + ", " + input_name(1) + "]");
    }
    if (output_name(0) != "depth" || output_name(1) != "depth_conf") {
        throw std::runtime_error("Expected outputs [depth, depth_conf], got [" +
            output_name(0) + ", " + output_name(1) + "]");
    }

    // Take the pairing from the graph's own ordering rather than reconstructing
    // names, then assert the past_/new_ suffixes line up. A silent misalignment
    // here would route one layer's K into another's V: no error, just wrong depth.
    m_past_names.reserve(kNumCacheTensors);
    m_new_names.reserve(kNumCacheTensors);
    for (int32_t i = 0; i < kNumCacheTensors; ++i) {
        std::string in_name  = input_name(kNumFixedInputs + i);
        std::string out_name = output_name(kNumFixedOutputs + i);

        if (in_name.rfind("past_", 0) != 0 || out_name.rfind("new_", 0) != 0) {
            throw std::runtime_error("Cache tensor " + std::to_string(i) + " naming unexpected: " +
                in_name + " / " + out_name);
        }
        if (in_name.substr(5) != out_name.substr(4)) {
            throw std::runtime_error("Cache tensor " + std::to_string(i) + " misaligned: " +
                in_name + " does not pair with " + out_name);
        }

        m_past_names.push_back(std::move(in_name));
        m_new_names.push_back(std::move(out_name));
    }

    // Built only after the name vectors are final, so the pointers stay valid.
    m_past_cstr.reserve(kNumCacheTensors);
    m_new_cstr.reserve(kNumCacheTensors);
    for (int32_t i = 0; i < kNumCacheTensors; ++i) {
        m_past_cstr.push_back(m_past_names[i].c_str());
        m_new_cstr.push_back(m_new_names[i].c_str());
    }

    LogMessage("StreamingONNXModel: {} cache tensors, {} <-> {} ... {} <-> {}",
        kNumCacheTensors, m_past_names.front(), m_new_names.front(),
        m_past_names.back(), m_new_names.back());
}

void StreamingONNXModel::_readGeometry() {
    // TypeInfo owns the shape/type view, so each must outlive the view taken from it.
    Ort::TypeInfo rgb_type_info       = m_session->GetInputTypeInfo(0);
    Ort::TypeInfo depth_in_type_info  = m_session->GetInputTypeInfo(1);
    Ort::TypeInfo depth_out_type_info = m_session->GetOutputTypeInfo(0);
    Ort::TypeInfo cache_type_info     = m_session->GetInputTypeInfo(kNumFixedInputs);

    // rgb: [B, 3, H, W]. H and W must be static; the batch dim may be a fixed
    // value or symbolic (reported as -1), in which case any batch is accepted
    // and each batch slot carries its own independent cache sequence.
    auto rgb_info = rgb_type_info.GetTensorTypeAndShapeInfo();
    auto rgb_shape = rgb_info.GetShape();
    if (rgb_shape.size() != 4 || rgb_shape[2] <= 0 || rgb_shape[3] <= 0) {
        throw std::runtime_error("rgb input must have [B,3,H,W] shape with static H and W");
    }
    m_model_h = rgb_shape[2];
    m_model_w = rgb_shape[3];
    m_graph_batch = rgb_shape[0] > 0 ? rgb_shape[0] : 0; // 0 = symbolic = any

    // The image tensors are read and written directly by the fp32 resize kernels,
    // so those must be fp32. Fail loudly here rather than reinterpret half floats
    // as single and hand back plausible-looking garbage.
    auto require_float = [](const auto& info, const char* what) {
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(std::string(what) + " must be float32, got ONNX element type " +
                std::to_string(static_cast<int>(info.GetElementType())) +
                " (convert with keep_io_types=True to leave the image tensors fp32)");
        }
    };
    require_float(rgb_info, "rgb input");
    require_float(depth_in_type_info.GetTensorTypeAndShapeInfo(), "sparse_depth input");
    require_float(depth_out_type_info.GetTensorTypeAndShapeInfo(), "depth output");

    // past_k_00: [1, heads, n_frames, tokens, head_dim]; n_frames is symbolic (-1).
    auto cache_info = cache_type_info.GetTensorTypeAndShapeInfo();
    auto cache_shape = cache_info.GetShape();
    if (cache_shape.size() != 5 || cache_shape[1] <= 0 || cache_shape[3] <= 0 || cache_shape[4] <= 0) {
        throw std::runtime_error("cache input must have shape [1,heads,n_frames,tokens,head_dim]");
    }
    m_num_heads  = cache_shape[1];
    m_num_tokens = cache_shape[3];
    m_head_dim   = cache_shape[4];
    // Any element type is fine here: the caches are opaque to this class.
    m_cache_type = cache_info.GetElementType();

    // The cache batch dim must agree with rgb's -- a graph that batches images
    // but not cache sequences cannot stream per-view.
    const int64_t cache_batch = cache_shape[0] > 0 ? cache_shape[0] : 0;
    if (cache_batch != m_graph_batch) {
        throw std::runtime_error("rgb and cache batch dims disagree (" +
            std::to_string(m_graph_batch) + " vs " + std::to_string(cache_batch) + ")");
    }

    m_cur_batch = m_graph_batch > 0 ? m_graph_batch : 1;

    LogMessage("StreamingONNXModel geometry: input {}x{}, batch {}, cache [B,{},n_frames,{},{}] element type {} ({} bytes)",
        m_model_h, m_model_w, m_graph_batch == 0 ? "dynamic" : std::to_string(m_graph_batch),
        m_num_heads, m_num_tokens, m_head_dim,
        static_cast<int>(m_cache_type), onnxElementSize(m_cache_type));
}

void StreamingONNXModel::_ensureScratch(int64_t batch) {
    if (batch <= m_alloc_batch) {
        return;
    }
    // Scratch is only live within a single runInference call (which syncs before
    // returning), so growing it between calls is safe.
    if (m_d_rgb)   { cudaFree(m_d_rgb);   m_d_rgb = nullptr; }
    if (m_d_depth) { cudaFree(m_d_depth); m_d_depth = nullptr; }
    checkCudaError(cudaMalloc(&m_d_rgb, batch * 3 * m_model_h * m_model_w * sizeof(float)),
        "cudaMalloc streaming model rgb scratch");
    checkCudaError(cudaMalloc(&m_d_depth, batch * m_model_h * m_model_w * sizeof(float)),
        "cudaMalloc streaming model depth scratch");
    if (!m_d_empty) {
        // Backing for the zero-length frame-0 caches. No elements are read; ORT
        // just wants a valid device pointer.
        checkCudaError(cudaMalloc(&m_d_empty, 256), "cudaMalloc streaming model empty cache backing");
    }
    m_alloc_batch = batch;
}

std::vector<Ort::Value> StreamingONNXModel::_makeEmptyCaches() const {
    std::vector<int64_t> shape{m_cur_batch, m_num_heads, 0, m_num_tokens, m_head_dim};
    std::vector<Ort::Value> caches;
    caches.reserve(kNumCacheTensors);
    for (int32_t i = 0; i < kNumCacheTensors; ++i) {
        // Untyped overload so the caches are created in whatever type the graph
        // declares. This is the only place cache dtype matters -- every later
        // generation is an ORT-owned value we pass straight back in -- which is
        // what makes an fp16 cache a ~10 line change rather than a rewrite.
        // Byte count is 0: no element is ever read from m_d_empty.
        caches.push_back(Ort::Value::CreateTensor(
            m_memory_info,
            m_d_empty,
            0,
            shape.data(),
            shape.size(),
            m_cache_type
        ));
    }
    return caches;
}

bool StreamingONNXModel::acquire(const void* owner) {
    const void* expected = nullptr;
    if (m_owner.compare_exchange_strong(expected, owner)) {
        return true;
    }
    // Re-acquiring from the same pipeline (e.g. stop/start) is fine.
    return expected == owner;
}

void StreamingONNXModel::release(const void* owner) {
    const void* expected = owner;
    m_owner.compare_exchange_strong(expected, nullptr);
}

void StreamingONNXModel::resetState() {
    if (m_binding) {
        m_binding->ClearBoundInputs();
        m_binding->ClearBoundOutputs();
    }
    m_cache = _makeEmptyCaches();
    m_frames_seen = 0;
    LogMessage("StreamingONNXModel: KV cache reset (next frame feeds zero-length caches)");
}

bool StreamingONNXModel::runInference(
    const float* input_data,
    const float* depth_data,
    float*       output_data,
    TensorShape  input_shape,
    TensorShape  depth_shape,
    TensorShape  output_shape
) {
    if (!m_session || !m_binding) {
        LogMessage("StreamingONNXModel: session not initialized");
        return false;
    }
    if (!input_data || !depth_data || !output_data) {
        LogMessage("StreamingONNXModel: null input, depth, or output pointer");
        return false;
    }

    auto time_start = std::chrono::high_resolution_clock::now();

    // Batch = images per step, one independent cache sequence per slot.
    const int64_t batch = static_cast<int64_t>(input_shape.N);
    if (!supportsBatch(static_cast<int32_t>(batch))) {
        LogMessage("StreamingONNXModel: batch {} not supported (graph batch: {})",
            batch, m_graph_batch == 0 ? std::string("dynamic") : std::to_string(m_graph_batch));
        return false;
    }
    if (batch != m_cur_batch) {
        // The cache's batch dim is part of the sequence state; a different batch
        // means a different set of view sequences, so restart rather than feed a
        // mismatched cache into the graph.
        LogMessage("StreamingONNXModel: batch changed {} -> {}, restarting sequence",
            m_cur_batch, batch);
        m_cur_batch = batch;
        resetState();
    }

    try {
        _ensureScratch(batch);

        // Camera resolution -> model resolution, all batch slots in one pass:
        // [B,3,H,W] is B*3 contiguous planes to the resampler. Depth is resampled
        // sparsely so invalid (zero) pixels are never blended into valid ones.
        // Range validation and orientation are handled inside the graph.
        checkCudaError(resize_bilinear_chw(
            input_data, m_d_rgb,
            static_cast<int>(input_shape.H), static_cast<int>(input_shape.W),
            static_cast<int>(m_model_h), static_cast<int>(m_model_w),
            static_cast<int>(batch * 3), 0
        ), "resize rgb to model resolution");

        const size_t depth_in_elems  = depth_shape.H * depth_shape.W;
        const size_t depth_mdl_elems = static_cast<size_t>(m_model_h * m_model_w);
        for (int64_t b = 0; b < batch; ++b) {
            checkCudaError(resize_sparse_depth(
                depth_data + b * depth_in_elems, m_d_depth + b * depth_mdl_elems,
                static_cast<int>(depth_shape.H), static_cast<int>(depth_shape.W),
                static_cast<int>(m_model_h), static_cast<int>(m_model_w),
                0
            ), "resize sparse depth to model resolution");
        }

        // ORT runs on its own stream; make sure the resamples are visible first.
        checkCudaError(cudaStreamSynchronize(0), "sync before streaming Run");

        std::vector<int64_t> rgb_tensor_shape{batch, 3, m_model_h, m_model_w};
        std::vector<int64_t> depth_tensor_shape{batch, 1, m_model_h, m_model_w};

        Ort::Value rgb_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, m_d_rgb, static_cast<size_t>(batch * 3 * m_model_h * m_model_w),
            rgb_tensor_shape.data(), rgb_tensor_shape.size()
        );
        Ort::Value depth_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, m_d_depth, static_cast<size_t>(batch * m_model_h * m_model_w),
            depth_tensor_shape.data(), depth_tensor_shape.size()
        );

        m_binding->ClearBoundInputs();
        m_binding->ClearBoundOutputs();

        m_binding->BindInput("rgb", rgb_tensor);
        m_binding->BindInput("sparse_depth", depth_tensor);
        for (int32_t i = 0; i < kNumCacheTensors; ++i) {
            m_binding->BindInput(m_past_cstr[i], m_cache[i]);
        }

        // Bound by MemoryInfo rather than to our own buffers: the graph decides
        // n_frames_out, so ORT must allocate these device-side at the size it
        // produced. The returned values are fed straight back in next frame --
        // the cache never touches host memory.
        m_binding->BindOutput("depth", m_memory_info);
        m_binding->BindOutput("depth_conf", m_memory_info);
        for (int32_t i = 0; i < kNumCacheTensors; ++i) {
            m_binding->BindOutput(m_new_cstr[i], m_memory_info);
        }

        m_session->Run(Ort::RunOptions{nullptr}, *m_binding);

        // Bind order, so index kNumFixedOutputs + i is m_new_names[i].
        std::vector<Ort::Value> outputs = m_binding->GetOutputValues();
        if (outputs.size() != static_cast<size_t>(kNumFixedOutputs + kNumCacheTensors)) {
            LogMessage("StreamingONNXModel: expected {} outputs, got {}",
                kNumFixedOutputs + kNumCacheTensors, outputs.size());
            return false;
        }

        // Drop the previous generation's bindings before those tensors are freed.
        m_binding->ClearBoundInputs();
        m_binding->ClearBoundOutputs();

        // depth is [B, H, W] in whatever orientation the graph emits; take the
        // dims it reports rather than assuming them.
        auto depth_out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (depth_out_shape.size() != 3 || depth_out_shape[0] != batch) {
            LogMessage("StreamingONNXModel: unexpected depth output shape (rank {}, batch {})",
                depth_out_shape.size(), depth_out_shape.empty() ? -1 : depth_out_shape[0]);
            return false;
        }
        const int64_t src_h = depth_out_shape[1];
        const int64_t src_w = depth_out_shape[2];

        // All batch slots in one pass: [B,H,W] in and [B,H',W'] out are both
        // contiguous single-channel planes.
        checkCudaError(resize_bilinear_chw(
            outputs[0].GetTensorData<float>(), output_data,
            static_cast<int>(src_h), static_cast<int>(src_w),
            static_cast<int>(output_shape.H), static_cast<int>(output_shape.W),
            static_cast<int>(batch), 0
        ), "resize model depth to output resolution");
        checkCudaError(cudaStreamSynchronize(0), "sync after streaming depth resize");

        auto cache_shape = outputs[kNumFixedOutputs].GetTensorTypeAndShapeInfo().GetShape();
        if (cache_shape.size() != 5 || cache_shape[0] != batch) {
            LogMessage("StreamingONNXModel: cache output shape unexpected (rank {}, batch {} vs {})",
                cache_shape.size(), cache_shape.empty() ? -1 : cache_shape[0], batch);
            resetState();
            return false;
        }
        // The graph slices to its own retention window, so this is just reported.
        const int64_t retained = cache_shape[2];

        // This step's caches become the next step's inputs; assigning here frees
        // the previous generation back to ORT's arena for reuse.
        m_cache.clear();
        m_cache.reserve(kNumCacheTensors);
        for (int32_t i = 0; i < kNumCacheTensors; ++i) {
            m_cache.push_back(std::move(outputs[kNumFixedOutputs + i]));
        }

        m_frames_seen++;

        auto time_end = std::chrono::high_resolution_clock::now();
        LogMessage("StreamingONNXModel frame {}: batch {}, {} frames retained, depth out [{}, {}], {} ms",
            m_frames_seen, batch, retained, src_h, src_w,
            std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count());

    } catch (const std::exception& e) {
        LogMessage("Error during streaming ONNX inference: {}", e.what());
        // The cache generation is now indeterminate; restart the sequence rather
        // than feeding a half-updated cache into the next frame.
        resetState();
        return false;
    }

    return true;
}

bool StreamingONNXModel::runInference(
    const uint8_t* /*input_data*/,
    const float*   /*depth_data*/,
    float*         /*output_data*/,
    TensorShape    /*input_shape*/,
    TensorShape    /*depth_shape*/,
    TensorShape    /*output_shape*/
) {
    LogMessage("StreamingONNXModel: uint8 input overload is not implemented; "
               "the pipeline converts to float before inference.");
    return false;
}

};