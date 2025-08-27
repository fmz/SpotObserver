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

};