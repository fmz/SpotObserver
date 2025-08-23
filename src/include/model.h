//
// Created by faisal on 5/12/2025.
//

#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>

namespace SOb {

struct TensorShape {
    size_t N, C, H, W;
    TensorShape(size_t n, size_t c, size_t h, size_t w) : N(n), C(c), H(h), W(w) {}
};

class MLModel {
public:
    virtual ~MLModel() = default;
    virtual bool runInference(
        const float* input_data,
        const float* depth_data,
        float*       output_data,
        TensorShape  input_shape,
        TensorShape  depth_shape,
        TensorShape  output_shape
    ) = 0;
    virtual bool runInference(
        const uint8_t* rgb_data,
        const float*   depth_data,
        float*         output_data,
        TensorShape    input_shape,
        TensorShape    depth_shape,
        TensorShape    output_shape
    ) = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class TorchModel : public MLModel {
    torch::jit::script::Module m_module;
    torch::Device m_device;

    torch::Tensor _run_inference(
        const torch::Tensor& input_tensor,
        const std::optional<torch::Tensor>& depth_tensor = std::nullopt
    );

public:
    explicit TorchModel(const std::string& model_path, const std::string& device_type = "cpu");
    ~TorchModel() override;

    bool runInference(
        const float* input_data,
        const float* depth_data,
        float*       output_data,
        TensorShape  input_shape,
        TensorShape  depth_shape,
        TensorShape  output_shape
    ) override;

    bool runInference(
        const uint8_t* rgb_data,
        const float*   depth_data,
        float*         output_data,
        TensorShape    input_shape,
        TensorShape    depth_shape,
        TensorShape    output_shape
    ) override;
    
    void setDevice(const std::string& device_type);
    std::string getDevice() const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

class ONNXModel : public MLModel {
    struct CudaMemoryDeleter {
        explicit CudaMemoryDeleter(Ort::Allocator* alloc) {
            alloc_ = alloc;
        }

        void operator()(void* ptr) const {
            alloc_->Free(ptr);
        }

        Ort::Allocator* alloc_;
    };

    std::unique_ptr<Ort::Session> m_session;
    Ort::Env m_env;
    Ort::SessionOptions m_sess_options;
    Ort::MemoryInfo m_memory_info;
    std::unique_ptr<Ort::Allocator> m_allocator;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<std::vector<int64_t>> m_input_shapes;
    std::vector<std::vector<int64_t>> m_output_shapes;
    void* m_input_buffer;
    void* m_output_buffer;

    int32_t m_num_model_inputs;

    std::unique_ptr<Ort::IoBinding> m_binding;

    bool m_use_cuda;
    bool m_is_warmed_up = false;

    void _warmupModel();

public:
    explicit ONNXModel(const std::string& model_path, const std::string& device_type = "cpu");
    ~ONNXModel() override;

    bool runInference(
        const float* input_data,
        const float* depth_data,
        float*       output_data,
        TensorShape  input_shape,
        TensorShape  depth_shape,
        TensorShape  output_shape
    ) override;

    bool runInference(
        const uint8_t* input_data,
        const float*   depth_data,
        float*         output_data,
        TensorShape    input_shape,
        TensorShape    depth_shape,
        TensorShape    output_shape
    ) override;

    void setDevice(const std::string& device_type);
    std::string getDevice() const;
};

} // namespace UB