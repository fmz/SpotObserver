//
// Created by faisal on 5/12/2025.
//

#pragma once

#include "utils.h"

#include <atomic>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>

namespace SOb {

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

    // Streaming models carry per-frame state (e.g. a KV cache) that is only valid
    // for a contiguous frame sequence. The pipeline calls this whenever the
    // sequence restarts. No-op for stateless models.
    virtual void resetState() {}

    // True when the model consumes full-resolution sparse metric depth directly
    // instead of the pipeline's downscaled depth.
    virtual bool wantsFullResDepth() const { return false; }

    // Models holding per-sequence state can serve exactly one pipeline at a time;
    // a second pipeline driving the same instance would interleave two camera
    // streams into one cache. Stateless models are freely shareable, so the
    // default always succeeds. Returns false if another owner holds the instance.
    virtual bool acquire(const void* owner) { (void)owner; return true; }
    virtual void release(const void* owner) { (void)owner; }

    // Whether the model can run a batch of n images per step. Streaming models
    // constrain this to their graph's batch dim (or accept any n when the export
    // declares it symbolic); stateless models take whatever they're given.
    virtual bool supportsBatch(int32_t n) const { return n >= 1; }
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

    // Declaration order is load-bearing: members are destroyed in reverse, and
    // Ort::Env must outlive every Session created from it (and the Session must
    // outlive the Allocator/IoBinding that reference it). Keep m_env first and
    // m_session ahead of m_allocator/m_binding.
    Ort::Env m_env;
    Ort::SessionOptions m_sess_options;
    Ort::MemoryInfo m_memory_info;
    std::unique_ptr<Ort::Session> m_session;
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

///////////////////////////////////////////////////////////////////////////////////////////////////

// Autoregressive depth model with a transformer KV cache.
//
// Graph I/O (50 in, 50 out):
//   in  : rgb [1,3,H,W], sparse_depth [1,1,H,W], then past_k_00, past_v_00,
//         past_k_01, ... interleaved K/V per layer, each [1,16,n_frames,P,64]
//   out : depth, depth_conf, then new_k_00, new_v_00, ... in the same interleaved
//         order, already sliced to the model's retention window
//
// The caches never leave the GPU. Each frame binds the previous step's outputs
// straight back as inputs via IoBinding, and binds the new outputs by MemoryInfo
// so ORT sizes them to whatever n_frames_out the graph produced -- they cannot be
// pre-allocated because the sequence axis grows before the window clamps it.
// Frame 0 feeds zero-length caches.
//
// The cache is per-frame-sequence state, so an instance must not be shared
// between pipelines; interleaving two camera streams into one cache silently
// corrupts it. Load a separate instance per pipeline.
class StreamingONNXModel : public MLModel {
    static constexpr int32_t kNumCacheTensors = 48; // 24 layers x {K, V}
    static constexpr int32_t kNumFixedInputs  = 2;  // rgb, sparse_depth
    static constexpr int32_t kNumFixedOutputs = 2;  // depth, depth_conf

    // Declaration order is load-bearing; see the note in ONNXModel.
    Ort::Env m_env;
    Ort::SessionOptions m_sess_options;
    Ort::MemoryInfo m_memory_info;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::IoBinding> m_binding;

    // Index-aligned by construction: m_past_names[i] pairs with m_new_names[i],
    // which is session output kNumFixedOutputs + i, which is m_cache[i].
    std::vector<std::string> m_past_names;
    std::vector<std::string> m_new_names;
    std::vector<const char*> m_past_cstr;
    std::vector<const char*> m_new_cstr;

    // Previous step's caches, device-resident and owned by ORT's CUDA allocator.
    std::vector<Ort::Value> m_cache;
    // Backing pointer for the zero-length frame-0 caches (no elements are read).
    void* m_d_empty{nullptr};

    // Model-native input geometry, read from the graph rather than assumed.
    int64_t m_model_h{0};
    int64_t m_model_w{0};
    int64_t m_num_heads{0};
    int64_t m_num_tokens{0};
    int64_t m_head_dim{0};

    // Batch handling. m_graph_batch is what the graph declares for the batch dim:
    // a positive value pins it; 0 means symbolic ("B"), so the model takes
    // whatever batch each pipeline stream delivers (one cache sequence per batch
    // slot -- slots are independent views, e.g. the two front cameras).
    // m_cur_batch is the batch of the active sequence; changing it invalidates
    // the cache, so a mid-stream change forces a sequence restart.
    int64_t m_graph_batch{1};
    int64_t m_cur_batch{1};
    int64_t m_alloc_batch{0}; // scratch capacity, grown on demand

    // Element type the graph declares for the caches. Their contents are never
    // read or written here -- they leave ORT and come straight back in -- so this
    // only decides how the zero-length frame-0 tensors are created. An fp16 cache
    // halves the largest allocation this model makes.
    ONNXTensorElementDataType m_cache_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

    // Scratch at model resolution.
    float* m_d_rgb{nullptr};
    float* m_d_depth{nullptr};

    int64_t m_frames_seen{0};
    bool m_use_cuda{false};
    // The pipeline currently driving this instance, or null when free.
    std::atomic<const void*> m_owner{nullptr};

    void _setDevice(const std::string& device_type);
    void _buildCacheNames();
    void _readGeometry();
    void _ensureScratch(int64_t batch);
    void _freeScratch();
    std::vector<Ort::Value> _makeEmptyCaches() const;

public:
    explicit StreamingONNXModel(const std::string& model_path, const std::string& device_type = "cuda");
    ~StreamingONNXModel() override;

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

    void resetState() override;
    bool wantsFullResDepth() const override { return true; }
    bool acquire(const void* owner) override;
    void release(const void* owner) override;
    bool supportsBatch(int32_t n) const override {
        if (n < 1) return false;
        return m_graph_batch == 0 || n == m_graph_batch;
    }

    std::string getDevice() const { return m_use_cuda ? "cuda" : "cpu"; }
    int64_t getModelHeight() const { return m_model_h; }
    int64_t getModelWidth() const { return m_model_w; }
};

} // namespace SOb