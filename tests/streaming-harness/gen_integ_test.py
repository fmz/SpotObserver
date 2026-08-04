"""Integration harness: runs the REAL StreamingONNXModel (verbatim from model.h /
model.cpp) plus the REAL resize kernels against a tiny stub model with the same IO
contract, on the ONNX Runtime CPU EP.

Only two substitutions are made, both documented in the output:
  * _setDevice        -> CPU EP (the CUDA EP is the one function not covered here)
  * cudaMalloc/Free   -> host malloc/free, so the kernels operate on host buffers
Everything else -- name pairing, geometry, zero-length frame 0, the bind/Run/
ping-pong loop, resetState -- is the shipping code.
"""
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_BUILD = _HERE / "build"
_BUILD.mkdir(exist_ok=True)
import re, pathlib

SRC = pathlib.Path(str(_REPO / "src"))
OUT = pathlib.Path(str(_BUILD))
OUT.mkdir(parents=True, exist_ok=True)

header = (SRC / "include/model.h").read_text(encoding="utf-8", errors="replace")
impl = (SRC / "model.cpp").read_text(encoding="utf-8", errors="replace")
cu = (SRC / "cuda_kernels.cu").read_text(encoding="utf-8", errors="replace")

h_start = header.index("class StreamingONNXModel")
h_rest = header[h_start:]
class_decl = h_rest[: re.search(r"^\};$", h_rest, re.M).end()]

i_start = impl.index("// Bytes per element for the types a KV cache")
i_rest = impl[i_start:]
class_impl = i_rest[: re.search(r"^\};\s*$", i_rest, re.M).start()]

kernels = cu[cu.index("__global__ void resize_bilinear_chw_kernel"):
             cu.index("cudaError_t resize_bilinear_chw(")]

# --- substitution 1: CPU execution provider -----------------------------------
sd_start = class_impl.index("void StreamingONNXModel::_setDevice")
sd_end = class_impl.index("void StreamingONNXModel::_buildCacheNames")
class_impl = class_impl[:sd_start] + """void StreamingONNXModel::_setDevice(const std::string&) {
    // TEST SUBSTITUTION: CPU EP instead of the CUDA EP.
    m_use_cuda = true;   // gate in the ctor; memory info stays CPU
}

""" + class_impl[sd_end:]

# --- test accessors (injected into the copy only) ------------------------------
class_decl = class_decl.replace(
    "public:\n    explicit StreamingONNXModel(",
    "public:\n"
    "    const std::vector<Ort::Value>& testCache() const { return m_cache; }\n"
    "    int64_t testFrames() const { return m_frames_seen; }\n"
    "    explicit StreamingONNXModel(", 1)

tu = r"""// GENERATED integration harness -- not part of the build.
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <set>
#include <cstring>

// ---- CUDA stubs: host memory, so the real kernels run on host buffers --------
using cudaError_t = int;
using cudaStream_t = void*;
constexpr cudaError_t cudaSuccess = 0;
static size_t g_alloc_bytes = 0, g_alloc_count = 0, g_free_count = 0;
static cudaError_t cudaMalloc(void** p, size_t n) { *p = malloc(n); g_alloc_bytes += n; g_alloc_count++; return *p ? 0 : 1; }
template <typename T> static cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
static cudaError_t cudaFree(void* p) { if (p) { free(p); g_free_count++; } return 0; }
static cudaError_t cudaGetDeviceCount(int* c) { *c = 1; return 0; }
static cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
static const char* cudaGetErrorString(cudaError_t) { return "stub"; }

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__
struct dim3 { unsigned x, y, z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
static dim3 blockIdx, threadIdx, blockDim;
static const float CUDART_INF_F = HUGE_VALF;
using std::min; using std::max;

namespace SOb {
namespace fs = std::filesystem;

struct TensorShape { size_t N, C, H, W; };

static bool g_quiet = true;
template <typename... Args>
void LogMessage(const std::format_string<Args...> fmt, Args&&... args) {
    if (!g_quiet) { printf("    [log] %s\n", std::format(fmt, std::forward<Args>(args)...).c_str()); }
}
inline void LogMessage(const std::string& s) { if (!g_quiet) printf("    [log] %s\n", s.c_str()); }
template <typename... Args>
void LogPerf(const std::format_string<Args...> fmt, Args&&... args) {
    if (!g_quiet) { printf("    [perf] %s\n", std::format(fmt, std::forward<Args>(args)...).c_str()); }
}
inline void checkCudaError(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) throw std::runtime_error(operation);
}

// ---- verbatim kernels from src/cuda_kernels.cu ------------------------------
KERNELS_HERE

// Host launchers matching the real .cu launcher signatures.
cudaError_t resize_bilinear_chw(const float* d_in, float* d_out, int in_h, int in_w,
                                int out_h, int out_w, int channels, cudaStream_t) {
    blockDim = dim3(1,1,1);
    for (int c = 0; c < channels; ++c)
      for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x) {
            blockIdx = dim3(x,y,c); threadIdx = dim3(0,0,0);
            resize_bilinear_chw_kernel(d_in, d_out, in_h, in_w, out_h, out_w, channels);
        }
    return 0;
}
cudaError_t resize_sparse_depth(const float* d_in, float* d_out, int in_h, int in_w,
                                int out_h, int out_w, cudaStream_t) {
    blockDim = dim3(1,1,1);
    for (int y = 0; y < out_h; ++y)
      for (int x = 0; x < out_w; ++x) {
          blockIdx = dim3(x,y,0); threadIdx = dim3(0,0,0);
          resize_sparse_depth_kernel(d_in, d_out, in_h, in_w, out_h, out_w);
      }
    return 0;
}

class MLModel {
public:
    virtual ~MLModel() = default;
    virtual bool runInference(const float*, const float*, float*, TensorShape, TensorShape, TensorShape) = 0;
    virtual bool runInference(const uint8_t*, const float*, float*, TensorShape, TensorShape, TensorShape) = 0;
    virtual void resetState() {}
    virtual bool wantsFullResDepth() const { return false; }
    virtual bool acquire(const void* owner) { (void)owner; return true; }
    virtual void release(const void* owner) { (void)owner; }
};

// ---- verbatim class declaration from src/include/model.h --------------------
CLASS_DECL_HERE

// ---- verbatim implementation from src/model.cpp -----------------------------
CLASS_IMPL_HERE

} // namespace SOb

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& detail = "") {
    printf("%-56s %s %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) g_fail++;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) { f = sign << 31; }
        else { uint32_t e = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; e--; }
               man &= 0x3FFu; f = (sign << 31) | (e << 23) | (man << 13); }
    } else if (exp == 31) { f = (sign << 31) | (0xFFu << 23) | (man << 13); }
    else { f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13); }
    float out; std::memcpy(&out, &f, 4); return out;
}

// Reads cache element 0 whatever the graph carries it as.
static float cache_elem0(const Ort::Value& v) {
    if (v.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        return half_to_float(*v.GetTensorData<uint16_t>());
    return *v.GetTensorData<float>();
}

static void run_suite(const char* model, bool fp16_cache) {
    using namespace SOb;
    const int WINDOW = 4;                 // matches make_stub_model.py
    const size_t CH = 12, CW = 16;        // "camera" resolution; model is 8x10
    const char* tag = fp16_cache ? "[fp16 cache] " : "[fp32 cache] ";
    printf("\n--- %s ---\n", fp16_cache ? "fp16 caches, fp32 image IO (hybrid recipe)"
                                        : "fp32 throughout (current export)");

    std::unique_ptr<StreamingONNXModel> m;
    try {
        m = std::make_unique<StreamingONNXModel>(model, "cuda");
    } catch (const std::exception& e) {
        printf("%sconstruction threw: %s\n", tag, e.what());
        g_fail++; return;
    }
    check(true, (std::string(tag) + "construct: graph accepted").c_str());

    const auto want = fp16_cache ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                                 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    check(m->testCache().size() == 48 &&
          m->testCache()[0].GetTensorTypeAndShapeInfo().GetElementType() == want,
          (std::string(tag) + "reset: empty caches match the graph's element type").c_str());
    check(m->testCache()[0].GetTensorTypeAndShapeInfo().GetShape()[2] == 0,
          (std::string(tag) + "reset: frame-0 cache has n_frames == 0").c_str());

    TensorShape in_s{1, 3, CH, CW}, dp_s{1, 1, CH, CW}, out_s{1, 1, CH, CW};
    std::vector<float> rgb(3 * CH * CW, 1.0f), depth(CH * CW, 2.0f), out(CH * CW, -1.f);

    bool growth_ok = true, content_ok = true, out_ok = true, type_ok = true;
    std::string seq;
    for (int f = 0; f < 7; ++f) {
        std::fill(out.begin(), out.end(), -1.f);
        if (!m->runInference(rgb.data(), depth.data(), out.data(), in_s, dp_s, out_s)) {
            check(false, (std::string(tag) + "runInference returned false").c_str(),
                  std::format("frame {}", f));
            return;
        }
        int64_t n = m->testCache()[0].GetTensorTypeAndShapeInfo().GetShape()[2];
        seq += std::to_string(n) + (f < 6 ? "," : "");
        if (n != std::min<int64_t>(f + 1, WINDOW)) growth_ok = false;

        for (int i = 0; i < 48; ++i) {
            if (cache_elem0(m->testCache()[i]) != float(i)) content_ok = false;
            if (m->testCache()[i].GetTensorTypeAndShapeInfo().GetElementType() != want) type_ok = false;
        }
        for (size_t i = 0; i < out.size(); ++i)
            if (std::fabs(out[i] - 1.0f) > 1e-4f) out_ok = false;
    }
    check(growth_ok, (std::string(tag) + "stream: n_frames grows then clamps").c_str(), "seq=" + seq);
    check(content_ok, (std::string(tag) + "stream: cache[i] carries layer i").c_str());
    check(type_ok, (std::string(tag) + "stream: cache element type stable across frames").c_str());
    check(out_ok, (std::string(tag) + "stream: depth round-trips camera->model->camera").c_str());

    m->resetState();
    check(m->testCache()[0].GetTensorTypeAndShapeInfo().GetShape()[2] == 0 && m->testFrames() == 0,
          (std::string(tag) + "reset: mid-stream reset returns to zero-length").c_str());
    m->runInference(rgb.data(), depth.data(), out.data(), in_s, dp_s, out_s);
    check(m->testCache()[0].GetTensorTypeAndShapeInfo().GetShape()[2] == 1,
          (std::string(tag) + "reset: sequence restarts cleanly").c_str());

    int pipeline_a = 0, pipeline_b = 0;
    check(m->acquire(&pipeline_a), (std::string(tag) + "ownership: first pipeline acquires").c_str());
    check(!m->acquire(&pipeline_b), (std::string(tag) + "ownership: second pipeline refused").c_str());
    m->release(&pipeline_a);
    check(m->acquire(&pipeline_b), (std::string(tag) + "ownership: released is reusable").c_str());
    m->release(&pipeline_b);

    size_t before = g_free_count;
    m.reset();
    check(g_free_count > before, (std::string(tag) + "teardown: scratch freed").c_str());
}

int main() {
    run_suite("STUB_PATH", false);
    run_suite("STUB_PATH_FP16", true);
    printf("\n%s (%d failure(s))\n",
           g_fail ? "FAILURES PRESENT" : "ALL INTEGRATION TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
"""

tu = tu.replace("KERNELS_HERE", kernels)
tu = tu.replace("CLASS_DECL_HERE", class_decl)
tu = tu.replace("CLASS_IMPL_HERE", class_impl)
tu = tu.replace("STUB_PATH_FP16", str(OUT / "stub_stream_fp16cache.onnx").replace("\\", "/"))
tu = tu.replace("STUB_PATH", str(OUT / "stub_stream.onnx").replace("\\", "/"))
(OUT / "integ_test.cpp").write_text(tu, encoding="utf-8")
print("wrote", OUT / "integ_test.cpp")
