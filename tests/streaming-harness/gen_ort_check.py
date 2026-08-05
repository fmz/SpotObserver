"""Extract StreamingONNXModel verbatim from the real sources into a single TU
that can be compiled against the real ONNX Runtime headers, with CUDA/torch/log
dependencies stubbed. Purpose: type-check the ~350 lines of Ort C++ API usage on
a machine that has no CUDA toolkit and no libtorch."""
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

# class decl: from "class StreamingONNXModel" up to the first line that is exactly "};"
h_start = header.index("class StreamingONNXModel")
h_rest = header[h_start:]
m = re.search(r"^\};$", h_rest, re.M)
class_decl = h_rest[: m.end()]

# impl: from the ctor to just before the final namespace close
i_start = impl.index("// Bytes per element for the types a KV cache")
i_rest = impl[i_start:]
m2 = re.search(r"^\};\s*$", i_rest, re.M)
class_impl = i_rest[: m2.start()]

print(f"extracted class decl: {class_decl.count(chr(10))} lines")
print(f"extracted impl      : {class_impl.count(chr(10))} lines")

tu = f"""// GENERATED compile-check harness -- not part of the build.
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

// ---- stubs standing in for the real project headers -------------------------
using cudaError_t = int;
using cudaStream_t = void*;
constexpr cudaError_t cudaSuccess = 0;
cudaError_t cudaMalloc(void** p, size_t n);
template <typename T> cudaError_t cudaMalloc(T** p, size_t n) {{ return cudaMalloc(reinterpret_cast<void**>(p), n); }}
cudaError_t cudaFree(void* p);
cudaError_t cudaGetDeviceCount(int* c);
cudaError_t cudaStreamSynchronize(cudaStream_t s);
const char* cudaGetErrorString(cudaError_t e);

namespace SOb {{

namespace fs = std::filesystem;

struct TensorShape {{
    size_t N, C, H, W;
}};

template <typename... Args>
void LogMessage(const std::format_string<Args...> fmt, Args&&... args) {{ (void)fmt; }}
inline void LogMessage(const std::string& s) {{ (void)s; }}
template <typename... Args>
void LogPerf(const std::format_string<Args...> fmt, Args&&... args) {{ (void)fmt; }}

inline void checkCudaError(cudaError_t error, const std::string& operation) {{
    if (error != cudaSuccess) throw std::runtime_error(operation);
}}

cudaError_t resize_bilinear_chw(const float* d_in, float* d_out, int in_h, int in_w,
    int out_h, int out_w, int channels, cudaStream_t stream);
cudaError_t resize_sparse_depth(const float* d_in, float* d_out, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream);

class MLModel {{
public:
    virtual ~MLModel() = default;
    virtual bool runInference(const float*, const float*, float*, TensorShape, TensorShape, TensorShape) = 0;
    virtual bool runInference(const uint8_t*, const float*, float*, TensorShape, TensorShape, TensorShape) = 0;
    virtual void resetState() {{}}
    virtual bool wantsFullResDepth() const {{ return false; }}
    virtual bool acquire(const void* owner) {{ (void)owner; return true; }}
    virtual void release(const void* owner) {{ (void)owner; }}
    virtual bool supportsBatch(int32_t n) const {{ return n >= 1; }}
}};

// ---- verbatim class declaration from src/include/model.h --------------------
{class_decl}

// ---- verbatim implementation from src/model.cpp -----------------------------
{class_impl}

}} // namespace SOb
"""

(OUT / "ort_check.cpp").write_text(tu, encoding="utf-8")
print("wrote", OUT / "ort_check.cpp")
