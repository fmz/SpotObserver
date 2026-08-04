"""Extract the two resize kernels verbatim from cuda_kernels.cu and wrap them in a
host-side harness so their math can be executed and checked without nvcc/a GPU."""
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_BUILD = _HERE / "build"
_BUILD.mkdir(exist_ok=True)
import pathlib

SRC = pathlib.Path(str(_REPO / "src/cuda_kernels.cu"))
OUT = pathlib.Path(str(_BUILD))
OUT.mkdir(parents=True, exist_ok=True)

txt = SRC.read_text(encoding="utf-8", errors="replace")
start = txt.index("__global__ void resize_bilinear_chw_kernel")
end = txt.index("cudaError_t resize_bilinear_chw(")
kernels = txt[start:end]
print(f"extracted {kernels.count(chr(10))} lines of kernel source")

harness = r"""// GENERATED kernel test harness -- not part of the build.
// Executes the real kernel bodies on the host with CUDA builtins stubbed, so the
// resize math can be validated on a machine with no CUDA toolkit and no GPU.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <set>

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__

struct dim3 { unsigned x, y, z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
static dim3 blockIdx, threadIdx, blockDim;
static const float CUDART_INF_F = HUGE_VALF;
using std::min; using std::max;

// ---- verbatim kernels from src/cuda_kernels.cu ------------------------------
KERNELS_HERE
// -----------------------------------------------------------------------------

static int g_failures = 0;
static void check(bool ok, const char* name, const char* detail = "") {
    printf("%-58s %s %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) g_failures++;
}

// Serial stand-ins for the <<<grid, block>>> launches.
static void launch_bilinear(const float* src, float* dst, int in_h, int in_w,
                            int out_h, int out_w, int ch) {
    blockDim = dim3(1,1,1);
    for (int c = 0; c < ch; ++c)
        for (int y = 0; y < out_h; ++y)
            for (int x = 0; x < out_w; ++x) {
                blockIdx = dim3(x, y, c); threadIdx = dim3(0,0,0);
                resize_bilinear_chw_kernel(src, dst, in_h, in_w, out_h, out_w, ch);
            }
}
static void launch_sparse(const float* src, float* dst, int in_h, int in_w,
                          int out_h, int out_w) {
    blockDim = dim3(1,1,1);
    for (int y = 0; y < out_h; ++y)
        for (int x = 0; x < out_w; ++x) {
            blockIdx = dim3(x, y, 0); threadIdx = dim3(0,0,0);
            resize_sparse_depth_kernel(src, dst, in_h, in_w, out_h, out_w);
        }
}

int main() {
    // 1. Identity resize must be exact (half-pixel centres collapse to integers).
    {
        const int H = 37, W = 53;
        std::vector<float> src(3*H*W), dst(3*H*W, -1.f);
        for (size_t i = 0; i < src.size(); ++i) src[i] = float(i % 97) * 0.37f;
        launch_bilinear(src.data(), dst.data(), H, W, H, W, 3);
        float worst = 0.f;
        for (size_t i = 0; i < src.size(); ++i) worst = max(worst, fabsf(src[i]-dst[i]));
        char d[64]; snprintf(d, sizeof d, "max|diff|=%.3e", worst);
        check(worst < 1e-4f, "bilinear: identity resize is exact", d);
    }

    // 2. Bilinear reproduces a linear ramp exactly (non-circular analytic check).
    //    Interior only: edge clamping intentionally breaks linearity at borders.
    {
        const int IH = 480, IW = 640, OH = 392, OW = 518;
        const float a = 0.013f, b = -0.007f, c0 = 1.5f;
        std::vector<float> src(IH*IW), dst(OH*OW, -1.f);
        for (int y = 0; y < IH; ++y) for (int x = 0; x < IW; ++x) src[y*IW+x] = a*x + b*y + c0;
        launch_bilinear(src.data(), dst.data(), IH, IW, OH, OW, 1);
        float worst = 0.f;
        for (int y = 1; y < OH-1; ++y) for (int x = 1; x < OW-1; ++x) {
            float sy = (y+0.5f)*IH/OH - 0.5f, sx = (x+0.5f)*IW/OW - 0.5f;
            worst = max(worst, fabsf(dst[y*OW+x] - (a*sx + b*sy + c0)));
        }
        char d[64]; snprintf(d, sizeof d, "max|diff|=%.3e", worst);
        check(worst < 1e-3f, "bilinear: reproduces linear ramp (480x640->392x518)", d);
    }

    // 4. Sparse depth must never invent a value: every output is 0 or a real input.
    {
        const int IH = 480, IW = 640, OH = 392, OW = 518;
        std::vector<float> src(IH*IW, 0.f), dst(OH*OW, -1.f);
        unsigned s = 12345;
        int n_valid_in = 0;
        for (int i = 0; i < IH*IW; ++i) {
            s = s*1664525u + 1013904223u;
            if ((s >> 16) % 100 < 30) { src[i] = 0.5f + float((s>>8)%1000)*0.0075f; n_valid_in++; }
        }
        std::set<float> allowed(src.begin(), src.end());
        allowed.insert(0.f);
        launch_sparse(src.data(), dst.data(), IH, IW, OH, OW);
        bool invented = false; int n_valid_out = 0;
        for (int i = 0; i < OH*OW; ++i) {
            if (dst[i] != 0.f) n_valid_out++;
            if (!allowed.count(dst[i])) invented = true;
        }
        char d[96];
        snprintf(d, sizeof d, "valid in=%d out=%d (%.0f%% kept)", n_valid_in, n_valid_out,
                 100.0*n_valid_out/ (double)(OH*OW) / 0.30);
        check(!invented, "sparse depth: never interpolates an invented value", d);
        check(n_valid_out > 0.6*OH*OW*0.30, "sparse depth: retains most valid samples", d);
    }

    // 5. Degenerate inputs: 0 and NaN are the only "no sample" markers.
    {
        const int IH = 64, IW = 64, OH = 50, OW = 50;
        std::vector<float> src(IH*IW, 0.f), dst(OH*OW, 9.f);
        launch_sparse(src.data(), dst.data(), IH, IW, OH, OW);
        bool all_zero = std::all_of(dst.begin(), dst.end(), [](float v){ return v == 0.f; });
        check(all_zero, "sparse depth: all-invalid input yields all-zero output");

        std::fill(src.begin(), src.end(), NAN);
        std::fill(dst.begin(), dst.end(), 9.f);
        launch_sparse(src.data(), dst.data(), IH, IW, OH, OW);
        all_zero = std::all_of(dst.begin(), dst.end(), [](float v){ return v == 0.f; });
        check(all_zero, "sparse depth: NaN rejected (negated compare)");
    }

    // 6. Upscaling must still write every output pixel (empty-footprint guard).
    {
        const int IH = 392, IW = 518, OH = 480, OW = 640;
        std::vector<float> src(IH*IW, 2.5f), dst(OH*OW, -1.f);
        launch_sparse(src.data(), dst.data(), IH, IW, OH, OW);
        bool all_set = std::all_of(dst.begin(), dst.end(), [](float v){ return v == 2.5f; });
        check(all_set, "sparse depth: upscale 392x518->480x640 fills every pixel");
    }

    // 7. Canary check for out-of-bounds writes on the model-resolution path.
    {
        const int IH = 480, IW = 640, OH = 392, OW = 518;
        std::vector<float> src(3*IH*IW, 1.f);
        std::vector<float> buf(3*OH*OW + 64, -12345.f);
        launch_bilinear(src.data(), buf.data(), IH, IW, OH, OW, 3);
        bool canary_ok = true;
        for (int i = 0; i < 64; ++i) if (buf[3*OH*OW + i] != -12345.f) canary_ok = false;
        check(canary_ok, "bilinear: no writes past the output buffer");
    }

    printf("\n%s (%d failure(s))\n", g_failures ? "FAILURES PRESENT" : "ALL KERNEL TESTS PASSED", g_failures);
    return g_failures ? 1 : 0;
}
"""

harness = harness.replace("KERNELS_HERE", kernels)
(OUT / "kern_test.cpp").write_text(harness, encoding="utf-8")
print("wrote", OUT / "kern_test.cpp")
