//
// Created by brown on 7/23/2025.
//

#pragma once

#include "logger.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

namespace SOb {

struct TensorShape {
    size_t N, C, H, W;
    TensorShape(size_t n, size_t c, size_t h, size_t w) : N(n), C(c), H(h), W(w) {}
    bool operator==(const TensorShape& other) const {
        return N == other.N && C == other.C && H == other.H && W == other.W;
    }
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
    size_t total_size() const {
        return N * C * H * W;
    }
    std::string to_string() const {
        return std::format("N={}, C={}, H={}, W={}", N, C, H, W);
    }
};


inline void checkCudaError(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(operation + " failed: " + cudaGetErrorString(error));
    }
}

inline bool checkHR(HRESULT hr, const std::string& fmt)
{
    if (FAILED(hr))
    {
        LogMessage("{} failed: ({:#X}) {}", fmt, hr, std::system_category().message(hr));
        return false;
    }
    return true;
}

inline bool checkCUDA(cudaError_t err, const std::string& fmt)
{
    if (err != cudaSuccess)
    {
        LogMessage("{} failed: ({}) {}", fmt, int32_t(err), cudaGetErrorString(err));
        return false;
    }
    return true;
}

inline uint32_t __num_set_bits(uint32_t bitmask) {
    return __popcnt(bitmask);
}

struct TimingInfo {
    std::chrono::high_resolution_clock::time_point last_run_time;
    double accum_diff_between_run_times;
    int32_t num_iterations;
};

// Number of iterations to accumulate before reporting an average time.
static constexpr int32_t TIMING_NUM_ITERATIONS = 100;

// Identity printed alongside a timing line, e.g. [thread=3] or [robot=0 stream=1].
// Up to two key=value pairs; leave a key null to omit it.
struct TimingId {
    const char* key_a = nullptr;
    int32_t     val_a = 0;
    const char* key_b = nullptr;
    int32_t     val_b = 0;
};

// Build the canonical identity suffix: " [key_a=val_a key_b=val_b]" (omitting null keys).
inline std::string __formatTimingId(const TimingId& id) {
    std::string s;
    if (id.key_a) s += std::format("{}={}", id.key_a, id.val_a);
    if (id.key_b) { if (!s.empty()) s += ' '; s += std::format("{}={}", id.key_b, id.val_b); }
    return s.empty() ? std::string{} : " [" + s + "]";
}

// Emit one canonical timing line and reset the accumulator. Routed through LogPerf so it is
// gated by LogLevel::PERF. Single-threaded into the library, so no synchronization.
inline void __reportTiming(TimingInfo& t, const char* name, const TimingId& id) {
    LogPerf("[timing] {}{}: {} ms (over {} iters)",
            name,
            __formatTimingId(id),
            t.accum_diff_between_run_times / t.num_iterations,
            t.num_iterations);
}

// Accumulate the gap since the previous call (inter-iteration cadence). The first call only
// seeds the clock; every TIMING_NUM_ITERATIONS thereafter it prints the mean and resets.
inline void accumTimingInterval(TimingInfo& t, const char* name, const TimingId& id) {
    auto now = std::chrono::high_resolution_clock::now();
    if (t.last_run_time.time_since_epoch().count() == 0) {
        t.last_run_time = now;  // first call: seed, don't count the gap
        return;
    }
    t.accum_diff_between_run_times += std::chrono::duration<double, std::milli>(now - t.last_run_time).count();
    t.last_run_time = now;
    if (++t.num_iterations == TIMING_NUM_ITERATIONS) {
        __reportTiming(t, name, id);
        t = {now, 0.0, 0};  // keep last_run_time so the next interval is continuous
    }
}

// Accumulate a self-contained per-iteration duration (e.g. active work, a copy span, a fence
// wait). Prints the mean and resets every TIMING_NUM_ITERATIONS.
inline void accumTimingSample(TimingInfo& t, double sample_ms, const char* name, const TimingId& id) {
    t.accum_diff_between_run_times += sample_ms;
    if (++t.num_iterations == TIMING_NUM_ITERATIONS) {
        __reportTiming(t, name, id);
        t = {{}, 0.0, 0};
    }
}

}