//
// Created by brown on 7/23/2025.
//

#pragma once

#include "logger.h"
#include <cuda_runtime.h>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace SOb {

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

}