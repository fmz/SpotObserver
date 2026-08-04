//
// Host-only diagonal-pairing helper for the four-pcs GPU path. Split out of
// four-pcs-gpu.cu (and NOT compiled by nvcc) because Eigen's JacobiSVD is
// tagged EIGEN_DEVICE_FUNC: instantiating it from a .cu file makes nvcc try
// to generate device code for it, which fails deep inside Eigen's own
// PermutationMatrix implementation even though nothing here ever runs it on
// the GPU. Keeping it in a plain .cpp sidesteps that entirely.
//

#pragma once

#include <cuda_runtime.h>

namespace SOb {

struct HostDiagonalPairing {
    bool found = false;
    int order[4] = {0, 1, 2, 3};
    double ratio_a = 0, ratio_b = 0, diag_a = 0, diag_b = 0;
};

// Same least-squares diagonal-pairing solve as diagonalPairingAndRatios() in
// four-pcs.cpp, just taking float3 in and converting to Eigen::Vector3d at
// the boundary so the actual math is byte-identical to the CPU version.
HostDiagonalPairing diagonalPairingAndRatiosHost(const float3 base_points[4]);

}  // namespace SOb
