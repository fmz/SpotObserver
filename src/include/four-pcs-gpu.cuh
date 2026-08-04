//
// GPU port of four-pcs.h/.cpp. The CPU/Eigen version stays the reference
// implementation -- this mirrors its control flow and math as closely as
// possible, just with the expensive per-iteration searches (which touch
// thousands of points) offloaded to CUDA kernels instead of looping on one
// core. Small, cheap steps (Kabsch on 4 points, the 3x2 diagonal-pairing
// solve) intentionally stay on the host, reusing the already-validated CPU
// code where possible -- there's nothing to gain from parallelizing math
// that's already microseconds.
//
// REPRODUCIBILITY NOTE (same caveat as four-pcs.h and the C# port): the
// per-thread RNG used here (a splitmix64 variant seeded per-thread) is a
// third, independent random sequence from numpy's default_rng and
// std::mt19937_64. A "matching" seed does not reproduce the same draws
// across any of these. Validate by result quality, not exact numbers.

#pragma once

#include "four-pcs.h"  // reuses SOb::RegistrationResult and SOb::kabsch

#include <cuda_runtime.h>

namespace SOb {

// A point cloud living in GPU memory. One float3 per point, nothing fancier.
struct DevicePointCloud {
    float3* d_points = nullptr;
    int count = 0;
};

// A uniform grid over a DevicePointCloud, used to quickly find "which points
// are near this location" without checking every single point. Points are
// grouped into cubic cells; d_sorted_keys / d_sorted_indices hold every
// point's cell key and original index, sorted together by key, so all points
// in the same cell end up next to each other in the array. GPU analog of the
// CPU SpatialHashGrid class in four-pcs.cpp.
struct DeviceSpatialGrid {
    float cell_size = 0.0f;
    long long* d_sorted_keys = nullptr;
    int* d_sorted_indices = nullptr;
    const float3* d_points = nullptr;  // not owned -- the cloud this grid indexes
    int point_count = 0;
};

void freeDeviceSpatialGrid(DeviceSpatialGrid& grid);

// Stage 1: turn a depth image already sitting in GPU memory into a 3D point
// cloud, also in GPU memory -- no round trip through the CPU. Mirrors
// backproject() in pyspotobserver. Allocates out.d_points (caller frees with
// cudaFree) and fills out.count with the number of valid (depth > 0) pixels.
cudaError_t backprojectDepthToPoints(
    const float* d_depth, int width, int height,
    float cx, float cy, float fx, float fy,
    DevicePointCloud& out, cudaStream_t stream = 0);

// Stage 2: group a point cloud's points into grid cells sized to cell_size,
// so nearby-point searches only have to check a handful of cells instead of
// every point. Allocates grid buffers (caller frees with
// freeDeviceSpatialGrid). cloud must outlive the returned grid.
cudaError_t buildSpatialGrid(
    const DevicePointCloud& cloud, float cell_size,
    DeviceSpatialGrid& out, cudaStream_t stream = 0);

// Stage 3: RANSAC dominant-plane fit, same idea as fitDominantPlane() in
// four-pcs.h, running on the GPU.
struct GpuPlaneFit {
    bool valid = false;
    float3 normal = {0, 0, 0};
    float offset = 0.0f;
    int inlier_count = -1;
};
cudaError_t fitDominantPlaneGPU(
    const DevicePointCloud& cloud, int num_hypotheses,
    float threshold, unsigned long long seed,
    GpuPlaneFit& out, cudaStream_t stream = 0);

// Top-level entry point: same job as SOb::fourPointCongruentSets() in
// four-pcs.h, but with the expensive per-iteration searches running on the
// GPU. Signature intentionally mirrors the CPU version (float instead of
// double, float3 instead of Eigen::Vector3d, otherwise identical parameters
// and identical meaning). source and target must already be in GPU memory
// (e.g. via backprojectDepthToPoints).
//
// iterations/max_spread/seed defaults below (70/5.0/5) are tuned against
// real capture data's known ground-truth transform (see
// four_pcs_gpu_param_sweep in tests/standalone-tests) -- not a universal
// best, just validated for that data.
RegistrationResult fourPointCongruentSetsGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    int iterations = 70, float max_distance = 0.1f,
    float min_spread = 0.3f, float max_spread = 5.0f, float coplanar_tol = 0.05f,
    float distance_tol = 0.03f, float e_tol = 0.05f, unsigned long long seed = 5,
    const float3* dominant_plane_normal = nullptr,
    float dominant_plane_offset = 0.0f,
    float plane_reject_thresh = 0.04f, float plane_reject_angle_cos = 0.94f,
    cudaStream_t stream = 0);

}  // namespace SOb
