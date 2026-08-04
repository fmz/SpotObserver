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
//
// target_plane_normal: pass the TARGET cloud's own independently-fitted
// dominant-plane normal to reject any candidate whose rotation doesn't map
// source's floor-normal close to this direction, within
// plane_alignment_cos_thresh. Compared as a SIGNED dot product, not
// |cos(angle)| -- the caller must pre-orient both normals to a shared
// convention first (e.g. export_four_pcs_test_data.py's
// canonicalize_plane_sign(), which points each toward its own cloud's frame
// origin), otherwise this can't tell a correct match from a genuine
// upside-down flip, since both give the same |cos(angle)|. Both robots
// stand on the same real floor, so a genuinely correct transform shouldn't
// rotate "up" by ~90 degrees, let alone flip it entirely -- catches both a
// wall-matched-to-floor tilt and an upside-down flip before the expensive
// whole-cloud scoring pass. Only active when both dominant_plane_normal and
// target_plane_normal are non-null.
RegistrationResult fourPointCongruentSetsGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    int iterations = 150, float max_distance = 0.1f,
    float min_spread = 0.3f, float max_spread = 5.0f, float coplanar_tol = 0.05f,
    float distance_tol = 0.03f, float e_tol = 0.05f, unsigned long long seed = 5,
    const float3* dominant_plane_normal = nullptr,
    float dominant_plane_offset = 0.0f,
    const float3* target_plane_normal = nullptr,
    float plane_alignment_cos_thresh = 0.866f,
    float plane_reject_thresh = 0.04f, float plane_reject_angle_cos = 0.94f,
    cudaStream_t stream = 0);

// GPU port of icp() in icp_align.py -- refines an initial rigid alignment
// (e.g. fourPointCongruentSetsGPU()'s output) by iterating: transform every
// source point by the current estimate, find its nearest target point
// (correspondence), then re-fit rotation/translation from those
// correspondences with kabsch(), composing the result into a running total.
// Repeats until max_iterations or the per-iteration correction drops below
// tolerance.
//
// Not just four-pcs with smaller steps: unlike fourPointCongruentSetsGPU()'s
// 4-point random sampling, every source point participates every iteration,
// so this converges to a locally-precise fit instead of searching for a
// global correspondence -- meant to be handed an already-roughly-correct
// starting guess (four-pcs's job), not run from scratch.
//
// dominant_plane_normal / dominant_plane_offset: same floor_mask idea as
// icp() -- SOURCE points near this plane still participate every iteration,
// just matched against target with the looser max_distance_floor tolerance
// instead of max_distance, since floor correspondences are noisier. Pass
// nullptr (default) to use max_distance for every point uniformly.
//
// kabsch() itself is called directly (defined in four-pcs.cpp, a plain .cpp
// never compiled by nvcc) -- same pattern fourPointCongruentSetsGPU() already
// uses for its own 4-point fit, so no separate host-only split was needed
// here despite ICP's kabsch() call operating on potentially thousands of
// correspondences instead of exactly 4.
RegistrationResult icpGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    const RegistrationResult& initial,
    int max_iterations = 100, float tolerance = 1e-6f, float max_distance = 0.1f,
    const float3* dominant_plane_normal = nullptr,
    float dominant_plane_offset = 0.0f,
    float max_distance_floor = 0.5f,
    cudaStream_t stream = 0);

}  // namespace SOb
