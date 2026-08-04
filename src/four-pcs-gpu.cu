// Workaround for a known Eigen/nvcc incompatibility (libeigen/eigen#2690):
// under nvcc device compilation, Eigen's arg_default_impl expects a
// global-scope ::arg, which newer MSVC STL no longer exposes globally. Bring
// std::arg into the global namespace ourselves before Eigen (pulled in
// transitively via four-pcs-gpu.cuh -> four-pcs.h) gets included. Needed
// whenever Eigen types are touched in this file at all, not just for the
// JacobiSVD call that used to live here.
#include <complex>
using std::arg;

#include "four-pcs-gpu.cuh"
#include "four-pcs-gpu-diagonal.h"
#include "utils.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace SOb {

// ===========================================================================
// Small math helpers shared by host code and device kernels.
// ===========================================================================

__host__ __device__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline float norm3(float3 a) {
    return sqrtf(dot3(a, a));
}
__host__ __device__ inline float dist3(float3 a, float3 b) {
    return norm3(sub3(a, b));
}
__host__ __device__ inline float3 addScaled3(float3 a, float3 b, float s) {
    // a + s * b
    return make_float3(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
}

void freeDeviceSpatialGrid(DeviceSpatialGrid& grid) {
    if (grid.d_sorted_keys) cudaFree(grid.d_sorted_keys);
    if (grid.d_sorted_indices) cudaFree(grid.d_sorted_indices);
    grid = DeviceSpatialGrid{};
}

// ===========================================================================
// Grid cell math + lookup, shared by every kernel that needs "which points
// are near this location". Same encoding scheme as the CPU SpatialHashGrid,
// so a given cell size behaves the same on both sides.
// ===========================================================================

__host__ __device__ inline void gridCellCoord(float3 p, float cell_size, int& cx, int& cy, int& cz) {
    cx = (int)floorf(p.x / cell_size);
    cy = (int)floorf(p.y / cell_size);
    cz = (int)floorf(p.z / cell_size);
}

__host__ __device__ inline long long gridEncodeKey(int x, int y, int z) {
    const long long bias = 1LL << 19;
    long long ex = (long long)x + bias, ey = (long long)y + bias, ez = (long long)z + bias;
    return (ex << 42) | (ey << 21) | ez;
}

// Binary search for the [lo, hi) range of entries in a sorted key array that
// equal `key`. This is what stands in for the CPU version's hash map lookup
// -- instead of hashing straight to a bucket, we binary-search the sorted
// array, which is the natural fit once the points are already sorted by key
// for other reasons (building the grid in the first place).
__device__ inline void gridFindCellRange(const long long* sorted_keys, int n, long long key, int& lo, int& hi) {
    int l = 0, r = n;
    while (l < r) { int m = (l + r) >> 1; if (sorted_keys[m] < key) l = m + 1; else r = m; }
    lo = l;
    l = lo; r = n;
    while (l < r) { int m = (l + r) >> 1; if (sorted_keys[m] <= key) l = m + 1; else r = m; }
    hi = l;
}

// Nearest point index within max_distance, or -1 if none found. Mirrors
// SpatialHashGrid::nearestWithin() / cKDTree.query(distance_upper_bound=...).
__device__ inline int gridNearestWithin(
    const long long* sorted_keys, const int* sorted_indices, const float3* points, int n_points,
    float cell_size, float3 query, float max_distance)
{
    int reach = (int)ceilf(max_distance / cell_size) + 1;
    int qx, qy, qz;
    gridCellCoord(query, cell_size, qx, qy, qz);
    int best = -1;
    float best_d = max_distance;

    for (int dx = -reach; dx <= reach; dx++)
        for (int dy = -reach; dy <= reach; dy++)
            for (int dz = -reach; dz <= reach; dz++) {
                long long key = gridEncodeKey(qx + dx, qy + dy, qz + dz);
                int lo, hi;
                gridFindCellRange(sorted_keys, n_points, key, lo, hi);
                for (int k = lo; k < hi; k++) {
                    int idx = sorted_indices[k];
                    float d = dist3(points[idx], query);
                    if (d < best_d) { best_d = d; best = idx; }
                }
            }
    return best;
}

namespace {

__global__ void computeCellKeysKernel(const float3* d_points, int n, float cell_size,
                                       long long* d_keys_out, int* d_indices_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx, cy, cz;
    gridCellCoord(d_points[i], cell_size, cx, cy, cz);
    d_keys_out[i] = gridEncodeKey(cx, cy, cz);
    d_indices_out[i] = i;
}

}  // namespace

cudaError_t buildSpatialGrid(const DevicePointCloud& cloud, float cell_size,
                              DeviceSpatialGrid& out, cudaStream_t stream) {
    out = DeviceSpatialGrid{};
    out.cell_size = fmaxf(cell_size, 1e-6f);
    out.point_count = cloud.count;
    out.d_points = cloud.d_points;

    if (cloud.count == 0) return cudaSuccess;

    cudaError_t err = cudaMalloc(&out.d_sorted_keys, cloud.count * sizeof(long long));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&out.d_sorted_indices, cloud.count * sizeof(int));
    if (err != cudaSuccess) return err;

    int threads = 256;
    int blocks = (cloud.count + threads - 1) / threads;
    computeCellKeysKernel<<<blocks, threads, 0, stream>>>(
        cloud.d_points, cloud.count, out.cell_size, out.d_sorted_keys, out.d_sorted_indices);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Sort (key, original index) pairs by key -- this is what makes all the
    // points in one cell sit next to each other, so gridFindCellRange's
    // binary search works.
    thrust::device_ptr<long long> keys_ptr(out.d_sorted_keys);
    thrust::device_ptr<int> idx_ptr(out.d_sorted_indices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys_ptr, keys_ptr + cloud.count, idx_ptr);

    return cudaGetLastError();
}

// ===========================================================================
// Stage 1: backprojection (depth image -> 3D points)
// ===========================================================================

namespace {

__global__ void backprojectKernel(
    const float* d_depth, int width, int height,
    float cx, float cy, float fx, float fy,
    float3* d_points_out, int* d_write_index)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    float z = d_depth[v * width + u];
    if (z <= 0.0f) return;  // invalid / no-return pixel, same as the CPU/Python `depth > 0` mask

    float x = (u - cx) * z / fx;
    float y = (v - cy) * z / fy;

    int slot = atomicAdd(d_write_index, 1);
    d_points_out[slot] = make_float3(x, y, z);
}

}  // namespace

cudaError_t backprojectDepthToPoints(
    const float* d_depth, int width, int height,
    float cx, float cy, float fx, float fy,
    DevicePointCloud& out, cudaStream_t stream)
{
    out = DevicePointCloud{};
    int max_points = width * height;

    cudaError_t err = cudaMalloc(&out.d_points, (size_t)max_points * sizeof(float3));
    if (err != cudaSuccess) return err;

    int* d_count = nullptr;
    err = cudaMalloc(&d_count, sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    if (err != cudaSuccess) return err;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    backprojectKernel<<<grid, block, 0, stream>>>(d_depth, width, height, cx, cy, fx, fy, out.d_points, d_count);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_count); return err; }

    // Need the final count on the host before this function returns it in `out`,
    // so this is a real sync point -- fine here since backprojection happens
    // once per frame, not once per 4PCS iteration.
    err = cudaMemcpyAsync(&out.count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    cudaFree(d_count);
    return err;
}

// ===========================================================================
// Stage 3: dominant-plane RANSAC fit
// ===========================================================================

namespace {

struct PlaneHypothesis { int i0, i1, i2; };

// One block per hypothesis: thread 0 computes that hypothesis's plane, then
// every thread in the block checks a slice of the cloud against it and the
// block reduces to a single inlier count. Many hypotheses run as separate
// blocks at the same time, which is the GPU replacement for the CPU
// version's "try 300 hypotheses one after another" loop.
__global__ void countPlaneInliersKernel(
    const float3* d_points, int n,
    const PlaneHypothesis* d_hyps, float threshold,
    float3* d_normals_out, float* d_offsets_out, int* d_counts_out)
{
    int h = blockIdx.x;

    __shared__ float3 s_normal;
    __shared__ float s_offset;
    __shared__ bool s_degenerate;

    if (threadIdx.x == 0) {
        PlaneHypothesis hyp = d_hyps[h];
        float3 p0 = d_points[hyp.i0], p1 = d_points[hyp.i1], p2 = d_points[hyp.i2];
        float3 n = cross3(sub3(p1, p0), sub3(p2, p0));
        float len = norm3(n);
        if (len < 1e-8f) {
            s_degenerate = true;
        } else {
            s_degenerate = false;
            n = make_float3(n.x / len, n.y / len, n.z / len);
            s_normal = n;
            s_offset = -dot3(n, p0);
        }
    }
    __syncthreads();

    if (s_degenerate) {
        if (threadIdx.x == 0) d_counts_out[h] = -1;
        return;
    }

    int local_count = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dist = fabsf(dot3(s_normal, d_points[i]) + s_offset);
        if (dist < threshold) local_count++;
    }

    // Standard shared-memory tree reduction: fold the block's per-thread
    // counts in half repeatedly until thread 0 holds the total.
    extern __shared__ int s_counts[];
    s_counts[threadIdx.x] = local_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_counts[threadIdx.x] += s_counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_counts_out[h] = s_counts[0];
        d_normals_out[h] = s_normal;
        d_offsets_out[h] = s_offset;
    }
}

}  // namespace

cudaError_t fitDominantPlaneGPU(
    const DevicePointCloud& cloud, int num_hypotheses,
    float threshold, unsigned long long seed,
    GpuPlaneFit& out, cudaStream_t stream)
{
    out = GpuPlaneFit{};
    if (cloud.count < 3) return cudaSuccess;  // out.valid stays false

    // Hypotheses (which 3 points to try) are cheap to generate, so this
    // happens on the host with an ordinary RNG -- no need for device
    // randomness here, unlike the per-trial base search below.
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, cloud.count - 1);
    std::vector<PlaneHypothesis> hyps(num_hypotheses);
    for (auto& h : hyps) h = {dist(rng), dist(rng), dist(rng)};

    PlaneHypothesis* d_hyps = nullptr;
    float3* d_normals = nullptr;
    float* d_offsets = nullptr;
    int* d_counts = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_hyps, num_hypotheses * sizeof(PlaneHypothesis)); if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_normals, num_hypotheses * sizeof(float3)); if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_offsets, num_hypotheses * sizeof(float)); if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_counts, num_hypotheses * sizeof(int)); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(d_hyps, hyps.data(), num_hypotheses * sizeof(PlaneHypothesis),
                           cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto cleanup;

    {
        int threads = 256;
        countPlaneInliersKernel<<<num_hypotheses, threads, threads * sizeof(int), stream>>>(
            cloud.d_points, cloud.count, d_hyps, threshold, d_normals, d_offsets, d_counts);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    {
        std::vector<int> counts(num_hypotheses);
        std::vector<float3> normals(num_hypotheses);
        std::vector<float> offsets(num_hypotheses);
        err = cudaMemcpyAsync(counts.data(), d_counts, num_hypotheses * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(normals.data(), d_normals, num_hypotheses * sizeof(float3), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(offsets.data(), d_offsets, num_hypotheses * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;

        int best_h = -1, best_count = -1;
        for (int h = 0; h < num_hypotheses; h++) {
            if (counts[h] > best_count) { best_count = counts[h]; best_h = h; }
        }
        if (best_h >= 0 && best_count >= 0) {
            out.valid = true;
            out.normal = normals[best_h];
            out.offset = offsets[best_h];
            out.inlier_count = best_count;
        }
    }

cleanup:
    if (d_hyps) cudaFree(d_hyps);
    if (d_normals) cudaFree(d_normals);
    if (d_offsets) cudaFree(d_offsets);
    if (d_counts) cudaFree(d_counts);
    return err;
}

// ===========================================================================
// Stage 6: score a candidate rigid transform against the whole target cloud.
// Same "one thread per point, reduce to a count" shape as plane-fitting
// above, just transforming each point first and doing a grid lookup instead
// of a flat distance check.
// ===========================================================================

// Row-major 3x3 rotation + translation, passed into kernels by value (plain
// POD struct, no pointers -- avoids a separate host->device copy just to
// hand a kernel twelve numbers).
struct RigidTransform { float r[9]; float t[3]; };

namespace {

__global__ void scoreTransformKernel(
    const float3* d_source, int n_source,
    const long long* d_target_keys, const int* d_target_indices, const float3* d_target_points, int n_target,
    float cell_size, RigidTransform xf, float max_distance,
    const unsigned char* d_off_plane_mask,  // nullptr = every point counts
    int* d_block_sums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hit = 0;

    if (i < n_source) {
        float3 p = d_source[i];
        float3 tp = make_float3(
            xf.r[0] * p.x + xf.r[1] * p.y + xf.r[2] * p.z + xf.t[0],
            xf.r[3] * p.x + xf.r[4] * p.y + xf.r[5] * p.z + xf.t[1],
            xf.r[6] * p.x + xf.r[7] * p.y + xf.r[8] * p.z + xf.t[2]);

        bool counts = (d_off_plane_mask == nullptr) || (d_off_plane_mask[i] != 0);
        if (counts) {
            int nearest = gridNearestWithin(d_target_keys, d_target_indices, d_target_points, n_target,
                                             cell_size, tp, max_distance);
            if (nearest >= 0) hit = 1;
        }
    }

    extern __shared__ int s_hits[];
    s_hits[threadIdx.x] = hit;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_hits[threadIdx.x] += s_hits[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_block_sums[blockIdx.x] = s_hits[0];
}

__global__ void markOffPlaneKernel(
    const float3* d_points, int n, float3 normal, float offset, float threshold,
    unsigned char* d_off_plane_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dist = fabsf(dot3(normal, d_points[i]) + offset);
    d_off_plane_out[i] = (dist >= threshold) ? 1 : 0;  // "off-plane" = NOT within `threshold` of the plane
}

}  // namespace

// Sums a small per-block array on the host -- num_blocks is at most a few
// thousand even for large clouds, so this isn't worth its own kernel.
static cudaError_t scoreCandidateTransform(
    const DevicePointCloud& source, const DeviceSpatialGrid& target_grid,
    const RigidTransform& xf, float max_distance,
    const unsigned char* d_off_plane_mask,
    long& out_score, cudaStream_t stream)
{
    out_score = 0;
    if (source.count == 0) return cudaSuccess;

    int threads = 256;
    int blocks = (source.count + threads - 1) / threads;

    int* d_block_sums = nullptr;
    cudaError_t err = cudaMalloc(&d_block_sums, blocks * sizeof(int));
    if (err != cudaSuccess) return err;

    scoreTransformKernel<<<blocks, threads, threads * sizeof(int), stream>>>(
        source.d_points, source.count,
        target_grid.d_sorted_keys, target_grid.d_sorted_indices, target_grid.d_points, target_grid.point_count,
        target_grid.cell_size, xf, max_distance, d_off_plane_mask, d_block_sums);
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_block_sums); return err; }

    std::vector<int> block_sums(blocks);
    err = cudaMemcpyAsync(block_sums.data(), d_block_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    cudaFree(d_block_sums);
    if (err != cudaSuccess) return err;

    long total = 0;
    for (int s : block_sums) total += s;
    out_score = total;
    return cudaSuccess;
}

// ===========================================================================
// Tiny per-thread random number generator (splitmix64). Each GPU thread
// needs its own independent random draws, and passing every thread a slice
// of one shared sequence isn't practical -- this seeds a separate, cheap
// sequence per thread from a single seed + the thread's own id, so two
// different threads (almost certainly) never draw the same numbers. No new
// library dependency (e.g. curand) needed for something this simple.
//
// Stream construction follows splitmix64's own documented splitting method
// (as used by e.g. Java's SplittableRandom): stream i's starting state is
// seed + i*GOLDEN_GAMMA, plain integer addition, no XOR. That's the
// construction the algorithm was actually designed and tested around --
// every next() call advances state by the same GOLDEN_GAMMA before mixing,
// so giving each thread a starting state on that same arithmetic ladder
// just hands it a distinct, non-overlapping segment of one long sequence,
// which the avalanche mix then decorrelates. An earlier version combined
// seed and stream_id with XOR instead of addition; that isn't the
// documented construction and doesn't carry the same independence
// guarantee (XOR after the fact can partially cancel the additive stride
// structure the mix step relies on).
// ===========================================================================

struct GpuRng {
    unsigned long long state;
    __device__ explicit GpuRng(unsigned long long seed, unsigned long long stream_id) {
        state = seed + stream_id * 0x9E3779B97F4A7C15ULL;
    }
    __device__ unsigned long long next() {
        state += 0x9E3779B97F4A7C15ULL;
        unsigned long long z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Slight modulo bias for non-power-of-2 n, same as the CPU's rejection
    // approach is itself an approximation of "uniform" -- not worth a
    // rejection loop for a value this disposable.
    __device__ int nextInt(int n) { return (int)(next() % (unsigned long long)n); }
};

// ===========================================================================
// Stage 4: coplanar base search. The CPU version retries one random
// quadruple at a time, up to 200 times, until one satisfies the spread and
// coplanarity checks. Here, many threads each try exactly one quadruple at
// the same time -- parallel breadth standing in for the CPU's sequential
// depth. Whichever valid thread claims a slot first is returned; this is
// the same "return on first success" semantics as the CPU version, just
// arrived at differently.
// ===========================================================================

struct GpuCoplanarBaseOut {
    int idx[4];
    float3 pts[4];
};

namespace {

__global__ void searchCoplanarBasesKernel(
    const float3* d_points, int n,
    float min_spread, float max_spread, float coplanar_tol,
    unsigned long long seed, int max_found,
    GpuCoplanarBaseOut* d_found_out, int* d_found_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    GpuRng rng(seed, (unsigned long long)tid);

    int idx[4];
    idx[0] = rng.nextInt(n);
    idx[1] = rng.nextInt(n);
    idx[2] = rng.nextInt(n);
    idx[3] = rng.nextInt(n);
    if (idx[0] == idx[1] || idx[0] == idx[2] || idx[0] == idx[3] ||
        idx[1] == idx[2] || idx[1] == idx[3] || idx[2] == idx[3]) return;

    float3 pts[4];
    for (int k = 0; k < 4; k++) pts[k] = d_points[idx[k]];

    float min_dist = 3.0e38f, max_dist = 0.0f;
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++) {
            float d = dist3(pts[a], pts[b]);
            min_dist = fminf(min_dist, d);
            max_dist = fmaxf(max_dist, d);
        }
    if (min_dist < min_spread || max_dist > max_spread) return;

    // normal of the plane through the first three points
    float3 normal = cross3(sub3(pts[1], pts[0]), sub3(pts[2], pts[0]));
    float len = norm3(normal);
    if (len < 1e-6f) return;
    normal = make_float3(normal.x / len, normal.y / len, normal.z / len);

    // how far the fourth point sits off that plane
    float residual = fabsf(dot3(sub3(pts[3], pts[0]), normal));
    if (residual > coplanar_tol) return;

    int slot = atomicAdd(d_found_count, 1);
    if (slot < max_found) {
        GpuCoplanarBaseOut out;
        for (int k = 0; k < 4; k++) { out.idx[k] = idx[k]; out.pts[k] = pts[k]; }
        d_found_out[slot] = out;
    }
}

}  // namespace

// Launches num_trials independent attempts and returns the first one that
// satisfied every check (if any). found=false means none of the num_trials
// attempts worked -- same meaning as the CPU version exhausting its retries.
static bool searchCoplanarBasesGPU(
    const DevicePointCloud& cloud, float min_spread, float max_spread, float coplanar_tol,
    int num_trials, unsigned long long seed, GpuCoplanarBaseOut& out, cudaStream_t stream)
{
    if (cloud.count < 4) return false;

    const int max_found = 64;
    GpuCoplanarBaseOut* d_found = nullptr;
    int* d_found_count = nullptr;
    if (cudaMalloc(&d_found, max_found * sizeof(GpuCoplanarBaseOut)) != cudaSuccess) return false;
    if (cudaMalloc(&d_found_count, sizeof(int)) != cudaSuccess) { cudaFree(d_found); return false; }
    cudaMemsetAsync(d_found_count, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (num_trials + threads - 1) / threads;
    searchCoplanarBasesKernel<<<blocks, threads, 0, stream>>>(
        cloud.d_points, cloud.count, min_spread, max_spread, coplanar_tol, seed, max_found, d_found, d_found_count);

    int found_count = 0;
    bool ok = (cudaGetLastError() == cudaSuccess) &&
              (cudaMemcpyAsync(&found_count, d_found_count, sizeof(int), cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
              (cudaStreamSynchronize(stream) == cudaSuccess);

    if (ok && found_count > 0) {
        ok = cudaMemcpy(&out, d_found, sizeof(GpuCoplanarBaseOut), cudaMemcpyDeviceToHost) == cudaSuccess;
    } else {
        ok = false;
    }

    cudaFree(d_found);
    cudaFree(d_found_count);
    return ok;
}

// ===========================================================================
// Stage 5a: distance_pairs() equivalent -- every pair of points in a cloud
// whose distance falls within a tolerance band around a target distance.
// Builds its OWN grid sized to this call's specific distance, same as the
// CPU version -- reusing one fixed-cell grid across very different target
// distances was a real, measured performance bug earlier in this project
// (a small cell size searched at a large radius means checking tens of
// thousands of neighboring cells per point), so this rebuilds one sized
// correctly every time instead. Point clouds here are small (thousands, not
// hundreds of thousands), so rebuilding is cheap.
//
// Matches the CPU version's behavior when there are more matching pairs than
// max_pairs: a UNIFORM random subsample of all matches, not just whichever
// ones happened to be found first (which followed cell-sort order and so
// was spatially clustered -- a real bias that could exclude the true
// congruent set's diagonal pairs from the candidate pool entirely).
//
// Implemented via random-priority sampling: every (i, j) pair gets a
// deterministic pseudo-random 64-bit priority from a hash of its indices and
// a seed; keeping the max_pairs smallest priorities is exactly a uniform
// random max_pairs-subset of all matches, without replacement -- the same
// guarantee as the CPU's std::sample. distancePairsGPU runs the kernel
// twice: once to count total matches (no writes), and -- only if that count
// exceeds max_pairs -- again with Bernoulli thinning (keeping ~2*max_pairs
// candidates, comfortably below the allocated buffer) followed by a
// thrust::sort_by_key on priority to take the max_pairs smallest.
// ===========================================================================

struct IntPair { int a, b; };

// Deterministic pseudo-random priority for an (i, j) pair -- splitmix64-style
// finalizer over the packed indices XOR the seed. Keeping the N smallest
// priorities is exactly a uniform random N-subset of all pairs.
__device__ inline unsigned long long pairPriority(int i, int j, unsigned long long seed) {
    unsigned long long z = ((((unsigned long long)(unsigned int)i) << 32) | (unsigned int)j) ^ seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

namespace {

__global__ void pairsInRangeKernel(
    const long long* sorted_keys, const int* sorted_indices, const float3* points, int n,
    float cell_size, float target_distance, float tolerance,
    unsigned long long sample_seed, unsigned long long keep_threshold,
    IntPair* d_pairs_out,                  // nullptr = count-only pass
    unsigned long long* d_priorities_out,  // nullptr = don't record priorities
    int buffer_capacity, int* d_pair_count)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int i = sorted_indices[k];  // order doesn't matter, just need to visit every point once
    float3 pi = points[i];

    float max_d = target_distance + tolerance;
    float min_d = fmaxf(0.0f, target_distance - tolerance);
    int reach = (int)ceilf(max_d / cell_size) + 1;
    int cx, cy, cz;
    gridCellCoord(pi, cell_size, cx, cy, cz);

    for (int dx = -reach; dx <= reach; dx++)
        for (int dy = -reach; dy <= reach; dy++)
            for (int dz = -reach; dz <= reach; dz++) {
                long long key = gridEncodeKey(cx + dx, cy + dy, cz + dz);
                int lo, hi;
                gridFindCellRange(sorted_keys, n, key, lo, hi);
                for (int t = lo; t < hi; t++) {
                    int j = sorted_indices[t];
                    if (j <= i) continue;  // undirected: avoid duplicates and self-pairs
                    float d = dist3(pi, points[j]);
                    if (d < min_d || d > max_d) continue;
                    unsigned long long pri = pairPriority(i, j, sample_seed);
                    if (pri > keep_threshold) continue;
                    int slot = atomicAdd(d_pair_count, 1);
                    if (d_pairs_out != nullptr && slot < buffer_capacity) {
                        d_pairs_out[slot] = {i, j};
                        if (d_priorities_out != nullptr) d_priorities_out[slot] = pri;
                    }
                }
        }
}

}  // namespace

// Returns device-owned buffers (caller frees with cudaFree); *out_count is
// the number of valid entries actually written (<= max_pairs).
static cudaError_t distancePairsGPU(
    const DevicePointCloud& cloud, float target_distance, float tolerance, int max_pairs,
    unsigned long long sample_seed,
    IntPair*& d_pairs_out, int& out_count, cudaStream_t stream)
{
    d_pairs_out = nullptr;
    out_count = 0;
    if (cloud.count == 0) return cudaSuccess;

    DeviceSpatialGrid grid;
    cudaError_t err = buildSpatialGrid(cloud, fmaxf(target_distance + tolerance, 1e-3f), grid, stream);
    if (err != cudaSuccess) return err;

    int* d_count = nullptr;
    err = cudaMalloc(&d_count, sizeof(int));
    if (err != cudaSuccess) { freeDeviceSpatialGrid(grid); return err; }

    int threads = 256;
    int blocks = (cloud.count + threads - 1) / threads;
    const unsigned long long keep_all = ~0ULL;

    // Pass 1: count every pair in the distance band (no writes).
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    pairsInRangeKernel<<<blocks, threads, 0, stream>>>(
        grid.d_sorted_keys, grid.d_sorted_indices, grid.d_points, grid.point_count,
        grid.cell_size, target_distance, tolerance,
        sample_seed, keep_all, nullptr, nullptr, 0, d_count);
    int total = 0;
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaMemcpyAsync(&total, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess || total == 0) {
        freeDeviceSpatialGrid(grid);
        cudaFree(d_count);
        return err;
    }

    if (total <= max_pairs) {
        // Everything fits: keep every pair, no sampling involved.
        err = cudaMalloc(&d_pairs_out, total * sizeof(IntPair));
        if (err == cudaSuccess) {
            cudaMemsetAsync(d_count, 0, sizeof(int), stream);
            pairsInRangeKernel<<<blocks, threads, 0, stream>>>(
                grid.d_sorted_keys, grid.d_sorted_indices, grid.d_points, grid.point_count,
                grid.cell_size, target_distance, tolerance,
                sample_seed, keep_all, d_pairs_out, nullptr, total, d_count);
            err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err == cudaSuccess) out_count = total;
        }
    } else {
        // Uniform subsample. Threshold targets ~2*max_pairs survivors; the
        // buffer holds 3*max_pairs, ~15 standard deviations above that mean
        // (binomial with p = 2/3 over the buffer capacity), so a clipped
        // (order-biased) overflow is effectively impossible.
        int buffer_capacity = 3 * max_pairs;
        double keep_fraction = (2.0 * max_pairs) / (double)total;
        unsigned long long keep_threshold = keep_fraction >= 1.0
            ? keep_all
            : (unsigned long long)(keep_fraction * (double)keep_all);

        unsigned long long* d_priorities = nullptr;
        err = cudaMalloc(&d_pairs_out, buffer_capacity * sizeof(IntPair));
        if (err == cudaSuccess) err = cudaMalloc(&d_priorities, buffer_capacity * sizeof(unsigned long long));
        if (err == cudaSuccess) {
            cudaMemsetAsync(d_count, 0, sizeof(int), stream);
            pairsInRangeKernel<<<blocks, threads, 0, stream>>>(
                grid.d_sorted_keys, grid.d_sorted_indices, grid.d_points, grid.point_count,
                grid.cell_size, target_distance, tolerance,
                sample_seed, keep_threshold, d_pairs_out, d_priorities, buffer_capacity, d_count);
            int survivors = 0;
            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaMemcpyAsync(&survivors, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err == cudaSuccess) {
                if (survivors > buffer_capacity) survivors = buffer_capacity;
                if (survivors > max_pairs) {
                    // The max_pairs smallest priorities = a uniform random
                    // max_pairs-subset of all band pairs.
                    thrust::device_ptr<unsigned long long> pri_ptr(d_priorities);
                    thrust::device_ptr<IntPair> pair_ptr(d_pairs_out);
                    thrust::sort_by_key(thrust::cuda::par.on(stream),
                                         pri_ptr, pri_ptr + survivors, pair_ptr);
                    err = cudaStreamSynchronize(stream);
                    if (err == cudaSuccess) out_count = max_pairs;
                } else {
                    out_count = survivors;
                }
            }
        }
        if (d_priorities) cudaFree(d_priorities);
    }

    freeDeviceSpatialGrid(grid);
    cudaFree(d_count);
    if (err != cudaSuccess && d_pairs_out) { cudaFree(d_pairs_out); d_pairs_out = nullptr; out_count = 0; }
    return err;
}

// ===========================================================================
// Stage 5b: find_congruent() equivalent -- given a base's two diagonal
// lengths and crossing ratios, search the target cloud for every 4-point
// subset that shares them. Ties stages 4-5a together: get candidate point
// pairs at each diagonal's distance, compute where the crossing point would
// land for each pair (in both directions, since we don't know which end is
// which), then grid-match the two candidate sets against each other.
// ===========================================================================

struct CandidateQuad { int P, Q, R, S; };

namespace {

// Writes 2 output points per input pair (both directions along the pair),
// each paired with which original (a, b) indices produced it.
__global__ void computeCrossingPointsKernel(
    const IntPair* d_pairs, int n_pairs, const float3* d_target, float ratio,
    float3* d_crossing_out, int2* d_direction_out)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_pairs) return;
    IntPair pr = d_pairs[k];
    float3 P = d_target[pr.a], Q = d_target[pr.b];

    d_crossing_out[2 * k] = addScaled3(P, sub3(Q, P), ratio);
    d_direction_out[2 * k] = make_int2(pr.a, pr.b);

    d_crossing_out[2 * k + 1] = addScaled3(Q, sub3(P, Q), ratio);
    d_direction_out[2 * k + 1] = make_int2(pr.b, pr.a);
}

// Capped first-N in scan order (not a random subsample) -- this matches
// Python's find_congruent(), which also just breaks out of its candidate
// loop once n_checked > max_candidates (four_pcs.py:289). Unlike the
// distance-pairs cap below, this isn't a port divergence, so it's left as
// a plain atomic-counted truncation.
__global__ void matchCrossingPointsKernel(
    const long long* e1_sorted_keys, const int* e1_sorted_indices, const float3* e1_points, int n_e1,
    float cell_size, const int2* e1_directions,
    const float3* e2_points, const int2* e2_directions, int n_e2,
    float e_tolerance,
    CandidateQuad* d_candidates_out, int max_candidates, int* d_candidate_count)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_e2) return;
    float3 q = e2_points[k];

    int reach = (int)ceilf(e_tolerance / cell_size) + 1;
    int cx, cy, cz;
    gridCellCoord(q, cell_size, cx, cy, cz);

    for (int dx = -reach; dx <= reach; dx++)
        for (int dy = -reach; dy <= reach; dy++)
            for (int dz = -reach; dz <= reach; dz++) {
                long long key = gridEncodeKey(cx + dx, cy + dy, cz + dz);
                int lo, hi;
                gridFindCellRange(e1_sorted_keys, n_e1, key, lo, hi);
                for (int t = lo; t < hi; t++) {
                    int e1_idx = e1_sorted_indices[t];
                    if (dist3(e1_points[e1_idx], q) <= e_tolerance) {
                        int slot = atomicAdd(d_candidate_count, 1);
                        if (slot < max_candidates) {
                            int2 dirA = e1_directions[e1_idx];
                            int2 dirB = e2_directions[k];
                            d_candidates_out[slot] = {dirA.x, dirA.y, dirB.x, dirB.y};
                        }
                    }
                }
            }
}

}  // namespace

static std::vector<CandidateQuad> findCongruentGPU(
    const DevicePointCloud& target,
    float ratio_a, float ratio_b, float diag_a, float diag_b,
    float distance_tol, float e_tol, int max_pairs_per_distance, int max_candidates,
    unsigned long long seed, cudaStream_t stream)
{
    std::vector<CandidateQuad> result;

    // Distinct seeds per reservoir (diag_a pairs, diag_b pairs, final
    // candidate match) so the three subsamples aren't correlated draws.
    IntPair *d_pairs_a = nullptr, *d_pairs_b = nullptr;
    int n_pairs_a = 0, n_pairs_b = 0;
    if (distancePairsGPU(target, diag_a, distance_tol, max_pairs_per_distance, seed, d_pairs_a, n_pairs_a, stream) != cudaSuccess ||
        distancePairsGPU(target, diag_b, distance_tol, max_pairs_per_distance, seed ^ 0x9E3779B97F4A7C15ULL, d_pairs_b, n_pairs_b, stream) != cudaSuccess ||
        n_pairs_a == 0 || n_pairs_b == 0) {
        if (d_pairs_a) cudaFree(d_pairs_a);
        if (d_pairs_b) cudaFree(d_pairs_b);
        return result;
    }

    int n_e1 = 2 * n_pairs_a, n_e2 = 2 * n_pairs_b;
    float3 *d_e1 = nullptr, *d_e2 = nullptr;
    int2 *d_e1_dir = nullptr, *d_e2_dir = nullptr;
    cudaMalloc(&d_e1, n_e1 * sizeof(float3));
    cudaMalloc(&d_e1_dir, n_e1 * sizeof(int2));
    cudaMalloc(&d_e2, n_e2 * sizeof(float3));
    cudaMalloc(&d_e2_dir, n_e2 * sizeof(int2));

    int threads = 256;
    computeCrossingPointsKernel<<<(n_pairs_a + threads - 1) / threads, threads, 0, stream>>>(
        d_pairs_a, n_pairs_a, target.d_points, ratio_a, d_e1, d_e1_dir);
    computeCrossingPointsKernel<<<(n_pairs_b + threads - 1) / threads, threads, 0, stream>>>(
        d_pairs_b, n_pairs_b, target.d_points, ratio_b, d_e2, d_e2_dir);

    DevicePointCloud e1_cloud{d_e1, n_e1};
    DeviceSpatialGrid e1_grid;
    cudaError_t err = buildSpatialGrid(e1_cloud, fmaxf(e_tol, 1e-6f), e1_grid, stream);

    CandidateQuad* d_candidates = nullptr;
    int* d_candidate_count = nullptr;
    if (err == cudaSuccess) err = cudaMalloc(&d_candidates, max_candidates * sizeof(CandidateQuad));
    if (err == cudaSuccess) err = cudaMalloc(&d_candidate_count, sizeof(int));
    if (err == cudaSuccess) cudaMemsetAsync(d_candidate_count, 0, sizeof(int), stream);

    int candidate_count = 0;
    if (err == cudaSuccess) {
        matchCrossingPointsKernel<<<(n_e2 + threads - 1) / threads, threads, 0, stream>>>(
            e1_grid.d_sorted_keys, e1_grid.d_sorted_indices, d_e1, n_e1,
            e1_grid.cell_size, d_e1_dir, d_e2, d_e2_dir, n_e2, e_tol,
            d_candidates, max_candidates, d_candidate_count);
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaMemcpyAsync(&candidate_count, d_candidate_count, sizeof(int),
                                                        cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (candidate_count > max_candidates) candidate_count = max_candidates;
    }

    if (err == cudaSuccess && candidate_count > 0) {
        result.resize(candidate_count);
        cudaMemcpy(result.data(), d_candidates, candidate_count * sizeof(CandidateQuad), cudaMemcpyDeviceToHost);
    }

    freeDeviceSpatialGrid(e1_grid);
    cudaFree(d_pairs_a); cudaFree(d_pairs_b);
    cudaFree(d_e1); cudaFree(d_e1_dir); cudaFree(d_e2); cudaFree(d_e2_dir);
    if (d_candidates) cudaFree(d_candidates);
    if (d_candidate_count) cudaFree(d_candidate_count);
    return result;
}

// ===========================================================================
// Stage 7: host orchestration. Same control flow, same order of checks, and
// the same parameters as SOb::fourPointCongruentSets() in four-pcs.cpp --
// the difference is entirely in what each step costs: the base search,
// distance-pair search, and candidate scoring below run as GPU kernels
// against thousands of points at once, instead of a CPU loop over one point
// at a time. Kabsch itself (four points, in and out) stays exactly the CPU
// function, called directly.
// ===========================================================================

RegistrationResult fourPointCongruentSetsGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    int iterations, float max_distance,
    float min_spread, float max_spread, float coplanar_tol,
    float distance_tol, float e_tol, unsigned long long seed,
    const float3* dominant_plane_normal, float dominant_plane_offset,
    const float3* target_plane_normal, float plane_alignment_cos_thresh,
    float plane_reject_thresh, float plane_reject_angle_cos,
    cudaStream_t stream)
{
    RegistrationResult best;  // rotation = identity, translation = 0 (four-pcs.h defaults)
    long best_score = -1;

    // Built once, reused by every candidate's score check this whole call --
    // this is the CPU version's `target_grid` (sized for max_distance).
    DeviceSpatialGrid target_score_grid;
    if (buildSpatialGrid(target, fmaxf(max_distance, 0.05f), target_score_grid, stream) != cudaSuccess) {
        return best;
    }

    bool plane_active = dominant_plane_normal != nullptr;
    unsigned char* d_source_off_plane = nullptr;
    if (plane_active) {
        if (cudaMalloc(&d_source_off_plane, source.count * sizeof(unsigned char)) == cudaSuccess) {
            int threads = 256, blocks = (source.count + threads - 1) / threads;
            markOffPlaneKernel<<<blocks, threads, 0, stream>>>(
                source.d_points, source.count, *dominant_plane_normal, dominant_plane_offset,
                plane_reject_thresh, d_source_off_plane);
        }
    }

    // Host-side RNG used only to hand each GPU stage its own seed for this
    // iteration -- the actual random sampling happens on the device (per
    // GpuRng above), this just keeps every iteration's kernels from reusing
    // identical seeds.
    std::mt19937_64 host_rng(seed);

    for (int it = 0; it < iterations; it++) {
        unsigned long long base_seed = host_rng();

        GpuCoplanarBaseOut base;
        if (!searchCoplanarBasesGPU(source, min_spread, max_spread, coplanar_tol,
                                     /*num_trials=*/2048, base_seed, base, stream)) {
            continue;
        }

        if (plane_active) {
            float3 base_normal = cross3(sub3(base.pts[1], base.pts[0]), sub3(base.pts[2], base.pts[0]));
            float base_normal_len = norm3(base_normal);
            if (base_normal_len > 1e-8f) {
                base_normal = make_float3(base_normal.x / base_normal_len, base_normal.y / base_normal_len,
                                           base_normal.z / base_normal_len);
                float3 centroid = make_float3(
                    (base.pts[0].x + base.pts[1].x + base.pts[2].x + base.pts[3].x) * 0.25f,
                    (base.pts[0].y + base.pts[1].y + base.pts[2].y + base.pts[3].y) * 0.25f,
                    (base.pts[0].z + base.pts[1].z + base.pts[2].z + base.pts[3].z) * 0.25f);
                float cos_angle = fabsf(dot3(base_normal, *dominant_plane_normal));
                float centroid_dist = fabsf(dot3(centroid, *dominant_plane_normal) + dominant_plane_offset);

                // this base is itself just a patch of the dominant plane -- skip it,
                // same degeneracy check as the CPU version
                if (cos_angle > plane_reject_angle_cos && centroid_dist < plane_reject_thresh * 3) continue;
            }
        }

        HostDiagonalPairing pairing = diagonalPairingAndRatiosHost(base.pts);
        if (!pairing.found) continue;

        float3 ordered_base_points[4] = {
            base.pts[pairing.order[0]], base.pts[pairing.order[1]],
            base.pts[pairing.order[2]], base.pts[pairing.order[3]]
        };

        auto candidates = findCongruentGPU(
            target, (float)pairing.ratio_a, (float)pairing.ratio_b, (float)pairing.diag_a, (float)pairing.diag_b,
            distance_tol, e_tol, /*max_pairs_per_distance=*/2000, /*max_candidates=*/250,
            host_rng(), stream);

        if (candidates.empty()) continue;

        // Pulling the matched target points down to the host is cheap (4
        // points per candidate) -- Kabsch itself is a 3x3 SVD, not worth
        // running on the GPU regardless of cloud size.
        std::vector<float3> target_points_host;
        if (target.count > 0) {
            target_points_host.resize(target.count);
            cudaMemcpyAsync(target_points_host.data(), target.d_points, target.count * sizeof(float3),
                             cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        for (auto& candidate : candidates) {
            int idxs[4] = {candidate.P, candidate.Q, candidate.R, candidate.S};

            PointCloud base_pts(4, 3), matched_pts(4, 3);
            for (int k = 0; k < 4; k++) {
                float3 bp = ordered_base_points[k];
                base_pts.row(k) = Eigen::Vector3d(bp.x, bp.y, bp.z).transpose();
                float3 mp = target_points_host[idxs[k]];
                matched_pts.row(k) = Eigen::Vector3d(mp.x, mp.y, mp.z).transpose();
            }
            auto [rotation, translation] = kabsch(base_pts, matched_pts);

            // Both robots stand on the same real floor -- a genuinely correct
            // transform shouldn't rotate source's floor-normal far from
            // target's own, independently-fitted floor-normal. Signed dot
            // product (no fabs): both normals are pre-oriented toward their
            // own cloud's frame origin (see export_four_pcs_test_data.py's
            // canonicalize_plane_sign()), so a real match points the SAME
            // way, not just along the same axis -- catches an upside-down
            // flip, not only a 90-degree wall-to-floor tilt, before the
            // expensive whole-cloud scoring pass below.
            if (plane_active && target_plane_normal != nullptr) {
                Eigen::Vector3d src_n(dominant_plane_normal->x, dominant_plane_normal->y, dominant_plane_normal->z);
                Eigen::Vector3d tgt_n(target_plane_normal->x, target_plane_normal->y, target_plane_normal->z);
                Eigen::Vector3d rotated_normal = rotation * src_n;
                double cos_angle = rotated_normal.dot(tgt_n);
                if (cos_angle < plane_alignment_cos_thresh) continue;
            }

            RigidTransform xf;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    xf.r[r * 3 + c] = (float)rotation(r, c);
            xf.t[0] = (float)translation(0); xf.t[1] = (float)translation(1); xf.t[2] = (float)translation(2);

            long score = 0;
            scoreCandidateTransform(source, target_score_grid, xf, max_distance, d_source_off_plane, score, stream);

            if (score > best_score) {
                best_score = score;
                best.rotation = rotation;
                best.translation = translation;
                best.score = best_score;
            }
        }
    }

    freeDeviceSpatialGrid(target_score_grid);
    if (d_source_off_plane) cudaFree(d_source_off_plane);
    return best;
}

// ===========================================================================
// ICP refinement (GPU port of icp() in icp_align.py). Correspondence search
// (every source point against the whole target cloud) is the expensive part
// and runs as a GPU kernel; kabsch() itself stays on the host (see
// four-pcs-gpu.cuh's note on why that's safe here despite operating on
// potentially thousands of points instead of exactly 4).
// ===========================================================================

namespace {

// One thread per source point: transform it by the current running estimate,
// then look up its nearest target point (within max_distance, or
// max_distance_floor if it's a floor point per d_off_plane_mask) -- same
// grid-lookup building block as scoreTransformKernel above, but this writes
// out WHICH point matched, not just whether one did, since ICP needs the
// actual correspondences to re-fit with kabsch().
__global__ void findCorrespondencesKernel(
    const float3* d_source, int n_source,
    RigidTransform xf,
    const long long* d_target_keys, const int* d_target_indices, const float3* d_target_points, int n_target,
    float cell_size, float max_distance, float max_distance_floor,
    const unsigned char* d_off_plane_mask,  // nullptr = every point uses max_distance
    int* d_match_target_idx)  // -1 if no match within threshold
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_source) return;

    float3 p = d_source[i];
    float3 tp = make_float3(
        xf.r[0] * p.x + xf.r[1] * p.y + xf.r[2] * p.z + xf.t[0],
        xf.r[3] * p.x + xf.r[4] * p.y + xf.r[5] * p.z + xf.t[1],
        xf.r[6] * p.x + xf.r[7] * p.y + xf.r[8] * p.z + xf.t[2]);

    bool off_plane = (d_off_plane_mask == nullptr) || (d_off_plane_mask[i] != 0);
    float threshold = off_plane ? max_distance : max_distance_floor;

    d_match_target_idx[i] = gridNearestWithin(d_target_keys, d_target_indices, d_target_points, n_target,
                                               cell_size, tp, threshold);
}

}  // namespace

RegistrationResult icpGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    const RegistrationResult& initial,
    int max_iterations, float tolerance, float max_distance,
    const float3* dominant_plane_normal, float dominant_plane_offset,
    float max_distance_floor,
    cudaStream_t stream)
{
    RegistrationResult best = initial;
    if (source.count == 0 || target.count == 0) return best;

    // source/target don't change across iterations, only the running
    // transform does -- keep plain host copies instead of round-tripping
    // through the GPU every iteration just to build kabsch()'s input.
    std::vector<float3> h_source(source.count), h_target(target.count);
    cudaMemcpy(h_source.data(), source.d_points, source.count * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_target.data(), target.d_points, target.count * sizeof(float3), cudaMemcpyDeviceToHost);

    DeviceSpatialGrid target_grid;
    if (buildSpatialGrid(target, fmaxf(fmaxf(max_distance, max_distance_floor), 0.05f), target_grid, stream)
            != cudaSuccess) {
        return best;
    }

    bool plane_active = dominant_plane_normal != nullptr;
    unsigned char* d_source_off_plane = nullptr;
    if (plane_active) {
        if (cudaMalloc(&d_source_off_plane, source.count * sizeof(unsigned char)) == cudaSuccess) {
            int threads = 256, blocks = (source.count + threads - 1) / threads;
            // 0.04f matches fourPointCongruentSetsGPU's plane_reject_thresh
            // default -- "near enough to the fitted plane to count as floor".
            markOffPlaneKernel<<<blocks, threads, 0, stream>>>(
                source.d_points, source.count, *dominant_plane_normal, dominant_plane_offset,
                /*threshold=*/0.04f, d_source_off_plane);
        }
    }

    int* d_match_idx = nullptr;
    cudaMalloc(&d_match_idx, source.count * sizeof(int));
    std::vector<int> h_match_idx(source.count);

    Eigen::Matrix3d rotation_total = initial.rotation;
    Eigen::Vector3d translation_total = initial.translation;

    int threads = 256;
    int blocks = (source.count + threads - 1) / threads;

    for (int it = 0; it < max_iterations; it++) {
        RigidTransform xf;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                xf.r[r * 3 + c] = (float)rotation_total(r, c);
        xf.t[0] = (float)translation_total(0); xf.t[1] = (float)translation_total(1);
        xf.t[2] = (float)translation_total(2);

        findCorrespondencesKernel<<<blocks, threads, 0, stream>>>(
            source.d_points, source.count, xf,
            target_grid.d_sorted_keys, target_grid.d_sorted_indices, target_grid.d_points, target_grid.point_count,
            target_grid.cell_size, max_distance, max_distance_floor, d_source_off_plane, d_match_idx);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(h_match_idx.data(), d_match_idx, source.count * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;

        std::vector<Eigen::Vector3d> eval_pts_vec, matched_pts_vec;
        eval_pts_vec.reserve(source.count);
        matched_pts_vec.reserve(source.count);
        for (int i = 0; i < source.count; i++) {
            if (h_match_idx[i] < 0) continue;
            float3 p = h_source[i];
            Eigen::Vector3d transformed = rotation_total * Eigen::Vector3d(p.x, p.y, p.z) + translation_total;
            float3 mp = h_target[h_match_idx[i]];
            eval_pts_vec.push_back(transformed);
            matched_pts_vec.push_back(Eigen::Vector3d(mp.x, mp.y, mp.z));
        }

        if ((int)eval_pts_vec.size() < 3) break;  // not enough correspondences, same as the CPU icp()

        PointCloud eval_pts((long)eval_pts_vec.size(), 3), matched_pts((long)matched_pts_vec.size(), 3);
        for (size_t k = 0; k < eval_pts_vec.size(); k++) {
            eval_pts.row((long)k) = eval_pts_vec[k].transpose();
            matched_pts.row((long)k) = matched_pts_vec[k].transpose();
        }

        auto [rotation_vec, translation_vec] = kabsch(eval_pts, matched_pts);

        rotation_total = rotation_vec * rotation_total;
        translation_total = rotation_vec * translation_total + translation_vec;
        best.score = (long)eval_pts_vec.size();

        if (translation_vec.norm() < tolerance &&
            (rotation_vec - Eigen::Matrix3d::Identity()).norm() < tolerance) {
            break;
        }
    }

    best.rotation = rotation_total;
    best.translation = translation_total;

    cudaFree(d_match_idx);
    if (d_source_off_plane) cudaFree(d_source_off_plane);
    freeDeviceSpatialGrid(target_grid);

    return best;
}

}  // namespace SOb
