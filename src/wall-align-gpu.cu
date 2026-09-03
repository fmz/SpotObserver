#include "wall-align-gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace SOb {

// Grid helpers duplicated with internal linkage: private copies per
// translation unit beat coordinating a shared header across files that get
// edited independently (the same reason four-pcs-gpu.cu keeps its own local
// copies rather than exporting them).
namespace {

constexpr float kPi = 3.14159265358979f;

__host__ __device__ inline float3 sub3f(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float dot3f(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float dist3f(float3 a, float3 b) {
    float3 d = sub3f(a, b);
    return sqrtf(dot3f(d, d));
}

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

__device__ inline void gridFindCellRange(const long long* sorted_keys, int n, long long key, int& lo, int& hi) {
    int l = 0, r = n;
    while (l < r) { int m = (l + r) >> 1; if (sorted_keys[m] < key) l = m + 1; else r = m; }
    lo = l;
    l = lo; r = n;
    while (l < r) { int m = (l + r) >> 1; if (sorted_keys[m] <= key) l = m + 1; else r = m; }
    hi = l;
}

// 12-float rigid transform POD (row-major R, then t), local copy as usual.
struct XForm { float r[9]; float t[3]; };

__device__ inline float3 applyXForm(const XForm& xf, float3 p) {
    return make_float3(
        xf.r[0] * p.x + xf.r[1] * p.y + xf.r[2] * p.z + xf.t[0],
        xf.r[3] * p.x + xf.r[4] * p.y + xf.r[5] * p.z + xf.t[1],
        xf.r[6] * p.x + xf.r[7] * p.y + xf.r[8] * p.z + xf.t[2]);
}

// ---- stage kernels -------------------------------------------------------

// aligned[i] = R * p[i] (rotation only -- the floor z-shift is applied
// separately once the inlier floor height is known).
__global__ void rotateKernel(const float3* d_in, int n, XForm xf, float3* d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_out[i] = applyXForm(xf, d_in[i]);
}

// mask[i] = 1 if point i lies within `threshold` of plane n.p + offset = 0.
__global__ void planeMaskKernel(
    const float3* d_points, int n, float3 normal, float offset, float threshold,
    unsigned char* d_mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dist = fabsf(dot3f(d_points[i], normal) + offset);
    d_mask[i] = (dist < threshold) ? 1 : 0;
}

// Accumulates (sum_z, count) over floor-inlier points of an aligned cloud so
// the host can place the floor at exactly z=0 regardless of the plane fit's
// normal/offset sign conventions.
__global__ void floorHeightKernel(
    const float3* d_aligned, const unsigned char* d_floor_mask, int n,
    float* d_sum_z, int* d_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !d_floor_mask[i]) return;
    atomicAdd(d_sum_z, d_aligned[i].z);
    atomicAdd(d_count, 1);
}

// shifts z in place by -floor_z, then marks structure points (inside the
// [z_min, z_max] band above the now-zeroed floor).
__global__ void shiftAndStructureMaskKernel(
    float3* d_aligned, int n, float floor_z, float z_min, float z_max,
    unsigned char* d_structure_mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_aligned[i].z -= floor_z;
    float z = d_aligned[i].z;
    d_structure_mask[i] = (z > z_min && z < z_max) ? 1 : 0;
}

// Local-PCA orientation histogram: for each structure point, fit the 2D (XY)
// covariance of its structure neighbors within `radius`; if the neighborhood
// is strongly linear (wall-like), vote its principal direction (mod 180 deg)
// into a shared 180-bin histogram, weighted by anisotropy. The 2x2
// eigenproblem is solved in closed form -- no Eigen anywhere near device
// code (see four-pcs-gpu-diagonal.h for why that matters).
__global__ void orientationHistKernel(
    const float3* d_aligned, const unsigned char* d_structure_mask, int n,
    const long long* d_grid_keys, const int* d_grid_indices, int grid_n, float grid_cell,
    float radius, float min_anisotropy,
    float* d_hist /* 180 bins */)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !d_structure_mask[i]) return;

    float3 p = d_aligned[i];
    int reach = (int)ceilf(radius / grid_cell) + 1;
    int qx, qy, qz;
    gridCellCoord(p, grid_cell, qx, qy, qz);

    // accumulate 2D moments over neighbors
    float sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    int count = 0;
    for (int dx = -reach; dx <= reach; dx++)
        for (int dy = -reach; dy <= reach; dy++)
            for (int dz = -reach; dz <= reach; dz++) {
                long long key = gridEncodeKey(qx + dx, qy + dy, qz + dz);
                int lo, hi;
                gridFindCellRange(d_grid_keys, grid_n, key, lo, hi);
                for (int k = lo; k < hi; k++) {
                    int j = d_grid_indices[k];
                    if (!d_structure_mask[j]) continue;
                    float3 q = d_aligned[j];
                    if (dist3f(p, q) > radius) continue;
                    sx += q.x; sy += q.y;
                    sxx += q.x * q.x; sxy += q.x * q.y; syy += q.y * q.y;
                    count++;
                }
            }
    if (count < 6) return;

    float inv = 1.0f / count;
    float mx = sx * inv, my = sy * inv;
    float a = sxx * inv - mx * mx;   // cov_xx
    float b = sxy * inv - mx * my;   // cov_xy
    float c = syy * inv - my * my;   // cov_yy

    // closed-form eigenvalues of [[a,b],[b,c]]
    float tr = a + c;
    float disc = sqrtf(fmaxf((a - c) * (a - c) + 4.0f * b * b, 0.0f));
    float lam_max = 0.5f * (tr + disc);
    float lam_min = 0.5f * (tr - disc);
    if (lam_max <= 1e-12f) return;

    float anis = 1.0f - lam_min / lam_max;  // 1 = perfectly linear
    if (anis < min_anisotropy) return;

    // principal direction angle, folded into [0, 180)
    float ang = 0.5f * atan2f(2.0f * b, a - c) * 180.0f / kPi;
    int bin = ((int)floorf(ang) % 180 + 180) % 180;
    atomicAdd(&d_hist[bin], anis);
}

// Dense translation voting for one yaw candidate: every (source structure
// point rotated by yaw, target structure point) pair whose heights agree
// within z_tol votes for the XY shift that would superimpose them. The
// argmax of the vote grid IS the best translation for this yaw -- same
// answer the prototype's FFT cross-correlation produced, computed directly
// (no cuFFT dependency), and O(n_src * n_tgt) is trivial at GPU speeds for
// coarse clouds.
__global__ void translationVoteKernel(
    const float3* d_src_aligned, const unsigned char* d_src_structure, int n_src,
    const float3* d_tgt_aligned, const unsigned char* d_tgt_structure, int n_tgt,
    float yaw_cos, float yaw_sin,
    float vote_cell, int vote_dim /* votes are vote_dim x vote_dim */, float vote_half_extent,
    float z_tol,
    unsigned int* d_votes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src || !d_src_structure[i]) return;

    float3 p = d_src_aligned[i];
    float rx = yaw_cos * p.x - yaw_sin * p.y;
    float ry = yaw_sin * p.x + yaw_cos * p.y;

    for (int j = 0; j < n_tgt; j++) {
        if (!d_tgt_structure[j]) continue;
        float3 q = d_tgt_aligned[j];
        if (fabsf(q.z - p.z) > z_tol) continue;
        float dx = q.x - rx, dy = q.y - ry;
        if (fabsf(dx) >= vote_half_extent || fabsf(dy) >= vote_half_extent) continue;
        int bi = (int)floorf((dx + vote_half_extent) / vote_cell);
        int bj = (int)floorf((dy + vote_half_extent) / vote_cell);
        if (bi < 0 || bi >= vote_dim || bj < 0 || bj >= vote_dim) continue;
        atomicAdd(&d_votes[bi * vote_dim + bj], 1u);
    }
}

// Scores a full-frame candidate transform: counts NON-FLOOR source points
// landing within max_distance of some target point (grid lookup over the
// original target cloud). Floor points are excluded from the count for the
// same reason fourPointCongruentSets excludes them when a plane is active --
// floor-on-floor matches say nothing about whether the alignment is right.
__global__ void scoreNonFloorKernel(
    const float3* d_source, const unsigned char* d_source_floor, int n_source,
    const long long* d_grid_keys, const int* d_grid_indices,
    const float3* d_target_points, int grid_n, float grid_cell,
    XForm xf, float max_distance,
    unsigned int* d_score)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_source || d_source_floor[i]) return;

    float3 tp = applyXForm(xf, d_source[i]);
    int reach = (int)ceilf(max_distance / grid_cell) + 1;
    int qx, qy, qz;
    gridCellCoord(tp, grid_cell, qx, qy, qz);

    for (int dx = -reach; dx <= reach; dx++)
        for (int dy = -reach; dy <= reach; dy++)
            for (int dz = -reach; dz <= reach; dz++) {
                long long key = gridEncodeKey(qx + dx, qy + dy, qz + dz);
                int lo, hi;
                gridFindCellRange(d_grid_keys, grid_n, key, lo, hi);
                for (int k = lo; k < hi; k++) {
                    if (dist3f(d_target_points[d_grid_indices[k]], tp) < max_distance) {
                        atomicAdd(d_score, 1u);
                        return;  // this source point is an inlier; next point
                    }
                }
            }
}

// ---- host-side helpers ---------------------------------------------------

// Rodrigues rotation taking unit vector n to +Z. Plain float math -- no
// Eigen needed for a formula this small, which also keeps this .cu free of
// anything nvcc might trip on.
void rotationToZ(const float3& normal, float R[9]) {
    float nx = normal.x, ny = normal.y, nz = normal.z;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= len; ny /= len; nz /= len;
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }  // point "up" first

    // axis = n x z = (ny, -nx, 0), sin = |axis|, cos = nz
    float vx = ny, vy = -nx;
    float s = sqrtf(vx * vx + vy * vy);
    float c = nz;
    if (s < 1e-9f) {  // already (anti)parallel to z; flipped above, so parallel
        R[0] = 1; R[1] = 0; R[2] = 0;
        R[3] = 0; R[4] = 1; R[5] = 0;
        R[6] = 0; R[7] = 0; R[8] = 1;
        return;
    }
    // R = I + [v]_x + [v]_x^2 * (1-c)/s^2 with v = (vx, vy, 0)
    float k = (1 - c) / (s * s);
    R[0] = 1 + k * (-vy * vy); R[1] = k * vx * vy;        R[2] = vy;
    R[3] = k * vx * vy;        R[4] = 1 + k * (-vx * vx); R[5] = -vx;
    R[6] = -vy;                R[7] = vx;                 R[8] = c;
}

void matMul3(const float A[9], const float B[9], float out[9]) {
    for (int r = 0; r < 3; r++)
        for (int col = 0; col < 3; col++) {
            float acc = 0;
            for (int k = 0; k < 3; k++) acc += A[r * 3 + k] * B[k * 3 + col];
            out[r * 3 + col] = acc;
        }
}

void matTVec3(const float A[9], const float v[3], float out[3]) {  // A^T * v
    for (int r = 0; r < 3; r++)
        out[r] = A[0 * 3 + r] * v[0] + A[1 * 3 + r] * v[1] + A[2 * 3 + r] * v[2];
}

// Constrained floor fit: RANSAC over host copies of the (coarse) cloud, but
// only accepting near-horizontal hypotheses (|unit normal z| >=
// min_normal_z). fitDominantPlaneGPU is deliberately NOT used here -- it
// maximizes inliers with no orientation constraint, and a large wall can
// out-count the floor on real captures (which happened, and silently broke
// the entire alignment). Host-side on purpose: the coarse clouds are ~10-20k
// points, so 300 hypotheses x 20k checks is microseconds -- not worth a
// kernel.
bool constrainedFloorFit(
    const std::vector<float3>& pts, int iterations, float threshold,
    float min_normal_z, unsigned long long seed,
    float3& out_normal, float& out_offset, int& out_inliers)
{
    if (pts.size() < 3) return false;
    // splitmix64 -- tiny, deterministic, no <random> engine state to drag around
    unsigned long long state = seed + 0x9E3779B97F4A7C15ULL;
    auto next = [&state]() {
        unsigned long long z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };

    int best_count = -1;
    for (int it = 0; it < iterations; it++) {
        size_t i0 = next() % pts.size(), i1 = next() % pts.size(), i2 = next() % pts.size();
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        const float3 &p0 = pts[i0], &p1 = pts[i1], &p2 = pts[i2];
        float ax = p1.x - p0.x, ay = p1.y - p0.y, az = p1.z - p0.z;
        float bx = p2.x - p0.x, by = p2.y - p0.y, bz = p2.z - p0.z;
        float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len < 1e-8f) continue;
        nx /= len; ny /= len; nz /= len;
        if (fabsf(nz) < min_normal_z) continue;  // not floor-like; skip before counting
        float offset = -(nx * p0.x + ny * p0.y + nz * p0.z);

        int count = 0;
        for (const float3& p : pts)
            if (fabsf(nx * p.x + ny * p.y + nz * p.z + offset) < threshold) count++;
        if (count > best_count) {
            best_count = count;
            out_normal = make_float3(nx, ny, nz);
            out_offset = offset;
        }
    }
    if (best_count < 3) return false;
    out_inliers = best_count;
    return true;
}

// Top-N peaks of a circular (mod 180) histogram with minimum separation,
// after light smoothing -- host-side port of the prototype's peaks_of().
std::vector<int> histogramPeaks(const std::vector<float>& hist, int n_peaks, int separation) {
    const int N = 180;
    std::vector<float> smooth(N, 0.0f);
    for (int i = 0; i < N; i++) {
        float acc = 0;
        for (int d = -5; d <= 5; d++) acc += hist[((i + d) % N + N) % N];
        smooth[i] = acc / 11.0f;
    }
    std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return smooth[a] > smooth[b]; });

    std::vector<int> peaks;
    for (int a : order) {
        bool far_enough = true;
        for (int p : peaks) {
            int d = std::abs(a - p);
            if (std::min(d, 180 - d) <= separation) { far_enough = false; break; }
        }
        if (far_enough) peaks.push_back(a);
        if ((int)peaks.size() >= n_peaks) break;
    }
    return peaks;
}

}  // namespace

// ---- public entry point --------------------------------------------------

WallAlignResult wallAlignGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    float structure_z_min, float structure_z_max,
    float orient_radius, float min_anisotropy,
    float vote_cell, float vote_half_extent, float vote_z_tol,
    int icp_iterations, float icp_max_distance,
    int plane_hypotheses, float plane_threshold,
    unsigned long long plane_seed,
    float min_floor_normal_z,
    cudaStream_t stream)
{
    WallAlignResult out;
    if (source.count == 0 || target.count == 0) return out;

    const int threads = 256;
    auto blocksFor = [&](int n) { return (n + threads - 1) / threads; };

    // ---- 1. floor plane per cloud (orientation-constrained -- see
    //         constrainedFloorFit above for why fitDominantPlaneGPU is not
    //         used here) ----
    std::vector<float3> h_source_pts(source.count), h_target_pts(target.count);
    if (cudaMemcpyAsync(h_source_pts.data(), source.d_points, source.count * sizeof(float3),
                         cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(h_target_pts.data(), target.d_points, target.count * sizeof(float3),
                         cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        std::cerr << "wallAlignGPU: cloud download for floor fit failed\n";
        return out;
    }

    GpuPlaneFit plane_src, plane_tgt;
    if (!constrainedFloorFit(h_source_pts, plane_hypotheses, plane_threshold, min_floor_normal_z,
                              plane_seed, plane_src.normal, plane_src.offset, plane_src.inlier_count) ||
        !constrainedFloorFit(h_target_pts, plane_hypotheses, plane_threshold, min_floor_normal_z,
                              plane_seed, plane_tgt.normal, plane_tgt.offset, plane_tgt.inlier_count)) {
        std::cerr << "wallAlignGPU: constrained floor fit failed -- no near-horizontal plane "
                     "found (is the floor visible in both clouds?)\n";
        return out;
    }
    plane_src.valid = plane_tgt.valid = true;

    std::cout << "  [wall-align] source floor: normal=(" << plane_src.normal.x << ","
              << plane_src.normal.y << "," << plane_src.normal.z << ") inliers="
              << plane_src.inlier_count << "/" << source.count << "\n"
              << "  [wall-align] target floor: normal=(" << plane_tgt.normal.x << ","
              << plane_tgt.normal.y << "," << plane_tgt.normal.z << ") inliers="
              << plane_tgt.inlier_count << "/" << target.count << "\n";

    // ---- 2. gravity-align both clouds (rotate floor normal to +Z) ----
    float R_A[9], R_B[9];
    rotationToZ(plane_src.normal, R_A);
    rotationToZ(plane_tgt.normal, R_B);

    struct AlignedCloud {
        float3* d_points = nullptr;
        unsigned char* d_floor = nullptr;      // floor-inlier mask (original frame)
        unsigned char* d_structure = nullptr;  // structure-band mask (aligned frame)
        float floor_z = 0.0f;
        int count = 0;
    };
    AlignedCloud a_src, a_tgt;
    a_src.count = source.count;
    a_tgt.count = target.count;

    auto setupAligned = [&](const DevicePointCloud& cloud, const GpuPlaneFit& plane,
                             const float R[9], AlignedCloud& a) -> bool {
        cudaError_t err = cudaMalloc(&a.d_points, cloud.count * sizeof(float3));
        if (err == cudaSuccess) err = cudaMalloc(&a.d_floor, cloud.count);
        if (err == cudaSuccess) err = cudaMalloc(&a.d_structure, cloud.count);
        if (err != cudaSuccess) return false;

        planeMaskKernel<<<blocksFor(cloud.count), threads, 0, stream>>>(
            cloud.d_points, cloud.count, plane.normal, plane.offset, plane_threshold, a.d_floor);

        XForm xf{};
        for (int i = 0; i < 9; i++) xf.r[i] = R[i];
        xf.t[0] = xf.t[1] = xf.t[2] = 0.0f;
        rotateKernel<<<blocksFor(cloud.count), threads, 0, stream>>>(
            cloud.d_points, cloud.count, xf, a.d_points);

        // floor height from the actual inliers (sign-convention-proof)
        float* d_sum = nullptr; int* d_cnt = nullptr;
        if (cudaMalloc(&d_sum, sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&d_cnt, sizeof(int)) != cudaSuccess) { cudaFree(d_sum); return false; }
        cudaMemsetAsync(d_sum, 0, sizeof(float), stream);
        cudaMemsetAsync(d_cnt, 0, sizeof(int), stream);
        floorHeightKernel<<<blocksFor(cloud.count), threads, 0, stream>>>(
            a.d_points, a.d_floor, cloud.count, d_sum, d_cnt);
        float h_sum = 0; int h_cnt = 0;
        cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_sum); cudaFree(d_cnt);
        if (h_cnt == 0) return false;
        a.floor_z = h_sum / h_cnt;

        shiftAndStructureMaskKernel<<<blocksFor(cloud.count), threads, 0, stream>>>(
            a.d_points, cloud.count, a.floor_z, structure_z_min, structure_z_max, a.d_structure);
        return true;
    };

    auto cleanupAligned = [](AlignedCloud& a) {
        if (a.d_points) cudaFree(a.d_points);
        if (a.d_floor) cudaFree(a.d_floor);
        if (a.d_structure) cudaFree(a.d_structure);
        a = AlignedCloud{};
    };

    if (!setupAligned(source, plane_src, R_A, a_src) || !setupAligned(target, plane_tgt, R_B, a_tgt)) {
        std::cerr << "wallAlignGPU: gravity alignment setup failed\n";
        cleanupAligned(a_src); cleanupAligned(a_tgt);
        return out;
    }

    // ---- 3. wall-direction histograms -> yaw candidates ----
    auto orientationHistogram = [&](AlignedCloud& a, std::vector<float>& hist_out) -> bool {
        DevicePointCloud aligned_cloud{a.d_points, a.count};
        DeviceSpatialGrid grid;
        if (buildSpatialGrid(aligned_cloud, orient_radius, grid, stream) != cudaSuccess) return false;

        float* d_hist = nullptr;
        if (cudaMalloc(&d_hist, 180 * sizeof(float)) != cudaSuccess) {
            freeDeviceSpatialGrid(grid);
            return false;
        }
        cudaMemsetAsync(d_hist, 0, 180 * sizeof(float), stream);
        orientationHistKernel<<<blocksFor(a.count), threads, 0, stream>>>(
            a.d_points, a.d_structure, a.count,
            grid.d_sorted_keys, grid.d_sorted_indices, grid.point_count, grid.cell_size,
            orient_radius, min_anisotropy, d_hist);
        hist_out.resize(180);
        cudaMemcpyAsync(hist_out.data(), d_hist, 180 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_hist);
        freeDeviceSpatialGrid(grid);
        return true;
    };

    std::vector<float> hist_src, hist_tgt;
    if (!orientationHistogram(a_src, hist_src) || !orientationHistogram(a_tgt, hist_tgt)) {
        std::cerr << "wallAlignGPU: orientation histogram failed\n";
        cleanupAligned(a_src); cleanupAligned(a_tgt);
        return out;
    }

    std::vector<int> peaks_src = histogramPeaks(hist_src, 2, 20);
    std::vector<int> peaks_tgt = histogramPeaks(hist_tgt, 2, 20);

    // yaws mapping some source wall direction onto some target wall
    // direction; each pairing is ambiguous mod 180, giving both options
    std::vector<int> yaw_candidates;
    for (int a : peaks_src)
        for (int b : peaks_tgt)
            for (int k : {0, 180}) {
                int yaw = ((b - a + k) % 360 + 360) % 360;
                bool dup = false;
                for (int y : yaw_candidates)
                    if (std::min(std::abs(y - yaw), 360 - std::abs(y - yaw)) <= 3) { dup = true; break; }
                if (!dup) yaw_candidates.push_back(yaw);
            }

    // ---- 4. per-candidate: translation vote -> full transform -> short ICP
    //         refinement -> non-floor inlier score; best wins ----
    const int vote_dim = (int)std::ceil(2.0f * vote_half_extent / vote_cell);
    unsigned int* d_votes = nullptr;
    unsigned int* d_score = nullptr;
    DeviceSpatialGrid target_grid;  // over the ORIGINAL target frame, for scoring
    float score_cell = std::max(icp_max_distance, 0.05f);
    bool setup_ok =
        cudaMalloc(&d_votes, vote_dim * vote_dim * sizeof(unsigned int)) == cudaSuccess &&
        cudaMalloc(&d_score, sizeof(unsigned int)) == cudaSuccess &&
        buildSpatialGrid(target, score_cell, target_grid, stream) == cudaSuccess;
    if (!setup_ok) {
        std::cerr << "wallAlignGPU: candidate-evaluation setup failed\n";
        if (d_votes) cudaFree(d_votes);
        if (d_score) cudaFree(d_score);
        cleanupAligned(a_src); cleanupAligned(a_tgt);
        return out;
    }

    std::vector<unsigned int> h_votes(vote_dim * vote_dim);
    out.candidates_tried = (int)yaw_candidates.size();

    for (int yaw : yaw_candidates) {
        float th = yaw * kPi / 180.0f;
        float c = std::cos(th), s = std::sin(th);

        cudaMemsetAsync(d_votes, 0, vote_dim * vote_dim * sizeof(unsigned int), stream);
        translationVoteKernel<<<blocksFor(a_src.count), threads, 0, stream>>>(
            a_src.d_points, a_src.d_structure, a_src.count,
            a_tgt.d_points, a_tgt.d_structure, a_tgt.count,
            c, s, vote_cell, vote_dim, vote_half_extent, vote_z_tol, d_votes);
        cudaMemcpyAsync(h_votes.data(), d_votes, vote_dim * vote_dim * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::cerr << "wallAlignGPU: translation vote failed at yaw " << yaw << "\n";
            continue;
        }

        int best_bin = (int)(std::max_element(h_votes.begin(), h_votes.end()) - h_votes.begin());
        float tx = (best_bin / vote_dim) * vote_cell - vote_half_extent + 0.5f * vote_cell;
        float ty = (best_bin % vote_dim) * vote_cell - vote_half_extent + 0.5f * vote_cell;

        // full-frame transform:
        //   q = R_B^T * ( Ryaw * (R_A p + [0,0,-fzA]) + [tx,ty,0] + [0,0,fzB] )
        // => R_full = R_B^T Ryaw R_A
        //    t_full = R_B^T (Ryaw [0,0,-fzA] + [tx, ty, fzB])
        float Ryaw[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
        float RyA[9], R_full_rowmajor[9], tmp[3], t_full[3];
        matMul3(Ryaw, R_A, RyA);
        // R_B^T * RyA: (B^T A)_{rc} = sum_k B_{kr} A_{kc}
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 3; col++) {
                float acc = 0;
                for (int k = 0; k < 3; k++) acc += R_B[k * 3 + r] * RyA[k * 3 + col];
                R_full_rowmajor[r * 3 + col] = acc;
            }
        tmp[0] = tx + Ryaw[2] * (-a_src.floor_z);
        tmp[1] = ty + Ryaw[5] * (-a_src.floor_z);
        tmp[2] = Ryaw[8] * (-a_src.floor_z) + a_tgt.floor_z;
        matTVec3(R_B, tmp, t_full);

        // short ICP refinement from this seed, via the icpGPU in
        // four-pcs-gpu.cu. The source floor plane is passed through so floor
        // correspondences get the looser max_distance_floor tolerance, same
        // idea as icp_align.py's floor_mask.
        RegistrationResult icp_seed;
        icp_seed.translation = Eigen::Vector3d(t_full[0], t_full[1], t_full[2]);
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 3; col++) icp_seed.rotation(r, col) = R_full_rowmajor[r * 3 + col];

        RegistrationResult refined = icpGPU(source, target, icp_seed,
                                             icp_iterations, 1e-6f, icp_max_distance,
                                             &plane_src.normal, plane_src.offset, 0.5f, stream);

        XForm xf{};
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 3; col++) xf.r[r * 3 + col] = (float)refined.rotation(r, col);
        xf.t[0] = (float)refined.translation(0);
        xf.t[1] = (float)refined.translation(1);
        xf.t[2] = (float)refined.translation(2);

        cudaMemsetAsync(d_score, 0, sizeof(unsigned int), stream);
        scoreNonFloorKernel<<<blocksFor(source.count), threads, 0, stream>>>(
            source.d_points, a_src.d_floor, source.count,
            target_grid.d_sorted_keys, target_grid.d_sorted_indices,
            target_grid.d_points, target_grid.point_count, target_grid.cell_size,
            xf, icp_max_distance, d_score);
        unsigned int h_score = 0;
        cudaMemcpyAsync(&h_score, d_score, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            std::cerr << "wallAlignGPU: scoring failed at yaw " << yaw << "\n";
            continue;
        }

        std::cout << "  [wall-align] yaw=" << yaw << " shift=(" << tx << "," << ty
                  << ") post-icp score=" << h_score << "\n";

        if ((long)h_score > out.best.score) {
            out.best.score = (long)h_score;
            out.best.rotation = refined.rotation;
            out.best.translation = refined.translation;
            out.best_yaw_deg = (float)yaw;
        }
    }

    // ---- 5. annealed final refinement of the winner ----
    // The per-candidate ICP above runs at a single, tight max_distance --
    // enough to rank candidates, but a tight radius can't correct residual
    // rotation: a few degrees of yaw error displaces far points by more
    // than max_distance (room-scale arm-length effect), so the very points
    // carrying the strongest rotation signal never form correspondences.
    // Re-refine the winner coarse-to-fine: a wide first radius recovers the
    // far correspondences, then tightening restores precision. Measured on
    // the Aug 4 capture: recovers ~4 deg of rotation single-radius ICP
    // leaves behind, with better inlier fractions at both 10cm and 5cm.
    if (out.best.score >= 0) {
        RegistrationResult annealed = out.best;
        for (float mult : {4.0f, 2.0f, 1.0f, 0.5f}) {
            annealed = icpGPU(source, target, annealed,
                               icp_iterations, 1e-6f, icp_max_distance * mult,
                               &plane_src.normal, plane_src.offset, 0.5f, stream);
        }

        XForm xf{};
        for (int r = 0; r < 3; r++)
            for (int col = 0; col < 3; col++) xf.r[r * 3 + col] = (float)annealed.rotation(r, col);
        xf.t[0] = (float)annealed.translation(0);
        xf.t[1] = (float)annealed.translation(1);
        xf.t[2] = (float)annealed.translation(2);

        cudaMemsetAsync(d_score, 0, sizeof(unsigned int), stream);
        scoreNonFloorKernel<<<blocksFor(source.count), threads, 0, stream>>>(
            source.d_points, a_src.d_floor, source.count,
            target_grid.d_sorted_keys, target_grid.d_sorted_indices,
            target_grid.d_points, target_grid.point_count, target_grid.cell_size,
            xf, icp_max_distance, d_score);
        unsigned int h_score = 0;
        cudaMemcpyAsync(&h_score, d_score, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) == cudaSuccess && (long)h_score >= out.best.score) {
            out.best.rotation = annealed.rotation;
            out.best.translation = annealed.translation;
            out.best.score = (long)h_score;
            std::cout << "  [wall-align] annealed refinement accepted, score=" << h_score << "\n";
        } else {
            std::cout << "  [wall-align] annealed refinement did not improve score; keeping single-radius result\n";
        }
    }

    // non-floor source point count for the fraction (host-side: count of
    // zeros in the floor mask)
    {
        std::vector<unsigned char> h_floor(source.count);
        cudaMemcpyAsync(h_floor.data(), a_src.d_floor, source.count, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        long non_floor = 0;
        for (unsigned char m : h_floor) non_floor += (m == 0);
        out.inlier_fraction = (non_floor > 0 && out.best.score > 0)
                                   ? (double)out.best.score / non_floor : 0.0;
    }

    std::cout << "  [wall-align] best: yaw=" << out.best_yaw_deg
              << " score=" << out.best.score
              << " inlier_fraction=" << out.inlier_fraction
              << (out.inlier_fraction < 0.10 && out.best.score >= 0
                      ? "  -- LOW: likely not a real alignment, consider recapturing"
                      : "")
              << "\n";

    cudaFree(d_votes);
    cudaFree(d_score);
    freeDeviceSpatialGrid(target_grid);
    cleanupAligned(a_src);
    cleanupAligned(a_tgt);
    return out;
}

}  // namespace SOb
