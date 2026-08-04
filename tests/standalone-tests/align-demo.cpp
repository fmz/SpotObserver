//
// Live-demo alignment runner: loads an exported capture pair (same file
// format as four-pcs-gpu-test.cpp -- header, SOURCE plane line, TARGET
// plane line, then points), runs BOTH coarse-alignment pipelines --
//   A) fourPointCongruentSetsGPU + icpGPU  (the tuned 4PCS recipe)
//   B) wallAlignGPU                        (deterministic wall-constrained)
// -- scores both with the SAME metric (fraction of source points within
// max_distance of a target point; each pipeline's internal score is not
// comparable across pipelines), prints both, and writes the winner to a
// small machine-readable result file for run_demo.py to render.
//
// Usage: align_demo [exported-data.txt] [result-out.txt]
//

#define _USE_MATH_DEFINES
#include "four-pcs.h"
#include "four-pcs-gpu.cuh"
#include "wall-align-gpu.cuh"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

double rotationAngleDegrees(const Eigen::Matrix3d& R) {
    double trace = R.trace();
    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) / 2.0));
    return std::acos(cos_angle) * 180.0 / M_PI;
}

// Minimal host-side uniform grid for the shared scoring metric. Not
// performance-critical (runs twice on ~10-20k-point coarse clouds).
struct HostGrid {
    float cell;
    std::unordered_map<long long, std::vector<int>> cells;
    const std::vector<float3>* pts;

    static long long key(int x, int y, int z) {
        const long long bias = 1LL << 19;
        return (((long long)x + bias) << 42) | (((long long)y + bias) << 21) | ((long long)z + bias);
    }
    void build(const std::vector<float3>& points, float cell_size) {
        cell = cell_size; pts = &points;
        cells.clear();
        for (int i = 0; i < (int)points.size(); i++) {
            const float3& p = points[i];
            cells[key((int)std::floor(p.x / cell), (int)std::floor(p.y / cell),
                       (int)std::floor(p.z / cell))].push_back(i);
        }
    }
    bool anyWithin(const float3& q, float max_d) const {
        int cx = (int)std::floor(q.x / cell), cy = (int)std::floor(q.y / cell),
            cz = (int)std::floor(q.z / cell);
        float md2 = max_d * max_d;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    auto it = cells.find(key(cx + dx, cy + dy, cz + dz));
                    if (it == cells.end()) continue;
                    for (int i : it->second) {
                        const float3& p = (*pts)[i];
                        float ex = p.x - q.x, ey = p.y - q.y, ez = p.z - q.z;
                        if (ex * ex + ey * ey + ez * ez < md2) return true;
                    }
                }
        return false;
    }
};

double inlierFraction(const std::vector<float3>& source, const HostGrid& target_grid,
                       const SOb::RegistrationResult& xf, float max_d) {
    long inliers = 0;
    for (const float3& p : source) {
        Eigen::Vector3d tp = xf.rotation * Eigen::Vector3d(p.x, p.y, p.z) + xf.translation;
        if (target_grid.anyWithin(make_float3((float)tp(0), (float)tp(1), (float)tp(2)), max_d))
            inliers++;
    }
    return source.empty() ? 0.0 : (double)inliers / source.size();
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_path = argc > 1 ? argv[1]
        : "PySpotObserver/pyspotobserver/tools/captures/four_pcs_test_data.txt";
    std::string result_path = argc > 2 ? argv[2] : "alignment_result.txt";

    std::ifstream f(data_path);
    if (!f) {
        std::cerr << "Could not open " << data_path << "\n"
                  << "Generate it first with:\n"
                  << "  python PySpotObserver/pyspotobserver/tools/export_four_pcs_test_data.py "
                  << data_path << "\n";
        return 1;
    }

    size_t n_src, n_tgt;
    f >> n_src >> n_tgt;
    double nx, ny, nz, offset;        // source dominant plane
    f >> nx >> ny >> nz >> offset;
    double tnx, tny, tnz, toffset;    // target dominant plane
    f >> tnx >> tny >> tnz >> toffset;

    std::vector<float3> h_source(n_src), h_target(n_tgt);
    for (size_t i = 0; i < n_src; i++) {
        double x, y, z; f >> x >> y >> z;
        h_source[i] = make_float3((float)x, (float)y, (float)z);
    }
    for (size_t i = 0; i < n_tgt; i++) {
        double x, y, z; f >> x >> y >> z;
        h_target[i] = make_float3((float)x, (float)y, (float)z);
    }
    std::cout << "Loaded " << n_src << " source / " << n_tgt << " target points\n";

    SOb::DevicePointCloud d_source{}, d_target{};
    d_source.count = (int)n_src;
    d_target.count = (int)n_tgt;
    cudaError_t err = cudaMalloc(&d_source.d_points, n_src * sizeof(float3));
    if (err == cudaSuccess) err = cudaMalloc(&d_target.d_points, n_tgt * sizeof(float3));
    if (err == cudaSuccess)
        err = cudaMemcpy(d_source.d_points, h_source.data(), n_src * sizeof(float3), cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_target.d_points, h_target.data(), n_tgt * sizeof(float3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA setup failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    const float kMaxDistance = 0.1f;
    HostGrid target_grid;
    target_grid.build(h_target, kMaxDistance);

    // ---- pipeline A: 4PCS + ICP (same recipe as four-pcs-gpu-test.cpp) ----
    float3 plane_normal_f = make_float3((float)nx, (float)ny, (float)nz);
    float3 target_plane_normal_f = make_float3((float)tnx, (float)tny, (float)tnz);

    auto tA0 = std::chrono::steady_clock::now();
    SOb::RegistrationResult coarse = SOb::fourPointCongruentSetsGPU(
        d_source, d_target,
        /*iterations=*/300, kMaxDistance,
        /*min_spread=*/1.2f, /*max_spread=*/8.0f, /*coplanar_tol=*/0.05f,
        /*distance_tol=*/0.03f, /*e_tol=*/0.05f, /*seed=*/5,
        &plane_normal_f, (float)offset, &target_plane_normal_f);
    SOb::RegistrationResult result_a = SOb::icpGPU(
        d_source, d_target, coarse,
        /*max_iterations=*/100, /*tolerance=*/1e-6f, kMaxDistance,
        &plane_normal_f, (float)offset, /*max_distance_floor=*/0.5f);
    cudaDeviceSynchronize();
    double elapsed_a = std::chrono::duration<double>(std::chrono::steady_clock::now() - tA0).count();
    double frac_a = inlierFraction(h_source, target_grid, result_a, kMaxDistance);

    // ---- pipeline B: wall-constrained alignment ----
    auto tB0 = std::chrono::steady_clock::now();
    SOb::WallAlignResult wall = SOb::wallAlignGPU(d_source, d_target);
    cudaDeviceSynchronize();
    double elapsed_b = std::chrono::duration<double>(std::chrono::steady_clock::now() - tB0).count();
    SOb::RegistrationResult result_b = wall.best;
    double frac_b = wall.best.score >= 0
                         ? inlierFraction(h_source, target_grid, result_b, kMaxDistance) : -1.0;

    cudaFree(d_source.d_points);
    cudaFree(d_target.d_points);

    std::cout << "\n=== comparison (shared metric: fraction of source points within "
              << kMaxDistance << "m of target) ===\n";
    std::cout << "A: 4PCS+ICP   fraction=" << frac_a
              << " rotation=" << rotationAngleDegrees(result_a.rotation) << " deg"
              << " |t|=" << result_a.translation.norm() << " elapsed=" << elapsed_a << "s\n";
    std::cout << "B: wall-align fraction=" << frac_b
              << " rotation=" << rotationAngleDegrees(result_b.rotation) << " deg"
              << " |t|=" << result_b.translation.norm() << " elapsed=" << elapsed_b << "s\n";

    bool use_a = frac_a >= frac_b;
    const SOb::RegistrationResult& winner = use_a ? result_a : result_b;
    double winner_frac = use_a ? frac_a : frac_b;
    std::cout << "winner: " << (use_a ? "4pcs_icp" : "wall_align")
              << " (fraction " << winner_frac << ")"
              << (winner_frac < 0.10 ? "  -- WARNING: below 0.10, likely NOT a real alignment" : "")
              << "\n";

    std::ofstream out(result_path);
    if (!out) {
        std::cerr << "Could not write " << result_path << "\n";
        return 1;
    }
    out.precision(10);
    out << "method " << (use_a ? "4pcs_icp" : "wall_align") << "\n";
    out << "fraction " << winner_frac << "\n";
    out << "R";
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) out << " " << winner.rotation(r, c);
    out << "\nt " << winner.translation(0) << " " << winner.translation(1) << " "
        << winner.translation(2) << "\n";
    out.close();
    std::cout << "wrote " << result_path << "\n";

    return winner_frac >= 0.10 ? 0 : 2;  // exit 2 = ran, but low-quality result
}
