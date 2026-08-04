//
// Standalone check for wallAlignGPU() on real exported capture data -- the
// same input file four-pcs-test.cpp / four-pcs-gpu-test.cpp use (see
// PySpotObserver/pyspotobserver/tools/export_four_pcs_test_data.py).
//
// KNOWN-GOOD REFERENCE (Aug 4 capture, validated visually + by ICP
// convergence in the Python prototype): yaw ~= 35 deg with, after ICP,
//   R ~ [[ 0.8137 -0.5812 -0.0074]
//        [ 0.5811  0.8132  0.0316]
//        [-0.0123 -0.0300  0.9995]]
//   t ~ [ 0.297 -1.296 -0.011 ]
// and a ~36% overall inlier fraction (non-floor fraction will differ; the
// point is it should be WELL above the ~0.10 garbage line). Exact numbers
// will vary with capture data -- if run against a different export, judge
// by inlier_fraction and by rendering the result, not these constants.
//
// Usage: wall_align_gpu_test [path-to-exported-data.txt]
//

#define _USE_MATH_DEFINES
#include "four-pcs.h"       // SOb::PointCloud (loader convenience only)
#include "wall-align-gpu.cuh"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

double rotationAngleDegrees(const Eigen::Matrix3d& R) {
    double trace = R.trace();
    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) / 2.0));
    return std::acos(cos_angle) * 180.0 / M_PI;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_path = argc > 1 ? argv[1]
        : "PySpotObserver/pyspotobserver/tools/captures/four_pcs_test_data.txt";

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
    double nx, ny, nz, offset;
    f >> nx >> ny >> nz >> offset;  // exported plane -- unused; wallAlignGPU fits its own

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
        std::cerr << "[FAIL] CUDA setup failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    SOb::WallAlignResult result = SOb::wallAlignGPU(d_source, d_target);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    cudaFree(d_source.d_points);
    cudaFree(d_target.d_points);

    if (result.best.score < 0) {
        std::cerr << "[FAIL] wallAlignGPU produced no candidate at all\n";
        return 1;
    }

    Eigen::IOFormat rowFmt(Eigen::StreamPrecision, 0, ", ", "\n", "  [", "]");
    std::cout << "\nBest: yaw=" << result.best_yaw_deg << " deg (of "
              << result.candidates_tried << " wall-consistent candidates)\n"
              << "rotation angle=" << rotationAngleDegrees(result.best.rotation) << " deg, "
              << "translation=" << result.best.translation.norm() << " m\n"
              << "rotation:\n" << result.best.rotation.format(rowFmt) << "\n"
              << "translation: [" << result.best.translation.transpose() << "]\n"
              << "score=" << result.best.score
              << " inlier_fraction=" << result.inlier_fraction << "\n"
              << "elapsed=" << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    // Soft gate: a real alignment on this project's captures scores well
    // above 0.10 non-floor inlier fraction after refinement. Below that,
    // treat as failure -- matches the empirical garbage line from the
    // Python-side session work.
    if (result.inlier_fraction < 0.10) {
        std::cerr << "[FAIL] inlier fraction " << result.inlier_fraction
                  << " below the 0.10 quality gate -- alignment not trustworthy\n";
        return 1;
    }

    std::cout << "[PASS] wall-align-gpu-test\n";
    return 0;
}
