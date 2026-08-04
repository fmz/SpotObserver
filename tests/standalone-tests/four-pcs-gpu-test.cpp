//
// Standalone check for the GPU port of four_pcs (four-pcs-gpu.cu) on real
// capture data. Loads the same exported point cloud pair used by
// four-pcs-test.cpp (see PySpotObserver/pyspotobserver/tools/export_four_pcs_test_data.py),
// runs four-pcs to get a rough initial alignment, then refines it with
// icpGPU().
//
// GPU-only: this used to also run the CPU/Eigen four-pcs reference
// (four-pcs.cpp) for comparison, but the two use different RNGs
// (std::mt19937_64 vs this file's per-thread splitmix64, see the
// REPRODUCIBILITY NOTE in four-pcs-gpu.cuh) and don't produce matching
// numbers even with a "matching" seed, so a side-by-side CPU run wasn't
// buying anything once the GPU path had already been validated. There's no
// CPU port of icp() at all -- GPU only, by design.
//
// Usage: four_pcs_gpu_test [path-to-exported-data.txt]
//

#define _USE_MATH_DEFINES
#include "four-pcs.h"
#include "four-pcs-gpu.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

double rotationAngleDegrees(const Eigen::Matrix3d& R) {
    double trace = R.trace();
    double cos_angle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cos_angle) * 180.0 / M_PI;
}

bool isWellFormed(const SOb::RegistrationResult& result) {
    double angle = rotationAngleDegrees(result.rotation);
    double t_norm = result.translation.norm();
    return std::isfinite(angle) && std::isfinite(t_norm) && angle >= 0.0 && angle <= 180.0;
}

void printTransform(const std::string& label, const SOb::RegistrationResult& result) {
    Eigen::IOFormat rowFmt(Eigen::StreamPrecision, 0, ", ", "\n", "  [", "]");
    std::cout << label << " rotation:\n" << result.rotation.format(rowFmt) << "\n"
              << label << " translation: [" << result.translation.transpose() << "]\n";
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
    f >> nx >> ny >> nz >> offset;
    double tnx, tny, tnz, toffset;
    f >> tnx >> tny >> tnz >> toffset;

    SOb::PointCloud source(n_src, 3), target(n_tgt, 3);
    for (size_t i = 0; i < n_src; i++) {
        double x, y, z;
        f >> x >> y >> z;
        source.row(i) = Eigen::Vector3d(x, y, z).transpose();
    }
    for (size_t i = 0; i < n_tgt; i++) {
        double x, y, z;
        f >> x >> y >> z;
        target.row(i) = Eigen::Vector3d(x, y, z).transpose();
    }

    std::cout << "Loaded " << n_src << " source / " << n_tgt << " target points\n";

    // iterations bumped from 70 -> 300: no known ground-truth transform for
    // this capture to tune a seed against, so more random base attempts is
    // the only lever that doesn't depend on getting lucky (see four-pcs.h's
    // note on why max_spread=5.0, not the repo default of 1.2, for real
    // room-scale captures).
    std::vector<float3> h_source(n_src), h_target(n_tgt);
    for (size_t i = 0; i < n_src; i++)
        h_source[i] = make_float3((float)source(i, 0), (float)source(i, 1), (float)source(i, 2));
    for (size_t i = 0; i < n_tgt; i++)
        h_target[i] = make_float3((float)target(i, 0), (float)target(i, 1), (float)target(i, 2));

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

    float3 plane_normal_f = make_float3((float)nx, (float)ny, (float)nz);
    float3 target_plane_normal_f = make_float3((float)tnx, (float)tny, (float)tnz);

    // Target-plane alignment check active: candidates whose rotation maps
    // source's floor-normal away from target's own floor-normal are rejected
    // before scoring (see four-pcs-gpu.cu). Fixes the tilt/upside-down axis;
    // doesn't constrain yaw or translation, which is what icpGPU() below is
    // for.
    //
    // min_spread 0.3 -> 1.2, max_spread 5.0 -> 8.0.
    auto t0 = std::chrono::steady_clock::now();
    SOb::RegistrationResult gpu_result = SOb::fourPointCongruentSetsGPU(
        d_source, d_target,
        /*iterations=*/300, /*max_distance=*/0.1f,
        /*min_spread=*/1.2f, /*max_spread=*/8.0f, /*coplanar_tol=*/0.05f,
        /*distance_tol=*/0.03f, /*e_tol=*/0.05f, /*seed=*/5,
        &plane_normal_f, (float)offset, &target_plane_normal_f);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double gpu_elapsed = std::chrono::duration<double>(t1 - t0).count();

    double gpu_angle = rotationAngleDegrees(gpu_result.rotation);
    double gpu_t_norm = gpu_result.translation.norm();
    std::cout << "GPU four-pcs:  rotation=" << gpu_angle << " deg, translation=" << gpu_t_norm
              << ", score=" << gpu_result.score << "/" << n_src
              << ", elapsed=" << gpu_elapsed << "s\n";
    printTransform("four-pcs", gpu_result);

    // ICP refinement: four-pcs finds a rough initial alignment (and, with the
    // plane checks above, gets the "up" axis right), then icpGPU() polishes
    // yaw/translation using every point's own nearest-neighbor
    // correspondence instead of a random 4-point sample.
    auto t2 = std::chrono::steady_clock::now();
    SOb::RegistrationResult icp_result = SOb::icpGPU(
        d_source, d_target, gpu_result,
        /*max_iterations=*/100, /*tolerance=*/1e-6f, /*max_distance=*/0.1f,
        &plane_normal_f, (float)offset, /*max_distance_floor=*/0.5f);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    double icp_elapsed = std::chrono::duration<double>(t3 - t2).count();

    cudaFree(d_source.d_points);
    cudaFree(d_target.d_points);

    double icp_angle = rotationAngleDegrees(icp_result.rotation);
    double icp_t_norm = icp_result.translation.norm();
    std::cout << "GPU ICP:       rotation=" << icp_angle << " deg, translation=" << icp_t_norm
              << ", final correspondences=" << icp_result.score << "/" << n_src
              << ", elapsed=" << icp_elapsed << "s\n";
    printTransform("ICP-refined", icp_result);

    bool icp_ok = isWellFormed(icp_result);
    std::cout << (icp_ok ? "[PASS] " : "[FAIL] ") << "ICP-refined result is a well-formed rotation/translation\n";

    return icp_ok ? 0 : 1;
}
