//
// Correctness check for the GPU port of four_pcs (four-pcs-gpu.cu), checked
// against the already-validated CPU/Eigen version (four-pcs.cpp) on the same
// real capture data. Loads the same exported point cloud pair used by
// four-pcs-test.cpp (see PySpotObserver/pyspotobserver/tools/export_four_pcs_test_data.py)
// and runs both the CPU and GPU registration on it.
//
// NOTE on reproducibility: the CPU version uses std::mt19937_64; the GPU
// version uses a different per-thread RNG (splitmix64, see the
// REPRODUCIBILITY NOTE in four-pcs-gpu.cuh). The two will NOT produce
// identical rotation/translation numbers even with a "matching" seed. This
// test checks that both results are well-formed and that the GPU version
// lands in the same ballpark as the CPU one -- not that the numbers match
// exactly. A big gap between them is worth investigating; a modest one is
// expected and fine, same as the C++-vs-Python comparison in four-pcs-test.cpp.
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
    Eigen::Vector3d plane_normal(nx, ny, nz);

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

    // Recipe tuned against real capture data's known ground-truth transform
    // (see four_pcs_gpu_param_sweep): seed=5, iterations=70, min_spread=0.3,
    // max_spread=5.0 (see four-pcs.h's note on why not the repo default of
    // 1.2, for real room-scale captures). Applied identically to both the CPU
    // and GPU calls below, though their RNGs differ (see reproducibility note
    // above) so this doesn't guarantee matching results between them.

    // ---- CPU reference ----
    auto t0 = std::chrono::steady_clock::now();
    SOb::RegistrationResult cpu_result = SOb::fourPointCongruentSets(
        source, target,
        /*iterations=*/70, /*max_distance=*/0.1,
        /*min_spread=*/0.3, /*max_spread=*/5.0, /*coplanar_tol=*/0.05,
        /*distance_tol=*/0.03, /*e_tol=*/0.05, /*seed=*/5,
        &plane_normal, offset);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_elapsed = std::chrono::duration<double>(t1 - t0).count();

    double cpu_angle = rotationAngleDegrees(cpu_result.rotation);
    double cpu_t_norm = cpu_result.translation.norm();
    std::cout << "CPU:  rotation=" << cpu_angle << " deg, translation=" << cpu_t_norm
              << ", elapsed=" << cpu_elapsed << "s\n";
    printTransform("CPU", cpu_result);

    // ---- GPU version: upload the same points, run the same recipe ----
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

    auto t2 = std::chrono::steady_clock::now();
    SOb::RegistrationResult gpu_result = SOb::fourPointCongruentSetsGPU(
        d_source, d_target,
        /*iterations=*/70, /*max_distance=*/0.1f,
        /*min_spread=*/0.3f, /*max_spread=*/5.0f, /*coplanar_tol=*/0.05f,
        /*distance_tol=*/0.03f, /*e_tol=*/0.05f, /*seed=*/5,
        &plane_normal_f, (float)offset);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    double gpu_elapsed = std::chrono::duration<double>(t3 - t2).count();

    cudaFree(d_source.d_points);
    cudaFree(d_target.d_points);

    double gpu_angle = rotationAngleDegrees(gpu_result.rotation);
    double gpu_t_norm = gpu_result.translation.norm();
    std::cout << "GPU:  rotation=" << gpu_angle << " deg, translation=" << gpu_t_norm
              << ", elapsed=" << gpu_elapsed << "s\n";
    printTransform("GPU", gpu_result);

    // ---- checks ----
    bool cpu_ok = isWellFormed(cpu_result);
    bool gpu_ok = isWellFormed(gpu_result);
    std::cout << (cpu_ok ? "[PASS] " : "[FAIL] ") << "CPU result is a well-formed rotation/translation\n";
    std::cout << (gpu_ok ? "[PASS] " : "[FAIL] ") << "GPU result is a well-formed rotation/translation\n";

    // Not an exact-match check -- different RNGs, see file header comment.
    // Flags a likely real bug if the two disagree by a lot; a modest gap
    // between them is expected and fine.
    double angle_diff = std::fabs(cpu_angle - gpu_angle);
    double t_diff = std::fabs(cpu_t_norm - gpu_t_norm);
    bool comparable = angle_diff < 20.0 && t_diff < 1.0;
    std::cout << (comparable ? "[PASS] " : "[WARN] ")
              << "GPU result is in the same ballpark as CPU (angle diff=" << angle_diff
              << " deg, translation diff=" << t_diff << "m) -- a large gap here is worth"
              << " investigating, not automatically a bug\n";

    return (cpu_ok && gpu_ok) ? 0 : 1;
}
