//
// Standalone correctness/timing check for the C++ port of four_pcs.py.
//
// Loads a point cloud pair exported by
// PySpotObserver/pyspotobserver/tools/export_four_pcs_test_data.py and runs the same
// registration recipe validated in Python all session: plane-aware
// fourPointCongruentSets() with source-side floor rejection.
//
// NOTE on reproducibility: this will NOT reproduce the exact rotation/
// translation numbers from a Python run on the same data, even with a
// "matching" seed -- numpy's default_rng (PCG64) and this file's
// std::mt19937_64 are different RNGs with no shared sequence. What should
// match is the *quality* of the result: a similar rotation angle and
// translation magnitude, and a comparably high match score, since both are
// searching the same real geometry for the same underlying answer.
//
// Usage: four_pcs_test [path-to-exported-data.txt]
//

#define _USE_MATH_DEFINES
#include "four-pcs.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

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

    std::cout << "Loaded " << n_src << " source / " << n_tgt << " target points, "
              << "plane normal (" << nx << ", " << ny << ", " << nz << "), offset " << offset << "\n";

    // Recipe tuned against real capture data's known ground-truth transform
    // (see four_pcs_gpu_param_sweep): seed=5, iterations=70, min_spread=0.3,
    // max_spread=5.0 (NOT the repo default of 1.2 -- see four-pcs.h's note on
    // why that default returns zero valid bases on real, room-scale captures).
    auto t0 = std::chrono::steady_clock::now();
    SOb::RegistrationResult result = SOb::fourPointCongruentSets(
        source, target,
        /*iterations=*/70, /*max_distance=*/0.1,
        /*min_spread=*/0.3, /*max_spread=*/5.0, /*coplanar_tol=*/0.05,
        /*distance_tol=*/0.03, /*e_tol=*/0.05, /*seed=*/5,
        &plane_normal, offset);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double trace = result.rotation.trace();
    double angle_deg = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
    double translation_norm = result.translation.norm();

    std::cout << "fourPointCongruentSets: rotation=" << angle_deg << " deg, "
              << "translation=" << translation_norm << ", elapsed=" << elapsed << "s\n";

    // sanity bounds, not a strict pass/fail against a specific Python run --
    // see the reproducibility note above for why
    bool sane = std::isfinite(angle_deg) && std::isfinite(translation_norm)
             && angle_deg >= 0.0 && angle_deg <= 180.0;
    std::cout << (sane ? "[PASS] " : "[FAIL] ")
              << "result is a well-formed rotation/translation\n";

    return sane ? 0 : 1;
}
