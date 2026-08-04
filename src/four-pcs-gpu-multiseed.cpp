#include "four-pcs-gpu-multiseed.h"

#include <iostream>

namespace SOb {

MultiSeedResult fourPointCongruentSetsGPUMultiSeed(
    const DevicePointCloud& source, const DevicePointCloud& target,
    const std::vector<unsigned long long>& seeds,
    int iterations_per_seed,
    float max_distance,
    float min_spread, float max_spread, float coplanar_tol,
    float distance_tol, float e_tol,
    const float3* dominant_plane_normal,
    float dominant_plane_offset,
    const float3* target_plane_normal,
    float plane_alignment_cos_thresh,
    float plane_reject_thresh, float plane_reject_angle_cos,
    cudaStream_t stream)
{
    MultiSeedResult out;
    out.per_seed_scores.reserve(seeds.size());

    for (unsigned long long seed : seeds) {
        RegistrationResult result = fourPointCongruentSetsGPU(
            source, target,
            iterations_per_seed, max_distance,
            min_spread, max_spread, coplanar_tol,
            distance_tol, e_tol, seed,
            dominant_plane_normal, dominant_plane_offset,
            target_plane_normal, plane_alignment_cos_thresh,
            plane_reject_thresh, plane_reject_angle_cos,
            stream);

        out.per_seed_scores.push_back(result.score);
        std::cout << "  [multi-seed] seed=" << seed << " score=" << result.score
                  << " (" << (source.count > 0 ? (double)result.score / source.count : 0.0)
                  << " of source)\n";

        if (result.score > out.best.score) {
            out.best = result;
            out.best_seed = seed;
        }
    }

    out.best_inlier_fraction =
        (source.count > 0 && out.best.score > 0) ? (double)out.best.score / source.count : 0.0;

    std::cout << "  [multi-seed] best: seed=" << out.best_seed
              << " score=" << out.best.score
              << " inlier_fraction=" << out.best_inlier_fraction
              << (out.best_inlier_fraction < 0.10
                      ? "  -- LOW: below ~0.10 has consistently meant a wrong alignment on this project's captures"
                      : "")
              << "\n";

    return out;
}

}  // namespace SOb
