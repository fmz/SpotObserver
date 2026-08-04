//
// Multi-seed best-of-N wrapper around fourPointCongruentSetsGPU().
//
// 4PCS is a RANSAC-style lottery: each seed explores a different random set
// of bases, and any single seed can get unlucky. Since the GPU makes one
// full run cheap (~1-2s), running several seeds and keeping the
// highest-scoring result is the most principled robustness improvement
// available without touching the kernels -- it compounds with simply raising
// iterations_per_seed, which the GPU also makes affordable.
//
// Deliberately a separate host-only file: nothing in four-pcs-gpu.cu is
// modified, so this stays merge-proof against independent edits to that
// file. Plain .cpp (not .cu) -- there's no device code here, just a loop of
// calls into the existing entry point.
//

#pragma once

#include "four-pcs-gpu.cuh"

#include <vector>

namespace SOb {

struct MultiSeedResult {
    RegistrationResult best;             // highest-scoring result across all seeds
    unsigned long long best_seed = 0;    // which seed produced it
    // Score per seed, parallel to the seeds vector passed in. Comparing
    // these tells you a lot: one seed far above the rest = a strong, real
    // alignment that most seeds missed; all seeds clustered low = the data
    // itself probably lacks enough shared structure for ANY seed to find
    // (recapture, don't re-tune).
    std::vector<long> per_seed_scores;
    // best.score / (number of source points), as a rough quality gate.
    // Confirmed-good alignments on this project's real captures have landed
    // around 0.15-0.20 at max_distance=0.1; results below ~0.10 have
    // consistently turned out to be wrong. Slightly underestimates the true
    // fraction when a dominant plane is active (denominator still counts
    // on-plane points that scoring excludes).
    double best_inlier_fraction = 0.0;
};

// Runs fourPointCongruentSetsGPU() once per seed (all other parameters
// forwarded unchanged, same defaults and meaning as the single-seed entry
// point) and returns the best result by score.
MultiSeedResult fourPointCongruentSetsGPUMultiSeed(
    const DevicePointCloud& source, const DevicePointCloud& target,
    const std::vector<unsigned long long>& seeds = {0, 1, 2, 3, 4, 5, 6, 7},
    int iterations_per_seed = 150,
    float max_distance = 0.1f,
    float min_spread = 0.3f, float max_spread = 5.0f, float coplanar_tol = 0.05f,
    float distance_tol = 0.03f, float e_tol = 0.05f,
    const float3* dominant_plane_normal = nullptr,
    float dominant_plane_offset = 0.0f,
    const float3* target_plane_normal = nullptr,
    float plane_alignment_cos_thresh = 0.866f,
    float plane_reject_thresh = 0.04f, float plane_reject_angle_cos = 0.94f,
    cudaStream_t stream = 0);

}  // namespace SOb
