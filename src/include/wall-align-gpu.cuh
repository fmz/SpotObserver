//
// GPU port of the wall-constrained, gravity-aligned coarse alignment that
// succeeded on the Aug 4 capture where 4PCS failed (see
// yaw_sweep_prototype.py / wall-constrained follow-ups from that session).
//
// Idea: both robots stand on the same floor, so the full 6-DOF search 4PCS
// performs is overkill -- three of the six unknowns can be eliminated
// outright:
//   1. fit each cloud's floor plane (fitDominantPlaneGPU) and rotate it to
//      horizontal -> roll and pitch are gone;
//   2. shift each cloud so its floor sits at z=0 -> Z is gone;
//   3. what remains is yaw + XY translation. Yaw is constrained to the small
//      set of angles that map the source's dominant WALL directions onto the
//      target's (walls are the longest, straightest, most reliable structure
//      in an indoor capture -- extracted via local-PCA orientation
//      histograms over the non-floor points). XY translation per yaw
//      candidate comes from dense point-pair voting (every source/target
//      structure-point pair votes for the shift that would superimpose it).
//   4. each of the few (typically <= 8) surviving candidates is refined with
//      a short icpGPU() run and scored by non-floor inlier count; best wins.
//
// Unlike 4PCS this is DETERMINISTIC -- no RANSAC lottery over random bases,
// so no seed sensitivity (the only RNG anywhere is the plane fit's
// hypothesis sampling, which is stable in practice). It exhaustively covers
// the reduced search space instead of gambling on drawing a lucky base.
//
// Assumptions worth knowing:
//   - both clouds actually contain the floor (it need not be the dominant
//     plane fit's winner in pathological cases -- but for these captures it
//     always has been);
//   - the robots share that floor (true for two Spots in one room);
//   - there are walls / wall-like vertical structure in view of both.
// Where those hold, this replaces fourPointCongruentSetsGPU() as the coarse
// stage; the result feeds icpGPU() exactly the same way.
//

#pragma once

#include "four-pcs-gpu.cuh"  // DevicePointCloud, DeviceSpatialGrid, fitDominantPlaneGPU, icpGPU, RegistrationResult

#include <cuda_runtime.h>

namespace SOb {

struct WallAlignResult {
    // Final transform (already ICP-refined) + non-floor inlier score, same
    // conventions as fourPointCongruentSetsGPU()'s RegistrationResult so it
    // can drop into the same downstream code.
    RegistrationResult best;
    float best_yaw_deg = 0.0f;   // winning yaw candidate (pre-ICP), for diagnostics
    int candidates_tried = 0;    // how many wall-consistent yaws were evaluated
    // best.score / (number of non-floor source points). Same quality gate as
    // the multi-seed wrapper: on this project's real captures, correct
    // alignments have landed well above 0.10 after ICP; wrong ones below.
    double inlier_fraction = 0.0;
};

// Wall-constrained coarse alignment + per-candidate ICP refinement.
// source/target must already be GPU-resident voxel-downsampled clouds (same
// input convention as fourPointCongruentSetsGPU).
//
// structure_z_min/max: band above the fitted floor (in meters) whose points
//     count as "structure" (walls/furniture) for orientation histograms and
//     translation voting. Floor and ceiling points are excluded.
// orient_radius: neighborhood radius for the local-PCA wall-direction
//     estimate at each structure point.
// min_anisotropy: 0..1; how strongly linear a point's neighborhood must be
//     to vote in the orientation histogram (walls are ~1, clutter is ~0).
// vote_cell / vote_half_extent: resolution and +/- range (meters) of the XY
//     translation voting grid.
// vote_z_tol: a source/target point pair only votes if their (gravity-
//     aligned) heights agree within this -- cheap pre-filter that keeps
//     table-edge points from voting with floor-lamp points.
// icp_iterations / icp_max_distance: per-candidate refinement depth. Keep
//     iterations modest; the winner can always be re-refined longer by the
//     caller.
// plane_*: forwarded to fitDominantPlaneGPU for each cloud's floor fit.
WallAlignResult wallAlignGPU(
    const DevicePointCloud& source, const DevicePointCloud& target,
    float structure_z_min = 0.15f, float structure_z_max = 2.5f,
    float orient_radius = 0.4f, float min_anisotropy = 0.7f,
    float vote_cell = 0.1f, float vote_half_extent = 16.0f, float vote_z_tol = 0.3f,
    int icp_iterations = 30, float icp_max_distance = 0.1f,
    int plane_hypotheses = 300, float plane_threshold = 0.04f,
    unsigned long long plane_seed = 0,
    cudaStream_t stream = 0);

}  // namespace SOb
