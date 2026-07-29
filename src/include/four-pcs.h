//
// Ported from pyspotobserver/four_pcs.py.
//

#pragma once

#include <Eigen/Dense>
#include <utility>

namespace SOb {

// A point cloud, one point per row, matching numpy's (N, 3) layout used by
// the original Python implementation (RowMajor so each point stays contiguous).
using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// Estimate the rigid transformation (rotation and translation) that aligns
// source points to target points.
//
// NOTE: this is defined here for now since four-pcs is the first ported
// module and needs it internally, mirroring how four_pcs.py imports kabsch
// from icp_align.py rather than redefining it. When icp() is ported next,
// move this into a shared header (e.g. icp.h) that both this file and
// icp.cpp include, and remove it from here, to avoid two definitions.
std::pair<Eigen::Matrix3d, Eigen::Vector3d> kabsch(
    const PointCloud& source_points, const PointCloud& target_points);

// Check which points lie within `threshold` of a plane defined by
// normal . x + offset = 0.
Eigen::Array<bool, Eigen::Dynamic, 1> pointsNearPlane(
    const PointCloud& points, const Eigen::Vector3d& normal, double offset,
    double threshold = 0.04);

// Fit the single largest planar surface in a point cloud using RANSAC.
struct PlaneFit {
    bool valid = false;  // false if `points` had fewer than 3 rows
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double offset = 0.0;
    Eigen::Array<bool, Eigen::Dynamic, 1> inlier_mask;
};

// seed selects an independent, deterministic C++ random sequence -- it does
// NOT reproduce the same draws as the same seed value in the Python version,
// since numpy's default_rng (PCG64) and std::mt19937_64 are different RNGs.
// Reproducibility only holds within one language, not across the two.
PlaneFit fitDominantPlane(const PointCloud& points, int iterations = 300,
                           double threshold = 0.04, unsigned seed = 0);

// Best rotation matrix (3x3) and translation vector found across all bases.
struct RegistrationResult {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

// Find an initial rigid alignment between two point clouds with no prior
// pose information, using 4-Points Congruent Sets. Meant to be run once,
// then handed to icp() for refinement.
//
// dominant_plane_normal / dominant_plane_offset: pass a plane from
// fitDominantPlane() (typically the floor) to reject candidate bases that
// are just a patch of that plane, and to exclude on-plane source points from
// scoring -- otherwise a transform that just slides the plane onto itself
// wins by default. Pass nullptr (default) for the original, plane-unaware
// behavior.
//
// max_spread default note (carried over from four_pcs.py): 1.2 was found to
// return zero valid bases at all on a real, room-scale, voxel-downsampled
// capture -- the sampled points were too sparse. That capture needed
// max_spread around 8 (room-scale) before the search could find bases at
// all. Treat 1.2 as a synthetic-density default, not a real-capture one.
RegistrationResult fourPointCongruentSets(
    const PointCloud& source, const PointCloud& target,
    int iterations = 200, double max_distance = 0.1,
    double min_spread = 0.3, double max_spread = 1.2, double coplanar_tol = 0.05,
    double distance_tol = 0.03, double e_tol = 0.05, unsigned seed = 0,
    const Eigen::Vector3d* dominant_plane_normal = nullptr,
    double dominant_plane_offset = 0.0,
    double plane_reject_thresh = 0.04, double plane_reject_angle_cos = 0.94);

}  // namespace SOb
