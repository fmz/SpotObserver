#include "four-pcs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SOb {

std::pair<Eigen::Matrix3d, Eigen::Vector3d> kabsch(
        const PointCloud& source_points, const PointCloud& target_points) {

    // compute centroids of both point sets
    Eigen::Vector3d p = source_points.colwise().mean();
    Eigen::Vector3d q = target_points.colwise().mean();

    // center the points around their centroids
    PointCloud source_centered = source_points.rowwise() - p.transpose();
    PointCloud target_centered = target_points.rowwise() - q.transpose();

    // compute the covariance matrix
    Eigen::Matrix3d covariance_matrix = source_centered.transpose() * target_centered;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // compute the rotation matrix using the kabsch algorithm
    Eigen::Matrix3d rotation_matrix = svd.matrixV() * svd.matrixU().transpose();

    // ensure a proper rotation (determinant = 1)
    if (rotation_matrix.determinant() < 0) {
        Eigen::Matrix3d v_mat = svd.matrixV();
        v_mat.col(2) *= -1;
        rotation_matrix = v_mat * svd.matrixU().transpose();
    }

    // compute the translation vector by aligning the centroids
    Eigen::Vector3d translation_vector = q - rotation_matrix * p;

    return {rotation_matrix, translation_vector};
}

Eigen::Array<bool, Eigen::Dynamic, 1> pointsNearPlane(
        const PointCloud& points, const Eigen::Vector3d& normal, double offset, double threshold) {
    return ((points * normal).array() + offset).abs() < threshold;
}

PlaneFit fitDominantPlane(const PointCloud& points, int iterations, double threshold, unsigned seed) {
    PlaneFit result;
    const long n = points.rows();
    if (n < 3) return result;  // valid stays false

    // NOTE: std::mt19937_64 here is an independent RNG from numpy's
    // default_rng (PCG64) -- see the seed comment in four-pcs.h. Also, unlike
    // rng.choice(n, 3, replace=False), which samples 3 distinct indices
    // directly, this draws 3 independent indices and retries on collision --
    // same guarantee (3 distinct points), different mechanism.
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long> dist(0, n - 1);
    long best_count = -1;

    for (int it = 0; it < iterations; it++) {
        long i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;

        Eigen::Vector3d p0 = points.row(i0).transpose();
        Eigen::Vector3d p1 = points.row(i1).transpose();
        Eigen::Vector3d p2 = points.row(i2).transpose();
        Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0);
        double norm_len = normal.norm();
        if (norm_len < 1e-8) continue;

        normal /= norm_len;
        double offset = -normal.dot(p0);
        auto inliers = pointsNearPlane(points, normal, offset, threshold);
        long count = inliers.count();

        if (count > best_count) {
            best_count = count;
            result.valid = true;
            result.normal = normal;
            result.offset = offset;
            result.inlier_mask = inliers;
        }
    }
    return result;
}

namespace {

// Select a coplanar base of 4 points from a point cloud. See
// select_coplanar_base() in four_pcs.py for the full parameter rationale.
struct CoplanarBase {
    bool found = false;
    std::array<int, 4> indices{};
    std::array<Eigen::Vector3d, 4> points{};
};

CoplanarBase selectCoplanarBase(const PointCloud& cloud, std::mt19937_64& rng,
                                 double min_spread, double max_spread,
                                 double coplanar_tol, int iterations) {
    const long n = cloud.rows();
    std::uniform_int_distribution<long> dist(0, n - 1);

    for (int it = 0; it < iterations; it++) {

        // randomly sample 4 points and check if they are coplanar
        std::array<long, 4> idx{};
        bool duplicate = false;
        for (int k = 0; k < 4; k++) {
            idx[k] = dist(rng);
            for (int j = 0; j < k; j++) if (idx[j] == idx[k]) duplicate = true;
        }
        if (duplicate) continue;

        std::array<Eigen::Vector3d, 4> pts{};
        for (int k = 0; k < 4; k++) pts[k] = cloud.row(idx[k]).transpose();

        double min_dist = std::numeric_limits<double>::infinity();
        double max_dist = 0.0;
        for (int a = 0; a < 4; a++)
            for (int b = a + 1; b < 4; b++) {
                double d = (pts[a] - pts[b]).norm();
                min_dist = std::min(min_dist, d);
                max_dist = std::max(max_dist, d);
            }
        if (min_dist < min_spread || max_dist > max_spread) continue;

        // compute the normal of the plane defined by the first three points
        Eigen::Vector3d normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        double normal_len = normal.norm();
        if (normal_len < 1e-6) continue;
        normal /= normal_len;

        // compute the residual for the fourth point to check coplanarity
        double residual = std::fabs((pts[3] - pts[0]).dot(normal));
        if (residual > coplanar_tol) continue;

        CoplanarBase result;
        result.found = true;
        result.indices = {(int)idx[0], (int)idx[1], (int)idx[2], (int)idx[3]};
        result.points = pts;
        return result;
    }
    return CoplanarBase{};
}

// Find the pairing of 4 coplanar points that makes them behave like the two
// diagonals of a quadrilateral, and compute the affine-invariant ratios where
// those diagonals cross. See diagonal_pairing_and_ratios() in four_pcs.py.
struct DiagonalPairing {
    bool found = false;
    std::array<int, 4> order{};  // (a, b, c, d): segment ab and segment cd cross
    double ratio_a = 0, ratio_b = 0, diag_a = 0, diag_b = 0;
};

DiagonalPairing diagonalPairingAndRatios(const std::array<Eigen::Vector3d, 4>& base_points) {
    // the 3 ways to split 4 points into two pairs -- only one will actually cross
    static const std::array<std::array<int, 4>, 3> possible_orderings = {{
        {0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}
    }};

    for (auto& ordering : possible_orderings) {
        int a = ordering[0], b = ordering[1], c = ordering[2], d = ordering[3];
        const Eigen::Vector3d& point_a = base_points[a];
        const Eigen::Vector3d& point_b = base_points[b];
        const Eigen::Vector3d& point_c = base_points[c];
        const Eigen::Vector3d& point_d = base_points[d];

        // solve for where segment ab and segment cd cross:
        // point_a + ratio_a*(point_b - point_a) = point_c + ratio_b*(point_d - point_c)
        Eigen::Matrix<double, 3, 2> coefficient_matrix;
        coefficient_matrix.col(0) = point_b - point_a;
        coefficient_matrix.col(1) = -(point_d - point_c);
        Eigen::Vector3d right_hand_side = point_c - point_a;

        // least-squares solve (3 equations, 2 unknowns), matching np.linalg.lstsq
        Eigen::Vector2d solution = coefficient_matrix.jacobiSvd(
            Eigen::ComputeThinU | Eigen::ComputeThinV).solve(right_hand_side);
        double ratio_a = solution(0), ratio_b = solution(1);

        // the crossing point must actually lie between the two endpoints on each segment
        if (ratio_a >= -0.05 && ratio_a <= 1.05 && ratio_b >= -0.05 && ratio_b <= 1.05) {
            Eigen::Vector3d crossing_point = point_a + ratio_a * (point_b - point_a);
            Eigen::Vector3d crossing_point_check = point_c + ratio_b * (point_d - point_c);

            if ((crossing_point - crossing_point_check).norm() < 0.02) {
                DiagonalPairing result;
                result.found = true;
                result.order = {a, b, c, d};
                result.ratio_a = ratio_a;
                result.ratio_b = ratio_b;
                result.diag_a = (point_b - point_a).norm();
                result.diag_b = (point_d - point_c).norm();
                return result;
            }
        }
    }
    return DiagonalPairing{};
}

// Minimal spatial hash grid over 3D points, standing in for scipy's cKDTree
// (query_pairs / query_ball_point / query w/ distance_upper_bound) since no
// KD-tree library is wired into this project yet. Correct at any query
// radius relative to cell_size (searches enough neighboring cells to cover
// it), though most efficient when radius is close to cell_size.
class SpatialHashGrid {
public:
    SpatialHashGrid(std::vector<Eigen::Vector3d> points, double cell_size)
            : points_(std::move(points)), cell_size_(std::max(cell_size, 1e-6)) {
        for (int i = 0; i < (int)points_.size(); i++) {
            cells_[cellKey(points_[i])].push_back(i);
        }
    }

    // all index pairs (i < j) whose distance falls within [target_distance -
    // tolerance, target_distance + tolerance]. Mirrors distance_pairs().
    std::vector<std::pair<int, int>> pairsInRange(double target_distance, double tolerance) const {
        std::vector<std::pair<int, int>> result;
        double max_d = target_distance + tolerance;
        double min_d = std::max(0.0, target_distance - tolerance);
        int reach = (int)std::ceil(max_d / cell_size_) + 1;

        for (int i = 0; i < (int)points_.size(); i++) {
            auto [cx, cy, cz] = cellCoord(points_[i]);
            for (int dx = -reach; dx <= reach; dx++)
                for (int dy = -reach; dy <= reach; dy++)
                    for (int dz = -reach; dz <= reach; dz++) {
                        auto it = cells_.find(encodeKey(cx + dx, cy + dy, cz + dz));
                        if (it == cells_.end()) continue;
                        for (int j : it->second) {
                            if (j <= i) continue;  // undirected, avoid duplicates/self-pairs
                            double d = (points_[i] - points_[j]).norm();
                            if (d >= min_d && d <= max_d) result.emplace_back(i, j);
                        }
                    }
        }
        return result;
    }

    // for each point in `query`, indices into this grid's points within `radius`.
    // Mirrors cKDTree.query_ball_point().
    std::vector<std::vector<int>> radiusQuery(const std::vector<Eigen::Vector3d>& query, double radius) const {
        std::vector<std::vector<int>> result(query.size());
        int reach = (int)std::ceil(radius / cell_size_) + 1;

        for (size_t q = 0; q < query.size(); q++) {
            auto [cx, cy, cz] = cellCoord(query[q]);
            for (int dx = -reach; dx <= reach; dx++)
                for (int dy = -reach; dy <= reach; dy++)
                    for (int dz = -reach; dz <= reach; dz++) {
                        auto it = cells_.find(encodeKey(cx + dx, cy + dy, cz + dz));
                        if (it == cells_.end()) continue;
                        for (int idx : it->second) {
                            if ((points_[idx] - query[q]).norm() <= radius) result[q].push_back(idx);
                        }
                    }
        }
        return result;
    }

    // nearest point index within max_distance, or -1 if none found.
    // Mirrors cKDTree.query(..., distance_upper_bound=max_distance).
    int nearestWithin(const Eigen::Vector3d& query, double max_distance) const {
        int reach = (int)std::ceil(max_distance / cell_size_) + 1;
        auto [cx, cy, cz] = cellCoord(query);
        int best = -1;
        double best_d = max_distance;

        for (int dx = -reach; dx <= reach; dx++)
            for (int dy = -reach; dy <= reach; dy++)
                for (int dz = -reach; dz <= reach; dz++) {
                    auto it = cells_.find(encodeKey(cx + dx, cy + dy, cz + dz));
                    if (it == cells_.end()) continue;
                    for (int idx : it->second) {
                        double d = (points_[idx] - query).norm();
                        if (d < best_d) { best_d = d; best = idx; }
                    }
                }
        return best;
    }

private:
    std::vector<Eigen::Vector3d> points_;
    double cell_size_;
    std::unordered_map<int64_t, std::vector<int>> cells_;

    std::tuple<int, int, int> cellCoord(const Eigen::Vector3d& p) const {
        return { (int)std::floor(p.x() / cell_size_),
                 (int)std::floor(p.y() / cell_size_),
                 (int)std::floor(p.z() / cell_size_) };
    }

    static int64_t encodeKey(int x, int y, int z) {
        auto enc = [](int v) -> int64_t { return (int64_t)v + (1 << 19); };
        return (enc(x) << 42) | (enc(y) << 21) | enc(z);
    }

    int64_t cellKey(const Eigen::Vector3d& p) const {
        auto [x, y, z] = cellCoord(p);
        return encodeKey(x, y, z);
    }
};

// Find all point pairs in a cloud whose distance falls within a tolerance
// band around a target distance, capped at max_pairs. Mirrors distance_pairs().
//
// Builds its own grid sized to THIS query's radius rather than reusing a
// fixed-cell grid tuned for a different (often much smaller) radius -- that
// mismatch is a real perf trap: a small-celled grid queried at a large
// radius (diagonals here can be several meters) means searching tens of
// thousands of neighboring cells per point. Rebuilding is cheap since these
// are the coarse, downsampled clouds (thousands of points, not hundreds of
// thousands).
std::vector<std::pair<int, int>> distancePairs(const std::vector<Eigen::Vector3d>& points,
                                                double target_distance, double tolerance,
                                                std::mt19937_64& rng, size_t max_pairs = 2000) {
    SpatialHashGrid grid(points, std::max(target_distance + tolerance, 1e-3));
    auto pairs = grid.pairsInRange(target_distance, tolerance);
    if (pairs.size() > max_pairs) {
        std::vector<std::pair<int, int>> sampled;
        sampled.reserve(max_pairs);
        std::sample(pairs.begin(), pairs.end(), std::back_inserter(sampled), max_pairs, rng);
        pairs = std::move(sampled);
    }
    return pairs;
}

// Search the target cloud for all 4-point subsets that are approximately
// congruent to a base -- share the same two diagonal lengths AND the same
// two crossing ratios, within tolerance. Mirrors find_congruent().
std::vector<std::array<int, 4>> findCongruent(
        const std::array<Eigen::Vector3d, 4>& ordered_base_points,
        double ratio_a, double ratio_b, double diag_a, double diag_b,
        const PointCloud& target, const std::vector<Eigen::Vector3d>& target_points,
        double distance_tolerance, double e_tolerance, std::mt19937_64& rng,
        size_t max_pairs_per_distance = 2000, size_t max_candidates = 250) {

    auto pairs_a = distancePairs(target_points, diag_a, distance_tolerance, rng, max_pairs_per_distance);
    auto pairs_b = distancePairs(target_points, diag_b, distance_tolerance, rng, max_pairs_per_distance);
    if (pairs_a.empty() || pairs_b.empty()) return {};

    // candidate crossing points implied by each pairs_a pair + ratio_a.
    // try both point orderings per pair, since we don't know which end plays "a" vs "b"
    std::vector<Eigen::Vector3d> e1_all;
    std::vector<std::array<int, 2>> pairs_a_directions;
    e1_all.reserve(pairs_a.size() * 2);
    pairs_a_directions.reserve(pairs_a.size() * 2);
    for (auto& [pi, qi] : pairs_a) {
        Eigen::Vector3d P = target.row(pi).transpose(), Q = target.row(qi).transpose();
        e1_all.push_back(P + ratio_a * (Q - P));
        pairs_a_directions.push_back({pi, qi});
        e1_all.push_back(Q + ratio_a * (P - Q));
        pairs_a_directions.push_back({qi, pi});
    }

    // candidate crossing points implied by each pairs_b pair + ratio_b
    std::vector<Eigen::Vector3d> e2_all;
    std::vector<std::array<int, 2>> pairs_b_directions;
    e2_all.reserve(pairs_b.size() * 2);
    pairs_b_directions.reserve(pairs_b.size() * 2);
    for (auto& [pi, qi] : pairs_b) {
        Eigen::Vector3d P = target.row(pi).transpose(), Q = target.row(qi).transpose();
        e2_all.push_back(P + ratio_b * (Q - P));
        pairs_b_directions.push_back({pi, qi});
        e2_all.push_back(Q + ratio_b * (P - Q));
        pairs_b_directions.push_back({qi, pi});
    }

    // build a grid over e1_all and query for nearby points in e2_all
    SpatialHashGrid e1_grid(e1_all, std::max(e_tolerance, 1e-6));
    auto e2_matches = e1_grid.radiusQuery(e2_all, e_tolerance);

    std::vector<std::array<int, 4>> candidates;
    size_t n_checked = 0;

    for (size_t e2_index = 0; e2_index < e2_matches.size(); e2_index++) {
        for (int e1_index : e2_matches[e2_index]) {
            n_checked++;
            if (n_checked > max_candidates) return candidates;

            int P = pairs_a_directions[e1_index][0], Q = pairs_a_directions[e1_index][1];
            int R = pairs_b_directions[e2_index][0], S = pairs_b_directions[e2_index][1];
            candidates.push_back({P, Q, R, S});
        }
    }
    return candidates;
}

}  // namespace

RegistrationResult fourPointCongruentSets(
        const PointCloud& source, const PointCloud& target,
        int iterations, double max_distance,
        double min_spread, double max_spread, double coplanar_tol,
        double distance_tol, double e_tol, unsigned seed,
        const Eigen::Vector3d* dominant_plane_normal, double dominant_plane_offset,
        double plane_reject_thresh, double plane_reject_angle_cos) {

    std::mt19937_64 rng(seed);

    // target as a plain point vector, passed to find_congruent()'s
    // distance_pairs() calls (which each build their own appropriately-sized
    // grid -- see distancePairs()'s comment), plus one grid sized for
    // max_distance specifically, reused across every candidate's scoring
    // pass below since that query radius never changes.
    std::vector<Eigen::Vector3d> target_points(target.rows());
    for (long i = 0; i < target.rows(); i++) target_points[i] = target.row(i);
    SpatialHashGrid target_grid(target_points, std::max(max_distance, 0.05));

    bool plane_active = dominant_plane_normal != nullptr;
    Eigen::Array<bool, Eigen::Dynamic, 1> source_off_plane;
    if (plane_active) {
        source_off_plane = !pointsNearPlane(source, *dominant_plane_normal, dominant_plane_offset, plane_reject_thresh);
    }

    RegistrationResult best;
    long best_score = -1;

    for (int it = 0; it < iterations; it++) {

        CoplanarBase base = selectCoplanarBase(source, rng, min_spread, max_spread, coplanar_tol, /*iterations=*/200);
        if (!base.found) continue;

        if (plane_active) {
            Eigen::Vector3d base_normal = (base.points[1] - base.points[0]).cross(base.points[2] - base.points[0]);
            double base_normal_len = base_normal.norm();

            if (base_normal_len > 1e-8) {
                base_normal /= base_normal_len;
                Eigen::Vector3d base_centroid = (base.points[0] + base.points[1] + base.points[2] + base.points[3]) * 0.25;
                double cos_angle = std::fabs(base_normal.dot(*dominant_plane_normal));
                double centroid_dist = std::fabs(base_centroid.dot(*dominant_plane_normal) + dominant_plane_offset);

                // this base is itself just a patch of the dominant plane -- skip it,
                // since any match found from it is exactly the degenerate case above
                if (cos_angle > plane_reject_angle_cos && centroid_dist < plane_reject_thresh * 3) continue;
            }
        }

        DiagonalPairing pairing = diagonalPairingAndRatios(base.points);
        if (!pairing.found) continue;

        std::array<Eigen::Vector3d, 4> ordered_base_points = {
            base.points[pairing.order[0]], base.points[pairing.order[1]],
            base.points[pairing.order[2]], base.points[pairing.order[3]]
        };

        auto candidates = findCongruent(ordered_base_points, pairing.ratio_a, pairing.ratio_b,
                                         pairing.diag_a, pairing.diag_b, target, target_points,
                                         distance_tol, e_tol, rng);

        // verify each candidate with a real rigid fit, scored against the whole cloud --
        // find_congruent() only guarantees affine invariance, not a genuine rigid match
        for (auto& candidate : candidates) {
            PointCloud base_pts(4, 3), matched_pts(4, 3);
            for (int k = 0; k < 4; k++) {
                base_pts.row(k) = ordered_base_points[k].transpose();
                matched_pts.row(k) = target.row(candidate[k]);
            }
            auto [rotation_matrix, translation_vector] = kabsch(base_pts, matched_pts);

            PointCloud transformed_source = (rotation_matrix * source.transpose()).transpose();
            transformed_source.rowwise() += translation_vector.transpose();

            long score = 0;
            for (long i = 0; i < transformed_source.rows(); i++) {
                Eigen::Vector3d query_point = transformed_source.row(i).transpose();
                bool matched = target_grid.nearestWithin(query_point, max_distance) >= 0;
                // points on the dominant plane don't count toward the score -- otherwise a
                // transform that just slides the plane onto itself wins by default
                if (matched && (!plane_active || source_off_plane(i))) score++;
            }

            if (score > best_score) {
                best_score = score;
                best.rotation = rotation_matrix;
                best.translation = translation_vector;
            }
        }
    }

    return best;
}

}  // namespace SOb
