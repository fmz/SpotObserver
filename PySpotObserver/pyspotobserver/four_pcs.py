
import numpy as np
from scipy.spatial import cKDTree

from pyspotobserver.icp_align import kabsch

def downsample_cloud(cloud, voxel_size):

    """
    Downsample a point cloud using a voxel grid filter.

    Args:
        cloud (np.ndarray): Input point cloud of shape (N, 3).
        voxel_size (float): Size of the voxel grid.

    Returns:
        np.ndarray: Downsampled point cloud.
    """

    # compute voxel indices
    voxel_indices = np.floor(cloud / voxel_size).astype(np.int32)

    # use a dictionary to store unique voxels
    unique_voxels = {}

    for idx, voxel in enumerate(voxel_indices):
        key = tuple(voxel)
        if key not in unique_voxels:
            unique_voxels[key] = cloud[idx]

    # return the downsampled points
    return np.array(list(unique_voxels.values()))

def select_coplanar_base(cloud, rng, min_spread = 0.3, max_spread = 1.2, coplanar_tol = 0.05, iterations = 200):

    """Select a coplanar base of 4 points from a point cloud.

    Args:
        cloud (np.ndarray): Input point cloud of shape (N, 3).
        rng (np.random.Generator): Seeded random generator, so results are reproducible.
        min_spread (float): Minimum allowed distance between any two of the 4 points.
            Points too close together make the diagonal-ratio math numerically unstable.
        max_spread (float): Maximum allowed distance between any two of the 4 points.
            Keeps the base local, so the later target-cloud distance search stays fast
            and the base is more likely to fall inside the region the clouds share.
        coplanar_tol (float): Tolerance for considering the 4th point coplanar with the
            other three.
        iterations (int): Number of random 4-point samples to attempt before giving up.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The 4 chosen point indices and their coordinates,
        or (None, None) if no valid base was found within `iterations` tries.
    """

    n = len(cloud)

    for _ in range(iterations):

        # randomly sample 4 points and check if they are coplanar
        indices = rng.choice(n, 4, replace = False)
        sample_points = cloud[indices]

        distances = np.linalg.norm(sample_points[:, None, :] - sample_points[None, :, :], axis = -1)

        if distances[distances > 0].min() < min_spread or distances.max() > max_spread:
            continue

        # compute the normal of the plane defined by the first three points
        version_a = sample_points[1] - sample_points[0]
        version_b = sample_points[2] - sample_points[0]
        normal = np.cross(version_a, version_b)
        normal_len = np.linalg.norm(normal)

        if normal_len < 1e-6:
            continue

        normal /= normal_len

        # compute the residual for the fourth point to check coplanarity
        residual = abs(np.dot(sample_points[3] - sample_points[0], normal))

        if residual > coplanar_tol:
            continue

        return indices, sample_points

    return None, None

def points_near_plane(points, normal, offset, threshold = 0.04):

    """Check which points lie within `threshold` of a plane defined by
    normal . x + offset = 0.

    Args:
        points (np.ndarray): Points to test, shape (N, 3).
        normal (np.ndarray): Unit normal of the plane, shape (3,).
        offset (float): Plane offset, as in normal . x + offset = 0.
        threshold (float): Maximum distance from the plane to count as "on" it.

    Returns:
        np.ndarray: Boolean mask of shape (N,), True where a point is within
        `threshold` of the plane.
    """

    return np.abs(points @ normal + offset) < threshold

def fit_dominant_plane(points, iterations = 300, threshold = 0.04, seed = 0):

    """Fit the single largest planar surface in a point cloud using RANSAC.

    Args:
        points (np.ndarray): Points to search, shape (N, 3).
        iterations (int): Number of random 3-point plane hypotheses to try.
        threshold (float): Distance from a hypothesis plane for a point to
            count as an inlier.
        seed (int): Random seed, so results are reproducible.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: (normal, offset, inlier_mask)
        for the best-scoring plane found, where normal . x + offset = 0.
        Returns (None, None, None) if `points` has fewer than 3 rows.
    """

    n = len(points)
    if n < 3:
        return None, None, None

    rng = np.random.default_rng(seed)
    best_count = -1
    best_normal, best_offset, best_inliers = None, None, None

    for _ in range(iterations):
        idx = rng.choice(n, 3, replace = False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)

        if norm_len < 1e-8:
            continue

        normal = normal / norm_len
        offset = -normal.dot(p0)
        inliers = points_near_plane(points, normal, offset, threshold)
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_normal, best_offset, best_inliers = normal, offset, inliers

    return best_normal, best_offset, best_inliers

def diagonal_pairing_and_ratios(base_points):

    """Find the pairing of 4 coplanar points that makes them behave like the two
    diagonals of a quadrilateral, and compute the affine-invariant ratios where
    those diagonals cross.

    Args:
        base_points (np.ndarray): The 4 base points, shape (4, 3).

    Returns:
        Tuple: (point_order, ratio_a, ratio_b, diag_a, diag_b), where point_order
        is a 4-tuple of indices into base_points ordered (a, b, c, d) so segment
        ab and segment cd are the crossing diagonals. Returns all None if no
        pairing of the 4 points produces crossing diagonals.
    """

    # the 3 ways to split 4 points into two pairs -- only one will actually cross
    possible_orderings = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]

    for a, b, c, d in possible_orderings:
        point_a, point_b, point_c, point_d = base_points[a], base_points[b], base_points[c], base_points[d]

        # solve for where segment ab and segment cd cross:
        # point_a + ratio_a*(point_b - point_a) = point_c + ratio_b*(point_d - point_c)
        coefficient_matrix = np.stack([point_b - point_a, -(point_d - point_c)], axis = 1)
        right_hand_side = point_c - point_a
        solution, *_ = np.linalg.lstsq(coefficient_matrix, right_hand_side, rcond = None)
        ratio_a, ratio_b = solution

        # the crossing point must actually lie between the two endpoints on each segment
        if -0.05 <= ratio_a <= 1.05 and -0.05 <= ratio_b <= 1.05:
            crossing_point = point_a + ratio_a * (point_b - point_a)
            crossing_point_check = point_c + ratio_b * (point_d - point_c)

            if np.linalg.norm(crossing_point - crossing_point_check) < 0.02:
                diag_a = np.linalg.norm(point_b - point_a)
                diag_b = np.linalg.norm(point_d - point_c)
                return (a, b, c, d), ratio_a, ratio_b, diag_a, diag_b

    return None, None, None, None, None

def distance_pairs(tree, points, target_distance, tolerance, rng, max_pairs = 2000):

    """Find all point pairs in a cloud whose distance falls within a tolerance
    band around a target distance.

    Args:
        tree (cKDTree): KD-tree already built over `points`.
        points (np.ndarray): The point cloud being searched, shape (M, 3).
        target_distance (float): The distance to search for.
        tolerance (float): How far a pair's actual distance can be from
            target_distance and still count as a match.
        rng (np.random.Generator): Seeded random generator, used only when
            subsampling down to max_pairs.
        max_pairs (int): Hard cap on how many pairs to return, to bound
            runtime on dense or repetitive regions of the cloud.

    Returns:
        np.ndarray: Array of shape (K, 2) of index pairs into `points`.
    """

    pairs = tree.query_pairs(target_distance + tolerance, output_type = 'ndarray')

    if len(pairs) == 0:
        return np.empty((0, 2), dtype = int)

    distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis = -1)
    keep = distances > (target_distance - tolerance)
    pairs = pairs[keep]

    if len(pairs) > max_pairs:
        # sample row indices into `pairs`, not the point-index values themselves
        selection = rng.choice(len(pairs), max_pairs, replace = False)
        pairs = pairs[selection]

    return pairs

def find_congruent(base_points,
                   invar_ratio_a, invar_ratio_b, diag_a, diag_b, target, target_tree,
                   distance_tolerance = 0.03, e_tolerance = 0.05,
                   rng = None, max_pairs_per_distance = 2000, max_candidates = 250):

    """Search the target cloud for all 4-point subsets that are approximately
    congruent to a base / share the same two diagonal lengths AND the
    same two crossing ratios, within tolerance.

    Args:
        base_points (np.ndarray): The 4 base points [a, b, c, d] in diagonal-pair
            order, so segment ab and segment cd are the crossing diagonals.
        invar_ratio_a (float): Invariant ratio along diagonal ab.
        invar_ratio_b (float): Invariant ratio along diagonal cd.
        diag_a (float): Length of diagonal ab.
        diag_b (float): Length of diagonal cd.
        target (np.ndarray): The target cloud to search, shape (M, 3).
        target_tree (cKDTree): KD-tree already built over `target`.
        distance_tolerance (float): Tolerance used when matching diagonal lengths.
        e_tolerance (float): Tolerance used when matching crossing points.
        rng (np.random.Generator): Seeded random generator, passed through to
            distance_pairs().
        max_pairs_per_distance (int): Cap on candidate pairs kept per distance search.
        max_candidates (int): Cap on total congruent-set candidates returned.

    Returns:
        list: Candidate 4-point sets as index tuples (P, Q, R, S) into `target`.
    """

    # compute the distances between the points in the base
    pairs_a = distance_pairs(target_tree, target, diag_a, distance_tolerance, rng, max_pairs_per_distance)
    pairs_b = distance_pairs(target_tree, target, diag_b, distance_tolerance, rng, max_pairs_per_distance)

    if len(pairs_a) == 0 or len(pairs_b) == 0:
        return []

    # candidate crossing points implied by each pairs_a pair + invar_ratio_a.
    # try both point orderings per pair, since we don't know which end plays "a" vs "b"
    P1 = target[pairs_a[:, 0]]; Q1 = target[pairs_a[:, 1]]
    e1_all = np.concatenate([P1 + invar_ratio_a * (Q1 - P1), Q1 + invar_ratio_a * (P1 - Q1)], axis = 0)
    pairs_a_directions = np.concatenate([pairs_a, pairs_a[:, ::-1]], axis = 0)

    # candidate crossing points implied by each pairs_b pair + invar_ratio_b
    P2 = target[pairs_b[:, 0]]; Q2 = target[pairs_b[:, 1]]
    e2_all = np.concatenate([P2 + invar_ratio_b * (Q2 - P2), Q2 + invar_ratio_b * (P2 - Q2)], axis = 0)
    pairs_b_directions = np.concatenate([pairs_b, pairs_b[:, ::-1]], axis = 0)

    # build a KD-tree for the first set of points and query for nearby points in the second set
    e1_tree = cKDTree(e1_all)
    e2_matches = e1_tree.query_ball_point(e2_all, r = e_tolerance)

    candidates = []
    n_checked = 0

    # iterate through the matches and collect candidate correspondences
    for e2_index, e1_index_list in enumerate(e2_matches):
        for e1_index in e1_index_list:

            n_checked += 1

            if n_checked > max_candidates:
                return candidates

            P, Q = pairs_a_directions[e1_index]
            R, S = pairs_b_directions[e2_index]
            candidates.append((P, Q, R, S))

    return candidates

def four_point_congruent_sets(source, target, iterations = 200, max_distance = 0.1,
                               min_spread = 0.3, max_spread = 1.2, coplanar_tol = 0.05,
                               distance_tol = 0.03, e_tol = 0.05, seed = 0,
                               dominant_plane_normal = None, dominant_plane_offset = None,
                               plane_reject_thresh = 0.04, plane_reject_angle_cos = 0.94):

    """Find an initial rigid alignment between two point clouds with no prior
    pose information, using 4-Points Congruent Sets. Meant to be run once,
    then handed to icp() for refinement.

    Args:
        source (np.ndarray): Source points of shape (N, 3).
        target (np.ndarray): Target points of shape (M, 3).
        
        iterations (int): Number of random bases to try. More bases means a
            better chance of finding the true alignment, at the cost of runtime.
        max_distance (float): Maximum distance for a transformed source point
            to count as matching a target point, when scoring a candidate transform.

        min_spread (float): Passed through to select_coplanar_base().
        max_spread (float): Passed through to select_coplanar_base(). NOTE:
            the default here (1.2) was found to return zero valid bases at
            all on a real, room-scale, voxel-downsampled capture -- the
            sampled points were too sparse for four points within 1.2m of
            each other to also be coplanar within tolerance. That capture
            needed max_spread somewhere around 8 (room-scale) before the
            search could find bases at all; a much larger or smaller scene
            would need its own value. Treat 1.2 as a synthetic-density
            default, not a real-capture one, and tune per scene.

        coplanar_tol (float): Passed through to select_coplanar_base().
        distance_tol (float): Passed through to find_congruent().
        e_tol (float): Passed through to find_congruent().
        seed (int): Random seed, so results are reproducible run to run.
        
        dominant_plane_normal (np.ndarray): Unit normal of a plane to reject
            and down-weight, typically the floor, from fit_dominant_plane().
            If None (default), no plane handling is applied and behavior is
            identical to the original function.
        dominant_plane_offset (float): Offset of that plane, as in
            normal . x + offset = 0. Required if dominant_plane_normal is set.

        plane_reject_thresh (float): Distance from the plane, in meters, for
            a point to be considered "on" it.
        plane_reject_angle_cos (float): A sampled base is rejected as a
            plane-patch if the cosine of the angle between its own normal and
            `dominant_plane_normal` exceeds this AND its centroid is within
            3x `plane_reject_thresh` of the plane. Default ~0.94 is about 20 degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Best rotation matrix (3x3) and
        translation vector found across all bases.
    """

    rng = np.random.default_rng(seed)
    target_tree = cKDTree(target)

    plane_active = dominant_plane_normal is not None and dominant_plane_offset is not None
    source_off_plane = (~points_near_plane(source, dominant_plane_normal, dominant_plane_offset, plane_reject_thresh)
                         if plane_active else None)

    best_score = -1
    best_translation = np.zeros(3)
    best_rotation = np.eye(3)

    for _ in range(iterations):

        base_indices, base_points = select_coplanar_base(source, rng, min_spread, max_spread, coplanar_tol)
        if base_indices is None:
            continue

        if plane_active:
            base_normal = np.cross(base_points[1] - base_points[0], base_points[2] - base_points[0])
            base_normal_len = np.linalg.norm(base_normal)

            if base_normal_len > 1e-8:

                base_normal = base_normal / base_normal_len
                base_centroid = base_points.mean(axis = 0)
                cos_angle = abs(np.dot(base_normal, dominant_plane_normal))
                centroid_dist = abs(base_centroid @ dominant_plane_normal + dominant_plane_offset)

                # this base is itself just a patch of the dominant plane -- skip it,
                # since any match found from it is exactly the degenerate case above
                if cos_angle > plane_reject_angle_cos and centroid_dist < plane_reject_thresh * 3:
                    continue

        point_order, ratio_a, ratio_b, diag_a, diag_b = diagonal_pairing_and_ratios(base_points)
        if point_order is None:
            continue

        a, b, c, d = point_order
        ordered_base_points = np.array([base_points[a], base_points[b], base_points[c], base_points[d]])

        candidates = find_congruent(ordered_base_points, ratio_a, ratio_b, diag_a, diag_b,
                                     target, target_tree, distance_tol, e_tol, rng)

        # verify each candidate with a real rigid fit, scored against the whole cloud --
        # find_congruent() only guarantees affine invariance, not a genuine rigid match
        for (P, Q, R, S) in candidates:
            matched_target_points = np.array([target[P], target[Q], target[R], target[S]])
            rotation_matrix, translation_vector = kabsch(ordered_base_points, matched_target_points)

            transformed_source = (rotation_matrix @ source.T).T + translation_vector
            _, nearest_target_indices = target_tree.query(transformed_source, distance_upper_bound = max_distance)
            matched = nearest_target_indices < len(target)

            # points on the dominant plane don't count toward the score -- otherwise a
            # transform that just slides the plane onto itself wins by default
            score = np.sum(matched & source_off_plane) if plane_active else np.sum(matched)

            if score > best_score:
                best_score = score
                best_rotation = rotation_matrix
                best_translation = translation_vector

    return best_rotation, best_translation