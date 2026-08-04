"""
Exports a source/target point cloud pair -- plus a dominant-plane fit for
EACH cloud -- to a plain-text file the C++ standalone tests
(four-pcs-test.cpp, four-pcs-gpu-test.cpp) can load without needing Python,
Eigen bindings, or a point cloud library: just whitespace-separated numbers.

Source and target clouds come from captures/<name>_points_body.npy, written by
capture_dual.py -- already-backprojected, real depth-camera points in each
robot's own body frame (see backproject_to_body() there), not synthetic data.

The clouds written out are voxel-downsampled, matching how the real
fourPointCongruentSets() call is used in practice -- the same coarse clouds
serve both the base search and the final whole-cloud scoring, not the raw
backprojected depth image (hundreds of thousands of points, far too many for
the brute-force base search to be practical).

The target cloud's own dominant-plane fit (in addition to source's) lets the
C++ side reject a candidate transform whose rotation doesn't map source's
floor-normal close to target's floor-normal -- both robots stand on the same
real floor, so a genuinely correct transform shouldn't rotate "up" by ~90
degrees. Catches a wall-matched-to-floor false positive before it's ever
scored.

Each fit's normal sign is arbitrary (RANSAC picks it up from whichever 3
points it happened to sample, independently for each cloud), so before
writing them out, each normal is re-oriented to point toward its own cloud's
frame origin -- the robot's own body, which sits above its own floor by
construction. Without this, the C++ side's alignment check would have to
compare normals direction-agnostically (via |cos|), which can't tell a
correct match from a genuine upside-down flip (both give |cos| close to 1).

Output format (all whitespace-separated, no fixed column widths):
    n_src n_tgt
    nx ny nz offset  (source dominant-plane fit)
    nx ny nz offset  (target dominant-plane fit)
    x y z            (repeated n_src times -- source cloud)
    x y z            (repeated n_tgt times -- target cloud)

Usage:
    python export_four_pcs_test_data.py <output.txt> \\
        [--source-name robotA] [--target-name robotB] [--voxel-size V]
"""

import argparse
from pathlib import Path

import numpy as np

from pyspotobserver.four_pcs import downsample_cloud, fit_dominant_plane

CAPTURES_DIR = Path(__file__).parent / "captures"


def canonicalize_plane_sign(normal, offset):
    """Orient (normal, offset) so the cloud's own frame origin -- the
    robot's own body, which sits above its own floor by construction --
    lies on the positive side of normal.x + offset = 0. Evaluating the
    plane equation at the origin gives exactly `offset`, so this is just:
    flip both if offset is negative."""
    if offset < 0:
        return -normal, -offset
    return normal, offset


def load_capture_points(name):
    path = CAPTURES_DIR / f"{name}_points_body.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run capture_dual.py first to record '{name}' "
            f"(or pass --source-name/--target-name matching an existing capture)."
        )
    return np.load(path)


def write_test_data(path, source, target,
                     src_plane_normal, src_plane_offset,
                     tgt_plane_normal, tgt_plane_offset):
    with open(path, "w") as f:
        f.write(f"{len(source)} {len(target)}\n")
        f.write(f"{src_plane_normal[0]} {src_plane_normal[1]} {src_plane_normal[2]} {src_plane_offset}\n")
        f.write(f"{tgt_plane_normal[0]} {tgt_plane_normal[1]} {tgt_plane_normal[2]} {tgt_plane_offset}\n")
        for p in source:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for p in target:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path to write the exported test data to")
    parser.add_argument("--source-name", default="robotA",
                         help="captures/<name>_points_body.npy to use as the source cloud "
                              "(default robotA)")
    parser.add_argument("--target-name", default="robotB",
                         help="captures/<name>_points_body.npy to use as the target cloud "
                              "(default robotB)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="downsample_cloud() voxel size for the coarse clouds used in "
                              "both the base search and final scoring (default 0.05)")
    parser.add_argument("--plane-iterations", type=int, default=300,
                         help="RANSAC iterations for fit_dominant_plane() on the source cloud "
                              "(default 300)")
    parser.add_argument("--plane-threshold", type=float, default=0.04,
                         help="inlier distance threshold for fit_dominant_plane() (default 0.04)")
    args = parser.parse_args()

    print(f"Loading '{args.source_name}' and '{args.target_name}' from {CAPTURES_DIR}...")
    src_full = load_capture_points(args.source_name)
    tgt_full = load_capture_points(args.target_name)
    print(f"  source: {len(src_full)} points, target: {len(tgt_full)} points (full resolution)")

    print(f"Downsampling with voxel size {args.voxel_size}...")
    src_coarse = downsample_cloud(src_full, args.voxel_size)
    tgt_coarse = downsample_cloud(tgt_full, args.voxel_size)
    print(f"  source: {len(src_coarse)} points, target: {len(tgt_coarse)} points (downsampled)")

    print("Fitting dominant plane on the source cloud...")
    src_normal, src_offset, src_inlier_mask = fit_dominant_plane(
        src_coarse, iterations=args.plane_iterations, threshold=args.plane_threshold)
    if src_normal is None:
        raise RuntimeError(
            "fit_dominant_plane found no plane (source cloud has fewer than 3 points after "
            "downsampling -- try a smaller --voxel-size)")
    src_normal, src_offset = canonicalize_plane_sign(src_normal, src_offset)
    print(f"  normal={src_normal}, offset={src_offset:.4f}, "
          f"inliers={int(src_inlier_mask.sum())}/{len(src_coarse)}")

    print("Fitting dominant plane on the target cloud...")
    tgt_normal, tgt_offset, tgt_inlier_mask = fit_dominant_plane(
        tgt_coarse, iterations=args.plane_iterations, threshold=args.plane_threshold)
    if tgt_normal is None:
        raise RuntimeError(
            "fit_dominant_plane found no plane (target cloud has fewer than 3 points after "
            "downsampling -- try a smaller --voxel-size)")
    tgt_normal, tgt_offset = canonicalize_plane_sign(tgt_normal, tgt_offset)
    print(f"  normal={tgt_normal}, offset={tgt_offset:.4f}, "
          f"inliers={int(tgt_inlier_mask.sum())}/{len(tgt_coarse)}")

    write_test_data(args.output, src_coarse, tgt_coarse,
                     src_normal, src_offset, tgt_normal, tgt_offset)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
