"""
Exports a source/target point cloud pair -- plus a dominant-plane fit for the
source cloud -- to a plain-text file the C++ standalone tests
(four-pcs-test.cpp, four-pcs-gpu-test.cpp) can load without needing Python,
Eigen bindings, or a point cloud library: just whitespace-separated numbers.

The clouds written out are voxel-downsampled, matching how the real
fourPointCongruentSets() call is used in practice -- the same coarse clouds
serve both the base search and the final whole-cloud scoring, not the raw
backprojected depth image (hundreds of thousands of points, far too many for
the brute-force base search to be practical).

Output format (all whitespace-separated, no fixed column widths):
    n_src n_tgt
    nx ny nz offset
    x y z            (repeated n_src times -- source cloud)
    x y z            (repeated n_tgt times -- target cloud)

Usage:
    python export_four_pcs_test_data.py <output.txt> \\
        [--source-index N] [--target-index N] [--voxel-size V]
"""

import argparse
import struct

import numpy as np

from pyspotobserver.four_pcs import downsample_cloud, fit_dominant_plane

DATA_DIR = "/Users/adannaobuba/Documents/Brown/GHOST/GHOST/Assets/PointClouds"


def load_mesh_array(path, width=640, height=480):
    """Matches test_icp_align.py's convention: a little-endian int32 point
    count followed by that many little-endian float32 depth values, one per
    pixel, row-major."""
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack("<i", data[:4])[0]
    assert length == width * height, (
        f"{path}: expected {width * height} depth values, file has {length}"
    )
    return np.frombuffer(data[4:4 + length * 4], dtype="<f4").reshape(height, width)


def backproject(depth, fx, fy, cx, cy):
    h, w = depth.shape
    v, u = np.indices((h, w))
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)  # camera-frame points, Nx3


def write_test_data(path, source, target, plane_normal, plane_offset):
    with open(path, "w") as f:
        f.write(f"{len(source)} {len(target)}\n")
        f.write(f"{plane_normal[0]} {plane_normal[1]} {plane_normal[2]} {plane_offset}\n")
        for p in source:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for p in target:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path to write the exported test data to")
    parser.add_argument("--source-index", type=int, default=0,
                         help="mesh_array_<N> to use as the source cloud (default 0)")
    parser.add_argument("--target-index", type=int, default=1,
                         help="mesh_array_<N> to use as the target cloud (default 1)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="downsample_cloud() voxel size for the coarse clouds used in "
                              "both the base search and final scoring (default 0.05)")
    parser.add_argument("--plane-iterations", type=int, default=300,
                         help="RANSAC iterations for fit_dominant_plane() on the source cloud "
                              "(default 300)")
    parser.add_argument("--plane-threshold", type=float, default=0.04,
                         help="inlier distance threshold for fit_dominant_plane() (default 0.04)")
    args = parser.parse_args()

    w, h = 640, 480
    fx = fy = max(w, h)
    cx, cy = w / 2, h / 2

    src_depth = load_mesh_array(f"{DATA_DIR}/mesh_array_{args.source_index}", w, h)
    tgt_depth = load_mesh_array(f"{DATA_DIR}/mesh_array_{args.target_index}", w, h)

    print(f"Backprojecting mesh_array_{args.source_index} and mesh_array_{args.target_index}...")
    src_full = backproject(src_depth, fx, fy, cx, cy)
    tgt_full = backproject(tgt_depth, fx, fy, cx, cy)
    print(f"  source: {len(src_full)} points, target: {len(tgt_full)} points (full resolution)")

    print(f"Downsampling with voxel size {args.voxel_size}...")
    src_coarse = downsample_cloud(src_full, args.voxel_size)
    tgt_coarse = downsample_cloud(tgt_full, args.voxel_size)
    print(f"  source: {len(src_coarse)} points, target: {len(tgt_coarse)} points (downsampled)")

    print("Fitting dominant plane on the source cloud...")
    normal, offset, inlier_mask = fit_dominant_plane(
        src_coarse, iterations=args.plane_iterations, threshold=args.plane_threshold)
    if normal is None:
        raise RuntimeError(
            "fit_dominant_plane found no plane (source cloud has fewer than 3 points after "
            "downsampling -- try a smaller --voxel-size)")
    print(f"  normal={normal}, offset={offset:.4f}, inliers={int(inlier_mask.sum())}/{len(src_coarse)}")

    write_test_data(args.output, src_coarse, tgt_coarse, normal, offset)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
