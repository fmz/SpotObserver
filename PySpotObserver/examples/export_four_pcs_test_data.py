"""
Dev utility: export a real captured point cloud pair to a plain text file that
tests/standalone-tests/four-pcs-test.cpp can load, so the C++ port of
four_pcs.py can be checked against real data without needing a live robot or
a Python interpreter at test time.

Not part of the production pipeline -- this is only a bridge for the C++
correctness test. Reuses the same mesh_array loading and backprojection
convention as tests/test_icp_align.py (DATA_DIR, load_mesh_array) rather than
inventing a new data path.

Usage:
    python export_four_pcs_test_data.py <output.txt> [--src N] [--tgt M]

Output format (plain text, whitespace-separated):
    <n_source_points> <n_target_points>
    <plane_normal_x> <plane_normal_y> <plane_normal_z> <plane_offset>
    <source point 1 x y z>
    ...
    <source point n x y z>
    <target point 1 x y z>
    ...
    <target point m x y z>.

The plane line is fit_dominant_plane()'s result on the FULL source cloud
(before downsampling), matching how it's used in the real pipeline -- plane
fitting needs the full-resolution cloud, base search runs on the downsampled
one. See four_pcs.py's docstring for why.
"""

import argparse
import struct
import sys

import numpy as np

from pyspotobserver.four_pcs import fit_dominant_plane, downsample_cloud

DATA_DIR = "/Users/adannaobuba/Documents/Brown/GHOST/GHOST/Assets/PointClouds"


def load_mesh_array(path, width=640, height=480):
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack("<i", data[:4])[0]
    assert length == width * height
    return np.frombuffer(data[4:4 + length * 4], dtype="<f4").reshape(height, width)


def backproject(depth, fx, fy, cx, cy):
    h, w = depth.shape
    v, u = np.indices((h, w))
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path to write the exported test data to")
    parser.add_argument("--source-index", type=int, default=0,
                         help="mesh_array_<N> to use as the source cloud (default 0)")
    parser.add_argument("--target-index", type=int, default=1,
                         help="mesh_array_<N> to use as the target cloud (default 1)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="downsample_cloud() voxel size for the coarse clouds "
                              "used in the base search (default 0.05)")
    args = parser.parse_args()

    w, h = 640, 480
    fx = fy = max(w, h)
    cx, cy = w / 2, h / 2

    src_depth = load_mesh_array(f"{DATA_DIR}/mesh_array_{args.source_index}", w, h)
    tgt_depth = load_mesh_array(f"{DATA_DIR}/mesh_array_{args.target_index}", w, h)

    src_full = backproject(src_depth, fx, fy, cx, cy)
    tgt_full = backproject(tgt_depth, fx, fy, cx, cy)

    # plane fit on the FULL source cloud, matching real usage
    normal, offset, _ = fit_dominant_plane(src_full, iterations=300, threshold=0.04, seed=1)
    if normal is None:
        print("fit_dominant_plane found nothing -- source cloud too small?", file=sys.stderr)
        sys.exit(1)

    src_coarse = downsample_cloud(src_full, args.voxel_size)
    tgt_coarse = downsample_cloud(tgt_full, args.voxel_size)

    with open(args.output, "w") as f:
        f.write(f"{len(src_coarse)} {len(tgt_coarse)}\n")
        f.write(f"{normal[0]} {normal[1]} {normal[2]} {offset}\n")
        for p in src_coarse:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        for p in tgt_coarse:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"wrote {args.output}: source coarse={len(src_coarse)} pts, "
          f"target coarse={len(tgt_coarse)} pts, plane normal={normal}, offset={offset}")


if __name__ == "__main__":
    main()
