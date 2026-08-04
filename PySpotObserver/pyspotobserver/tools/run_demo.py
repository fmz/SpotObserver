"""
One-command demo pipeline: export -> align (C++ GPU) -> render.

Assumes capture_dual.py has ALREADY been run (fresh robotA_*/robotB_* files
in captures/). Then:

  1. exports the coarse cloud pair + plane fits via
     export_four_pcs_test_data.py (subprocess, same interpreter);
  2. runs the align_demo executable (both 4PCS+ICP and wall-align, winner by
     shared inlier metric -- its stdout streams through live so the audience
     can watch the scores come in);
  3. reads the winning transform and renders a true-color combined PLY
     (robotA transformed onto robotB, real RGB from the capture images) plus
     a quick top-down PNG sanity view (if matplotlib is available).

Outputs land in captures/demo_out/:
    demo_latest.ply / demo_latest.png     -- always overwritten, open these
    demo_<timestamp>.ply / .png           -- kept per run

Usage:
    python run_demo.py --exe path/to/align_demo[.exe]
    python run_demo.py --exe ... --coarse      # fast preview-quality PLY

Exit codes: 0 ok, 1 pipeline error, 2 aligned but below the 0.10 quality
gate (result rendered anyway -- eyeball it, but expect it to be wrong;
usually means the capture pair lacks shared structure).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS_DIR = Path(__file__).parent
CAPTURES_DIR = TOOLS_DIR / "captures"
OUT_DIR = CAPTURES_DIR / "demo_out"


def run_step(name, cmd, cwd=None):
    print(f"\n=== {name}: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    return proc.returncode


def read_result(path):
    method, fraction, R, t = None, None, None, None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "method":
                method = parts[1]
            elif parts[0] == "fraction":
                fraction = float(parts[1])
            elif parts[0] == "R":
                R = np.array([float(v) for v in parts[1:10]]).reshape(3, 3)
            elif parts[0] == "t":
                t = np.array([float(v) for v in parts[1:4]])
    if R is None or t is None:
        raise ValueError(f"{path} is missing R/t lines")
    return method, fraction, R, t


def backproject_with_color(name, tag):
    depth = np.load(CAPTURES_DIR / f"{name}_{tag}_depth.npy")
    rgb = np.array(Image.open(CAPTURES_DIR / f"{name}_{tag}_rgb.png"))
    with open(CAPTURES_DIR / f"{name}_{tag}_calib.json") as f:
        calib = json.load(f)
    h, w = depth.shape
    v, u = np.indices((h, w))
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - calib["cx"]) * z / calib["fx"]
    y = (v[valid] - calib["cy"]) * z / calib["fy"]
    pts = np.stack([x, y, z], -1) @ np.array(calib["rot"]).T + np.array(calib["trans"])
    return pts, rgb[valid]


def load_robot_colored(name):
    parts = [backproject_with_color(name, t) for t in ("frontleft", "frontright")]
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]))


def write_colored_ply(path, points, colors):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.hstack([points, colors.astype(int)]), fmt="%.5f %.5f %.5f %d %d %d")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True,
                         help="path to the built align_demo executable")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                         help="export voxel size (default 0.05, matching the tests)")
    parser.add_argument("--coarse", action="store_true",
                         help="render the PLY from a voxel-downsampled cloud (much faster to "
                              "write and load; preview quality)")
    args = parser.parse_args()

    exe = Path(args.exe)
    if not exe.exists():
        print(f"align_demo executable not found at {exe} -- build it first "
              f"(cmake target: align_demo)")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    data_txt = OUT_DIR / "demo_export.txt"
    result_txt = OUT_DIR / "demo_result.txt"
    t0 = time.time()

    # ---- 1. export ----
    rc = run_step("export", [sys.executable, TOOLS_DIR / "export_four_pcs_test_data.py",
                              data_txt, "--voxel-size", args.voxel_size])
    if rc != 0:
        print("export failed -- did capture_dual.py run successfully?")
        return 1

    # ---- 2. align ----
    rc = run_step("align", [exe, data_txt, result_txt])
    if rc == 1:
        print("align_demo failed")
        return 1
    low_quality = rc == 2

    method, fraction, R, t = read_result(result_txt)
    print(f"\nwinner: {method}, inlier fraction {fraction:.3f}  [{time.time()-t0:.1f}s]")

    # ---- 3. render ----
    src_pts, src_cols = load_robot_colored("robotA")
    tgt_pts, tgt_cols = load_robot_colored("robotB")
    if args.coarse:
        keep_s = np.random.default_rng(0).random(len(src_pts)) < 30000 / len(src_pts)
        keep_t = np.random.default_rng(1).random(len(tgt_pts)) < 30000 / len(tgt_pts)
        src_pts, src_cols = src_pts[keep_s], src_cols[keep_s]
        tgt_pts, tgt_cols = tgt_pts[keep_t], tgt_cols[keep_t]

    src_aligned = src_pts @ R.T + t
    pts = np.vstack([src_aligned, tgt_pts])
    cols = np.vstack([src_cols, tgt_cols])

    ply_stamped = OUT_DIR / f"demo_{stamp}.ply"
    write_colored_ply(ply_stamped, pts, cols)
    write_colored_ply(OUT_DIR / "demo_latest.ply", pts, cols)
    print(f"wrote {ply_stamped} and demo_latest.ply  [{time.time()-t0:.1f}s]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        step = max(1, len(src_aligned) // 40000)
        ax.scatter(tgt_pts[::step, 0], tgt_pts[::step, 1], s=0.5, c="blue",
                   alpha=0.5, label="robotB")
        ax.scatter(src_aligned[::step, 0], src_aligned[::step, 1], s=0.5, c="red",
                   alpha=0.5, label="robotA (aligned)")
        ax.set_aspect("equal")
        ax.legend(markerscale=10)
        ax.set_title(f"{method}  fraction={fraction:.2f}  ({stamp})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"demo_{stamp}.png", dpi=90)
        fig.savefig(OUT_DIR / "demo_latest.png", dpi=90)
        print(f"wrote demo_{stamp}.png and demo_latest.png")
    except ImportError:
        print("matplotlib not installed -- skipping PNG (PLY still written)")

    if low_quality:
        print("\nWARNING: inlier fraction below 0.10 -- this alignment is probably wrong. "
              "Reposition the robots so both cameras share distinctive, non-floor structure "
              "and recapture.")
        return 2
    print(f"\ndone in {time.time()-t0:.1f}s -- open demo_latest.ply (MeshLab/CloudCompare) "
          f"or demo_latest.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
