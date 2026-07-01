"""Front camera stitching via point-cloud projection."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.ndimage import distance_transform_edt

from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

STITCH_OUT_W = 1800
STITCH_OUT_H = 1000


@dataclass
class CamStitchParams:
    rot: np.ndarray    # (3, 3) body_R_cam rotation matrix
    trans: np.ndarray  # (3,) body_t_cam translation vector
    fx: float
    fy: float
    cx: float
    cy: float


def extract_stitch_params(response: image_pb2.ImageResponse) -> CamStitchParams:
    """Extract body-to-camera transform and pinhole intrinsics from an ImageResponse."""
    snapshot = response.shot.transforms_snapshot
    frame_name = response.shot.frame_name_image_sensor
    pose = get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name)
    if pose is None:
        raise ValueError(f"No transform from body to {frame_name!r}")
    M = pose.to_matrix()

    src = response.source
    which = src.WhichOneof("camera_models")
    if which == "pinhole":
        intr = src.pinhole.intrinsics
    elif which == "pinhole_brown_conrady":
        intr = src.pinhole_brown_conrady.intrinsics.pinhole_intrinsics
    elif which == "kannala_brandt":
        intr = src.kannala_brandt.intrinsics.pinhole_intrinsics
    else:
        raise ValueError(f"No pinhole intrinsics for {frame_name!r} (model={which!r})")

    return CamStitchParams(
        rot=M[:3, :3].copy(),
        trans=M[:3, 3].copy(),
        fx=intr.focal_length.x,
        fy=intr.focal_length.y,
        cx=intr.principal_point.x,
        cy=intr.principal_point.y,
    )


def _complete_depth(dep: np.ndarray, valid_threshold: float = 0, max_dist: float = None) -> np.ndarray:
    mask = (dep <= valid_threshold) | ~np.isfinite(dep)

    total = dep.size
    n_holes = int(mask.sum())
    print(f"[_complete_depth] shape={dep.shape} total={total} "
          f"holes={n_holes} ({100.0 * n_holes / total:.1f}%) "
          f"valid_threshold={valid_threshold} max_dist={max_dist}")

    if not mask.any() or mask.all():
        print(f"[_complete_depth] early return: mask.any()={mask.any()} "
              f"mask.all()={mask.all()} -> NOT filling any holes")
        return dep

    distances, nearest_idx = distance_transform_edt(
        mask, return_distances=True, return_indices=True)

    fill = mask if max_dist is None else mask & (distances <= max_dist)
    n_fill = int(fill.sum())
    n_skipped = n_holes - n_fill
    if n_skipped:
        print(f"[_complete_depth] {n_skipped} holes beyond max_dist={max_dist} "
              f"(max hole distance={distances[mask].max():.1f}px) left UNFILLED")

    out = dep.copy()
    out[fill] = dep[tuple(nearest_idx)][fill]

    # Verify no holes remain after completion
    remaining_mask = (out <= valid_threshold) | ~np.isfinite(out)
    n_remaining = int(remaining_mask.sum())
    print(f"[_complete_depth] filled={n_fill} remaining_holes={n_remaining} "
          f"all_finite={np.isfinite(out).all()} all_positive={(out > valid_threshold).all()}")
    if n_remaining:
        # A remaining hole means the nearest source pixel was itself invalid,
        # or the hole was skipped by max_dist.
        print(f"[_complete_depth] WARNING: {n_remaining} holes NOT filled "
              f"(nearest-source-invalid or max_dist skip)")

    return out


def _densify_output(out_rgb: np.ndarray, out_dep: np.ndarray, max_dist: float = 2.0) -> None:
    """Close sparse forward-splat gaps in the reprojected canvas, in place.

    Reprojection scatters ~2x(HxW) points into a much larger canvas, so painted
    pixels are separated by 1-2px gaps. Fill each empty pixel from its nearest
    painted neighbour (both depth and colour, using the same indices so they stay
    consistent), bounded by max_dist so genuinely empty background stays black.
    """
    empty = out_dep <= 0
    if not empty.any() or empty.all():
        return

    distances, (ii, jj) = distance_transform_edt(
        empty, return_distances=True, return_indices=True)
    fill = empty & (distances <= max_dist)

    out_dep[fill] = out_dep[ii[fill], jj[fill]]
    out_rgb[fill] = out_rgb[ii[fill], jj[fill]]

    n_before = int(empty.sum())
    n_after = int((out_dep <= 0).sum())
    print(f"[_densify_output] max_dist={max_dist} holes {n_before} -> {n_after} "
          f"(filled {n_before - n_after})")


def _backproject(
    rgb: np.ndarray, dep: np.ndarray, params: CamStitchParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project valid pixels to body-frame 3D points."""
    v_idx, u_idx = np.indices(dep.shape)
    valid = (dep > 0) & np.isfinite(dep)
    z = dep[valid]
    u = u_idx[valid].astype(np.float32)
    v = v_idx[valid].astype(np.float32)
    x = (u - params.cx) * z / params.fx
    y = (v - params.cy) * z / params.fy
    cam_pts = np.stack((x, y, z), axis=-1)
    body_pts = cam_pts @ params.rot.T + params.trans
    return body_pts, rgb[valid]


def _stitch_pointcloud(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray, use_nearest: bool = True
) -> None:
    """Depth-based stitcher: back-projects pixels to body-frame point cloud, then re-projects."""
    out_h, out_w = out_rgb.shape[:2]
    out_rgb[:] = 0.0
    out_dep[:] = 0.0

    if use_nearest:
        print("[_stitch_pointcloud] completing LEFT depth")
        l_dep = _complete_depth(l_dep)
        print("[_stitch_pointcloud] completing RIGHT depth")
        r_dep = _complete_depth(r_dep)

    pts_l, cols_l = _backproject(l_rgb, l_dep, l_params)
    pts_r, cols_r = _backproject(r_rgb, r_dep, r_params)

    pts = np.vstack((pts_l, pts_r))
    cols = np.vstack((cols_l, cols_r))

    # Keep only points in front of robot (body X = forward)
    fwd = pts[:, 0]
    keep = fwd > 0.1
    pts, cols, fwd = pts[keep], cols[keep], fwd[keep]

    # Virtual frontal projection: u = -Y/X * f + cx_v,  v = -Z/X * f + cy_v
    # cy_v is anchored to the camera's own principal point so the horizon stays
    # at the same canvas row regardless of how tall the output buffer is.
    f_virt = l_params.fx
    cx_v, cy_v = out_w / 2.0, l_params.cy
    u = (-pts[:, 1] * f_virt / pts[:, 0] + cx_v).astype(np.int32)
    v = (-pts[:, 2] * f_virt / pts[:, 0] + cy_v).astype(np.int32)

    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, cols, fwd = u[in_bounds], v[in_bounds], cols[in_bounds], fwd[in_bounds]

    # Paint far-to-near so closer points win
    order = np.argsort(fwd)[::-1]
    out_rgb[v[order], u[order]] = cols[order]
    out_dep[v[order], u[order]] = fwd[order]

    # Even with fully-completed input depth, the output canvas can have holes:
    # reprojection scatters sparse points, so neighbouring output pixels may be
    # skipped entirely. This is distinct from input-depth holes.
    out_total = out_dep.size
    out_holes = int((out_dep <= 0).sum())
    print(f"[_stitch_pointcloud] painted {len(order)} pts -> out_dep "
          f"holes={out_holes}/{out_total} ({100.0 * out_holes / out_total:.1f}%) "
          f"unique_pixels_painted={len(np.unique(v * out_w + u))}")

    # if use_nearest:
    #     _densify_output(out_rgb, out_dep)


def _stitch_cylindrical(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray,
) -> None:
    """
    Depth-free stitcher: for each virtual canvas pixel, back-computes the body-frame
    ray direction from the virtual K, rotates it into each camera frame using R only
    (translation ignored — assumes scene at infinity), then samples each camera with
    cv2.remap. Blends 50/50 in the overlap region. Depth is warped with the same
    maps using nearest-neighbour sampling to avoid blending across discontinuities.
    """
    out_h, out_w = out_rgb.shape[:2]

    f_virt = l_params.fx
    cx_v, cy_v = out_w / 2.0, l_params.cy

    # Body-frame ray for each virtual canvas pixel: d = [1, -(u-cx)/f, -(v-cy)/f]
    ug, vg = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    d_body = np.stack([
        np.ones((out_h, out_w), dtype=np.float32),
        -(ug - cx_v) / f_virt,
        -(vg - cy_v) / f_virt,
    ], axis=-1)  # (out_h, out_w, 3)

    def _make_warp(params: CamStitchParams, img_h: int, img_w: int):
        # Rotate body ray into camera frame (row-vector form: d_cam = d_body @ rot,
        # since rot = body_R_cam so rot.T maps body→cam, and (rot.T @ col).T = row @ rot)
        d_cam = d_body @ params.rot  # (out_h, out_w, 3)
        dz = d_cam[..., 2]
        safe_dz = np.where(dz > 0, dz, 1.0)  # avoid div-by-zero for behind-camera rays
        map_x = np.where(dz > 0, params.fx * d_cam[..., 0] / safe_dz + params.cx, -1.0).astype(np.float32)
        map_y = np.where(dz > 0, params.fy * d_cam[..., 1] / safe_dz + params.cy, -1.0).astype(np.float32)
        valid = (dz > 0) & (map_x >= 0) & (map_x < img_w) & (map_y >= 0) & (map_y < img_h)
        return map_x, map_y, valid

    l_h, l_w = l_rgb.shape[:2]
    r_h, r_w = r_rgb.shape[:2]
    map_xl, map_yl, valid_l = _make_warp(l_params, l_h, l_w)
    map_xr, map_yr, valid_r = _make_warp(r_params, r_h, r_w)

    warped_l = cv2.remap(l_rgb, map_xl, map_yl, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_r = cv2.remap(r_rgb, map_xr, map_yr, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    out_rgb[:] = 0.0
    out_rgb[valid_l & ~valid_r] = warped_l[valid_l & ~valid_r]
    out_rgb[valid_r & ~valid_l] = warped_r[valid_r & ~valid_l]
    both = valid_l & valid_r
    out_rgb[both] = 0.5 * (warped_l[both] + warped_r[both])

    # Warp depth with the same maps (nearest-neighbour avoids blending across discontinuities)
    warped_dl = cv2.remap(l_dep, map_xl, map_yl, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_dr = cv2.remap(r_dep, map_xr, map_yr, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    dep_valid_l = valid_l & (warped_dl > 0)
    dep_valid_r = valid_r & (warped_dr > 0)
    out_dep[:] = 0.0
    out_dep[dep_valid_l & ~dep_valid_r] = warped_dl[dep_valid_l & ~dep_valid_r]
    out_dep[dep_valid_r & ~dep_valid_l] = warped_dr[dep_valid_r & ~dep_valid_l]
    dep_both = dep_valid_l & dep_valid_r
    out_dep[dep_both] = 0.5 * (warped_dl[dep_both] + warped_dr[dep_both])


def compute_stitch(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray,
    use_depth: bool = True,
) -> None:
    """
    Stitch left and right front camera images onto a virtual forward-facing canvas.

    use_depth=True  — point-cloud projection (accurate at all ranges, requires depth).
    use_depth=False — direction-only inverse warp (no depth needed, parallax error
                      grows as objects get closer than ~3 m).

    Results written in-place to out_rgb (H, W, 3) float32 and out_dep (H, W) float32.
    out_dep is zeroed when use_depth=False.
    """
    if use_depth:
        _stitch_pointcloud(l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep)
    else:
        _stitch_cylindrical(l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep)
