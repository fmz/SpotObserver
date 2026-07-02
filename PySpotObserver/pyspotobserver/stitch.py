"""Front camera stitching via point-cloud projection."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

STITCH_OUT_W = 1800
STITCH_OUT_H = 1000

# Horizontal field of view the cylindrical canvas spans. Sets the angular scale
# (pixels per radian) so the two toed-out front cameras fill the rectangle; widen
# if the sides are clipped, narrow if there are large black bands.
STITCH_HFOV_DEG = 150.0


@dataclass
class CamStitchParams:
    rot: np.ndarray    # (3, 3) body_R_cam rotation matrix
    trans: np.ndarray  # (3,) body_t_cam translation vector
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"                 # "pinhole" | "brown_conrady" | "kannala_brandt"
    dist: Optional[np.ndarray] = None      # OpenCV distortion coeffs for the model
    # Cached (H, W, 2) grid of distortion-corrected normalized rays (x/z, y/z).
    _unproj: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # Cached cylindrical panorama remap: ((out_h, out_w, img_h, img_w), (map_x, map_y, valid)).
    _pano: Optional[tuple] = field(default=None, repr=False, compare=False)


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
        model, dist = "pinhole", None
    elif which == "pinhole_brown_conrady":
        di = src.pinhole_brown_conrady.intrinsics
        intr = di.pinhole_intrinsics
        model = "brown_conrady"
        dist = np.array([di.k1, di.k2, di.p1, di.p2, di.k3], dtype=np.float64)
    elif which == "kannala_brandt":
        di = src.kannala_brandt.intrinsics
        intr = di.pinhole_intrinsics
        model = "kannala_brandt"
        dist = np.array([di.k1, di.k2, di.k3, di.k4], dtype=np.float64)
    else:
        raise ValueError(f"No pinhole intrinsics for {frame_name!r} (model={which!r})")

    return CamStitchParams(
        rot=M[:3, :3].copy(),
        trans=M[:3, 3].copy(),
        fx=intr.focal_length.x,
        fy=intr.focal_length.y,
        cx=intr.principal_point.x,
        cy=intr.principal_point.y,
        model=model,
        dist=dist,
    )


def _normalized_rays(params: CamStitchParams, h: int, w: int) -> np.ndarray:
    """Per-pixel distortion-corrected normalized camera rays (x/z, y/z), shape (h, w, 2).

    Unprojects every pixel through the camera's real distortion model so back-projection
    is geometrically correct for fisheye (kannala_brandt) and brown-conrady lenses, not
    just the pinhole approximation. Cached on the params (intrinsics are fixed) so the
    iterative undistortion runs once per camera, not per frame.
    """
    cache = params._unproj
    if cache is not None and cache.shape[0] == h and cache.shape[1] == w:
        return cache

    v_idx, u_idx = np.indices((h, w), dtype=np.float32)
    pts = np.stack((u_idx.ravel(), v_idx.ravel()), axis=-1).reshape(-1, 1, 2).astype(np.float32)
    K = np.array([[params.fx, 0.0, params.cx],
                  [0.0, params.fy, params.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    if params.model == "kannala_brandt" and params.dist is not None:
        norm = cv2.fisheye.undistortPoints(pts, K, params.dist.reshape(4, 1))
    elif params.model == "brown_conrady" and params.dist is not None:
        norm = cv2.undistortPoints(pts, K, params.dist)
    else:  # pinhole (or missing coeffs): plain linear unprojection
        norm = np.empty_like(pts)
        norm[:, 0, 0] = (pts[:, 0, 0] - params.cx) / params.fx
        norm[:, 0, 1] = (pts[:, 0, 1] - params.cy) / params.fy

    rays = norm.reshape(h, w, 2).astype(np.float32)
    params._unproj = rays
    return rays


def _backproject(
    rgb: np.ndarray, dep: np.ndarray, params: CamStitchParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project valid pixels to body-frame 3D points using the camera's distortion model."""
    h, w = dep.shape
    valid = (dep > 0) & np.isfinite(dep)
    rays = _normalized_rays(params, h, w)
    z = dep[valid]
    xn = rays[..., 0][valid]
    yn = rays[..., 1][valid]
    cam_pts = np.stack((xn * z, yn * z, z), axis=-1)
    body_pts = cam_pts @ params.rot.T + params.trans
    return body_pts, rgb[valid]


def _cylindrical_color_map(
    params: CamStitchParams, out_h: int, out_w: int, img_h: int, img_w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed canvas->source remap for the cylindrical RGB panorama (rotation-only warp).

    For each canvas pixel, build the body-frame ray (the exact inverse of the depth
    path's cylindrical projection), rotate it into this camera, and project through the
    camera's real distortion model to a source pixel. Depends only on the fixed canvas
    geometry, pose, and intrinsics, so it is computed once and cached. Because it shares
    the depth path's projection, the colour panorama and the metric depth land on the
    same canvas — aligned for the far scene, offset by parallax up close.
    """
    cache = params._pano
    if cache is not None and cache[0] == (out_h, out_w, img_h, img_w):
        return cache[1]

    f_virt = (out_w / 2.0) / np.radians(STITCH_HFOV_DEG / 2.0)
    cx_v, cy_v = out_w / 2.0, out_h / 2.0
    ug, vg = np.meshgrid(np.arange(out_w, dtype=np.float32),
                         np.arange(out_h, dtype=np.float32))
    phi = (ug - cx_v) / f_virt
    # Inverse of the depth projection: azimuth phi about vertical, scaled elevation.
    d_body = np.stack([np.cos(phi), -np.sin(phi), (cy_v - vg) / f_virt], axis=-1)
    d_cam = d_body @ params.rot  # rotate body ray into camera frame

    z = d_cam[..., 2]
    in_front = z > 1e-6
    map_x = np.full((out_h, out_w), -1.0, np.float32)
    map_y = np.full((out_h, out_w), -1.0, np.float32)
    K = np.array([[params.fx, 0.0, params.cx],
                  [0.0, params.fy, params.cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    pts = d_cam[in_front].astype(np.float64).reshape(-1, 1, 3)
    zero = np.zeros(3)
    if params.model == "kannala_brandt" and params.dist is not None:
        proj, _ = cv2.fisheye.projectPoints(pts, zero, zero, K, params.dist.reshape(4, 1))
    elif params.model == "brown_conrady" and params.dist is not None:
        proj, _ = cv2.projectPoints(pts, zero, zero, K, params.dist)
    else:
        proj, _ = cv2.projectPoints(pts, zero, zero, K, None)
    proj = proj.reshape(-1, 2).astype(np.float32)
    map_x[in_front] = proj[:, 0]
    map_y[in_front] = proj[:, 1]
    valid = in_front & (map_x >= 0) & (map_x < img_w) & (map_y >= 0) & (map_y < img_h)

    result = (map_x, map_y, valid)
    params._pano = ((out_h, out_w, img_h, img_w), result)
    return result


def _stitch_hybrid(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray,
) -> None:
    """Hybrid stitcher: metric depth from the point cloud, colour from the RGB panorama.

    Depth: back-project both cameras (distortion-corrected) and splat Euclidean range
    onto a cylindrical canvas — sparse, but every value is a real measurement.
    Colour: warp the full RGB of both cameras onto the SAME cylindrical canvas by
    rotation only (like a phone panorama) — complete, no depth holes, but with parallax
    error up close. The two layers share the canvas geometry, so they align for the far
    scene; near objects show a small colour/depth offset (the panorama's parallax).
    """
    out_h, out_w = out_rgb.shape[:2]
    out_rgb[:] = 0.0
    out_dep[:] = 0.0

    f_virt = (out_w / 2.0) / np.radians(STITCH_HFOV_DEG / 2.0)
    cx_v, cy_v = out_w / 2.0, out_h / 2.0

    # --- Depth: cylindrical splat of the measured point cloud ---
    pts_l, _ = _backproject(l_rgb, l_dep, l_params)
    pts_r, _ = _backproject(r_rgb, r_dep, r_params)
    pts = np.vstack((pts_l, pts_r))
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    keep = X > 0.05                          # forward hemisphere
    X, Y, Z = X[keep], Y[keep], Z[keep]
    rho = np.sqrt(X * X + Y * Y)             # ground-plane (horizontal) distance
    rng = np.sqrt(rho * rho + Z * Z)         # Euclidean range along the ray
    phi = np.arctan2(-Y, X)                  # azimuth, + to the right (body -Y)
    u = (f_virt * phi + cx_v).astype(np.int32)
    v = (-f_virt * (Z / rho) + cy_v).astype(np.int32)
    ib = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, rng = u[ib], v[ib], rng[ib]
    order = np.argsort(rng)[::-1]            # far-to-near so nearer wins
    out_dep[v[order], u[order]] = rng[order]

    # --- Colour: cylindrical RGB panorama (rotation-only inverse warp) ---
    l_h, l_w = l_rgb.shape[:2]
    r_h, r_w = r_rgb.shape[:2]
    map_xl, map_yl, valid_l = _cylindrical_color_map(l_params, out_h, out_w, l_h, l_w)
    map_xr, map_yr, valid_r = _cylindrical_color_map(r_params, out_h, out_w, r_h, r_w)
    warped_l = cv2.remap(l_rgb, map_xl, map_yl, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_r = cv2.remap(r_rgb, map_xr, map_yr, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # Seam: canvas left half from FRONTRIGHT (body +Y), right half from FRONTLEFT,
    # falling back to the other camera where the preferred one has no coverage.
    left_half = np.zeros((out_h, out_w), dtype=bool)
    left_half[:, : out_w // 2] = True
    out_rgb[valid_r & left_half] = warped_r[valid_r & left_half]
    out_rgb[valid_l & ~left_half] = warped_l[valid_l & ~left_half]
    out_rgb[~valid_r & valid_l & left_half] = warped_l[~valid_r & valid_l & left_half]
    out_rgb[~valid_l & valid_r & ~left_half] = warped_r[~valid_l & valid_r & ~left_half]


def _stitch_direction_only(
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

    # Hard seam cut: left half of canvas from r (FRONTRIGHT), right half from l (FRONTLEFT).
    # Spot names cameras from the perspective of someone facing the robot, so FRONTLEFT sits
    # on the robot's right side (body -Y) and FRONTRIGHT on the robot's left (body +Y).
    # valid_r therefore covers the canvas left half and valid_l covers the canvas right half.
    # Fall back to the other camera where the preferred one has no valid coverage.
    left_half = np.zeros((out_h, out_w), dtype=bool)
    left_half[:, : out_w // 2] = True

    out_rgb[:] = 0.0
    out_rgb[valid_r & left_half] = warped_r[valid_r & left_half]
    out_rgb[valid_l & ~left_half] = warped_l[valid_l & ~left_half]
    out_rgb[~valid_r & valid_l & left_half] = warped_l[~valid_r & valid_l & left_half]
    out_rgb[~valid_l & valid_r & ~left_half] = warped_r[~valid_l & valid_r & ~left_half]

    # Warp depth with the same maps (nearest-neighbour avoids blending across discontinuities)
    warped_dl = cv2.remap(l_dep, map_xl, map_yl, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_dr = cv2.remap(r_dep, map_xr, map_yr, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    dep_valid_l = valid_l & (warped_dl > 0)
    dep_valid_r = valid_r & (warped_dr > 0)
    out_dep[:] = 0.0
    out_dep[dep_valid_r & left_half] = warped_dr[dep_valid_r & left_half]
    out_dep[dep_valid_l & ~left_half] = warped_dl[dep_valid_l & ~left_half]
    out_dep[~dep_valid_r & dep_valid_l & left_half] = warped_dl[~dep_valid_r & dep_valid_l & left_half]
    out_dep[~dep_valid_l & dep_valid_r & ~left_half] = warped_dr[~dep_valid_l & dep_valid_r & ~left_half]


def compute_stitch(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray,
    use_depth: bool = True,
) -> None:
    """
    Stitch left and right front camera images onto a virtual canvas.

    use_depth=True  — hybrid: metric depth from the cylindrical point-cloud splat
                      (sparse, distortion-corrected) + complete RGB from the cylindrical
                      rotation-only panorama. out_dep holds Euclidean range along each
                      cylindrical ray; colour is a full panorama with parallax error up close.
    use_depth=False — direction-only inverse warp (no depth, flat projection, parallax
                      error grows as objects get closer than ~3 m).

    Results written in-place to out_rgb (H, W, 3) float32 and out_dep (H, W) float32.
    """
    if use_depth:
        _stitch_hybrid(l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep)
    else:
        _stitch_direction_only(l_rgb, l_dep, l_params, r_rgb, r_dep, r_params, out_rgb, out_dep)
