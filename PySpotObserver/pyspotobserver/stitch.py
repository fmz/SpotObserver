"""Front camera stitching via point-cloud projection."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

STITCH_OUT_H = 480
STITCH_OUT_W = 960


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


def compute_stitch(
    l_rgb: np.ndarray, l_dep: np.ndarray, l_params: CamStitchParams,
    r_rgb: np.ndarray, r_dep: np.ndarray, r_params: CamStitchParams,
    out_rgb: np.ndarray, out_dep: np.ndarray,
) -> None:
    """
    Stitch left and right front camera images via point-cloud projection.

    Projects both cameras into the body frame, then re-projects onto a virtual
    forward-facing plane centred on the robot. Results written in-place to
    out_rgb (H, W, 3) float32 and out_dep (H, W) float32.
    Depth values in out_dep are body-frame X (forward distance in metres).
    """
    out_h, out_w = out_rgb.shape[:2]
    out_rgb[:] = 0.0
    out_dep[:] = 0.0

    pts_l, cols_l = _backproject(l_rgb, l_dep, l_params)
    pts_r, cols_r = _backproject(r_rgb, r_dep, r_params)

    pts = np.vstack((pts_l, pts_r))
    cols = np.vstack((cols_l, cols_r))

    # Keep only points in front of robot (body X = forward)
    fwd = pts[:, 0]
    keep = fwd > 0.1
    pts, cols, fwd = pts[keep], cols[keep], fwd[keep]

    # Virtual frontal projection: u = -Y/X * f + cx_v,  v = -Z/X * f + cy_v
    f_virt = l_params.fx
    cx_v, cy_v = out_w / 2.0, out_h / 2.0
    u = (-pts[:, 1] * f_virt / pts[:, 0] + cx_v).astype(np.int32)
    v = (-pts[:, 2] * f_virt / pts[:, 0] + cy_v).astype(np.int32)

    in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    u, v, cols, fwd = u[in_bounds], v[in_bounds], cols[in_bounds], fwd[in_bounds]

    # Paint far-to-near so closer points win
    order = np.argsort(fwd)[::-1]
    out_rgb[v[order], u[order]] = cols[order]
    out_dep[v[order], u[order]] = fwd[order]

    # Fill single-pixel gaps via dilation
    kernel = np.ones((3, 3), np.uint8)
    rgb_u8 = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_rgb[:] = cv2.dilate(rgb_u8, kernel, iterations=1).astype(np.float32) / 255.0
    dep_dilated = cv2.dilate(out_dep, kernel, iterations=1)
    out_dep[:] = np.where(out_dep > 0, out_dep, dep_dilated)
