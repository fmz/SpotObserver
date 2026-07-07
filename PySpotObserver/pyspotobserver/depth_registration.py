"""Client-side registration of raw depth images into the RGB camera frame.

Spot's ``*_depth_in_visual_frame`` sources are the robot doing this registration
onboard, at the cost of shipping RGB-sized depth images over the wire. Requesting
the raw depth-sensor sources (e.g. ``frontleft_depth``) is substantially cheaper,
but the pixels live in the depth sensor's frame. This module reprojects them into
the RGB camera's image plane so downstream consumers see the same aligned,
RGB-sized depth they would get from the robot-registered sources.

Mirrors the C++ implementation (``register_depth_to_rgb`` in cuda_kernels.cu).
"""

from dataclasses import dataclass

import numpy as np
from bosdyn.api import image_pb2  # type: ignore[import-untyped]
from bosdyn.client.frame_helpers import (  # type: ignore[import-untyped]
    BODY_FRAME_NAME,
    get_a_tform_b,
)


@dataclass
class DepthRegistrationParams:
    """Folded reprojection from a raw depth image into an RGB image plane.

    For a depth pixel (u, v) with depth z (meters):

        p = z * rays[v, u] + t

    where ``rays[v, u] = M @ [u, v, 1]`` with ``M = K_rgb @ R @ inv(K_depth)`` and
    ``t = K_rgb @ translation`` (R, translation from rgb_T_depth). Then
    ``(p[0] / p[2], p[1] / p[2])`` is the RGB pixel and ``p[2]`` is the depth in
    the RGB camera's frame.
    """

    rays: np.ndarray  # (H_src, W_src, 3) float32
    t: np.ndarray  # (3,) float32
    src_shape: tuple[int, int]  # (H_src, W_src)
    dst_shape: tuple[int, int]  # (H_dst, W_dst)


def _pinhole_intrinsics(source: image_pb2.ImageSource) -> np.ndarray:
    """Return the 3x3 pinhole camera matrix of an ImageSource."""
    which = source.WhichOneof("camera_models")
    if which == "pinhole":
        intr = source.pinhole.intrinsics
    elif which == "pinhole_brown_conrady":
        intr = source.pinhole_brown_conrady.intrinsics.pinhole_intrinsics
    elif which == "kannala_brandt":
        intr = source.kannala_brandt.intrinsics.pinhole_intrinsics
    else:
        raise ValueError(f"No pinhole intrinsics for {source.name!r} (model={which!r})")

    return np.array(
        [
            [intr.focal_length.x, intr.skew.x, intr.principal_point.x],
            [intr.skew.y, intr.focal_length.y, intr.principal_point.y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _body_t_sensor(response: image_pb2.ImageResponse) -> np.ndarray:
    snapshot = response.shot.transforms_snapshot
    frame_name = response.shot.frame_name_image_sensor
    pose = get_a_tform_b(snapshot, BODY_FRAME_NAME, frame_name)
    if pose is None:
        raise ValueError(f"No transform from body to {frame_name!r}")
    return pose.to_matrix()


def extract_registration_params(
    rgb_response: image_pb2.ImageResponse,
    depth_response: image_pb2.ImageResponse,
    dst_shape: tuple[int, int],
) -> DepthRegistrationParams:
    """Build registration params from a time-synced RGB + raw depth response pair.

    Intrinsics and extrinsics are fixed by the mechanical mounting, so this is
    computed once per stream and reused for every frame.
    """
    body_T_rgb = _body_t_sensor(rgb_response)
    body_T_depth = _body_t_sensor(depth_response)
    rgb_T_depth = np.linalg.inv(body_T_rgb) @ body_T_depth

    k_rgb = _pinhole_intrinsics(rgb_response.source)
    k_depth = _pinhole_intrinsics(depth_response.source)

    M = k_rgb @ rgb_T_depth[:3, :3] @ np.linalg.inv(k_depth)
    t = k_rgb @ rgb_T_depth[:3, 3]

    src_h = depth_response.shot.image.rows
    src_w = depth_response.shot.image.cols
    v, u = np.indices((src_h, src_w), dtype=np.float64)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H_src, W_src, 3)
    rays = (uv1 @ M.T).astype(np.float32)

    return DepthRegistrationParams(
        rays=rays,
        t=t.astype(np.float32),
        src_shape=(src_h, src_w),
        dst_shape=(int(dst_shape[0]), int(dst_shape[1])),
    )


def register_depth(
    raw_depth: np.ndarray,
    params: DepthRegistrationParams,
    out: np.ndarray,
) -> None:
    """Reproject a raw depth image into the RGB frame, in-place into ``out``.

    Args:
        raw_depth: (H_src, W_src) float32 depth in meters, 0 = invalid.
        params: Registration params for this camera pair.
        out: (H_dst, W_dst) float32 output, fully overwritten. Pixels that
            receive no depth sample are 0.
    """
    if raw_depth.shape != params.src_shape:
        raise ValueError(f"Raw depth shape {raw_depth.shape} != expected {params.src_shape}")
    if out.shape != params.dst_shape:
        raise ValueError(f"Output shape {out.shape} != expected {params.dst_shape}")

    out.fill(0.0)

    valid = raw_depth > 0
    if not valid.any():
        return

    z = raw_depth[valid]
    p = z[:, None] * params.rays[valid] + params.t
    pz = p[:, 2]

    front = pz > 0  # discard points behind the RGB camera
    p, pz = p[front], pz[front]
    if pz.size == 0:
        return

    u = np.floor(p[:, 0] / pz).astype(np.int32)
    v = np.floor(p[:, 1] / pz).astype(np.int32)

    # 2x2 splat: the raw depth image is lower resolution than the RGB target, so
    # covering the four nearest pixels reduces holes without a dilation pass.
    xs = np.concatenate((u, u + 1, u, u + 1))
    ys = np.concatenate((v, v, v + 1, v + 1))
    zs = np.concatenate((pz, pz, pz, pz))

    dst_h, dst_w = params.dst_shape
    in_bounds = (xs >= 0) & (xs < dst_w) & (ys >= 0) & (ys < dst_h)
    xs, ys, zs = xs[in_bounds], ys[in_bounds], zs[in_bounds]

    # Paint far-to-near so the nearest surface wins where samples overlap.
    order = np.argsort(zs)[::-1]
    out[ys[order], xs[order]] = zs[order]
