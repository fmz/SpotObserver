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


@dataclass
class DepthRegistrationWorkspace:
    """Reusable scratch buffers for :func:`register_depth`."""

    raw_depth: np.ndarray
    points: np.ndarray
    projected_x: np.ndarray
    projected_y: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    splat_valid: np.ndarray
    bool_scratch: np.ndarray
    z_buffer: np.ndarray
    dst_empty: np.ndarray


def create_registration_workspace(
    params: DepthRegistrationParams,
) -> DepthRegistrationWorkspace:
    """Allocate scratch storage once for repeated registration with ``params``."""
    src_shape = params.src_shape
    dst_h, dst_w = params.dst_shape
    return DepthRegistrationWorkspace(
        raw_depth=np.empty(src_shape, dtype=np.float32),
        points=np.empty((*src_shape, 3), dtype=np.float32),
        projected_x=np.empty(src_shape, dtype=np.float32),
        projected_y=np.empty(src_shape, dtype=np.float32),
        pixel_x=np.empty(src_shape, dtype=np.int32),
        pixel_y=np.empty(src_shape, dtype=np.int32),
        indices=np.empty(src_shape, dtype=np.intp),
        valid=np.empty(src_shape, dtype=np.bool_),
        splat_valid=np.empty(src_shape, dtype=np.bool_),
        bool_scratch=np.empty(src_shape, dtype=np.bool_),
        z_buffer=np.empty(dst_h * dst_w + 1, dtype=np.float32),
        dst_empty=np.empty((dst_h, dst_w), dtype=np.bool_),
    )


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
    workspace: DepthRegistrationWorkspace | None = None,
) -> None:
    """Reproject a raw depth image into the RGB frame, in-place into ``out``.

    Args:
        raw_depth: (H_src, W_src) float32 depth in meters, 0 = invalid.
        params: Registration params for this camera pair.
        out: (H_dst, W_dst) float32 output, fully overwritten. Pixels that
            receive no depth sample are 0.
        workspace: Optional reusable scratch storage. Supplying one avoids
            per-frame allocations in streaming callers.
    """
    if raw_depth.shape != params.src_shape:
        raise ValueError(f"Raw depth shape {raw_depth.shape} != expected {params.src_shape}")
    if out.shape != params.dst_shape:
        raise ValueError(f"Output shape {out.shape} != expected {params.dst_shape}")
    if workspace is None:
        workspace = create_registration_workspace(params)
    if (
        workspace.raw_depth.shape != params.src_shape
        or workspace.dst_empty.shape != params.dst_shape
    ):
        raise ValueError("Registration workspace shape does not match params")

    valid = workspace.valid
    np.greater(raw_depth, 0.0, out=valid)
    if not valid.any():
        out.fill(0.0)
        return

    points = workspace.points
    np.multiply(params.rays, raw_depth[..., None], out=points)
    np.add(points, params.t, out=points)
    pz = points[..., 2]
    np.greater(pz, 0.0, out=workspace.bool_scratch)
    np.logical_and(valid, workspace.bool_scratch, out=valid)
    if not valid.any():
        out.fill(0.0)
        return

    np.divide(points[..., 0], pz, out=workspace.projected_x, where=valid)
    np.divide(points[..., 1], pz, out=workspace.projected_y, where=valid)
    np.floor(workspace.projected_x, out=workspace.projected_x)
    np.floor(workspace.projected_y, out=workspace.projected_y)
    workspace.pixel_x.fill(0)
    workspace.pixel_y.fill(0)
    np.copyto(workspace.pixel_x, workspace.projected_x, where=valid, casting="unsafe")
    np.copyto(workspace.pixel_y, workspace.projected_y, where=valid, casting="unsafe")

    dst_h, dst_w = params.dst_shape
    sentinel_index = dst_h * dst_w
    workspace.z_buffer.fill(np.inf)

    # A z-buffered 2x2 splat. Invalid samples all target the extra sentinel
    # element, avoiding boolean-indexed temporaries and a global depth sort.
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        splat_valid = workspace.splat_valid
        np.copyto(splat_valid, valid)
        np.greater_equal(workspace.pixel_x, -du, out=workspace.bool_scratch)
        np.logical_and(splat_valid, workspace.bool_scratch, out=splat_valid)
        np.less(workspace.pixel_x, dst_w - du, out=workspace.bool_scratch)
        np.logical_and(splat_valid, workspace.bool_scratch, out=splat_valid)
        np.greater_equal(workspace.pixel_y, -dv, out=workspace.bool_scratch)
        np.logical_and(splat_valid, workspace.bool_scratch, out=splat_valid)
        np.less(workspace.pixel_y, dst_h - dv, out=workspace.bool_scratch)
        np.logical_and(splat_valid, workspace.bool_scratch, out=splat_valid)

        np.multiply(workspace.pixel_y, dst_w, out=workspace.indices, casting="unsafe")
        np.add(workspace.indices, workspace.pixel_x, out=workspace.indices)
        np.add(workspace.indices, dv * dst_w + du, out=workspace.indices)
        np.logical_not(splat_valid, out=workspace.bool_scratch)
        np.copyto(workspace.indices, sentinel_index, where=workspace.bool_scratch)
        np.minimum.at(workspace.z_buffer, workspace.indices, pz)

    np.copyto(out, workspace.z_buffer[:-1].reshape(params.dst_shape))
    np.isinf(out, out=workspace.dst_empty)
    np.copyto(out, 0.0, where=workspace.dst_empty)
