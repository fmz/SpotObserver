"""
Unit tests for client-side depth registration.
"""

import numpy as np
import pytest
from bosdyn.api import image_pb2  # type: ignore[import-untyped]
from pyspotobserver.depth_registration import (
    DepthRegistrationParams,
    extract_registration_params,
    register_depth,
)


def make_params(M, t, src_shape, dst_shape) -> DepthRegistrationParams:
    """Build params directly from a projection matrix, bypassing proto extraction."""
    v, u = np.indices(src_shape, dtype=np.float64)
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
    rays = (uv1 @ np.asarray(M, dtype=np.float64).T).astype(np.float32)
    return DepthRegistrationParams(
        rays=rays,
        t=np.asarray(t, dtype=np.float32),
        src_shape=src_shape,
        dst_shape=dst_shape,
    )


class TestRegisterDepth:
    def test_identity_maps_constant_plane(self):
        """With an identity projection, a constant plane fills the whole output."""
        params = make_params(np.eye(3), np.zeros(3), src_shape=(6, 8), dst_shape=(6, 8))
        raw = np.full((6, 8), 2.0, dtype=np.float32)
        out = np.empty((6, 8), dtype=np.float32)

        register_depth(raw, params, out)

        np.testing.assert_allclose(out, 2.0)

    def test_nearest_surface_wins(self):
        """Two samples landing on the same pixel keep the nearer depth."""
        # Both source pixels project along the optical axis to dst pixel (0, 0).
        params = make_params(np.zeros((3, 3)), np.zeros(3), src_shape=(1, 2), dst_shape=(2, 2))
        params.rays[0, 0] = (0.0, 0.0, 1.0)
        params.rays[0, 1] = (0.0, 0.0, 1.0)
        raw = np.array([[5.0, 1.0]], dtype=np.float32)
        out = np.empty((2, 2), dtype=np.float32)

        register_depth(raw, params, out)

        np.testing.assert_allclose(out, 1.0)  # 2x2 splat covers the full output

    def test_invalid_depth_leaves_holes(self):
        params = make_params(np.eye(3), np.zeros(3), src_shape=(4, 4), dst_shape=(4, 4))
        raw = np.zeros((4, 4), dtype=np.float32)
        raw[1, 1] = 3.0
        out = np.empty((4, 4), dtype=np.float32)

        register_depth(raw, params, out)

        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1:3, 1:3] = 3.0  # the sample plus its 2x2 splat
        np.testing.assert_allclose(out, expected)

    def test_points_behind_camera_are_skipped(self):
        params = make_params(-np.eye(3), np.zeros(3), src_shape=(4, 4), dst_shape=(4, 4))
        raw = np.full((4, 4), 2.0, dtype=np.float32)
        out = np.empty((4, 4), dtype=np.float32)

        register_depth(raw, params, out)

        np.testing.assert_allclose(out, 0.0)

    def test_shape_mismatch_raises(self):
        params = make_params(np.eye(3), np.zeros(3), src_shape=(4, 4), dst_shape=(4, 4))
        with pytest.raises(ValueError, match="Raw depth shape"):
            register_depth(np.zeros((2, 2), dtype=np.float32), params, np.zeros((4, 4), np.float32))
        with pytest.raises(ValueError, match="Output shape"):
            register_depth(np.zeros((4, 4), dtype=np.float32), params, np.zeros((2, 2), np.float32))


def _make_response(
    frame_name: str,
    *,
    rows: int,
    cols: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> image_pb2.ImageResponse:
    """Build a minimal ImageResponse with pinhole intrinsics and a body->sensor edge."""
    response = image_pb2.ImageResponse()
    response.shot.frame_name_image_sensor = frame_name
    response.shot.image.rows = rows
    response.shot.image.cols = cols

    intr = response.source.pinhole.intrinsics
    intr.focal_length.x = fx
    intr.focal_length.y = fy
    intr.principal_point.x = cx
    intr.principal_point.y = cy

    snapshot = response.shot.transforms_snapshot
    snapshot.child_to_parent_edge_map["body"].SetInParent()  # tree root
    edge = snapshot.child_to_parent_edge_map[frame_name]
    edge.parent_frame_name = "body"
    edge.parent_tform_child.rotation.w = 1.0
    edge.parent_tform_child.position.x = translation[0]
    edge.parent_tform_child.position.y = translation[1]
    edge.parent_tform_child.position.z = translation[2]
    return response


class TestExtractRegistrationParams:
    def test_coincident_cameras_reproject_in_place(self):
        """Same pose and intrinsics for both cameras => registration is a no-op."""
        rgb = _make_response("rgb", rows=20, cols=30, fx=50.0, fy=50.0, cx=0.0, cy=0.0)
        depth = _make_response("depth", rows=20, cols=30, fx=50.0, fy=50.0, cx=0.0, cy=0.0)

        params = extract_registration_params(rgb, depth, dst_shape=(20, 30))

        raw = np.full((20, 30), 2.0, dtype=np.float32)
        out = np.empty((20, 30), dtype=np.float32)
        register_depth(raw, params, out)

        np.testing.assert_allclose(out, 2.0, rtol=1e-5)

    def test_translated_depth_camera_shifts_projection(self):
        """A depth camera offset by baseline b shifts pixels by fx * b / z."""
        fx = 100.0
        rgb = _make_response("rgb", rows=20, cols=40, fx=fx, fy=fx, cx=0.0, cy=0.0)
        depth = _make_response(
            "depth",
            rows=20,
            cols=30,
            fx=fx,
            fy=fx,
            cx=0.0,
            cy=0.0,
            translation=(0.1, 0.0, 0.0),
        )

        params = extract_registration_params(rgb, depth, dst_shape=(20, 40))

        raw = np.zeros((20, 30), dtype=np.float32)
        raw[7, 10] = 2.0
        out = np.empty((20, 40), dtype=np.float32)
        register_depth(raw, params, out)

        # Expected shift: fx * 0.1 / 2.0 = 5 pixels in u, none in v.
        expected = np.zeros((20, 40), dtype=np.float32)
        expected[7:9, 15:17] = 2.0  # sample at (u=15, v=7) plus its 2x2 splat
        np.testing.assert_allclose(out, expected, rtol=1e-5)
