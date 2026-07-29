"""
Robot-in-the-loop validation of client-side depth registration.

Registers the raw ``*_depth`` sources into the RGB frame with
``pyspotobserver.depth_registration`` and compares the result against the
robot's own ``*_depth_in_visual_frame`` output for the same (time-synced)
capture. Both images come from a single GetImage RPC, so they describe the
exact same sensor readings.

Skipped unless a robot is reachable. To run:

    SPOT_ROBOT_IP=192.168.80.3 SPOT_USERNAME=user SPOT_PASSWORD=pass \
        pytest tests/test_depth_registration_robot.py -v -s
"""

import os

import numpy as np
import pytest
from bosdyn.api import image_pb2  # type: ignore[import-untyped]
from bosdyn.client.auth import InvalidLoginError  # type: ignore[import-untyped]
from bosdyn.client.exceptions import (  # type: ignore[import-untyped]
    ResponseError,
    RpcError,
)
from bosdyn.client.image import (  # type: ignore[import-untyped]
    ImageClient,
    UnknownImageSourceError,
    build_image_request,
)
from pyspotobserver.depth_registration import extract_registration_params, register_depth

ROBOT_IP = os.environ.get("SPOT_ROBOT_IP")
USERNAME = os.environ.get("SPOT_USERNAME")
PASSWORD = os.environ.get("SPOT_PASSWORD")

_REQUIRED_ENV_VARS = ("SPOT_ROBOT_IP", "SPOT_USERNAME", "SPOT_PASSWORD")
_MISSING_ENV_VARS = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]

requires_robot = pytest.mark.skipif(
    bool(_MISSING_ENV_VARS),
    reason=(
        f"Robot validation skipped: missing environment variable(s) "
        f"{', '.join(_MISSING_ENV_VARS)}. Run with e.g.: "
        f"SPOT_ROBOT_IP=192.168.80.3 SPOT_USERNAME=user SPOT_PASSWORD=pass "
        f"pytest tests/test_depth_registration_robot.py -v -s"
    ),
)

# camera -> (rgb source, raw depth source, robot-registered reference source)
_SOURCES = {
    "back": ("back_fisheye_image", "back_depth", "back_depth_in_visual_frame"),
    "frontleft": ("frontleft_fisheye_image", "frontleft_depth", "frontleft_depth_in_visual_frame"),
    "frontright": (
        "frontright_fisheye_image",
        "frontright_depth",
        "frontright_depth_in_visual_frame",
    ),
    "left": ("left_fisheye_image", "left_depth", "left_depth_in_visual_frame"),
    "right": ("right_fisheye_image", "right_depth", "right_depth_in_visual_frame"),
    "hand": ("hand_color_image", "hand_depth", "hand_depth_in_hand_color_frame"),
}


@pytest.fixture(scope="module")
def image_client() -> ImageClient:
    import bosdyn.client  # type: ignore[import-untyped]

    sdk = bosdyn.client.create_standard_sdk("DepthRegistrationValidation")
    robot = sdk.create_robot(ROBOT_IP)
    try:
        robot.authenticate(USERNAME, PASSWORD)
    except InvalidLoginError:
        pytest.fail(
            f"Authentication failed for user {USERNAME!r} on robot {ROBOT_IP} — "
            f"check SPOT_USERNAME and SPOT_PASSWORD"
        )
    except RpcError as exc:
        pytest.fail(
            f"Could not reach robot at {ROBOT_IP} ({exc}) — "
            f"check SPOT_ROBOT_IP and that the robot is powered on and on the network"
        )
    return robot.ensure_client(ImageClient.default_service_name)


def _decode_depth(response: image_pb2.ImageResponse) -> np.ndarray:
    img = response.shot.image
    src_name = response.source.name
    if img.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise AssertionError(f"{src_name}: unexpected depth pixel format: {img.pixel_format}")
    data = np.frombuffer(img.data, dtype=np.uint16)
    if data.size != img.rows * img.cols:
        raise AssertionError(
            f"{src_name}: depth payload has {data.size} pixels, "
            f"expected {img.rows}x{img.cols} = {img.rows * img.cols}"
        )
    depth_scale = response.source.depth_scale if response.source.depth_scale > 0 else 1.0
    return data.reshape(img.rows, img.cols).astype(np.float32) / depth_scale


@requires_robot
@pytest.mark.parametrize("camera", list(_SOURCES))
def test_registration_matches_robot_registered_depth(
    image_client: ImageClient, camera: str
) -> None:
    rgb_src, raw_src, ref_src = _SOURCES[camera]
    requests = [
        build_image_request(
            rgb_src,
            quality_percent=100.0,
            image_format=image_pb2.Image.FORMAT_JPEG,
            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
        ),
        build_image_request(
            raw_src,
            image_format=image_pb2.Image.FORMAT_RAW,
            pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
        ),
        build_image_request(
            ref_src,
            image_format=image_pb2.Image.FORMAT_RAW,
            pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
        ),
    ]

    try:
        responses = image_client.get_image(requests)
    except UnknownImageSourceError:
        # Expected for cameras the robot doesn't have (e.g. hand without a gripper).
        pytest.skip(
            f"{camera}: source(s) not available on this robot: {rgb_src}, {raw_src}, {ref_src}"
        )
    except ResponseError as exc:
        pytest.fail(f"{camera}: robot rejected the image request: {exc}")
    except RpcError as exc:
        pytest.fail(f"{camera}: lost connection to robot at {ROBOT_IP} during request: {exc}")

    assert len(responses) == 3
    rgb_response, raw_response, ref_response = responses

    raw = _decode_depth(raw_response)
    ref = _decode_depth(ref_response)

    ref_valid = ref > 0
    if ref_valid.sum() < 1000:
        pytest.skip(f"{camera}: reference depth too sparse to validate ({ref_valid.sum()} px)")

    params = extract_registration_params(rgb_response, raw_response, dst_shape=ref.shape)
    out = np.empty(ref.shape, dtype=np.float32)
    register_depth(raw, params, out)

    out_valid = out > 0
    both_valid = out_valid & ref_valid

    # Alignment/coverage: valid regions should mostly coincide. Normalizing by the
    # smaller region tolerates density differences between the two splats.
    overlap = both_valid.sum() / min(out_valid.sum(), ref_valid.sum())

    # Agreement where both have data: same sensor readings, so depths should
    # match closely except at occlusion boundaries.
    abs_diff = np.abs(out[both_valid] - ref[both_valid])
    median_abs = float(np.median(abs_diff))
    rel_close = abs_diff < np.maximum(0.1, 0.05 * ref[both_valid])
    frac_close = float(np.mean(rel_close))

    print(
        f"[{camera}] raw_valid={int((raw > 0).sum())} ref_valid={int(ref_valid.sum())} "
        f"out_valid={int(out_valid.sum())} overlap={overlap:.3f} "
        f"median_abs={median_abs * 100:.2f}cm frac_close={frac_close:.3f}"
    )

    assert overlap > 0.6, f"{camera}: valid-pixel overlap {overlap:.3f} suggests misalignment"
    assert median_abs < 0.05, f"{camera}: median depth error {median_abs:.3f}m too high"
    assert frac_close > 0.85, f"{camera}: only {frac_close:.1%} of pixels within tolerance"
