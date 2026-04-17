"""
Benchmark allocation vs. in-place conversion using one captured response set.

This measures CPU decode + conversion + allocation overhead.
It does not include network latency or stream-loop scheduling.
"""

import argparse
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import UnknownImageSourceError, build_image_request

from pyspotobserver import CameraType, SpotConnection
from common_cli import add_common_connection_arguments, build_config_from_args, parse_camera_list


def _convert_alloc(response: image_pb2.ImageResponse, is_depth: bool) -> np.ndarray:
    image_proto = response.shot.image

    if image_proto.format == image_pb2.Image.FORMAT_JPEG:
        img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode JPEG image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    if image_proto.format == image_pb2.Image.FORMAT_RAW:
        rows = image_proto.rows
        cols = image_proto.cols
        pixel_format = image_proto.pixel_format

        if is_depth:
            if pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                raise RuntimeError(f"Unexpected depth pixel format: {pixel_format}")
            img_data = np.frombuffer(image_proto.data, dtype=np.uint16)
            img = img_data.reshape((rows, cols))
            depth_scale = response.source.depth_scale if response.source.depth_scale > 0 else 1.0
            return img.astype(np.float32) / depth_scale

        if pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
            img = img_data.reshape((rows, cols, 3))
            return img.astype(np.float32) / 255.0

        if pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
            img = img_data.reshape((rows, cols, 4))
            return img[:, :, :3].astype(np.float32) / 255.0

        if pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
            img = img_data.reshape((rows, cols))
            img = np.stack([img] * 3, axis=-1)
            return img.astype(np.float32) / 255.0

        raise RuntimeError(f"Unsupported pixel format: {pixel_format}")

    raise RuntimeError(f"Unsupported image format: {image_proto.format}")


def _resolve_camera_pairs(
    cameras: Sequence[CameraType],
    available_sources: set[str] | None,
) -> List[Tuple[CameraType, str, str]]:
    pairs: List[Tuple[CameraType, str, str]] = []
    skipped: List[Tuple[CameraType, str, str]] = []

    for cam in cameras:
        rgb = CameraType.get_source_name(cam, depth=False)
        depth = CameraType.get_source_name(cam, depth=True)
        if available_sources is not None and (rgb not in available_sources or depth not in available_sources):
            skipped.append((cam, rgb, depth))
            continue
        pairs.append((cam, rgb, depth))

    if skipped:
        print("Skipping unavailable camera source pairs:")
        for cam, rgb, depth in skipped:
            print(f"  {cam.name}: rgb='{rgb}', depth='{depth}'")

    if not pairs:
        raise RuntimeError(
            "No requested camera had both RGB+depth sources available on this robot."
        )
    return pairs


def _build_requests(pairs: Sequence[Tuple[CameraType, str, str]]) -> List[image_pb2.ImageRequest]:
    requests: List[image_pb2.ImageRequest] = []
    for _, rgb_source, depth_source in pairs:
        requests.append(
            build_image_request(
                rgb_source,
                quality_percent=100.0,
                image_format=image_pb2.Image.FORMAT_JPEG,
            )
        )
        requests.append(
            build_image_request(
                depth_source,
                image_format=image_pb2.Image.FORMAT_RAW,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
            )
        )
    return requests


def _time_it(fn, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_connection_arguments(parser, include_buffer_size=False)
    parser.add_argument("--cameras", default="frontleft,frontright")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    config = build_config_from_args(args)
    requested_cameras = parse_camera_list(args.cameras)

    with SpotConnection(config) as conn:
        stream = conn.create_cam_stream(stream_id="benchmark_stream")

        available_sources: set[str] | None = None
        try:
            source_protos = conn.image_client.list_image_sources()
            available_sources = {src.name for src in source_protos}
        except Exception as exc:
            print(f"Warning: failed to list image sources ({exc}). Proceeding without pre-validation.")

        pairs = _resolve_camera_pairs(requested_cameras, available_sources)
        requests = _build_requests(pairs)

        try:
            responses = conn.image_client.get_image(
                requests,
                timeout=config.request_timeout_seconds,
            )
        except UnknownImageSourceError as exc:
            raise RuntimeError(
                "Robot rejected one or more image source names. "
                "Try running with different cameras or inspect list_image_sources()."
            ) from exc

        expected = len(pairs) * 2
        if len(responses) != expected:
            raise RuntimeError(f"Expected {expected} responses, got {len(responses)}")

        # First inference pass is used to determine shapes for preallocation.
        decoded_first_pass: List[np.ndarray] = []
        for i in range(len(pairs)):
            decoded_first_pass.append(_convert_alloc(responses[i * 2], is_depth=False))
            decoded_first_pass.append(_convert_alloc(responses[i * 2 + 1], is_depth=True))

        rgb_buffers = [np.zeros(decoded_first_pass[i * 2].shape, dtype=np.float32) for i in range(len(pairs))]
        depth_buffers = [np.zeros(decoded_first_pass[i * 2 + 1].shape, dtype=np.float32) for i in range(len(pairs))]

        def inplace_once() -> None:
            for i in range(len(pairs)):
                stream._convert_image_response_inplace(
                    responses[i * 2],
                    is_depth=False,
                    out_array=rgb_buffers[i],
                )
                stream._convert_image_response_inplace(
                    responses[i * 2 + 1],
                    is_depth=True,
                    out_array=depth_buffers[i],
                )

        def alloc_once() -> None:
            for i in range(len(pairs)):
                _ = _convert_alloc(responses[i * 2], is_depth=False)
                _ = _convert_alloc(responses[i * 2 + 1], is_depth=True)

        inplace_once()
        alloc_once()

        t_inplace = _time_it(inplace_once, args.iters)
        t_alloc = _time_it(alloc_once, args.iters)

        per_iter_inplace_ms = (t_inplace / args.iters) * 1000.0
        per_iter_alloc_ms = (t_alloc / args.iters) * 1000.0
        speedup = (per_iter_alloc_ms / per_iter_inplace_ms) if per_iter_inplace_ms > 0 else float("inf")

        print("Benchmark results (single captured response set):")
        print("Cameras used:", ", ".join(cam.name for cam, _, _ in pairs))
        print(f"In-place: {per_iter_inplace_ms:.3f} ms/iter")
        print(f"Alloc   : {per_iter_alloc_ms:.3f} ms/iter")
        print(f"Speedup : {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
