"""
Unified streaming example for Spot camera feeds.

This example combines the old basic, async, and multi-stream demos into one CLI.
Use `--async-mode` to switch to async retrieval. Add `--secondary-cameras` for a
second stream, and optionally `--secondary-robot-ip` to run that stream on a
second robot using the same credentials.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
import logging
import time
from typing import Sequence


import cv2
import numpy as np

from pyspotobserver import CameraType, SpotConfig, SpotConnection
from examples.common_cli import (
    add_common_connection_arguments,
    build_camera_mask,
    build_config_from_args,
    parse_camera_list,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    frames: int = 0
    fetch_seconds: float = 0.0
    display_seconds: float = 0.0
    loop_seconds: float = 0.0

    def add(self, *, fetch_seconds: float, display_seconds: float, loop_seconds: float) -> None:
        self.frames += 1
        self.fetch_seconds += fetch_seconds
        self.display_seconds += display_seconds
        self.loop_seconds += loop_seconds


@dataclass
class FetchResult:
    rgb_images: Sequence[np.ndarray]
    depth_images: Sequence[np.ndarray]
    fetch_seconds: float


@dataclass
class StreamSpec:
    label: str
    stream_label: str
    stream_id: str
    cameras: list[CameraType]
    robot_label: str

    @property
    def display_label(self) -> str:
        return f"{self.stream_label} [{self.robot_label}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_connection_arguments(parser)
    parser.add_argument(
        "--cameras",
        default="frontleft,frontright",
        help="Comma-separated cameras for the primary stream.",
    )
    parser.add_argument(
        "--secondary-cameras",
        help="Optional comma-separated cameras for a second stream configuration on each connected robot.",
    )
    parser.add_argument(
        "--secondary-robot-ip",
        help="Optional second robot IP. When provided, the configured stream set is started on this robot too.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum streaming duration in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Per-frame retrieval timeout in seconds.",
    )
    parser.add_argument(
        "--stream-id",
        default="primary_stream",
        help="Stream identifier for the primary stream.",
    )
    parser.add_argument(
        "--secondary-stream-id",
        default="secondary_stream",
        help="Stream identifier for the optional second stream.",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use async connection management and async frame retrieval.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows and only log frame metadata.",
    )
    parser.add_argument(
        "--print-timing",
        action="store_true",
        help="Print per-stream timing summary at the end of the run.",
    )
    parser.add_argument(
        "--vision-pipeline",
        action="store_true",
        help="Run the vision pipeline on the Spot outputs"
    )
    return parser.parse_args()


def build_stream_specs(args: argparse.Namespace) -> list[StreamSpec]:
    primary_cameras = parse_camera_list(args.cameras)
    stream_templates = [
        ("primary", args.stream_id, primary_cameras),
    ]
    if args.secondary_cameras:
        stream_templates.append(
            ("secondary", args.secondary_stream_id, parse_camera_list(args.secondary_cameras))
        )

    robot_labels = ["primary"]
    if args.secondary_robot_ip:
        robot_labels.append("secondary")

    specs = []
    for robot_label in robot_labels:
        for stream_label, stream_id, cameras in stream_templates:
            specs.append(
                StreamSpec(
                    label=f"{robot_label}:{stream_label}",
                    stream_label=stream_label,
                    stream_id=stream_id,
                    cameras=cameras.copy(),
                    robot_label=robot_label,
                )
            )
    return specs


def display_images(window_prefix: str, stream, rgb_images: Sequence[np.ndarray], depth_images: Sequence[np.ndarray]) -> bool:
    for i, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
        camera_name = stream.get_camera_order()[i].name
        rgb_display = (rgb * 255).astype(np.uint8)
        rgb_display = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
        
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized[0], cv2.COLORMAP_JET)
        cv2.imshow(f"{window_prefix} - {camera_name} - RGB", rgb_display)
        cv2.imshow(f"{window_prefix} - {camera_name} - Depth", depth_colored)

    return bool(cv2.waitKey(1) & 0xFF == ord("q"))


def print_timing_summary(
    timing_by_label: dict[str, TimingStats],
    specs: Sequence[StreamSpec],
    connections: dict[str, SpotConnection],
    elapsed_seconds: float,
) -> None:
    print("Timing summary:")
    print(f"Elapsed wall time: {elapsed_seconds:.3f} s")
    timing_by_robot: dict[str, TimingStats] = {}
    for spec in specs:
        stats = timing_by_label[spec.label]
        avg_fetch_ms = (stats.fetch_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
        avg_display_ms = (stats.display_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
        avg_loop_ms = (stats.loop_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
        fps = stats.frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
        print(
            f"{spec.display_label} ({connections[spec.robot_label].config.robot_ip}): "
            f"frames={stats.frames}, avg_fetch={avg_fetch_ms:.3f} ms, "
            f"avg_display={avg_display_ms:.3f} ms, avg_loop={avg_loop_ms:.3f} ms, fps={fps:.2f}"
        )
        robot_stats = timing_by_robot.setdefault(spec.robot_label, TimingStats())
        robot_stats.frames += stats.frames
        robot_stats.fetch_seconds += stats.fetch_seconds
        robot_stats.display_seconds += stats.display_seconds
        robot_stats.loop_seconds += stats.loop_seconds

    if len(timing_by_robot) > 1:
        print("Per-robot aggregate throughput:")
        for robot_label, stats in timing_by_robot.items():
            avg_fetch_ms = (stats.fetch_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
            avg_display_ms = (stats.display_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
            avg_loop_ms = (stats.loop_seconds / stats.frames) * 1000.0 if stats.frames else 0.0
            fps = stats.frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
            print(
                f"{robot_label} robot ({connections[robot_label].config.robot_ip}): "
                f"frames={stats.frames}, avg_fetch={avg_fetch_ms:.3f} ms, "
                f"avg_display={avg_display_ms:.3f} ms, avg_loop={avg_loop_ms:.3f} ms, fps={fps:.2f}"
            )


def build_connection_configs(args: argparse.Namespace) -> dict[str, SpotConfig]:
    primary_config = build_config_from_args(args)
    configs: dict[str, SpotConfig] = {"primary": primary_config}
    if args.secondary_robot_ip:
        secondary_config = type(primary_config)(**vars(primary_config))
        secondary_config.robot_ip = args.secondary_robot_ip
        configs["secondary"] = secondary_config
    return configs


def start_streams(connections: dict[str, SpotConnection], specs: list[StreamSpec]) -> dict[str, object]:
    streams = {}
    for spec in specs:
        conn = connections[spec.robot_label]
        stream = conn.create_cam_stream(stream_id=spec.stream_id)
        stream.start_streaming(build_camera_mask(spec.cameras))
        streams[spec.label] = stream
        logger.info(
            "%s stream on %s robot (%s) cameras: %s",
            spec.display_label,
            spec.robot_label,
            conn.config.robot_ip,
            stream.get_camera_order(),
        )
    return streams


def run_sync(args: argparse.Namespace, specs: list[StreamSpec]) -> int:
    connection_configs = build_connection_configs(args)
    timing_by_label = {spec.label: TimingStats() for spec in specs}
    overall_start = time.perf_counter()

    with ExitStack() as stack:
        connections = {
            label: stack.enter_context(SpotConnection(config))
            for label, config in connection_configs.items()
        }
        for label, conn in connections.items():
            logger.info("Connected to %s robot: %s", label, conn)

        streams = start_streams(connections, specs)

        try:
            start_time = time.perf_counter()
            should_quit = False
            while time.perf_counter() - start_time < args.duration and not should_quit:
                for spec in specs:
                    loop_start = time.perf_counter()
                    stream = streams[spec.label]

                    fetch_start = time.perf_counter()
                    rgb_images, depth_images = stream.get_current_images(timeout=args.timeout)
                    fetch_elapsed = time.perf_counter() - fetch_start

                    display_elapsed = 0.0
                    if not args.no_display:
                        display_start = time.perf_counter()
                        should_quit = display_images(spec.display_label, stream, rgb_images, depth_images)
                        display_elapsed = time.perf_counter() - display_start

                    timing_by_label[spec.label].add(
                        fetch_seconds=fetch_elapsed,
                        display_seconds=display_elapsed,
                        loop_seconds=fetch_elapsed + display_elapsed,
                    )

                    if should_quit:
                        logger.info("User requested quit")
                        break
        finally:
            finalize_sync(connections, streams, specs, timing_by_label, args.print_timing, overall_start)
    return 0


async def fetch_stream_async(stream, timeout: float, vision_pipeline: bool) -> FetchResult:
    fetch_start = time.perf_counter()
    rgb_images, depth_images = await stream.async_get_current_images(timeout=timeout, run_pipeline=vision_pipeline)
        
    return FetchResult(
        rgb_images=rgb_images,
        depth_images=depth_images,
        fetch_seconds=time.perf_counter() - fetch_start,
    )


async def run_async(args: argparse.Namespace, specs: list[StreamSpec]) -> int:
    connection_configs = build_connection_configs(args)
    timing_by_label = {spec.label: TimingStats() for spec in specs}
    overall_start = time.perf_counter()

    async with AsyncExitStack() as stack:
        connections = {}
        for label, config in connection_configs.items():
            connections[label] = await stack.enter_async_context(SpotConnection(config))
        for label, conn in connections.items():
            logger.info("Connected to %s robot: %s", label, conn)

        streams = start_streams(connections, specs)

        try:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < args.duration:
                results = await asyncio.gather(
                    *(fetch_stream_async(streams[spec.label], args.timeout, args.vision_pipeline) for spec in specs)
                )

                should_quit = False
                for spec, result in zip(specs, results):
                    display_elapsed = 0.0
                    if not args.no_display:
                        display_start = time.perf_counter()
                        should_quit = display_images(
                            spec.display_label,
                            streams[spec.label],
                            result.rgb_images,
                            result.depth_images,
                        ) or should_quit
                        display_elapsed = time.perf_counter() - display_start

                    timing_by_label[spec.label].add(
                        fetch_seconds=result.fetch_seconds,
                        display_seconds=display_elapsed,
                        loop_seconds=result.fetch_seconds + display_elapsed,
                    )

                if should_quit:
                    logger.info("User requested quit")
                    break

                await asyncio.sleep(0.01)
        finally:
            finalize_async(connections, streams, specs, timing_by_label, args.print_timing, overall_start)
    return 0


def finalize_sync(
    connections,
    streams,
    specs: Sequence[StreamSpec],
    timing_by_label: dict[str, TimingStats],
    print_timing: bool,
    overall_start: float,
) -> int:
    for label, stream in streams.items():
        stream.stop_streaming()
        logger.info(
            "%s stream stats: frames=%s, errors=%s",
            label,
            stream.frame_count,
            stream.error_count,
        )
    cv2.destroyAllWindows()
    for label, conn in connections.items():
        logger.info("%s robot active streams: %s", label, conn.list_streams())
    logger.info("Disconnected from robot(s)")
    if print_timing:
        print_timing_summary(timing_by_label, specs, connections, time.perf_counter() - overall_start)
    return 0


def finalize_async(
    connections,
    streams,
    specs: Sequence[StreamSpec],
    timing_by_label: dict[str, TimingStats],
    print_timing: bool,
    overall_start: float,
) -> int:
    for label, stream in streams.items():
        stream.stop_streaming()
        logger.info(
            "%s stream stats: frames=%s, errors=%s",
            label,
            stream.frame_count,
            stream.error_count,
        )
    cv2.destroyAllWindows()
    for label, conn in connections.items():
        logger.info("%s robot active streams: %s", label, conn.list_streams())
    logger.info("Disconnected from robot(s)")
    if print_timing:
        print_timing_summary(timing_by_label, specs, connections, time.perf_counter() - overall_start)
    return 0


def main() -> int:
    args = parse_args()
    specs = build_stream_specs(args)
    if args.async_mode:
        return asyncio.run(run_async(args, specs))
    return run_sync(args, specs)


if __name__ == "__main__":
    raise SystemExit(main())
