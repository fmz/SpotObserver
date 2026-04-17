"""
Shared command-line helpers for example scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from pyspotobserver import CameraType, SpotConfig


CAMERA_NAME_MAP: dict[str, CameraType] = {
    "back": CameraType.BACK,
    "frontleft": CameraType.FRONTLEFT,
    "frontright": CameraType.FRONTRIGHT,
    "left": CameraType.LEFT,
    "right": CameraType.RIGHT,
    "hand": CameraType.HAND,
}


def add_common_connection_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: str = "config_example.yaml",
    include_buffer_size: bool = True,
) -> None:
    default_config_path = Path(__file__).with_name(default_config)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help="YAML config file to load first.",
    )
    parser.add_argument(
        "--robot-ip",
        help="Robot IP address. Overrides the config file when provided.",
    )
    parser.add_argument(
        "--username",
        help="Robot username. Overrides the config file when provided.",
    )
    parser.add_argument(
        "--password",
        help="Robot password. Overrides the config file when provided.",
    )
    if include_buffer_size:
        parser.add_argument(
            "--image-buffer-size",
            type=int,
            help="Frame buffer size. Overrides the config file when provided.",
        )


def add_stream_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_cameras: str,
    default_duration: float = 30.0,
    include_no_display: bool = True,
) -> None:
    parser.add_argument(
        "--cameras",
        default=default_cameras,
        help=(
            "Comma-separated camera list. "
            f"Choices: {', '.join(sorted(CAMERA_NAME_MAP))}."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=default_duration,
        help="Maximum streaming duration in seconds.",
    )
    parser.add_argument(
        "--stream-id",
        default="example_stream",
        help="Stream identifier to create on the connection.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Per-frame retrieval timeout in seconds.",
    )
    if include_no_display:
        parser.add_argument(
            "--no-display",
            action="store_true",
            help="Disable OpenCV windows and only log frame metadata.",
        )


def build_config_from_args(args: argparse.Namespace) -> SpotConfig:
    config_path = getattr(args, "config", None)
    if config_path:
        config = SpotConfig.from_yaml(config_path)
    else:
        config = SpotConfig(robot_ip=args.robot_ip or "")

    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.username:
        config.username = args.username
    if args.password:
        config.password = args.password
    if hasattr(args, "image_buffer_size") and args.image_buffer_size is not None:
        config.image_buffer_size = args.image_buffer_size

    if not config.robot_ip:
        raise ValueError("Robot IP must be set in --config or with --robot-ip.")

    return config


def parse_camera_list(text: str) -> list[CameraType]:
    cameras: list[CameraType] = []
    seen: set[CameraType] = set()

    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        try:
            camera = CAMERA_NAME_MAP[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown camera '{name}'. Valid choices: {', '.join(sorted(CAMERA_NAME_MAP))}."
            ) from exc
        if camera not in seen:
            cameras.append(camera)
            seen.add(camera)

    if not cameras:
        raise ValueError("No cameras selected.")

    return cameras


def build_camera_mask(cameras: Sequence[CameraType]) -> CameraType:
    mask = CameraType(0)
    for camera in cameras:
        mask |= camera
    return mask


def camera_names(cameras: Iterable[CameraType]) -> str:
    return ", ".join(camera.name for camera in cameras)
