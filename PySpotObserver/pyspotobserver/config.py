"""
Configuration dataclasses and enums for PySpotObserver.
"""

from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path
from typing import Dict, Any
import yaml


class CameraType(IntFlag):
    """
    Camera types available on Spot robot.
    Uses IntFlag to support bitwise operations for camera masks.
    """
    BACK = 0x1
    FRONTLEFT = 0x2
    FRONTRIGHT = 0x4
    LEFT = 0x8
    RIGHT = 0x10
    HAND = 0x20

    @classmethod
    def get_source_name(cls, camera: "CameraType", depth: bool = False) -> str:
        """Get the Boston Dynamics API source name for a camera."""
        rgb_names = {
            cls.BACK: "back_fisheye_image",
            cls.FRONTLEFT: "frontleft_fisheye_image",
            cls.FRONTRIGHT: "frontright_fisheye_image",
            cls.LEFT: "left_fisheye_image",
            cls.RIGHT: "right_fisheye_image",
            # Hand camera uses color sensor naming, not fisheye naming.
            cls.HAND: "hand_color_image",
        }
        depth_names = {
            cls.BACK: "back_depth_in_visual_frame",
            cls.FRONTLEFT: "frontleft_depth_in_visual_frame",
            cls.FRONTRIGHT: "frontright_depth_in_visual_frame",
            cls.LEFT: "left_depth_in_visual_frame",
            cls.RIGHT: "right_depth_in_visual_frame",
            # Hand depth is aligned to hand color frame.
            cls.HAND: "hand_depth_in_hand_color_frame",
        }

        if camera not in rgb_names:
            raise ValueError(f"Invalid camera type: {camera}")
        return depth_names[camera] if depth else rgb_names[camera]


@dataclass
class SpotConfig:
    """
    Configuration for Spot robot connection and camera streaming.

    Can be loaded from YAML file or instantiated directly.
    """
    # Connection settings
    robot_ip: str
    username: str = ""
    password: str = ""

    # Streaming settings
    image_buffer_size: int = 5
    """Maximum number of image frames to buffer (FIFO queue)"""

    image_quality_percent: float = 100.0
    """JPEG quality for RGB images (0-100)"""

    request_timeout_seconds: float = 5.0
    """Timeout for image requests"""

    # Advanced settings
    sdk_name: str = "PySpotObserver"
    """Name to identify this SDK client"""

    connection_retry_attempts: int = 3
    """Number of connection retry attempts"""

    connection_retry_delay_ms: int = 100
    """Delay between connection retry attempts (milliseconds)"""

    extra_params: Dict[str, Any] = field(default_factory=dict)
    """Additional user-defined parameters"""

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "SpotConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            SpotConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(**data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            yaml_path: Path where YAML file should be saved
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, excluding extra_params if empty
        data = {
            'robot_ip': self.robot_ip,
            'username': self.username,
            'password': self.password,
            'image_buffer_size': self.image_buffer_size,
            'image_quality_percent': self.image_quality_percent,
            'request_timeout_seconds': self.request_timeout_seconds,
            'sdk_name': self.sdk_name,
            'connection_retry_attempts': self.connection_retry_attempts,
            'connection_retry_delay_ms': self.connection_retry_delay_ms,
        }

        if self.extra_params:
            data['extra_params'] = self.extra_params

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation with password redacted."""
        return (
            f"SpotConfig(robot_ip='{self.robot_ip}', username='{self.username}', "
            f"password='***', buffer_size={self.image_buffer_size}, "
            f"quality={self.image_quality_percent}%)"
        )
