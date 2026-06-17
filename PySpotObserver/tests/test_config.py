"""
Unit tests for configuration module.
"""

import os
import pytest
from pathlib import Path
import tempfile

from pyspotobserver.config import SpotConfig, CameraType


class TestCameraType:
    """Tests for CameraType enum."""

    def test_camera_mask_combinations(self):
        """Test bitwise operations on camera types."""
        mask = CameraType.FRONTLEFT | CameraType.FRONTRIGHT
        assert mask & CameraType.FRONTLEFT
        assert mask & CameraType.FRONTRIGHT
        assert not (mask & CameraType.BACK)

    def test_get_source_name_rgb(self):
        """Test RGB camera source name generation."""
        assert CameraType.get_source_name(CameraType.FRONTLEFT) == "frontleft_fisheye_image"
        assert CameraType.get_source_name(CameraType.BACK) == "back_fisheye_image"
        assert CameraType.get_source_name(CameraType.HAND) == "hand_color_image"

    def test_get_source_name_depth(self):
        """Test depth camera source name generation."""
        assert CameraType.get_source_name(CameraType.FRONTLEFT, depth=True) == \
               "frontleft_depth_in_visual_frame"
        assert CameraType.get_source_name(CameraType.RIGHT, depth=True) == \
               "right_depth_in_visual_frame"
        assert CameraType.get_source_name(CameraType.HAND, depth=True) == \
               "hand_depth_in_hand_color_frame"


class TestSpotConfig:
    """Tests for SpotConfig dataclass."""

    def test_default_values(self):
        """Test configuration with default values."""
        config = SpotConfig(robot_ip="192.168.80.3")
        assert config.robot_ip == "192.168.80.3"
        assert config.username == ""
        assert config.password == ""
        assert config.image_buffer_size == 5
        assert config.image_quality_percent == 100.0
        assert config.request_timeout_seconds == 10.0
        assert config.vision_model_path is None
        assert config.vision_providers is None

    def test_request_timeout_must_be_positive(self):
        """Test that non-positive request timeouts are rejected."""
        with pytest.raises(ValueError, match="request_timeout_seconds"):
            SpotConfig(robot_ip="192.168.80.3", request_timeout_seconds=0)

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = SpotConfig(
            robot_ip="10.0.0.1",
            username="admin",
            password="secret",
            image_buffer_size=10,
        )
        assert config.robot_ip == "10.0.0.1"
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.image_buffer_size == 10

    def test_repr_redacts_password(self):
        """Test that __repr__ redacts password."""
        config = SpotConfig(robot_ip="192.168.80.3", password="secret")
        repr_str = repr(config)
        assert "secret" not in repr_str
        assert "***" in repr_str

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        config = SpotConfig(
            robot_ip="192.168.80.3",
            username="testuser",
            password="testpass",
            image_buffer_size=7,
        )

        fd, raw_path = tempfile.mkstemp(suffix=".yaml")
        yaml_path = Path(raw_path)
        try:
            os.close(fd)
            yaml_path.unlink()

            # Save to YAML
            config.to_yaml(yaml_path)
            assert yaml_path.exists()

            # Load from YAML
            loaded_config = SpotConfig.from_yaml(yaml_path)
            assert loaded_config.robot_ip == config.robot_ip
            assert loaded_config.username == config.username
            assert loaded_config.password == config.password
            assert loaded_config.image_buffer_size == config.image_buffer_size
        finally:
            try:
                yaml_path.unlink()
            except FileNotFoundError:
                pass

    def test_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            SpotConfig.from_yaml("nonexistent.yaml")

    def test_extra_params(self):
        """Test extra_params field."""
        config = SpotConfig(
            robot_ip="192.168.80.3",
            extra_params={"location": "lab", "experiment": "test1"}
        )
        assert config.extra_params["location"] == "lab"
        assert config.extra_params["experiment"] == "test1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
