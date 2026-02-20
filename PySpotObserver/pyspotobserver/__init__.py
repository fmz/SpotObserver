"""
PySpotObserver - Python interface for Boston Dynamics Spot robot camera streaming.

This package provides a clean, Pythonic interface for connecting to Spot robots
and streaming camera data with support for both synchronous and asynchronous patterns.
"""

from .config import SpotConfig, CameraType
from .connection import SpotConnection
from .camera_stream import SpotCamStream

__version__ = "0.1.0"
__all__ = [
    "SpotConfig",
    "CameraType",
    "SpotConnection",
    "SpotCamStream",
]
