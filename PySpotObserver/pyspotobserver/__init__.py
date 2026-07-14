"""
PySpotObserver - Python interface for Boston Dynamics Spot robot camera streaming.

This package provides a clean, Pythonic interface for connecting to Spot robots
and streaming camera data with support for both synchronous and asynchronous patterns.
"""

from .camera_stream import SpotCamStream, SpotCamStreamError
from .config import CameraType, SpotConfig
from .connection import SpotAuthenticationError, SpotConnection, SpotConnectionError

__version__ = "0.1.0"
__all__ = [
    "CameraType",
    "SpotAuthenticationError",
    "SpotCamStream",
    "SpotCamStreamError",
    "SpotConfig",
    "SpotConnection",
    "SpotConnectionError",
]
