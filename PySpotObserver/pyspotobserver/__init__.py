"""
PySpotObserver - Python interface for Boston Dynamics Spot robot camera streaming.

This package provides a clean, Pythonic interface for connecting to Spot robots
and streaming camera data with support for both synchronous and asynchronous patterns.
"""

from .config import SpotConfig, CameraType
from .connection import SpotAuthenticationError, SpotConnection, SpotConnectionError
from .camera_stream import SpotCamStream, SpotCamStreamError

__version__ = "0.1.0"
__all__ = [
    "SpotConfig",
    "CameraType",
    "SpotConnection",
    "SpotConnectionError",
    "SpotAuthenticationError",
    "SpotCamStream",
    "SpotCamStreamError",
]
