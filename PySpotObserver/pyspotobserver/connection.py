"""
SpotConnection - Manages connection to Boston Dynamics Spot robot.
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from types import TracebackType

import bosdyn.client
from bosdyn.client import Robot, create_standard_sdk
from bosdyn.client.image import ImageClient

from .config import SpotConfig
from .camera_stream import SpotCamStream

logger = logging.getLogger(__name__)


class SpotConnectionError(Exception):
    """Base exception for SpotConnection errors."""
    pass


class SpotAuthenticationError(SpotConnectionError):
    """Raised when authentication fails."""
    pass


class SpotConnection:
    """
    Manages connection lifecycle and authentication with a Spot robot.

    Supports both synchronous and asynchronous usage patterns, as well as
    context manager protocol for automatic cleanup.

    Example (synchronous):
        >>> config = SpotConfig(robot_ip="192.168.80.3")
        >>> connection = SpotConnection(config)
        >>> connection.connect()
        >>> stream = connection.create_cam_stream()
        >>> connection.disconnect()

    Example (context manager):
        >>> with SpotConnection(config) as conn:
        ...     stream = conn.create_cam_stream()

    Example (async):
        >>> async with SpotConnection(config) as conn:
        ...     await conn.async_connect()
        ...     stream = conn.create_cam_stream()
    """

    def __init__(self, config: SpotConfig):
        """
        Initialize connection with configuration.

        Args:
            config: SpotConfig instance with connection parameters

        Note:
            Does not automatically connect. Call connect() or use as context manager.
        """
        self.config = config
        self._sdk: Optional[bosdyn.client.Sdk] = None
        self._robot: Optional[Robot] = None
        self._image_client: Optional[ImageClient] = None
        self._connected: bool = False
        self._cam_streams: Dict[str, SpotCamStream] = {}

        # Metrics and diagnostics
        self._last_connect_time_s: Optional[float] = None
        self._last_connect_error: Optional[Exception] = None
        self._last_connect_retries: int = 0

        logger.info(f"SpotConnection initialized for robot at {config.robot_ip}")

    @property
    def connected(self) -> bool:
        """Returns True if currently connected and authenticated."""
        return self._connected

    @property
    def robot(self) -> Robot:
        """
        Access to underlying Robot instance.

        Returns:
            Robot instance

        Raises:
            SpotConnectionError: If not connected
        """
        if not self._connected or self._robot is None:
            raise SpotConnectionError("Not connected to robot. Call connect() first.")
        return self._robot

    @property
    def image_client(self) -> ImageClient:
        """
        Access to ImageClient for direct API calls.

        Returns:
            ImageClient instance

        Raises:
            SpotConnectionError: If not connected
        """
        if not self._connected or self._image_client is None:
            raise SpotConnectionError("Not connected to robot. Call connect() first.")
        return self._image_client

    def connect(self) -> None:
        """
        Connect and authenticate with the robot.

        Raises:
            SpotConnectionError: If connection fails
            SpotAuthenticationError: If authentication fails
        """
        if self._connected:
            if self._robot is None or self._image_client is None:
                logger.warning(
                    "Connection marked as connected but internal state is missing. "
                    "Reinitializing connection."
                )
                self._cleanup()
            else:
                logger.warning("Already connected to robot")
                return

        try:
            # Create SDK instance
            self._sdk = create_standard_sdk(self.config.sdk_name)
            logger.debug(f"Created SDK instance: {self.config.sdk_name}")

            # Create robot instance with retry logic
            last_error = None
            self._last_connect_retries = 0
            for attempt in range(self.config.connection_retry_attempts):
                try:
                    self._robot = self._sdk.create_robot(self.config.robot_ip)
                    logger.debug(f"Created robot instance for {self.config.robot_ip}")
                    break
                except Exception as e:
                    last_error = e
                    self._last_connect_retries = attempt + 1
                    if attempt < self.config.connection_retry_attempts - 1:
                        delay_s = self.config.connection_retry_delay_ms / 1000.0
                        logger.warning(
                            f"Connection attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay_s}s..."
                        )
                        time.sleep(delay_s)
                    else:
                        raise SpotConnectionError(
                            f"Failed to create robot instance after "
                            f"{self.config.connection_retry_attempts} attempts: {e}"
                        ) from last_error

            # Authenticate
            try:
                self._robot.authenticate(self.config.username, self.config.password)
                logger.info(f"Authenticated as {self.config.username}")
            except Exception as e:
                raise SpotAuthenticationError(
                    f"Authentication failed for user '{self.config.username}': {e}"
                ) from e

            # Get image client
            try:
                self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
                logger.debug("Obtained ImageClient service")
            except Exception as e:
                raise SpotConnectionError(f"Failed to get ImageClient: {e}") from e

            self._connected = True
            self._last_connect_time_s = time.monotonic()
            self._last_connect_error = None
            logger.info(f"Successfully connected to Spot robot at {self.config.robot_ip}")

        except (SpotConnectionError, SpotAuthenticationError) as e:
            # Clean up on failure
            self._last_connect_error = e
            self._cleanup()
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            self._last_connect_error = e
            self._cleanup()
            raise SpotConnectionError(f"Unexpected error during connection: {e}") from e

    async def async_connect(self) -> None:
        """
        Async version of connect().

        Runs connection in executor to avoid blocking event loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.connect)

    def disconnect(self) -> None:
        """
        Disconnect from robot and clean up resources.

        Safe to call multiple times.
        """
        if not self._connected:
            logger.debug("Already disconnected")
            return

        logger.info("Disconnecting from robot...")

        # Stop all camera streams first
        for stream_id, stream in list(self._cam_streams.items()):
            try:
                logger.debug(f"Stopping stream: {stream_id}")
                stream.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")

        self._cam_streams.clear()

        # Clean up SDK resources
        self._cleanup()

        logger.info("Disconnected from robot")

    async def async_disconnect(self) -> None:
        """
        Async version of disconnect().

        Runs disconnection in executor to avoid blocking event loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.disconnect)

    def _cleanup(self) -> None:
        """Internal cleanup of SDK resources."""
        self._connected = False
        self._image_client = None
        self._robot = None
        self._sdk = None

    def create_cam_stream(
        self,
        stream_id: Optional[str] = None,
        auto_start_mask: Optional[int] = None,
    ) -> SpotCamStream:
        """
        Create a new camera stream.

        Args:
            stream_id: Optional unique identifier for this stream.
                      If None, a default ID will be generated.
            auto_start_mask: If provided, immediately start streaming with this mask.

        Returns:
            SpotCamStream instance

        Raises:
            SpotConnectionError: If not connected
            ValueError: If stream_id already exists
        """
        if not self._connected:
            raise SpotConnectionError("Must be connected before creating streams")

        if stream_id is None:
            stream_id = f"stream_{len(self._cam_streams)}"

        if stream_id in self._cam_streams:
            raise ValueError(f"Stream with id '{stream_id}' already exists")

        try:
            stream = SpotCamStream(
                image_client=self._image_client,
                config=self.config,
                stream_id=stream_id,
            )
        except Exception:
            logger.error(f"Failed to create camera stream: {stream_id}", exc_info=True)
            raise

        self._cam_streams[stream_id] = stream
        logger.info(f"Created camera stream: {stream_id}")

        if auto_start_mask is not None:
            logger.info(
                f"Auto-starting stream '{stream_id}' with mask: {auto_start_mask}"
            )
            try:
                stream.start_streaming(auto_start_mask)
            except Exception:
                logger.error(
                    f"Failed to auto-start stream '{stream_id}'",
                    exc_info=True,
                )
                del self._cam_streams[stream_id]
                raise

        return stream

    def remove_cam_stream(self, stream_id: str) -> None:
        """
        Remove and stop a camera stream.

        Args:
            stream_id: ID of stream to remove

        Raises:
            ValueError: If stream_id doesn't exist
        """
        if stream_id not in self._cam_streams:
            raise ValueError(f"Stream with id '{stream_id}' does not exist")

        stream = self._cam_streams[stream_id]
        try:
            stream.stop_streaming()
        except Exception as e:
            logger.error(f"Error stopping stream {stream_id}: {e}")

        del self._cam_streams[stream_id]
        logger.info(f"Removed camera stream: {stream_id}")

    def get_stream(self, stream_id: str) -> SpotCamStream:
        """
        Get an existing camera stream by ID.

        Args:
            stream_id: ID of stream to retrieve

        Returns:
            SpotCamStream instance

        Raises:
            ValueError: If stream_id doesn't exist
        """
        if stream_id not in self._cam_streams:
            raise ValueError(f"Stream with id '{stream_id}' does not exist")
        return self._cam_streams[stream_id]

    def list_streams(self) -> list[str]:
        """
        Get list of all active stream IDs.

        Returns:
            List of stream IDs
        """
        return list(self._cam_streams.keys())

    def stop_all_streams(self) -> None:
        """
        Stop all camera streams without disconnecting.
        """
        for stream_id, stream in list(self._cam_streams.items()):
            try:
                logger.debug(f"Stopping stream: {stream_id}")
                stream.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")

    def get_or_create_stream(
        self,
        stream_id: str,
        auto_start_mask: Optional[int] = None,
    ) -> SpotCamStream:
        """
        Get an existing stream or create it if missing.
        """
        if stream_id in self._cam_streams:
            return self._cam_streams[stream_id]
        return self.create_cam_stream(
            stream_id=stream_id,
            auto_start_mask=auto_start_mask,
        )

    def close(self) -> None:
        """Alias for disconnect()."""
        self.disconnect()

    @property
    def last_connect_time_s(self) -> Optional[float]:
        """Monotonic time of last successful connect."""
        return self._last_connect_time_s

    @property
    def last_connect_error(self) -> Optional[Exception]:
        """Last error encountered during connection, if any."""
        return self._last_connect_error

    @property
    def last_connect_retries(self) -> int:
        """Number of retries used during last connect attempt."""
        return self._last_connect_retries

    # Context manager support (synchronous)
    def __enter__(self) -> "SpotConnection":
        """Enter context manager - connects if not already connected."""
        if not self._connected:
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager - always disconnects."""
        self.disconnect()

    # Async context manager support
    async def __aenter__(self) -> "SpotConnection":
        """Enter async context manager - connects if not already connected."""
        if not self._connected:
            await self.async_connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context manager - always disconnects."""
        await self.async_disconnect()

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        stream_count = len(self._cam_streams)
        return (
            f"SpotConnection(robot_ip='{self.config.robot_ip}', "
            f"status='{status}', streams={stream_count})"
        )
