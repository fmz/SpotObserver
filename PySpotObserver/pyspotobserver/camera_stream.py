"""
SpotCamStream - Manages camera streaming from Spot robot.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, List, Tuple

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

from .config import SpotConfig, CameraType
from .color_correction import _ROBOT_CCMS

logger = logging.getLogger(__name__)


@dataclass
class ImageFrame:
    """
    Container for a single frame with RGB and depth images.

    Attributes:
        rgb_images: List of RGB images as numpy arrays (H, W, 3) in float32 [0, 1]
        depth_images: List of depth images as numpy arrays (H, W) in float32 (meters)
        camera_order: List of CameraType enums indicating order of images
        timestamp: Time when frame was captured (monotonic clock)
        acquisition_time: Robot's acquisition time from image response
    """
    rgb_images: List[np.ndarray]
    depth_images: List[np.ndarray]
    camera_order: List[CameraType]
    timestamp: float
    acquisition_time: Optional[float] = None


class SpotCamStreamError(Exception):
    """Base exception for SpotCamStream errors."""
    pass


class SpotCamStream:
    """
    Manages camera streaming from Spot robot in a background thread.

    Streams images from specified cameras and maintains a FIFO queue of frames.
    Supports both synchronous and asynchronous image retrieval.

    Example:
        >>> stream = connection.create_cam_stream()
        >>> stream.start_streaming(CameraType.FRONTLEFT | CameraType.FRONTRIGHT)
        >>> rgb_images, depth_images = stream.get_current_images()
        >>> stream.stop_streaming()
    """

    def __init__(
        self,
        image_client: ImageClient,
        config: SpotConfig,
        stream_id: str,
    ):
        """
        Initialize camera stream.

        Args:
            image_client: ImageClient from connected robot
            config: SpotConfig with streaming parameters
            stream_id: Unique identifier for this stream
        """
        self._image_client = image_client
        self._config = config
        self._stream_id = stream_id

        # Threading control
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Image buffer (FIFO queue)
        self._image_queue: Queue[ImageFrame] = Queue(maxsize=config.image_buffer_size)

        # Current camera configuration
        self._camera_mask: int = 0
        self._camera_order: List[CameraType] = []

        # Preallocated frame pool (initialized once we know image shapes)
        self._frame_pool: List[ImageFrame] = []
        self._frame_pool_index: int = 0

        # Color correction matrices (None if robot IP is not recognized)
        self._ccms: Optional[dict] = _ROBOT_CCMS.get(config.robot_ip)
        if self._ccms is not None:
            logger.info(f"SpotCamStream '{stream_id}': Color correction enabled for {config.robot_ip}")
        else:
            logger.info(f"SpotCamStream '{stream_id}': No color correction (unrecognized IP {config.robot_ip})")

        # Statistics
        self._frame_count = 0
        self._error_count = 0

        logger.info(f"SpotCamStream '{stream_id}' initialized")

    @property
    def streaming(self) -> bool:
        """Returns True if currently streaming."""
        return self._streaming

    @property
    def stream_id(self) -> str:
        """Returns stream ID."""
        return self._stream_id

    @property
    def camera_mask(self) -> int:
        """Returns current camera mask."""
        return self._camera_mask

    @property
    def frame_count(self) -> int:
        """Returns total frames captured."""
        return self._frame_count

    @property
    def error_count(self) -> int:
        """Returns total errors encountered."""
        return self._error_count

    def start_streaming(self, camera_mask: int) -> None:
        """
        Start streaming from specified cameras.

        Args:
            camera_mask: Bitwise OR of CameraType flags

        Raises:
            SpotCamStreamError: If already streaming or invalid camera mask
        """
        if self._streaming:
            raise SpotCamStreamError("Already streaming. Stop before restarting.")

        if camera_mask == 0:
            raise SpotCamStreamError("Camera mask cannot be zero")
        if camera_mask & ~self._valid_camera_mask():
            raise SpotCamStreamError(
                f"Camera mask contains unknown bits: {camera_mask:#x}"
            )

        # Parse camera mask to get ordered list of cameras
        self._camera_mask = camera_mask
        self._camera_order = self._parse_camera_mask(camera_mask)

        logger.info(
            f"Stream '{self._stream_id}': Starting with cameras: "
            f"{[cam.name for cam in self._camera_order]}"
        )

        # Clear queue and reset statistics
        self._clear_queue()
        self._frame_pool = []
        self._frame_pool_index = 0
        self._frame_count = 0
        self._error_count = 0

        # Start producer thread
        self._stop_event.clear()
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name=f"SpotCamStream-{self._stream_id}",
            daemon=True,
        )
        self._stream_thread.start()

        logger.info(f"Stream '{self._stream_id}': Started")

    def stop_streaming(self) -> None:
        """
        Stop streaming and clean up thread.

        Safe to call multiple times.
        """
        if not self._streaming:
            logger.debug(f"Stream '{self._stream_id}': Already stopped")
            return

        logger.info(f"Stream '{self._stream_id}': Stopping...")

        # Signal thread to stop
        self._streaming = False
        self._stop_event.set()

        # Wait for thread to finish. The SDK request timeout bounds how long the
        # producer can stay blocked in get_image().
        join_timeout = max(1.0, self._config.request_timeout_seconds + 1.0)
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=join_timeout)
            if self._stream_thread.is_alive():
                raise SpotCamStreamError(
                    f"Stream '{self._stream_id}': Thread did not stop within {join_timeout:.1f}s"
                )

        self._stream_thread = None
        self._clear_queue()

        logger.info(
            f"Stream '{self._stream_id}': Stopped "
            f"(frames={self._frame_count}, errors={self._error_count})"
        )

    def get_current_images(
        self,
        timeout: Optional[float] = None,
        copy: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the most recent frame of images.

        Blocks until images are available or timeout occurs.

        Note:
            Returned arrays are backed by a preallocated frame pool and may be
            overwritten by subsequent frames. Copy if you need to retain them.

        Args:
            timeout: Maximum seconds to wait for images. None = wait forever.
            copy: If True, returns deep copies of image arrays.

        Returns:
            Tuple of (rgb_images, depth_images) lists in camera_order

        Raises:
            SpotCamStreamError: If not streaming or timeout occurs
        """
        if not self._streaming and self._image_queue.empty():
            raise SpotCamStreamError("Not currently streaming")

        deadline = None if timeout is None else time.monotonic() + timeout
        poll_interval = 0.1

        while True:
            if not self._streaming and self._image_queue.empty():
                raise SpotCamStreamError("Stream stopped before images were available")

            frame = self._peek_latest_frame()
            if frame is not None:
                if not copy:
                    return frame.rgb_images, frame.depth_images
                return (
                    [img.copy() for img in frame.rgb_images],
                    [img.copy() for img in frame.depth_images],
                )

            wait_timeout = poll_interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SpotCamStreamError(
                        f"Timeout waiting for images (timeout={timeout}s)"
                    )
                wait_timeout = min(wait_timeout, remaining)

            try:
                frame = self._image_queue.get(timeout=wait_timeout)
            except Empty:
                continue

            if not copy:
                return frame.rgb_images, frame.depth_images
            return (
                [img.copy() for img in frame.rgb_images],
                [img.copy() for img in frame.depth_images],
            )

    async def async_get_current_images(
        self,
        timeout: Optional[float] = None,
        copy: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Async version of get_current_images().

        Note:
            Returned arrays are backed by a preallocated frame pool and may be
            overwritten by subsequent frames. Copy if you need to retain them.

        Args:
            timeout: Maximum seconds to wait for images. None = wait forever.
            copy: If True, returns deep copies of image arrays.

        Returns:
            Tuple of (rgb_images, depth_images) lists in camera_order

        Raises:
            SpotCamStreamError: If not streaming or timeout occurs
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_current_images, timeout, copy
        )

    def get_camera_order(self) -> List[CameraType]:
        """
        Get the ordered list of cameras being streamed.

        Returns:
            List of CameraType enums in the order images are returned
        """
        return self._camera_order.copy()

    def _parse_camera_mask(self, mask: int) -> List[CameraType]:
        """
        Parse camera mask into ordered list of CameraType enums.

        Args:
            mask: Bitwise OR of CameraType flags

        Returns:
            List of CameraType enums in consistent order
        """
        cameras = []
        for camera_type in CameraType:
            if camera_type == 0:  # Skip the 0 value if present
                continue
            if mask & camera_type:
                cameras.append(camera_type)

        return cameras

    def _stream_loop(self) -> None:
        """
        Producer thread main loop.

        Continuously requests images and pushes to queue.
        """
        logger.debug(f"Stream '{self._stream_id}': Thread started")

        while not self._stop_event.is_set() and self._streaming:
            try:
                # Build image requests for all cameras (RGB + depth)
                image_requests = self._build_image_requests()

                # Request images from robot
                start_time = time.monotonic()
                image_responses = self._image_client.get_image(
                    image_requests,
                    timeout=self._config.request_timeout_seconds,
                )
                request_time = time.monotonic() - start_time

                # Initialize the frame pool from the first successful response,
                # then reuse it for all subsequent frames.
                if not self._frame_pool:
                    initial_decoded = self._decode_initial_responses(image_responses)
                    self._initialize_frame_pool(initial_decoded)
                    frame = self._next_frame_from_pool()
                    self._fill_frame_from_decoded(initial_decoded, frame)
                else:
                    frame = self._next_frame_from_pool()
                    self._fill_frame_from_responses(image_responses, frame)
                self._enqueue_frame(frame)
                self._frame_count += 1

                if self._frame_count % 100 == 0:
                    logger.debug(
                        f"Stream '{self._stream_id}': "
                        f"Frame {self._frame_count}, "
                        f"request_time={request_time:.3f}s"
                    )

            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Stream '{self._stream_id}': Error in stream loop: {e}",
                    exc_info=True,
                )
                # Back off on error
                time.sleep(0.01)

        logger.debug(f"Stream '{self._stream_id}': Thread exiting")

    def _build_image_requests(self) -> List[image_pb2.ImageRequest]:
        """
        Build list of ImageRequest protos for all cameras (RGB + depth).

        Returns:
            List of ImageRequest protobuf objects
        """
        requests = []

        for camera in self._camera_order:
            # RGB request
            rgb_source = CameraType.get_source_name(camera, depth=False)
            requests.append(
                build_image_request(
                    rgb_source,
                    quality_percent=self._config.image_quality_percent,
                    image_format=image_pb2.Image.FORMAT_JPEG,
                )
            )

            # Depth request
            depth_source = CameraType.get_source_name(camera, depth=True)
            requests.append(
                build_image_request(
                    depth_source,
                    image_format=image_pb2.Image.FORMAT_RAW,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                )
            )

        return requests

    def _initialize_frame_pool(
        self,
        decoded: List[np.ndarray],
    ) -> None:
        """
        Initialize a preallocated pool of frames based on initial responses.

        Args:
            decoded: List of decoded arrays ordered as RGB, Depth, RGB, Depth...

        Raises:
            SpotCamStreamError: If responses are malformed
        """
        n_cameras = len(self._camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(decoded) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} decoded arrays, got {len(decoded)}"
            )

        # Infer shapes from decoded arrays
        rgb_shapes: List[Tuple[int, ...]] = []
        depth_shapes: List[Tuple[int, ...]] = []

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1

            rgb_shape = decoded[rgb_idx].shape
            depth_shape = decoded[depth_idx].shape
            rgb_shapes.append(rgb_shape)
            depth_shapes.append(depth_shape)

        # Preallocate a static pool of frames (one per queue slot)
        pool_size = max(1, self._config.image_buffer_size)
        self._frame_pool = []
        self._frame_pool_index = 0

        for _ in range(pool_size):
            rgb_images = [
                np.zeros(shape, dtype=np.float32) for shape in rgb_shapes
            ]
            depth_images = [
                np.zeros(shape, dtype=np.float32) for shape in depth_shapes
            ]
            self._frame_pool.append(
                ImageFrame(
                    rgb_images=rgb_images,
                    depth_images=depth_images,
                    camera_order=self._camera_order.copy(),
                    timestamp=0.0,
                    acquisition_time=None,
                )
            )

    def _next_frame_from_pool(self) -> ImageFrame:
        """
        Return the next preallocated frame from the pool in round-robin order.
        """
        if not self._frame_pool:
            raise SpotCamStreamError("Frame pool not initialized")
        frame = self._frame_pool[self._frame_pool_index]
        self._frame_pool_index = (self._frame_pool_index + 1) % len(self._frame_pool)
        return frame

    def _enqueue_frame(self, frame: ImageFrame) -> None:
        """
        Enqueue a frame, dropping the oldest if the queue is full.
        """
        try:
            self._image_queue.put(frame, timeout=0.001)
        except Exception:
            try:
                _ = self._image_queue.get_nowait()
            except Empty:
                pass
            try:
                self._image_queue.put(frame, timeout=0.001)
            except Exception as e:
                logger.warning(
                    f"Stream '{self._stream_id}': Failed to enqueue frame: {e}"
                )

    def _peek_latest_frame(self) -> Optional[ImageFrame]:
        """
        Return the newest buffered frame without consuming queued frames.
        """
        with self._image_queue.mutex:
            if not self._image_queue.queue:
                return None
            return self._image_queue.queue[-1]

    def _fill_frame_from_responses(
        self,
        responses: List[image_pb2.ImageResponse],
        frame: ImageFrame,
    ) -> None:
        """
        Fill a preallocated frame with image data from responses.
        """
        n_cameras = len(self._camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(responses) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} responses, got {len(responses)}"
            )

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1

            self._convert_image_response_inplace(
                responses[rgb_idx],
                is_depth=False,
                out_array=frame.rgb_images[i],
            )
            if self._ccms is not None:
                self._apply_ccm_inplace(frame.rgb_images[i], self._ccms[self._camera_order[i]])
            self._convert_image_response_inplace(
                responses[depth_idx],
                is_depth=True,
                out_array=frame.depth_images[i],
            )

        # Update timestamps
        frame.timestamp = time.monotonic()
        if responses and responses[0].HasField("shot"):
            frame.acquisition_time = (
                responses[0].shot.acquisition_time.seconds
                + responses[0].shot.acquisition_time.nanos / 1e9
            )
        else:
            frame.acquisition_time = None

    def _fill_frame_from_decoded(
        self,
        decoded: List[np.ndarray],
        frame: ImageFrame,
    ) -> None:
        """
        Fill a preallocated frame with already-decoded arrays.
        """
        n_cameras = len(self._camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(decoded) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} decoded arrays, got {len(decoded)}"
            )

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1
            np.copyto(frame.rgb_images[i], decoded[rgb_idx])
            np.copyto(frame.depth_images[i], decoded[depth_idx])

        frame.timestamp = time.monotonic()
        frame.acquisition_time = None

    def _decode_initial_responses(
        self,
        responses: List[image_pb2.ImageResponse],
    ) -> List[np.ndarray]:
        """
        Decode initial responses once for shape inference and first frame fill.
        """
        n_cameras = len(self._camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(responses) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} responses, got {len(responses)}"
            )

        decoded: List[np.ndarray] = []
        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1
            rgb = self._convert_image_response_alloc(responses[rgb_idx], is_depth=False)
            if self._ccms is not None:
                self._apply_ccm_inplace(rgb, self._ccms[self._camera_order[i]])
            decoded.append(rgb)
            decoded.append(self._convert_image_response_alloc(responses[depth_idx], is_depth=True))
        return decoded

    def _convert_image_response_alloc(
        self,
        response: image_pb2.ImageResponse,
        is_depth: bool,
    ) -> np.ndarray:
        """
        Convert ImageResponse protobuf to a newly allocated numpy array.
        """
        image_proto = response.shot.image

        if image_proto.format == image_pb2.Image.FORMAT_JPEG:
            img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise SpotCamStreamError("Failed to decode JPEG image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32) / 255.0

        if image_proto.format == image_pb2.Image.FORMAT_RAW:
            rows = image_proto.rows
            cols = image_proto.cols
            pixel_format = image_proto.pixel_format

            if is_depth:
                if pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    raise SpotCamStreamError(
                        f"Unexpected depth pixel format: {pixel_format}"
                    )
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

            raise SpotCamStreamError(
                f"Unsupported pixel format: {pixel_format}"
            )

        raise SpotCamStreamError(
            f"Unsupported image format: {image_proto.format}"
        )

    def _convert_image_response_inplace(
        self,
        response: image_pb2.ImageResponse,
        is_depth: bool,
        out_array: np.ndarray,
    ) -> None:
        """
        Convert ImageResponse protobuf to numpy array in-place.

        Args:
            response: ImageResponse protobuf
            is_depth: True if this is a depth image
            out_array: Preallocated output array

        Raises:
            SpotCamStreamError: If image format is unsupported
        """
        image_proto = response.shot.image

        # Decode based on format
        if image_proto.format == image_pb2.Image.FORMAT_JPEG:
            # Decode JPEG
            img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise SpotCamStreamError("Failed to decode JPEG image")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            if out_array.shape != img.shape or out_array.dtype != np.float32:
                raise SpotCamStreamError(
                    f"Output array shape mismatch: {out_array.shape} vs {img.shape}"
                )
            np.multiply(img, 1.0 / 255.0, out=out_array, casting="unsafe")
            return

        elif image_proto.format == image_pb2.Image.FORMAT_RAW:
            # Handle raw pixel data
            rows = image_proto.rows
            cols = image_proto.cols
            pixel_format = image_proto.pixel_format

            if is_depth:
                # Depth image (16-bit unsigned)
                if pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    raise SpotCamStreamError(
                        f"Unexpected depth pixel format: {pixel_format}"
                    )

                # Parse as uint16
                img_data = np.frombuffer(image_proto.data, dtype=np.uint16)
                img = img_data.reshape((rows, cols))

                if out_array.shape != img.shape or out_array.dtype != np.float32:
                    raise SpotCamStreamError(
                        f"Output array shape mismatch: {out_array.shape} vs {img.shape}"
                    )

                # Convert to meters using depth scale
                depth_scale = response.source.depth_scale if response.source.depth_scale > 0 else 1.0
                np.multiply(img, 1.0 / depth_scale, out=out_array, casting="unsafe")
                return

            else:
                # RGB/grayscale image
                if pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
                    img = img_data.reshape((rows, cols, 3))
                    if out_array.shape != img.shape or out_array.dtype != np.float32:
                        raise SpotCamStreamError(
                            f"Output array shape mismatch: {out_array.shape} vs {img.shape}"
                        )
                    np.multiply(img, 1.0 / 255.0, out=out_array, casting="unsafe")
                    return

                elif pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
                    img = img_data.reshape((rows, cols, 4))
                    # Drop alpha channel
                    img = img[:, :, :3]
                    if out_array.shape != img.shape or out_array.dtype != np.float32:
                        raise SpotCamStreamError(
                            f"Output array shape mismatch: {out_array.shape} vs {img.shape}"
                        )
                    np.multiply(img, 1.0 / 255.0, out=out_array, casting="unsafe")
                    return

                elif pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    img_data = np.frombuffer(image_proto.data, dtype=np.uint8)
                    img = img_data.reshape((rows, cols))
                    if out_array.shape != (rows, cols, 3) or out_array.dtype != np.float32:
                        raise SpotCamStreamError(
                            f"Output array shape mismatch: {out_array.shape} vs {(rows, cols, 3)}"
                        )
                    channel = out_array[:, :, 0]
                    np.multiply(img, 1.0 / 255.0, out=channel, casting="unsafe")
                    out_array[:, :, 1] = channel
                    out_array[:, :, 2] = channel
                    return

                else:
                    raise SpotCamStreamError(
                        f"Unsupported pixel format: {pixel_format}"
                    )

        else:
            raise SpotCamStreamError(
                f"Unsupported image format: {image_proto.format}"
            )

    @staticmethod
    def _apply_ccm_inplace(img: np.ndarray, matrix: np.ndarray) -> None:
        """
        Apply a 3x3 color correction matrix to an (H, W, 3) float32 image in-place.

        Computes corrected = matrix @ pixel for each pixel (column-vector form),
        equivalent to img @ matrix.T in numpy row-vector form, then clips to [0, 1].
        """
        img[:] = img @ matrix.T
        np.clip(img, 0.0, 1.0, out=img)

    @staticmethod
    def _valid_camera_mask() -> int:
        """Return a bitmask of all known camera types."""
        mask = 0
        for camera_type in CameraType:
            if camera_type != 0:
                mask |= camera_type
        return int(mask)

    def _clear_queue(self) -> None:
        """Clear all items from the queue."""
        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except Empty:
                break

    def __repr__(self) -> str:
        """String representation."""
        status = "streaming" if self._streaming else "stopped"
        cameras = [cam.name for cam in self._camera_order] if self._camera_order else []
        return (
            f"SpotCamStream(id='{self._stream_id}', status='{status}', "
            f"cameras={cameras}, frames={self._frame_count})"
        )
