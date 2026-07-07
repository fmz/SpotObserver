"""
SpotCamStream - Manages camera streaming from Spot robot.
"""

import asyncio
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from bosdyn.api import image_pb2  # type: ignore[import-untyped]
from bosdyn.client.image import ImageClient, build_image_request  # type: ignore[import-untyped]

from .color_correction import _ROBOT_CCMS
from .config import CameraType, SpotConfig
from .stitch import (
    STITCH_OUT_H,
    STITCH_OUT_W,
    CamStitchParams,
    compute_stitch,
    extract_btw_params,
    extract_ctb_params,
    extract_stitch_params,
)

if TYPE_CHECKING:
    from .vision_pipeline import VisionPipeline

_VIRTUAL_CAMERAS = frozenset({CameraType.FRONTSTITCHED})


logger = logging.getLogger(__name__)


def _sanitize_dump_name(name: str) -> str:
    sanitized = "".join(
        char.lower() if char.isalnum() else "_" for char in name if char.isalnum() or char in ".-_"
    )
    return sanitized or "unknown_robot"


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
        body_to_world: Body poses in the vision/world frame aligned with _sdk_camera_order
        frame_id: Monotonic frame index assigned by the producer thread. Used to
            key dumped files so producer-side (RGB/depth/pose) and consumer-side
            (pipeline output) dumps for the same capture share a filename.
    """

    rgb_images: list[np.ndarray]
    depth_images: list[np.ndarray]
    camera_order: list[CameraType]
    timestamp: float
    acquisition_time: float | None = None
    body_to_world: list[np.ndarray] | None = None
    frame_id: int = 0


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
        >>> rgb_images, depth_images, body_T_worlds = stream.get_current_images()
        >>> stream.stop_streaming()
    """

    def __init__(
        self,
        image_client: ImageClient,
        config: SpotConfig,
        stream_id: str,
        robot_dump_name: str | None = None,
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
        self.dumps_enabled = config.dumps_enabled
        self.save_dir = (
            os.path.join(
                config.save_dir,
                robot_dump_name or _sanitize_dump_name(config.robot_ip),
            )
            if config.save_dir is not None
            else None
        )

        # Threading control
        self._streaming = False
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Image buffer (FIFO queue)
        self._image_queue: Queue[ImageFrame] = Queue(maxsize=config.image_buffer_size)

        # Current camera configuration
        self._camera_mask: int = 0
        self._camera_order: list[CameraType] = []
        self._sdk_camera_order: list[CameraType] = []  # _camera_order minus virtual cameras
        self._sdk_to_frame_idx: dict = {}  # sdk camera index -> frame slot index

        # Stitch state (active when FRONTSTITCHED is in camera mask)
        self._stitch_enabled: bool = False
        self._stitch_params_l: CamStitchParams | None = None
        self._stitch_params_r: CamStitchParams | None = None
        self._stitch_fl_idx: int = 0
        self._stitch_fr_idx: int = 0
        self._stitch_out_idx: int = 0
        self._last_body_to_worlds: list[np.ndarray] | None = None

        # Preallocated frame pool (initialized once we know image shapes)
        self._frame_pool: list[ImageFrame] = []
        self._frame_pool_index: int = 0
        self._image_requests: list[image_pb2.ImageRequest] = []

        # Optional per-stream vision pipeline, imported lazily.
        self._vision_pipeline: VisionPipeline | None = None

        # Scratch buffers for color correction by image shape.
        self._ccm_scratch_by_shape: dict[tuple[int, ...], np.ndarray] = {}

        # Color correction matrices (None if robot IP is not recognized)
        self._ccms: dict | None = _ROBOT_CCMS.get(config.robot_ip)
        if self._ccms is not None:
            logger.info(
                f"SpotCamStream '{stream_id}': Color correction enabled for {config.robot_ip}"
            )
        else:
            logger.info(
                f"SpotCamStream '{stream_id}': No color correction (unrecognized IP {config.robot_ip})"
            )

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
            raise SpotCamStreamError(f"Camera mask contains unknown bits: {camera_mask:#x}")

        # Parse camera mask to get ordered list of cameras
        self._camera_mask = camera_mask
        self._camera_order = self._parse_camera_mask(camera_mask)
        self._sdk_camera_order = [c for c in self._camera_order if c not in _VIRTUAL_CAMERAS]
        self._sdk_to_frame_idx = {
            i: self._camera_order.index(cam) for i, cam in enumerate(self._sdk_camera_order)
        }
        self._stitch_enabled = CameraType.FRONTSTITCHED in self._camera_order
        self._stitch_params_l = None
        self._stitch_params_r = None
        if self._stitch_enabled:
            self._stitch_fl_idx = self._camera_order.index(CameraType.FRONTLEFT)
            self._stitch_fr_idx = self._camera_order.index(CameraType.FRONTRIGHT)
            self._stitch_out_idx = self._camera_order.index(CameraType.FRONTSTITCHED)
        self._image_requests = self._build_image_requests()

        logger.info(
            f"Stream '{self._stream_id}': Starting with cameras: "
            f"{[cam.name or cam.value for cam in self._camera_order]}"
        )

        # Clear queue and reset statistics
        self._clear_queue()
        self._frame_pool = []
        self._frame_pool_index = 0
        self._ccm_scratch_by_shape = {}
        self._frame_count = 0
        self._error_count = 0
        self._last_body_to_worlds = None

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

    def _dump_data(
        self,
        rgb: list[np.ndarray] | None = None,
        depth: list[np.ndarray] | None = None,
        b2w: list[np.ndarray] | None = None,
        c2b_responses: list[image_pb2.ImageResponse] | None = None,
        depth_dir_name: str = "output_depth",
        frame_id: int | None = None,
    ) -> None:
        """Dump streaming data as individual files.

        Args:
            frame_id: Frame index used to name the output files. When None, falls
                back to the current producer frame count. Callers on the producer
                thread pass the frame's own id so files stay aligned across the
                producer (RGB/depth/pose) and consumer (pipeline output) dumps.
        """
        # Keep this guard here so future dump call sites cannot bypass config.
        if not self.dumps_enabled:
            return
        if self.save_dir is None:
            logger.warning(
                f"Stream '{self._stream_id}': dumps_enabled=True but save_dir is not set. "
                "Skipping save."
            )
            return

        frame_id_str = str(self._frame_count if frame_id is None else frame_id)

        if rgb is not None:
            for camera, image in zip(self._camera_order, rgb):
                camera_name = (camera.name or f"camera_{camera.value}").lower()
                rgb_dir = os.path.join(self.save_dir, camera_name, "rgb")
                os.makedirs(rgb_dir, exist_ok=True)
                file_name = f"{frame_id_str}.png"
                image_u8 = np.clip(image, 0.0, 1.0)
                image_u8 = (image_u8 * 255.0).astype(np.uint8)
                image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
                file_path = os.path.join(rgb_dir, file_name)
                if not cv2.imwrite(file_path, image_bgr):
                    raise SpotCamStreamError(f"Failed to write RGB dump: {file_path}")

        if depth is not None:
            for camera, image in zip(self._camera_order, depth):
                camera_name = (camera.name or f"camera_{camera.value}").lower()
                depth_dir = os.path.join(self.save_dir, camera_name, depth_dir_name)
                os.makedirs(depth_dir, exist_ok=True)
                file_name = frame_id_str
                depth_f32 = np.ascontiguousarray(image, dtype=np.float32)
                with open(os.path.join(depth_dir, file_name), "wb") as f:
                    f.write(struct.pack("<I", depth_f32.size))
                    depth_f32.tofile(f)

        if b2w is not None:
            for camera, b2w_camera in zip(self._sdk_camera_order, b2w):
                camera_name = (camera.name or f"camera_{camera.value}").lower()
                transform_dir = os.path.join(self.save_dir, camera_name, "camera_transforms")
                os.makedirs(transform_dir, exist_ok=True)
                torch.save(
                    torch.from_numpy(np.ascontiguousarray(b2w_camera, dtype=np.float32)),
                    os.path.join(transform_dir, f"world_T_body_{frame_id_str}.pt"),
                )

        if c2b_responses is not None:
            for i, camera in enumerate(self._sdk_camera_order):
                camera_name = (camera.name or f"camera_{camera.value}").lower()
                transform_dir = os.path.join(self.save_dir, camera_name, "camera_transforms")
                os.makedirs(transform_dir, exist_ok=True)
                response_idx = i * 2
                try:
                    body_T_camera = extract_ctb_params(c2b_responses[response_idx])
                except ValueError as exc:
                    logger.warning(
                        f"Stream '{self._stream_id}': Failed to extract camera-to-body "
                        f"pose for {camera_name}: {exc}"
                    )
                    continue

                file_name = "body_T_camera.pt"
                torch.save(
                    torch.from_numpy(np.ascontiguousarray(body_T_camera, dtype=np.float32)),
                    os.path.join(transform_dir, file_name),
                )

    def get_current_images(
        self,
        timeout: float | None = None,
        copy: bool = False,
        run_pipeline: bool = False,
        dump: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
            Tuple of (rgb_images, depth_images, b2w). RGB/depth lists are in camera_order.
                b2w is in SDK camera order, which matches camera_order except virtual cameras
                such as FRONTSTITCHED are omitted.

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
                rgb, depth = frame.rgb_images, frame.depth_images

                if frame.body_to_world is None:
                    raise SpotCamStreamError(
                        "Body-to-world pose is unavailable for the current frame"
                    )

                b2w = frame.body_to_world

                if copy:
                    rgb = [img.copy() for img in rgb]
                    depth = [img.copy() for img in depth]
                    b2w = [pose.copy() for pose in b2w]

                if run_pipeline:
                    rgb, depth = self._run_vision_pipeline(rgb, depth)
                    # Raw RGB/depth/pose are dumped by the producer thread every
                    # frame; here we only persist the pipeline output, keyed to the
                    # same frame id so it lines up with the producer's dumps.
                    if dump and self.dumps_enabled:
                        self._dump_data(
                            depth=depth,
                            depth_dir_name="output_depth",
                            frame_id=frame.frame_id,
                        )

                # Should be consistent, but will log all to be sure
                b2w_shapes = set([e.shape for e in b2w])

                logger.info(
                    f"b2w received. length {len(b2w)}, shape(s): {b2w_shapes}, sample b2w: {b2w[0]}"
                )

                return rgb, depth, b2w

            wait_timeout = poll_interval
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SpotCamStreamError(f"Timeout waiting for images (timeout={timeout}s)")
                wait_timeout = min(wait_timeout, remaining)

            try:
                frame = self._image_queue.get(timeout=wait_timeout)
            except Empty:
                continue

            rgb, depth = frame.rgb_images, frame.depth_images

            if frame.body_to_world is None:
                raise SpotCamStreamError("Body-to-world pose is unavailable for the current frame")

            b2w = frame.body_to_world

            if copy:
                rgb = [img.copy() for img in rgb]
                depth = [img.copy() for img in depth]
                b2w = [pose.copy() for pose in b2w]

            if run_pipeline:
                rgb, depth = self._run_vision_pipeline(rgb, depth)
                # Raw RGB/depth/pose are dumped by the producer thread every
                # frame; here we only persist the pipeline output, keyed to the
                # same frame id so it lines up with the producer's dumps.
                if dump and self.dumps_enabled:
                    self._dump_data(
                        depth=depth,
                        depth_dir_name="output_depth",
                        frame_id=frame.frame_id,
                    )

            # Should be consistent, but will log all to be sure
            b2w_shapes = set([e.shape for e in b2w])

            logger.info(
                f"b2w received. length {len(b2w)}, shape(s): {b2w_shapes}, sample b2w: {b2w[0]}"
            )

            return rgb, depth, b2w

    async def async_get_current_images(
        self,
        timeout: float | None = None,
        copy: bool = False,
        run_pipeline: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Async version of get_current_images().

        Note:
            Returned arrays are backed by a preallocated frame pool and may be
            overwritten by subsequent frames. Copy if you need to retain them.

        Args:
            timeout: Maximum seconds to wait for images. None = wait forever.
            copy: If True, returns deep copies of image arrays.

        Returns:
            Tuple of (rgb_images, depth_images, b2w). RGB/depth lists are in camera_order.
                b2w is in SDK camera order, which matches camera_order except virtual cameras
                such as FRONTSTITCHED are omitted.

        Raises:
            SpotCamStreamError: If not streaming or timeout occurs
        """
        loop = asyncio.get_event_loop()

        # get_current_images already runs the vision pipeline and the pipeline-output
        # dump when requested; running the whole call in the executor keeps the event
        # loop unblocked without duplicating that logic here.
        return await loop.run_in_executor(
            None, self.get_current_images, timeout, copy, run_pipeline
        )

    def _run_vision_pipeline(
        self,
        rgb_images: list[np.ndarray],
        depth_images: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        try:
            from .vision_pipeline import VisionPipeline, VisionPipelineError
        except ImportError as exc:
            raise SpotCamStreamError(
                "Vision pipeline support is unavailable. "
                'Install PySpotObserver with: pip install -e ".[vision]"'
            ) from exc

        if self._vision_pipeline is None:
            try:
                self._vision_pipeline = VisionPipeline.from_config(self._config)
            except VisionPipelineError as exc:
                raise SpotCamStreamError(str(exc)) from exc

        try:
            pipeline = self._vision_pipeline
            if pipeline is None:
                raise SpotCamStreamError("Vision pipeline failed to initialize")
            return pipeline.run(rgb_images, depth_images)
        except VisionPipelineError as exc:
            raise SpotCamStreamError(str(exc)) from exc

    def get_camera_order(self) -> list[CameraType]:
        """
        Get the ordered list of cameras being streamed.

        Returns:
            List of CameraType enums in the order images are returned
        """
        return self._camera_order.copy()

    def _parse_camera_mask(self, mask: int) -> list[CameraType]:
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

        if CameraType.FRONTSTITCHED in cameras:
            if CameraType.FRONTLEFT not in cameras or CameraType.FRONTRIGHT not in cameras:
                raise SpotCamStreamError(
                    "FRONTSTITCHED requires both FRONTLEFT and FRONTRIGHT in camera mask"
                )

        return cameras

    def _stream_loop(self) -> None:
        """
        Producer thread main loop.

        Continuously requests images and pushes to queue.
        """
        logger.debug(f"Stream '{self._stream_id}': Thread started")

        while not self._stop_event.is_set() and self._streaming:
            try:
                # Request images from robot
                start_time = time.monotonic()
                image_responses = self._image_client.get_image(
                    self._image_requests,
                    timeout=self._config.request_timeout_seconds,
                )
                request_time = time.monotonic() - start_time

                # Initialize the frame pool from the first successful response,
                # then reuse it for all subsequent frames.
                if not self._frame_pool:
                    initial_decoded = self._decode_initial_responses(image_responses)
                    self._initialize_frame_pool(initial_decoded)
                    self._dump_data(c2b_responses=image_responses)
                    if self._stitch_enabled:
                        self._cache_stitch_params(image_responses)
                    frame = self._next_frame_from_pool()
                    self._fill_frame_from_decoded(initial_decoded, frame)
                    self._fill_frame_metadata_from_responses(image_responses, frame)
                else:
                    frame = self._next_frame_from_pool()
                    self._fill_frame_from_responses(image_responses, frame)
                if self._stitch_enabled:
                    self._apply_stitch(frame)

                frame.frame_id = self._frame_count
                self._enqueue_frame(frame)

                # Dump raw capture data (RGB, input depth, body-to-world pose) from
                # the producer thread so every streamed frame is saved as soon as it
                # arrives, instead of only when a consumer happens to call
                # get_current_images(). This mirrors the C++ producer
                # (ReaderWriterCBuf::push).
                if self.dumps_enabled:
                    self._dump_data(
                        rgb=frame.rgb_images,
                        depth=frame.depth_images,
                        b2w=frame.body_to_world,
                        depth_dir_name="input-depth",
                        frame_id=frame.frame_id,
                    )

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

    def _build_image_requests(self) -> list[image_pb2.ImageRequest]:
        """
        Build list of ImageRequest protos for all cameras (RGB + depth).

        Returns:
            List of ImageRequest protobuf objects
        """
        requests = []

        for camera in self._sdk_camera_order:
            # RGB request
            rgb_source = CameraType.get_source_name(camera, depth=False)
            requests.append(
                build_image_request(
                    rgb_source,
                    quality_percent=self._config.image_quality_percent,
                    image_format=image_pb2.Image.FORMAT_JPEG,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
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
        decoded: list[np.ndarray],
    ) -> None:
        """
        Initialize a preallocated pool of frames based on initial responses.

        Args:
            decoded: List of decoded arrays ordered as RGB, Depth, RGB, Depth...

        Raises:
            SpotCamStreamError: If responses are malformed
        """
        n_cameras = len(self._sdk_camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(decoded) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} decoded arrays, got {len(decoded)}"
            )

        # Infer shapes from decoded arrays
        rgb_shapes: list[tuple[int, ...]] = []
        depth_shapes: list[tuple[int, ...]] = []

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1

            rgb_shape = decoded[rgb_idx].shape
            depth_shape = decoded[depth_idx].shape
            rgb_shapes.append(rgb_shape)
            depth_shapes.append(depth_shape)

        # Add output shape for the virtual FRONTSTITCHED slot.
        # Height matches the front camera images so the full vertical FOV is captured.
        if self._stitch_enabled:
            rgb_shapes.insert(self._stitch_out_idx, (STITCH_OUT_H, STITCH_OUT_W, 3))
            depth_shapes.insert(self._stitch_out_idx, (STITCH_OUT_H, STITCH_OUT_W))

        # Preallocate a static pool of frames (one per queue slot)
        pool_size = max(1, self._config.image_buffer_size)
        self._frame_pool = []
        self._frame_pool_index = 0

        for _ in range(pool_size):
            rgb_images = [np.zeros(shape, dtype=np.float32) for shape in rgb_shapes]
            depth_images = [np.zeros(shape, dtype=np.float32) for shape in depth_shapes]
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
                logger.warning(f"Stream '{self._stream_id}': Failed to enqueue frame: {e}")

    def _peek_latest_frame(self) -> ImageFrame | None:
        """
        Return the newest buffered frame without consuming queued frames.
        """
        with self._image_queue.mutex:
            if not self._image_queue.queue:
                return None
            return self._image_queue.queue[-1]

    @staticmethod
    def _copy_frame(frame: ImageFrame) -> ImageFrame:
        body_to_world = None
        if frame.body_to_world is not None:
            body_to_world = [pose.copy() for pose in frame.body_to_world]
        return ImageFrame(
            rgb_images=[img.copy() for img in frame.rgb_images],
            depth_images=[img.copy() for img in frame.depth_images],
            camera_order=frame.camera_order.copy(),
            timestamp=frame.timestamp,
            acquisition_time=frame.acquisition_time,
            body_to_world=body_to_world,
            frame_id=frame.frame_id,
        )

    def _fill_frame_from_responses(
        self,
        responses: list[image_pb2.ImageResponse],
        frame: ImageFrame,
    ) -> None:
        """
        Fill a preallocated frame with image data from responses.
        """
        n_cameras = len(self._sdk_camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(responses) != expected_count:
            raise SpotCamStreamError(f"Expected {expected_count} responses, got {len(responses)}")

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1
            frame_i = self._sdk_to_frame_idx[i]

            ccm = None
            if self._ccms is not None:
                ccm = self._ccms[self._sdk_camera_order[i]]
            self._convert_image_response_inplace(
                responses[rgb_idx], is_depth=False, out_array=frame.rgb_images[frame_i], ccm=ccm
            )

            self._convert_image_response_inplace(
                responses[depth_idx],
                is_depth=True,
                out_array=frame.depth_images[frame_i],
            )

        # Update timestamps and camera extrinsics params
        self._fill_frame_metadata_from_responses(responses, frame)

    def _fill_frame_from_decoded(
        self,
        decoded: list[np.ndarray],
        frame: ImageFrame,
    ) -> None:
        """
        Fill a preallocated frame with already-decoded arrays.
        """
        n_cameras = len(self._sdk_camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(decoded) != expected_count:
            raise SpotCamStreamError(
                f"Expected {expected_count} decoded arrays, got {len(decoded)}"
            )

        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1
            frame_i = self._sdk_to_frame_idx[i]
            np.copyto(frame.rgb_images[frame_i], decoded[rgb_idx])
            np.copyto(frame.depth_images[frame_i], decoded[depth_idx])

        frame.timestamp = time.monotonic()
        frame.acquisition_time = None
        frame.body_to_world = None

    def _decode_initial_responses(
        self,
        responses: list[image_pb2.ImageResponse],
    ) -> list[np.ndarray]:
        """
        Decode initial responses once for shape inference and first frame fill.
        """
        n_cameras = len(self._sdk_camera_order)
        expected_count = n_cameras * 2  # RGB + depth per camera

        if len(responses) != expected_count:
            raise SpotCamStreamError(f"Expected {expected_count} responses, got {len(responses)}")

        decoded: list[np.ndarray] = []
        for i in range(n_cameras):
            rgb_idx = i * 2
            depth_idx = i * 2 + 1
            rgb = self._convert_image_response_alloc(responses[rgb_idx], is_depth=False)
            if self._ccms is not None:
                self._apply_ccm_inplace(rgb, self._ccms[self._sdk_camera_order[i]])
            decoded.append(rgb)
            decoded.append(self._convert_image_response_alloc(responses[depth_idx], is_depth=True))
        return decoded

    def _cache_stitch_params(self, responses: list[image_pb2.ImageResponse]) -> None:
        """Extract and cache body-to-camera transforms and intrinsics for stitching."""
        fl_sdk_i = self._sdk_camera_order.index(CameraType.FRONTLEFT)
        fr_sdk_i = self._sdk_camera_order.index(CameraType.FRONTRIGHT)
        self._stitch_params_l = extract_stitch_params(responses[fl_sdk_i * 2])
        self._stitch_params_r = extract_stitch_params(responses[fr_sdk_i * 2])
        logger.info(f"Stream '{self._stream_id}': Stitch params cached for front cameras")

    def _fill_frame_metadata_from_responses(
        self,
        responses: list[image_pb2.ImageResponse],
        frame: ImageFrame,
    ) -> None:
        """Fill timestamp and body-to-world metadata from the response batch."""
        frame.timestamp = time.monotonic()
        if responses and responses[0].HasField("shot"):
            frame.acquisition_time = (
                responses[0].shot.acquisition_time.seconds
                + responses[0].shot.acquisition_time.nanos / 1e9
            )
            try:
                body_to_worlds: list[np.ndarray] = []
                for i in range(len(self._sdk_camera_order)):
                    body_to_worlds.append(extract_btw_params(responses[i * 2]))
                frame.body_to_world = body_to_worlds
                self._last_body_to_worlds = [pose.copy() for pose in frame.body_to_world]
            except ValueError as exc:
                logger.warning(
                    f"Stream '{self._stream_id}': Failed to extract body-to-world "
                    f"pose; using previous pose if available: {exc}"
                )
                if self._last_body_to_worlds is None:
                    frame.body_to_world = None
                else:
                    frame.body_to_world = [pose.copy() for pose in self._last_body_to_worlds]
        else:
            frame.acquisition_time = None
            frame.body_to_world = None

    def _apply_stitch(self, frame: "ImageFrame") -> None:
        """Compute the stitched front view and write it into the FRONTSTITCHED frame slot."""
        if self._stitch_params_l is None or self._stitch_params_r is None:
            return
        compute_stitch(
            frame.rgb_images[self._stitch_fl_idx],
            frame.depth_images[self._stitch_fl_idx],
            self._stitch_params_l,
            frame.rgb_images[self._stitch_fr_idx],
            frame.depth_images[self._stitch_fr_idx],
            self._stitch_params_r,
            frame.rgb_images[self._stitch_out_idx],
            frame.depth_images[self._stitch_out_idx],
        )

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
                    raise SpotCamStreamError(f"Unexpected depth pixel format: {pixel_format}")
                img_data = np.frombuffer(image_proto.data, dtype=np.uint16)
                img = img_data.reshape((rows, cols))
                depth_scale = (
                    response.source.depth_scale if response.source.depth_scale > 0 else 1.0
                )
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

            raise SpotCamStreamError(f"Unsupported pixel format: {pixel_format}")

        raise SpotCamStreamError(f"Unsupported image format: {image_proto.format}")

    def _convert_image_response_inplace(
        self,
        response: image_pb2.ImageResponse,
        is_depth: bool,
        out_array: np.ndarray,
        ccm: np.ndarray | None = None,
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
            if ccm is not None:
                self._apply_ccm_inplace(
                    out_array,
                    ccm,
                    scratch=self._get_ccm_scratch(out_array.shape),
                )
            return

        elif image_proto.format == image_pb2.Image.FORMAT_RAW:
            # Handle raw pixel data
            rows = image_proto.rows
            cols = image_proto.cols
            pixel_format = image_proto.pixel_format

            if is_depth:
                # Depth image (16-bit unsigned)
                if pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    raise SpotCamStreamError(f"Unexpected depth pixel format: {pixel_format}")

                # Parse as uint16
                img_data = np.frombuffer(image_proto.data, dtype=np.uint16)
                img = img_data.reshape((rows, cols))

                if out_array.shape != img.shape or out_array.dtype != np.float32:
                    raise SpotCamStreamError(
                        f"Output array shape mismatch: {out_array.shape} vs {img.shape}"
                    )

                # Convert to meters using depth scale
                depth_scale = (
                    response.source.depth_scale if response.source.depth_scale > 0 else 1.0
                )
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
                    if ccm is not None:
                        self._apply_ccm_inplace(
                            out_array,
                            ccm,
                            scratch=self._get_ccm_scratch(out_array.shape),
                        )
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
                    if ccm is not None:
                        self._apply_ccm_inplace(
                            out_array,
                            ccm,
                            scratch=self._get_ccm_scratch(out_array.shape),
                        )
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
                    if ccm is not None:
                        self._apply_ccm_inplace(
                            out_array,
                            ccm,
                            scratch=self._get_ccm_scratch(out_array.shape),
                        )
                    return

                else:
                    raise SpotCamStreamError(f"Unsupported pixel format: {pixel_format}")

        else:
            raise SpotCamStreamError(f"Unsupported image format: {image_proto.format}")

    def _get_ccm_scratch(self, shape: tuple[int, ...]) -> np.ndarray:
        scratch = self._ccm_scratch_by_shape.get(shape)
        if scratch is None:
            scratch = np.empty(shape, dtype=np.float32)
            self._ccm_scratch_by_shape[shape] = scratch
        return scratch

    @staticmethod
    def _apply_ccm_inplace(
        img: np.ndarray,
        matrix: np.ndarray,
        scratch: np.ndarray | None = None,
    ) -> None:
        """
        Apply a 3x3 color correction matrix to an (H, W, 3) float32 image in-place.

        Matrices use row-vector convention: corrected = pixel @ matrix.
        The CCM was calibrated in linear light, so we gamma-decode before
        applying and gamma-encode after (matching the extract_matrices.py pipeline).
        """
        linear = scratch if scratch is not None else np.empty_like(img)
        np.copyto(linear, img)

        low = linear <= 0.04045
        linear[low] /= 12.92
        linear[~low] = ((linear[~low] + 0.055) / 1.055) ** 2.4

        np.matmul(linear, matrix, out=img)
        np.clip(img, 0.0, 1.0, out=img)

        low = img <= 0.0031308
        img[low] *= 12.92
        img[~low] = 1.055 * img[~low] ** (1.0 / 2.4) - 0.055
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
        cameras = (
            [cam.name or cam.value for cam in self._camera_order] if self._camera_order else []
        )
        return (
            f"SpotCamStream(id='{self._stream_id}', status='{status}', "
            f"cameras={cameras}, frames={self._frame_count})"
        )
