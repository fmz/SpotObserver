"""
Optional ONNX Runtime vision pipeline support.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import SpotConfig

DEFAULT_MODEL_ENV_VAR = "PYSPOTOBSERVER_VISION_MODEL"
DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
DEFAULT_DEPTH_SIZE = (120, 160)  # (height, width)


class VisionPipelineError(Exception):
    """Raised when the optional vision pipeline cannot run."""


def _normalize_providers(providers: Optional[Union[Sequence[str], str]]) -> Tuple[str, ...]:
    if providers is None:
        return DEFAULT_PROVIDERS
    if isinstance(providers, str):
        return tuple(part.strip() for part in providers.split(",") if part.strip())
    return tuple(provider for provider in providers if provider)


def _dtype_for_onnx_type(type_name: str) -> np.dtype:
    normalized = type_name.lower()
    if "float16" in normalized:
        return np.dtype(np.float16)
    if "double" in normalized or "float64" in normalized:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def _depth_list_from_output(output: np.ndarray, batch_size: int) -> List[np.ndarray]:
    output = np.asarray(output)

    if batch_size == 1 and output.ndim == 2:
        output = output[np.newaxis, :, :]
    elif output.ndim == 4 and output.shape[1] == 1:
        output = output[:, 0, :, :]
    elif output.ndim == 4 and output.shape[-1] == 1:
        output = output[:, :, :, 0]

    if output.ndim != 3 or output.shape[0] != batch_size:
        raise VisionPipelineError(
            "Vision model output must be shaped as (B, H, W), "
            "(B, 1, H, W), or (B, H, W, 1); "
            f"got {tuple(output.shape)} for batch size {batch_size}"
        )

    return [np.asarray(output[i]) for i in range(batch_size)]


class VisionPipeline:
    """
    Per-stream ONNX vision pipeline with reusable input buffers.

    ONNX Runtime is imported lazily so camera streaming remains available without
    installing the optional vision extra.
    """

    def __init__(
        self,
        model_path: Union[str, os.PathLike[str]],
        providers: Optional[Union[Sequence[str], str]] = None,
        depth_size: Tuple[int, int] = DEFAULT_DEPTH_SIZE,
    ):
        self.model_path = Path(model_path).expanduser()
        self.providers = _normalize_providers(providers)
        self.depth_size = depth_size
        self._lock = threading.Lock()
        self._session: Optional[Any] = None
        self._input_names: List[str] = []
        self._rgb_hw: Optional[Tuple[int, int]] = (
            None  # expected (H, W) from model; None = use input size
        )
        self._depth_hw: Optional[Tuple[int, int]] = (
            None  # expected depth (H, W) from model; None = use depth_size
        )
        self._rgb_dtype = np.dtype(np.float32)
        self._depth_dtype = np.dtype(np.float32)
        self._rgb_buffer: Optional[np.ndarray] = None
        self._depth_buffer: Optional[np.ndarray] = None
        self._depth_resize_buffer: Optional[np.ndarray] = None
        self._rgb_resize_buffer: Optional[np.ndarray] = None

        if not self.model_path.exists():
            raise VisionPipelineError(f"Vision model not found: {self.model_path}")

    @classmethod
    def from_config(cls, config: SpotConfig) -> VisionPipeline:
        extra_params = config.extra_params or {}
        model_path = (
            config.vision_model_path
            or extra_params.get("vision_model_path")
            or os.environ.get(DEFAULT_MODEL_ENV_VAR)
        )
        if not model_path:
            raise VisionPipelineError(
                "Vision model path is required for run_pipeline=True. "
                "Set SpotConfig.vision_model_path, extra_params['vision_model_path'], "
                f"or {DEFAULT_MODEL_ENV_VAR}."
            )

        providers = (
            config.vision_providers
            or extra_params.get("vision_providers")
            or extra_params.get("vision_provider")
        )
        depth_size = extra_params.get("vision_depth_size", DEFAULT_DEPTH_SIZE)
        if len(depth_size) != 2:
            raise VisionPipelineError("vision_depth_size must contain height and width")

        return cls(
            model_path=model_path,
            providers=providers,
            depth_size=(int(depth_size[0]), int(depth_size[1])),
        )

    def run(
        self,
        rgb_images: List[np.ndarray],
        depth_images: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not rgb_images or not depth_images:
            raise VisionPipelineError("Vision pipeline requires at least one RGB/depth pair")
        if len(rgb_images) != len(depth_images):
            raise VisionPipelineError(
                f"RGB/depth batch mismatch: {len(rgb_images)} RGB, {len(depth_images)} depth"
            )

        with self._lock:
            self._init_session()

            self._ensure_buffers(rgb_images, depth_images)
            self._fill_buffers(rgb_images, depth_images)

            feed = {
                self._input_names[0]: self._rgb_buffer,
                self._input_names[1]: self._depth_buffer,
            }
            output = self._session.run(None, feed)[0]

        return rgb_images, _depth_list_from_output(output, len(rgb_images))

    def _init_session(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VisionPipelineError(
                "Vision pipeline requires ONNX Runtime. "
                "Install PySpotObserver with the vision extra: "
                'pip install -e ".[vision]"'
            ) from exc

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        available = set(ort.get_available_providers())
        selected = [provider for provider in self.providers if provider in available]
        if not selected:
            raise VisionPipelineError(
                "None of the requested ONNX Runtime providers are available. "
                f"requested={list(self.providers)}, available={sorted(available)}"
            )

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=selected,
        )

        inputs = self._session.get_inputs()
        if len(inputs) < 2:
            raise VisionPipelineError(
                f"Vision model must expose at least 2 inputs; got {len(inputs)}"
            )

        self._input_names = [inputs[0].name, inputs[1].name]
        self._rgb_dtype = _dtype_for_onnx_type(inputs[0].type)
        self._depth_dtype = _dtype_for_onnx_type(inputs[1].type)
        shape = inputs[0].shape
        depth_shape = inputs[1].shape

        # Read expected spatial dimensions from model graph
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self._rgb_hw = (shape[2], shape[3])
        if (
            len(depth_shape) == 4
            and isinstance(depth_shape[2], int)
            and isinstance(depth_shape[3], int)
        ):
            self._depth_hw = (depth_shape[2], depth_shape[3])

    def _ensure_buffers(
        self,
        rgb_images: Sequence[np.ndarray],
        depth_images: Sequence[np.ndarray],
    ) -> None:
        batch_size = len(rgb_images)
        in_h, in_w, channels = rgb_images[0].shape
        if channels != 3:
            raise VisionPipelineError(f"Expected RGB images with 3 channels, got {channels}")

        # Use model's expected spatial size if known; otherwise use input size.
        h_rgb, w_rgb = self._rgb_hw if self._rgb_hw is not None else (in_h, in_w)

        # Use model's expected depth size if detected; fall back to constructor argument.
        depth_h, depth_w = self._depth_hw if self._depth_hw is not None else self.depth_size
        rgb_shape = (batch_size, 3, h_rgb, w_rgb)
        depth_shape = (batch_size, 1, depth_h, depth_w)

        if (
            self._rgb_buffer is None
            or self._rgb_buffer.shape != rgb_shape
            or self._rgb_buffer.dtype != self._rgb_dtype
        ):
            self._rgb_buffer = np.empty(rgb_shape, dtype=self._rgb_dtype)

        if (
            self._depth_buffer is None
            or self._depth_buffer.shape != depth_shape
            or self._depth_buffer.dtype != self._depth_dtype
        ):
            self._depth_buffer = np.empty(depth_shape, dtype=self._depth_dtype)

        if self._depth_dtype == np.dtype(np.float16):
            if self._depth_resize_buffer is None or self._depth_resize_buffer.shape != depth_shape:
                self._depth_resize_buffer = np.empty(depth_shape, dtype=np.float32)
        else:
            self._depth_resize_buffer = None

        # Pre-allocate float32 intermediate for RGB resize if needed
        if self._rgb_hw is not None and (in_h, in_w) != self._rgb_hw:
            resize_shape = (h_rgb, w_rgb, 3)
            if self._rgb_resize_buffer is None or self._rgb_resize_buffer.shape != resize_shape:
                self._rgb_resize_buffer = np.empty(resize_shape, dtype=np.float32)
        else:
            self._rgb_resize_buffer = None

        for index, depth in enumerate(depth_images):
            if depth.ndim != 2:
                raise VisionPipelineError(
                    f"Depth image {index} must be 2D before pipeline processing; got {depth.shape}"
                )

    def _fill_buffers(
        self,
        rgb_images: Sequence[np.ndarray],
        depth_images: Sequence[np.ndarray],
    ) -> None:
        depth_h, depth_w = self._depth_hw if self._depth_hw is not None else self.depth_size

        if self._rgb_buffer is not None:
            for index, rgb in enumerate(rgb_images):
                if self._rgb_resize_buffer is not None:
                    if self._rgb_hw is not None:
                        cv2.resize(
                            rgb, (self._rgb_hw[1], self._rgb_hw[0]), dst=self._rgb_resize_buffer
                        )
                    rgb = self._rgb_resize_buffer
                self._rgb_buffer[index, 0] = rgb[:, :, 0]
                self._rgb_buffer[index, 1] = rgb[:, :, 1]
                self._rgb_buffer[index, 2] = rgb[:, :, 2]

        depth_resize_buffer = self._depth_resize_buffer
        if self._depth_buffer is not None:
            for index, depth in enumerate(depth_images):
                depth_dst = self._depth_buffer[index, 0]
                resize_dst = (
                    depth_resize_buffer[index, 0] if depth_resize_buffer is not None else depth_dst
                )
                cv2.resize(
                    depth,
                    (depth_w, depth_h),
                    dst=resize_dst,
                    interpolation=cv2.INTER_NEAREST,
                )
                if resize_dst is not depth_dst:
                    depth_dst[...] = resize_dst


_default_pipeline: Optional[VisionPipeline] = None
_default_pipeline_key: Optional[Tuple[str, Tuple[str, ...], Tuple[int, int]]] = None
_default_pipeline_lock = threading.Lock()


def run_vision_pipeline(
    rgb_images: List[np.ndarray],
    depth_images: List[np.ndarray],
    *,
    model_path: Optional[str] = None,
    providers: Optional[Union[Sequence[str], str]] = None,
    depth_size: Tuple[int, int] = DEFAULT_DEPTH_SIZE,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Backwards-compatible functional entry point using a cached default pipeline.
    """
    resolved_model_path = model_path or os.environ.get(DEFAULT_MODEL_ENV_VAR)
    if not resolved_model_path:
        raise VisionPipelineError(
            f"Vision model path is required. Pass model_path or set {DEFAULT_MODEL_ENV_VAR}."
        )

    provider_tuple = _normalize_providers(providers)
    key = (str(Path(resolved_model_path).expanduser()), provider_tuple, depth_size)

    global _default_pipeline, _default_pipeline_key
    with _default_pipeline_lock:
        if _default_pipeline is None or _default_pipeline_key != key:
            _default_pipeline = VisionPipeline(
                model_path=resolved_model_path,
                providers=provider_tuple,
                depth_size=depth_size,
            )
            _default_pipeline_key = key

    return _default_pipeline.run(rgb_images, depth_images)
