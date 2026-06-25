"""
Optional ONNX Runtime vision pipeline support.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
        self._session = None
        self._input_names: List[str] = []
        self._rgb_dtype = np.dtype(np.float32)
        self._depth_dtype = np.dtype(np.float32)
        self._rgb_buffer: Optional[np.ndarray] = None
        self._depth_buffer: Optional[np.ndarray] = None
        self._depth_resize_buffer: Optional[np.ndarray] = None

        if not self.model_path.exists():
            raise VisionPipelineError(f"Vision model not found: {self.model_path}")

    @classmethod
    def from_config(cls, config: SpotConfig) -> "VisionPipeline":
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

            output = self._session.run(
                None,
                {
                    self._input_names[0]: self._rgb_buffer,
                    self._input_names[1]: self._depth_buffer,
                },
            )[0]

        completed_depth = _complete_sparse_using_nearest(output)

        raw_list = _depth_list_from_output(output, len(rgb_images))
        completed_list = _depth_list_from_output(completed_depth, len(rgb_images))
        for i, raw in enumerate(raw_list):
            
            print(f"dtype = {raw.dtype}")
            print(f" nan count = {np.sum(np.isnan(raw))}")
            print(f"percentiles: {np.nanpercentile(raw, [0, 1, 5, 25, 50, 75, 99, 100])}")

        return rgb_images, _depth_list_from_output(completed_depth, len(rgb_images))

    def _init_session(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
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

    def _ensure_buffers(
        self,
        rgb_images: Sequence[np.ndarray],
        depth_images: Sequence[np.ndarray],
    ) -> None:
        batch_size = len(rgb_images)
        h_rgb, w_rgb, channels = rgb_images[0].shape
        if channels != 3:
            raise VisionPipelineError(f"Expected RGB images with 3 channels, got {channels}")

        depth_h, depth_w = self.depth_size
        rgb_shape = (batch_size, 3, h_rgb, w_rgb)
        depth_shape = (batch_size, 1, depth_h, depth_w)

        if self._rgb_buffer is None or self._rgb_buffer.shape != rgb_shape:
            self._rgb_buffer = np.empty(rgb_shape, dtype=self._rgb_dtype)
        elif self._rgb_buffer.dtype != self._rgb_dtype:
            self._rgb_buffer = np.empty(rgb_shape, dtype=self._rgb_dtype)

        if self._depth_buffer is None or self._depth_buffer.shape != depth_shape:
            self._depth_buffer = np.empty(depth_shape, dtype=self._depth_dtype)
        elif self._depth_buffer.dtype != self._depth_dtype:
            self._depth_buffer = np.empty(depth_shape, dtype=self._depth_dtype)

        if self._depth_dtype == np.dtype(np.float16):
            if (
                self._depth_resize_buffer is None
                or self._depth_resize_buffer.shape != depth_shape
            ):
                self._depth_resize_buffer = np.empty(depth_shape, dtype=np.float32)
        else:
            self._depth_resize_buffer = None

        for index, rgb in enumerate(rgb_images):
            if rgb.shape != (h_rgb, w_rgb, 3):
                raise VisionPipelineError(
                    "All RGB images in a pipeline batch must have the same shape; "
                    f"image 0 is {(h_rgb, w_rgb, 3)}, image {index} is {rgb.shape}"
                )
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
        depth_h, depth_w = self.depth_size

        for index, rgb in enumerate(rgb_images):
            self._rgb_buffer[index, 0] = rgb[:, :, 0]
            self._rgb_buffer[index, 1] = rgb[:, :, 1]
            self._rgb_buffer[index, 2] = rgb[:, :, 2]

        for index, depth in enumerate(depth_images):
            depth_dst = self._depth_buffer[index, 0]
            resize_dst = (
                self._depth_resize_buffer[index, 0]
                if self._depth_resize_buffer is not None
                else depth_dst
            )
            cv2.resize(
                depth,
                (depth_w, depth_h),
                dst=resize_dst,
                interpolation=cv2.INTER_NEAREST,
            )
            if resize_dst is not depth_dst:
                depth_dst[...] = resize_dst

def _complete_sparse_using_nearest(sparse_depth: np.ndarray):
    
    batch_size = sparse_depth.shape[0]
    completed = np.empty_like(sparse_depth)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # iterating over other dimension
    for i in range(batch_size):

        frame = sparse_depth[i]
        original_shape = frame.shape
        depth_frame = frame.squeeze().astype(np.float32)
        mask = (depth_frame < 0.01)

        if mask.any():

            # applying kernel w/ dilation then replacing original vals w/ dilated vals
            dilated_version = cv2.dilate(depth_frame, dilation_kernel, iterations = 3)
            depth_frame[mask] = dilated_version[mask]  
            completed[i] = depth_frame.reshape(original_shape)

        else:
            completed[i] = frame

    return completed

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
            "Vision model path is required. Pass model_path or set "
            f"{DEFAULT_MODEL_ENV_VAR}."
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
