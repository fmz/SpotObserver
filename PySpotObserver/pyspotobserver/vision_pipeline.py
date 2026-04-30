from typing import List, Tuple
import numpy as np
import threading
import onnxruntime as ort
import cv2

# -------------------------
# Global singleton state
# -------------------------
_session = None
_input_names = None
_lock = threading.Lock()

# Preallocated buffers (resized only if needed)
_rgb_buffer = None
_depth_buffer = None

def _init_session():
    global _session, _input_names

    if _session is None:
        ort.preload_dlls()

        _session = ort.InferenceSession(
            "C:/Users/jtoribio/Documents/aanya/fileexchange/promptda-vitl-rotated_optimized_batch2_fp16.onnx",
            providers=["CUDAExecutionProvider"],
        )

        _input_names = [i.name for i in _session.get_inputs()]


def _ensure_buffers(batch_size, rgb_shape, depth_shape):
    """
    Allocate buffers only if size/shape changed.
    """
    global _rgb_buffer, _depth_buffer

    # Expected shapes:
    # rgb: (B, 3, H, W)
    # depth: (B, 1, H, W)

    if (
        _rgb_buffer is None
        or _rgb_buffer.shape != (batch_size, 3, rgb_shape[0], rgb_shape[1])
    ):
        _rgb_buffer = np.empty(
            (batch_size, 3, rgb_shape[0], rgb_shape[1]),
            dtype=np.float32,
        )

    if (
        _depth_buffer is None
        or _depth_buffer.shape != (batch_size, 1, depth_shape[0], depth_shape[1])
    ):
        _depth_buffer = np.empty(
            (batch_size, 1, depth_shape[0], depth_shape[1]),
            dtype=np.float32,
        )


def run_vision_pipeline(
    rgb_images: List[np.ndarray],
    depth_images: List[np.ndarray],
):
    """
    High-performance pipeline:
    - No per-call allocations (after warmup)
    - No np.stack
    - Thread-safe ONNX execution
    """

    _init_session()

    batch_size = len(rgb_images)

    # Assume all images same shape
    h_rgb, w_rgb, _ = rgb_images[0].shape
    h_d, w_d = depth_images[0].shape[:2]

    MODEL_DEPTH_SHAPE = (120, 160)

    _ensure_buffers(batch_size, (h_rgb, w_rgb), MODEL_DEPTH_SHAPE)

    # -------------------------
    # Fill buffers (no stack)
    # -------------------------
    for i in range(batch_size):
        # RGB: HWC -> CHW
        _rgb_buffer[i, 0] = rgb_images[i][:, :, 0]
        _rgb_buffer[i, 1] = rgb_images[i][:, :, 1]
        _rgb_buffer[i, 2] = rgb_images[i][:, :, 2]

        # Depth: HW -> 1HW
        resized = cv2.resize(
            depth_images[i],
            (160, 120),  # (width, height) for OpenCV
            interpolation=cv2.INTER_NEAREST,  # important for depth!
        )

        _depth_buffer[i, 0] = resized

    # -------------------------
    # Inference (thread-safe)
    # -------------------------
    with _lock:
        output = _session.run(
            None,
            {
                _input_names[0]: _rgb_buffer,
                _input_names[1]: _depth_buffer,
            },
        )[0]

    # # -------------------------
    # # Fast normalization (in-place)
    # # -------------------------
    # out_min = output.min()
    # out_max = output.max()
    # scale = 1.0 / (out_max - out_min + 1e-8)
    # output = (output - out_min) * scale

    return rgb_images, output