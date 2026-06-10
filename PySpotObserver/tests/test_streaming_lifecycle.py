import threading
import time

import numpy as np
import pytest
import cv2
from bosdyn.api import image_pb2

from pyspotobserver.camera_stream import ImageFrame, SpotCamStream, SpotCamStreamError
from pyspotobserver.config import CameraType, SpotConfig
from pyspotobserver.connection import SpotConnection, SpotConnectionError


class DummyImageClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def get_image(self, image_requests, timeout):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_stream(image_client=None) -> SpotCamStream:
    return SpotCamStream(
        image_client=image_client or DummyImageClient([]),
        config=SpotConfig(robot_ip="192.168.80.3", request_timeout_seconds=0.05),
        stream_id="test_stream",
    )


def test_get_current_images_returns_latest_frame():
    stream = make_stream()
    stream._streaming = True

    older = ImageFrame(
        rgb_images=[np.full((1, 1, 3), 1.0, dtype=np.float32)],
        depth_images=[np.full((1, 1), 1.0, dtype=np.float32)],
        camera_order=[CameraType.FRONTLEFT],
        timestamp=1.0,
    )
    newer = ImageFrame(
        rgb_images=[np.full((1, 1, 3), 2.0, dtype=np.float32)],
        depth_images=[np.full((1, 1), 2.0, dtype=np.float32)],
        camera_order=[CameraType.FRONTLEFT],
        timestamp=2.0,
    )

    stream._image_queue.put(older)
    stream._image_queue.put(newer)

    rgb_images, depth_images = stream.get_current_images(timeout=0.01)

    assert float(rgb_images[0][0, 0, 0]) == 2.0
    assert float(depth_images[0][0, 0]) == 2.0
    assert stream._image_queue.qsize() == 2


def test_get_current_images_run_pipeline_requires_model_path():
    stream = make_stream()
    stream._streaming = True
    frame = ImageFrame(
        rgb_images=[np.full((1, 1, 3), 1.0, dtype=np.float32)],
        depth_images=[np.full((1, 1), 1.0, dtype=np.float32)],
        camera_order=[CameraType.FRONTLEFT],
        timestamp=1.0,
    )
    stream._image_queue.put(frame)

    with pytest.raises(SpotCamStreamError, match="Vision model path is required"):
        stream.get_current_images(timeout=0.01, run_pipeline=True)


def test_get_current_images_run_pipeline_uses_lazy_pipeline(monkeypatch):
    import pyspotobserver.vision_pipeline as vision_pipeline

    class FakeVisionPipeline:
        @classmethod
        def from_config(cls, config):
            return cls()

        def run(self, rgb_images, depth_images):
            return rgb_images, [depth + 10.0 for depth in depth_images]

    monkeypatch.setattr(vision_pipeline, "VisionPipeline", FakeVisionPipeline)

    stream = make_stream()
    stream._streaming = True
    frame = ImageFrame(
        rgb_images=[np.full((1, 1, 3), 1.0, dtype=np.float32)],
        depth_images=[np.full((1, 1), 2.0, dtype=np.float32)],
        camera_order=[CameraType.FRONTLEFT],
        timestamp=1.0,
    )
    stream._image_queue.put(frame)

    rgb_images, depth_images = stream.get_current_images(
        timeout=0.01,
        run_pipeline=True,
    )

    assert float(rgb_images[0][0, 0, 0]) == 1.0
    assert float(depth_images[0][0, 0]) == 12.0


def test_get_current_images_unblocks_when_stream_stops():
    stream = make_stream()
    stream._streaming = True

    result = {}

    def reader():
        try:
            stream.get_current_images(timeout=None)
        except Exception as exc:  # pragma: no cover - exercised by assertion
            result["exc"] = exc

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.15)

    stream._streaming = False
    stream._stop_event.set()

    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(result["exc"], SpotCamStreamError)
    assert "stopped" in str(result["exc"]).lower()


def test_stream_loop_recovers_after_initial_request_error():
    stream = make_stream(DummyImageClient([RuntimeError("boom"), ["ok"]]))
    stream._streaming = True
    stream._camera_order = [CameraType.FRONTLEFT]

    decoded = [
        np.zeros((1, 1, 3), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    ]

    stream._build_image_requests = lambda: []
    stream._decode_initial_responses = lambda responses: decoded

    original_enqueue = stream._enqueue_frame

    def enqueue_and_stop(frame):
        original_enqueue(frame)
        stream._streaming = False
        stream._stop_event.set()

    stream._enqueue_frame = enqueue_and_stop

    stream._stream_loop()

    assert stream.frame_count == 1
    assert stream.error_count == 1
    assert len(stream._frame_pool) == stream._config.image_buffer_size


def test_jpeg_inplace_color_correction_updates_output_array():
    stream = make_stream()
    bgr = np.full((2, 2, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    response = image_pb2.ImageResponse()
    response.shot.image.format = image_pb2.Image.FORMAT_JPEG
    response.shot.image.data = encoded.tobytes()

    out = np.empty((2, 2, 3), dtype=np.float32)
    zero_ccm = np.zeros((3, 3), dtype=np.float32)

    stream._convert_image_response_inplace(
        response,
        is_depth=False,
        out_array=out,
        ccm=zero_ccm,
    )

    assert np.allclose(out, 0.0)


def test_disconnect_raises_if_stream_does_not_stop():
    class BrokenStream:
        def stop_streaming(self):
            raise SpotCamStreamError("still alive")

    conn = SpotConnection(SpotConfig(robot_ip="192.168.80.3"))
    conn._connected = True
    conn._cam_streams = {"broken": BrokenStream()}

    with pytest.raises(SpotConnectionError, match="Failed to stop all streams cleanly"):
        conn.disconnect()

    assert conn.connected is True
