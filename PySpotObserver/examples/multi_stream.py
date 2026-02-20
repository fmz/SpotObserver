"""
Advanced example showing multiple concurrent camera streams.

This demonstrates managing multiple streams with different camera configurations.
"""

import logging
import time
import cv2
import numpy as np

from pyspotobserver import SpotConfig, SpotConnection, CameraType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    config = SpotConfig(
        robot_ip="128.148.138.22",
        username="user",
        password="bigbubbabigbubba",
        image_buffer_size=3,
    )

    with SpotConnection(config) as conn:
        logger.info(f"Connected to robot: {conn}")

        # Create two separate streams with different camera configurations
        front_stream = conn.create_cam_stream(stream_id="front_cameras")
        side_stream = conn.create_cam_stream(stream_id="side_cameras")

        # Start front cameras stream
        front_stream.start_streaming(CameraType.FRONTLEFT | CameraType.FRONTRIGHT)
        logger.info(f"Front stream cameras: {front_stream.get_camera_order()}")

        # Start side cameras stream
        side_stream.start_streaming(CameraType.LEFT | CameraType.RIGHT)
        logger.info(f"Side stream cameras: {side_stream.get_camera_order()}")

        try:
            # Process both streams for 30 seconds
            start_time = time.time()

            while time.time() - start_time < 30.0:
                # Get images from both streams
                try:
                    front_rgb, front_depth = front_stream.get_current_images(timeout=1.0)
                    logger.debug(f"Front stream: {len(front_rgb)} images")
                except Exception as e:
                    logger.warning(f"Front stream error: {e}")
                    continue

                try:
                    side_rgb, side_depth = side_stream.get_current_images(timeout=1.0)
                    logger.debug(f"Side stream: {len(side_rgb)} images")
                except Exception as e:
                    logger.warning(f"Side stream error: {e}")
                    continue

                # Display front camera images
                for i, rgb in enumerate(front_rgb):
                    camera_name = front_stream.get_camera_order()[i].name
                    rgb_display = (rgb * 255).astype(np.uint8)
                    cv2.imshow(f"Front - {camera_name}", rgb_display)

                # Display side camera images
                for i, rgb in enumerate(side_rgb):
                    camera_name = side_stream.get_camera_order()[i].name
                    rgb_display = (rgb * 255).astype(np.uint8)
                    cv2.imshow(f"Side - {camera_name}", rgb_display)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break

        finally:
            # Clean up both streams
            front_stream.stop_streaming()
            side_stream.stop_streaming()
            cv2.destroyAllWindows()

            logger.info(
                f"Front stream stats: frames={front_stream.frame_count}, "
                f"errors={front_stream.error_count}"
            )
            logger.info(
                f"Side stream stats: frames={side_stream.frame_count}, "
                f"errors={side_stream.error_count}"
            )

        # List all streams
        logger.info(f"Active streams: {conn.list_streams()}")

    logger.info("Disconnected from robot")


if __name__ == "__main__":
    main()
