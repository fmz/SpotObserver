"""
Basic example of streaming camera images from Spot robot.

This demonstrates synchronous usage with context managers.
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
    # Option 1: Create config from parameters
    config = SpotConfig(
        robot_ip="128.148.138.22",
        username="user",
        password="bigbubbabigbubba",
        image_buffer_size=5,
    )

    # Option 2: Load from YAML file (uncomment to use)
    # config = SpotConfig.from_yaml("examples/config_example.yaml")

    # Connect to robot using context manager
    with SpotConnection(config) as conn:
        logger.info(f"Connected to robot: {conn}")

        # Create a camera stream
        stream = conn.create_cam_stream(stream_id="example_stream")

        # Start streaming from front cameras
        camera_mask = CameraType.FRONTLEFT | CameraType.FRONTRIGHT
        stream.start_streaming(camera_mask)

        logger.info(f"Streaming from cameras: {stream.get_camera_order()}")

        try:
            # Stream for 30 seconds
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < 30.0:
                # Get current images (blocks until available)
                rgb_images, depth_images = stream.get_current_images(timeout=2.0)

                frame_count += 1
                logger.info(
                    f"Frame {frame_count}: "
                    f"RGB shapes: {[img.shape for img in rgb_images]}, "
                    f"Depth shapes: {[img.shape for img in depth_images]}"
                )

                # Display images using OpenCV
                for i, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
                    camera_name = stream.get_camera_order()[i].name

                    # Convert RGB from float [0,1] to uint8 [0,255] for display
                    rgb_display = (rgb * 255).astype(np.uint8)

                    # Normalize depth for visualization
                    depth_normalized = cv2.normalize(
                        depth, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    # Show images
                    cv2.imshow(f"{camera_name} - RGB", rgb_display)
                    cv2.imshow(f"{camera_name} - Depth", depth_colored)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break

        finally:
            # Clean up
            stream.stop_streaming()
            cv2.destroyAllWindows()
            logger.info(
                f"Stream statistics: "
                f"frames={stream.frame_count}, errors={stream.error_count}"
            )

    logger.info("Disconnected from robot")


if __name__ == "__main__":
    main()
