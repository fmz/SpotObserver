"""
Async example of streaming camera images from Spot robot.

This demonstrates asynchronous usage with async/await patterns.
"""

import asyncio
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


async def process_images(stream, duration_seconds: float):
    """
    Async function to process images from stream.

    Args:
        stream: SpotCamStream instance
        duration_seconds: How long to stream for
    """
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration_seconds:
        try:
            # Async get images (non-blocking)
            rgb_images, depth_images = await stream.async_get_current_images(timeout=2.0)

            frame_count += 1
            logger.info(
                f"Frame {frame_count}: "
                f"RGB shapes: {[img.shape for img in rgb_images]}, "
                f"Depth shapes: {[img.shape for img in depth_images]}"
            )

            # Display images (OpenCV operations are blocking, but fast)
            for i, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
                camera_name = stream.get_camera_order()[i].name

                # Convert RGB from float [0,1] to uint8 [0,255]
                rgb_display = (rgb * 255).astype(np.uint8)

                # Normalize depth for visualization
                depth_normalized = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # Show images
                cv2.imshow(f"{camera_name} - RGB", rgb_display)
                cv2.imshow(f"{camera_name} - Depth", depth_colored)

            # Check for quit (non-blocking waitKey)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break

            # Small async sleep to yield control
            await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    return frame_count


async def main():
    # Create config
    config = SpotConfig(
        robot_ip="192.168.80.3",
        username="user",
        password="bigbubbabigbubba",
        image_buffer_size=5,
    )

    # Connect to robot using async context manager
    async with SpotConnection(config) as conn:
        logger.info(f"Connected to robot: {conn}")

        # Create a camera stream
        stream = conn.create_cam_stream(stream_id="async_stream")

        # Start streaming from all front cameras
        camera_mask = CameraType.FRONTLEFT | CameraType.FRONTRIGHT
        stream.start_streaming(camera_mask)

        logger.info(f"Streaming from cameras: {stream.get_camera_order()}")

        try:
            # Process images for 30 seconds
            frame_count = await process_images(stream, duration_seconds=30.0)

            logger.info(f"Processed {frame_count} frames")

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
    # Run async main
    asyncio.run(main())
