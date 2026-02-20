# PySpotObserver Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   cd PySpotObserver
   pip install -r requirements.txt
   ```

2. **Install package in development mode:**
   ```bash
   pip install -e .
   ```

## Your First Stream

### Step 1: Create a configuration

**Option A: Direct instantiation**
```python
from pyspotobserver import SpotConfig

config = SpotConfig(
    robot_ip="192.168.80.3",  # Replace with your robot's IP
    username="user",
    password="bigbubbabigbubba"
)
```

**Option B: Load from YAML**
```python
config = SpotConfig.from_yaml("config.yaml")
```

### Step 2: Connect to robot

```python
from pyspotobserver import SpotConnection

with SpotConnection(config) as conn:
    print(f"Connected: {conn}")
    # Your code here...
```

### Step 3: Create and start a camera stream

```python
from pyspotobserver import CameraType

# Inside the connection context:
stream = conn.create_cam_stream()

# Choose cameras (bitwise OR to combine)
cameras = CameraType.FRONTLEFT | CameraType.FRONTRIGHT

stream.start_streaming(cameras)
```

### Step 4: Get images

```python
# Get current frame
rgb_images, depth_images = stream.get_current_images(timeout=2.0)

# Process images
for i, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
    camera = stream.get_camera_order()[i]
    print(f"{camera.name}: RGB shape={rgb.shape}, Depth shape={depth.shape}")
```

### Step 5: Display with OpenCV (optional)

```python
import cv2
import numpy as np

# Convert to uint8 for display
rgb_display = (rgb_images[0] * 255).astype(np.uint8)
cv2.imshow("Front Left", rgb_display)

# Normalize depth for visualization
depth_norm = cv2.normalize(depth_images[0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
cv2.imshow("Depth", depth_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Step 6: Clean up

```python
stream.stop_streaming()
# Connection automatically closes when exiting 'with' block
```

## Complete Example

```python
from pyspotobserver import SpotConfig, SpotConnection, CameraType
import cv2

config = SpotConfig(robot_ip="192.168.80.3")

with SpotConnection(config) as conn:
    stream = conn.create_cam_stream()
    stream.start_streaming(CameraType.FRONTLEFT)

    try:
        while True:
            rgb, depth = stream.get_current_images(timeout=1.0)

            # Display
            rgb_display = (rgb[0] * 255).astype(np.uint8)
            cv2.imshow("Front Left", rgb_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop_streaming()
        cv2.destroyAllWindows()
```

## Async Example

```python
import asyncio
from pyspotobserver import SpotConfig, SpotConnection, CameraType

async def main():
    config = SpotConfig(robot_ip="192.168.80.3")

    async with SpotConnection(config) as conn:
        stream = conn.create_cam_stream()
        stream.start_streaming(CameraType.FRONTLEFT)

        # Async image retrieval
        rgb, depth = await stream.async_get_current_images()
        print(f"Got {len(rgb)} images")

        stream.stop_streaming()

asyncio.run(main())
```

## Available Cameras

```python
from pyspotobserver import CameraType

# Individual cameras
CameraType.BACK
CameraType.FRONTLEFT
CameraType.FRONTRIGHT
CameraType.LEFT
CameraType.RIGHT
CameraType.HAND

# Combine with bitwise OR
front_cameras = CameraType.FRONTLEFT | CameraType.FRONTRIGHT
all_cameras = (CameraType.BACK | CameraType.FRONTLEFT |
               CameraType.FRONTRIGHT | CameraType.LEFT | CameraType.RIGHT)
```

## Image Format

- **RGB Images**: NumPy arrays with shape `(height, width, 3)`, dtype `float32`, range [0, 1]
- **Depth Images**: NumPy arrays with shape `(height, width)`, dtype `float32`, values in meters

To convert RGB for OpenCV display:
```python
rgb_uint8 = (rgb * 255).astype(np.uint8)
```

## Error Handling

```python
from pyspotobserver import SpotConnectionError, SpotAuthenticationError

try:
    with SpotConnection(config) as conn:
        # ...
except SpotAuthenticationError as e:
    print(f"Failed to authenticate: {e}")
except SpotConnectionError as e:
    print(f"Connection error: {e}")
```

## Logging

Enable debug logging to see detailed information:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Next Steps

- Check out `examples/` directory for more complete examples
- Read `README.md` for full API documentation
- See `examples/config_example.yaml` for configuration options
- Run tests: `pytest tests/`

## Common Issues

**Connection timeout**: Ensure robot IP is correct and robot is powered on and connected to network

**Authentication failed**: Check username and password in configuration

**Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**No images received**: Check that cameras are not already in use by another client
