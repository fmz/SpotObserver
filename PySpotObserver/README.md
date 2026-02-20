# PySpotObserver

A clean, Pythonic interface for streaming camera data from Boston Dynamics Spot robots.

## Features

- **Clean API**: Simple, intuitive interface following Python best practices
- **Type-safe**: Full type hints for better IDE support and type checking
- **Async Support**: Both synchronous and async/await patterns supported
- **Context Managers**: Automatic resource cleanup with `with` statements
- **YAML Configuration**: Load settings from config files or pass as parameters
- **Multi-stream**: Support for multiple concurrent camera streams
- **Thread-safe**: Background streaming with thread-safe image buffering

## Installation

### From source

```bash
cd PySpotObserver
pip install -e .
```

### With development tools

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Synchronous Usage

```python
from pyspotobserver import SpotConfig, SpotConnection, CameraType

# Create configuration
config = SpotConfig(
    robot_ip="192.168.80.3",
    username="user",
    password="bigbubbabigbubba"
)

# Connect and stream
with SpotConnection(config) as conn:
    stream = conn.create_cam_stream()

    # Start streaming from front cameras
    stream.start_streaming(CameraType.FRONTLEFT | CameraType.FRONTRIGHT)

    # Get images
    rgb_images, depth_images = stream.get_current_images()

    # Process images...

    stream.stop_streaming()
```

### Async Usage

```python
import asyncio
from pyspotobserver import SpotConfig, SpotConnection, CameraType

async def main():
    config = SpotConfig(robot_ip="192.168.80.3")

    async with SpotConnection(config) as conn:
        stream = conn.create_cam_stream()
        stream.start_streaming(CameraType.FRONTLEFT)

        # Async image retrieval
        rgb_images, depth_images = await stream.async_get_current_images()

        stream.stop_streaming()

asyncio.run(main())
```

### Using YAML Configuration

```python
from pyspotobserver import SpotConfig, SpotConnection

# Load from YAML file
config = SpotConfig.from_yaml("config.yaml")

with SpotConnection(config) as conn:
    # ...
```

Example `config.yaml`:

```yaml
robot_ip: "192.168.80.3"
username: "user"
password: "bigbubbabigbubba"
image_buffer_size: 5
image_quality_percent: 100.0
```

## Architecture

### SpotConnection

Manages the robot connection lifecycle and authentication:

- **connect()** / **async_connect()**: Establish connection
- **disconnect()** / **async_disconnect()**: Clean shutdown
- **create_cam_stream()**: Create a new camera stream
- **remove_cam_stream()**: Remove an existing stream

### SpotCamStream

Handles camera streaming in a background thread:

- **start_streaming(camera_mask)**: Begin streaming from specified cameras
- **stop_streaming()**: Stop the stream
- **get_current_images()** / **async_get_current_images()**: Retrieve latest frame
- **get_camera_order()**: Get list of cameras being streamed

### CameraType

Enum for specifying cameras (use bitwise OR to combine):

- `CameraType.BACK`
- `CameraType.FRONTLEFT`
- `CameraType.FRONTRIGHT`
- `CameraType.LEFT`
- `CameraType.RIGHT`
- `CameraType.HAND`

### Image Format

Images are returned as NumPy arrays:

- **RGB**: `(H, W, 3)` float32 in range [0, 1]
- **Depth**: `(H, W)` float32 in meters

## Examples

See the `examples/` directory for complete working examples:

- **basic_streaming.py**: Simple synchronous streaming with OpenCV display
- **async_streaming.py**: Async/await pattern demonstration
- **multi_stream.py**: Multiple concurrent streams with different cameras
- **config_example.yaml**: Example configuration file
- **benchmark_allocation.py**: Allocation vs in-place conversion benchmark

## Design Principles

This implementation follows modern Python best practices:

1. **Type Safety**: Full type hints using Python 3.9+ syntax
2. **Dataclasses**: Configuration uses `@dataclass` for clean structure
3. **Context Managers**: Automatic cleanup with `with` statements
4. **Async Support**: Async variants for non-blocking operations
5. **CPU-Only**: NumPy arrays for simplicity and portability (no CUDA)
6. **FIFO Buffering**: Simple Queue-based buffering (vs. LIFO circular buffer in C++)
7. **Logging**: Structured logging throughout for debugging
8. **Error Handling**: Custom exception types for different failure modes

## Differences from C++ Version

The Python implementation differs in these ways:

- **No CUDA**: Uses CPU-only NumPy arrays instead of GPU memory
- **FIFO Queue**: Simpler queue.Queue instead of custom circular buffer
- **Threading**: Python threading instead of C++ jthread
- **No ML Pipeline**: Focuses on camera streaming only (no inference)
- **Simplified**: Removes Unity plugin and DLL export complexity

## Requirements

- Python 3.9+
- Boston Dynamics Spot SDK (`bosdyn-client`)
- NumPy
- OpenCV (opencv-python)
- PyYAML

## Contributing

When contributing, please follow these guidelines:

1. Use `black` for code formatting
2. Use `mypy` for type checking
3. Add tests for new features
4. Update documentation

## License

See LICENSE file for details.

## Related Projects

- [Boston Dynamics Spot SDK](https://github.com/boston-dynamics/spot-sdk)
- Original C++ SpotObserver (parent directory)
