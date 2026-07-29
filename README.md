# SpotObserver

SpotObserver is a C++/CUDA library for streaming and processing camera data from
[Boston Dynamics Spot](https://bostondynamics.com/products/spot/) robots. It is built
as a native Unity plugin (`SpotObserverLib.dll`) and exposes a C API for:

- Connecting to one or more Spot robots and streaming RGB + depth feeds from any of the
  onboard cameras (back, front-left, front-right, left, right, hand)
- A GPU vision pipeline (for depth completion at the moment) powered by LibTorch and ONNX Runtime
- Low-overhead handoff of images into Unity via CUDA/DX12 interop.
- Per-frame body-to-world transforms alongside the images

The full C API is declared in [`include/spot-observer.h`](include/spot-observer.h).

## Python Implementation (PySpotObserver)

A Pythonic implementation of the same functionality lives in
[`PySpotObserver/`](PySpotObserver/) — see its [README](PySpotObserver/README.md) and
[QUICKSTART](PySpotObserver/QUICKSTART.md). It provides a clean, type-hinted API with
sync and async usage, context-managed connections, YAML configuration, and an optional
ONNX vision pipeline:

```python
from pyspotobserver import SpotConfig, SpotConnection, CameraType

with SpotConnection(SpotConfig(robot_ip="192.168.80.3", username="...", password="...")) as conn:
    stream = conn.create_cam_stream()
    stream.start_streaming(CameraType.FRONTLEFT | CameraType.FRONTRIGHT)
    rgb_images, depth_images, body_T_worlds = stream.get_current_images()
```

## Repository layout

| Path | Contents |
| --- | --- |
| `include/` | Public C API header (`spot-observer.h`) |
| `src/` | Library implementation (connection, vision pipeline, CUDA kernels, Unity/DX12 interop) |
| `PySpotObserver/` | Python bindings and examples |
| `extern/` | Third-party dependencies (Spot C++ SDK, OpenCV, LibTorch, ONNX Runtime) |
| `tests/` | Integration tests (`integ-test`, `integ-test-dx12`) and standalone tests |
| `scripts/` | Utility scripts (e.g. timing statistics) |

## Building SpotObserver

### Prerequisites

- Windows 10/11 x64 with Visual Studio 2022 (MSVC v143 toolset)
- CMake **3.28+**
- NVIDIA CUDA Toolkit (currently only tested on Ampere and Ada GPUs, compute 8.6 / 8.9)
- A Unity Editor installation (the CUDA/DX12 interop module uses Unity's Native Plugin API headers;
  the build auto-detects the newest Unity Hub install — override with
  `-DUnityPluginAPI_INCLUDE=<Unity>/Editor/Data/PluginAPI` if needed)

### 1. Set up dependencies

All third-party dependencies live under `extern/`. Follow
[`extern/README.md`](extern/README.md) for the detailed steps.

### 2. Configure and build

Use `CMAKE_PREFIX_PATH` (pointing at the vcpkg installed tree), **not**
`CMAKE_TOOLCHAIN_FILE`:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="<vcpkg-abs-path>/installed/x64-windows" \
  -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=TRUE
cmake --build build --config Release
```

This produces `SpotObserverLib.dll` and automatically copies all required runtime DLLs
(vcpkg, OpenCV, LibTorch, ONNX Runtime) next to the built binaries. Set
`-DCOPY_DLLS=OFF` to disable the automatic copying.

Useful options:

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTS` | `ON` | Build the test executables under `tests/` |
| `INSTALL_TESTS` | `ON` | Install test executables |
| `COPY_DLLS` | `ON` | Copy required DLLs next to the built library |

### 3. Install (optional)

```bash
cmake --install build --config Release
```

By default this installs the library, public headers, and runtime DLLs into
`<repo>/install`. Override with `-DCMAKE_INSTALL_PREFIX=<path>` at configure time.

## Tests

With `BUILD_TESTS=ON` (the default), the build also produces:

- `tests/integ-test` — end-to-end streaming test against a real robot
- `tests/integ-test-dx12` — DirectX 12 interop / readback test
- `tests/standalone-tests` — smaller isolated tests

## Notes

- Robot credentials are passed to `SOb_ConnectToSpot()` at runtime; nothing is
  hardcoded in the library.
- The Boston Dynamics C++ SDK is still in beta