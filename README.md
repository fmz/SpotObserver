# SpotObserver

A C++ application for observing and interacting with Boston Dynamics Spot robot.

## Project Structure

- `src/main.cpp` - Full implementation using Boston Dynamics SDK
- `src/main_simple.cpp` - Simplified version without SDK dependencies for testing compilation
- `extern/spot-cpp-sdk/` - Boston Dynamics Spot C++ SDK submodule

## Building

This requires installing dependencies and builds the full version that can actually communicate with Spot:

#### Prerequisites

1. **Install vcpkg dependencies** (as described in `extern/spot-cpp-sdk/docs/cpp/quickstart.md`):
   ```bash
   git clone https://github.com/microsoft/vcpkg
   cd vcpkg
   git checkout 3b213864579b6fa686e38715508f7cd41a50900f
   
   # On Windows:
   .\bootstrap-vcpkg.bat
   .\vcpkg.exe install grpc:x64-windows
   .\vcpkg.exe install eigen3:x64-windows
   .\vcpkg.exe install cli11:x64-windows
   
2. **Build with SDK**:
   ```bash
   mkdir build
   cd build
   cmake .. -DBUILD_WITH_SPOT_SDK=ON -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
   cmake --build .
   ```

## Usage

### Simple Version
```bash
./SpotObserver username 192.168.80.3 password
```

### Full Version (with SDK)
```bash
./SpotObserver username <spot_ip_address> password
```

## Features

- **SpotConfig**: Configuration structure for robot connection parameters
- **LeaseGuard**: RAII wrapper for automatic lease management
- **Authentication**: Handles robot authentication using username/password
- **Robot State**: Retrieves and displays robot status information
- **Lease Management**: Automatically acquires and releases robot control lease

## Code Structure

The main application follows this flow:
1. Parse command line arguments (username, IP, password)
2. Create gRPC channel to robot
3. Authenticate with the robot
4. Acquire a lease for robot control
5. Use RAII LeaseGuard to ensure lease is properly released
6. Query robot state and display information
7. Clean up and exit

The `LeaseGuard` class ensures that robot leases are properly released even if the program exits unexpectedly, following RAII principles.

## Requirements

- C++17 or later
- CMake 3.10+
- For full build: gRPC, Protobuf, Eigen3, CLI11 (via vcpkg)

## Boston Dynamics SDK

This project uses the Boston Dynamics Spot C++ SDK v5.0.0. The SDK is included as a git submodule in the `extern/spot-cpp-sdk` directory.

For detailed information about the SDK, see:
- [C++ SDK Documentation](extern/spot-cpp-sdk/docs/cpp/README.md)
- [QuickStart Guide](extern/spot-cpp-sdk/docs/cpp/quickstart.md)
