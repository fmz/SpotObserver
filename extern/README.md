# Extern directory instructions

## OpenCV
Download and extract OpenCV (grab only the 'build' directory)
https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path

## Spot SDK
Follow the instructions here:
https://github.com/boston-dynamics/spot-cpp-sdk/blob/master/docs/cpp/quickstart.md
Important!!! When running cmake on the SDK, use CMAKE_PREFIX_PATH instead of CMAKE_TOOLCHAIN_FILE:
cmake -B build -DCMAKE_PREFIX_PATH="C:/Users/brown/Documents/fmz/vcpkg/installed/x64-windows" -DCMAKE_INSTALL_PREFIX="C:/Users/brown/Documents/fmz/SpotObserver/extern/spot-sdk-install" -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=TRUE cpp

Also, change this in protos/bosdyn/api/spot/choreography_sequence.proto:
STATUS_INTERRUPTED -> STATUS_SPOT_INTERRUPTED

## LibTorch
Download and extract libtorch for windows:
https://pytorch.org/get-started/locally/

## ONNX
Download and extract the latest ONNX Runtime GPU edition:
https://github.com/microsoft/onnxruntime/releases -> onnxruntime-win-x64-gpu-1.<x>.<y>.zip
(make sure include and lib directories are under the top-level extracted onnx dir)
