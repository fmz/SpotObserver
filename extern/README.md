# Extern directory instructions

## OpenCV
Download and extract OpenCV (grab only the 'build' directory)
https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path

## Spot SDK
Follow the instructions here:
https://github.com/boston-dynamics/spot-cpp-sdk/blob/master/docs/cpp/quickstart.md

Important!!! 

1. Install vcpkg in a directory adjacent to SpotObserver (next to it).
2. Do NOT git checkout the version number that is recommended in the spot-cpp-sdk quickstart, and instead add a file `vcpkg-configuration.json` inside the vcpkg directory with the following contents:
```{
  "default-registry": {
    "kind": "git",
    "repository": "https://github.com/Microsoft/vcpkg",
    "baseline": "3b213864579b6fa686e38715508f7cd41a50900f"
  }
}
```
3. To specify the dependency versions, add another file in `vcpkg`, name it `vcpkg.json`, and put the following in it:
```
{
  "dependencies": [
    {
      "name": "grpc",
      "version>=": "1.51.1"
    },
    {
      "name": "protobuf",
      "version>=": "3.21.12"
    },
    {
      "name": "eigen3",
      "version>=": "3.4.0#2"
    },
    {
      "name": "cli11",
      "version>=": "2.3.1"
    }
  ]
}
```

When running cmake on the SDK, use 
> CMAKE_PREFIX_PATH instead of CMAKE_TOOLCHAIN_FILE:
cmake -B build -DCMAKE_PREFIX_PATH="\<vcpkg-abs-path\>/installed/x64-windows" -DCMAKE_INSTALL_PREFIX="\<SpotObserver-abs-path\>/extern/spot-sdk-install" -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=TRUE cpp

Also, apply the patch provided to spot-cpp-sdk:
> cd extern/spot-cpp-sdk
> git apply ../spot-cpp-sdk.patch

## LibTorch
Download and extract libtorch for windows:
https://pytorch.org/get-started/locally/

## ONNX
Download and extract the latest ONNX Runtime GPU edition:
https://github.com/microsoft/onnxruntime/releases -> onnxruntime-win-x64-gpu-1.<x>.<y>.zip
(make sure include and lib directories are under the top-level extracted onnx dir)
