cmake_minimum_required(VERSION 3.28)
project(SpotCameraStream VERSION 0.1.0 LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build the test executable" ON)
option(INSTALL_TESTS "Install test executables" ON)
set(PLUGIN_NAME "SpotObserverLib")

# All runtime artifacts (plugin DLL, test executables, copied dependency DLLs)
# land in the top-level build directory, so dependency DLLs are copied exactly
# once and every executable runs from the same place. Multi-config generators
# (Visual Studio) append a per-config subdir, e.g. build/Release.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set default install directory to SpotObserver/install if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Default install directory" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(Protobuf_ROOT "${CMAKE_PREFIX_PATH}")
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/libtorch;${CMAKE_PREFIX_PATH})
message(STATUS "Using CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Find required packages
find_package(CUDAToolkit REQUIRED)
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDAToolkit not found")
endif()
message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")

set(SPOT_SDK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/extern/spot-sdk-install)
set(OpenCV_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/extern/opencv/include)
set(OpenCV_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/opencv/x64/vc16/lib)
set(OpenCV_LIBS opencv_world4110.lib)

# Force protobuf to come from vcpkg, not libtorch
find_package(Protobuf REQUIRED PATHS "${Protobuf_ROOT}" NO_DEFAULT_PATH)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

# Support Ampere and Ada architectures - set BEFORE finding Torch.
# NOTE: TorchConfig (Caffe2/public/cuda.cmake) force-sets CMAKE_CUDA_ARCHITECTURES
# to OFF and instead injects -gencode flags derived from TORCH_CUDA_ARCH_LIST into
# CMAKE_CUDA_FLAGS for every .cu in the project. TORCH_CUDA_ARCH_LIST is therefore
# what actually selects the built architectures, so derive it from our arch list
# ("Common" here used to make every .cu build for ~10 archs, sm_35 through sm_90).
set(CMAKE_CUDA_ARCHITECTURES 86;89 CACHE STRING "CUDA archs" FORCE)
set(TORCH_CUDA_ARCH_LIST "")
foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
    string(REGEX REPLACE "([0-9])$" ".\\1" _arch_dotted "${_arch}")
    list(APPEND TORCH_CUDA_ARCH_LIST "${_arch_dotted}")
endforeach()
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

set(USE_SYSTEM_NVTX ON)
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Libtorch not found. Please set CMAKE_PREFIX_PATH to the libtorch directory (e.g., extern/libtorch) containing the Torch cmake files.")
endif()
message(STATUS "Found libtorch: ${TORCH_LIBRARIES}")
message(STATUS "Torch install prefix: ${TORCH_INSTALL_PREFIX}")
# Torch's config appended its arch flags to CMAKE_CUDA_FLAGS; surface them so a
# wrong -gencode set is visible at configure time.
message(STATUS "CMAKE_CUDA_FLAGS after Torch: ${CMAKE_CUDA_FLAGS}")

set(ONNX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/onnxruntime-win-x64-gpu-1.22.0)
# Check if ONNX_DIR exists
if(NOT EXISTS ${ONNX_DIR})
    message(FATAL_ERROR "ONNX directory not found: ${ONNX_DIR}. Please ensure ONNX Runtime is correctly extracted.")
endif()
set(ONNX_INCLUDE_DIR ${ONNX_DIR}/include)
set(ONNX_LIB_DIR ${ONNX_DIR}/lib)
if(NOT EXISTS ${ONNX_INCLUDE_DIR} OR NOT EXISTS ${ONNX_LIB_DIR})
    message(FATAL_ERROR "ONNX include or lib directory not found: ${ONNX_INCLUDE_DIR} or ${ONNX_LIB_DIR}. Please ensure ONNX Runtime is correctly extracted.")
else()
    message(STATUS "Using ONNX library: ${ONNX_LIB_DIR}")
endif()


# Add DirectX 12 dependencies
if(WIN32)
    # DirectX 12 libraries
    set(DX12_LIBS
        d3d12.lib
        dxgi.lib
        d3dcompiler.lib
    )
endif()

# Create the main SpotObserver library
# For now, create a header-only library since src/ is empty
add_library(${PLUGIN_NAME} SHARED
    src/spot-observer.cpp
    src/spot-connection.cpp
    src/dumper.cpp
    src/include/unity-cuda-interop.h
    src/unity-cuda-interop.cpp
    src/model.cpp
    src/include/vision-pipeline.h
    src/vision-pipeline.cpp
    src/cuda_kernels.cu
    src/load_image_to_cuda.cu
)

# Unity Native-plugin API headers (IUnityInterface.h etc.).
# Auto-detects the newest Unity Hub install; override with -DUnityPluginAPI_INCLUDE=<path>.
if(NOT UnityPluginAPI_INCLUDE)
    file(GLOB _unity_editor_dirs "C:/Program Files/Unity/Hub/Editor/*")
    list(SORT _unity_editor_dirs COMPARE NATURAL ORDER DESCENDING)
    foreach(_unity_dir ${_unity_editor_dirs})
        if(EXISTS "${_unity_dir}/Editor/Data/PluginAPI/IUnityInterface.h")
            set(UnityPluginAPI_INCLUDE "${_unity_dir}/Editor/Data/PluginAPI")
            break()
        endif()
    endforeach()
endif()
set(UnityPluginAPI_INCLUDE "${UnityPluginAPI_INCLUDE}" CACHE PATH "Path to Unity's Native Plugin API headers")
if(NOT EXISTS "${UnityPluginAPI_INCLUDE}/IUnityInterface.h")
    message(FATAL_ERROR "Unity Native Plugin API headers not found (IUnityInterface.h). "
        "Set -DUnityPluginAPI_INCLUDE=<Unity>/Editor/Data/PluginAPI for your installed Unity version.")
endif()
message(STATUS "Using Unity Plugin API headers: ${UnityPluginAPI_INCLUDE}")

# Include directories
# TODO: Use target_include_directories instead of include_directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${OpenCV_INCLUDES}
    ${SPOT_SDK_ROOT}/include
    ${UnityPluginAPI_INCLUDE}
    ${ONNX_INCLUDE_DIR}
)

message(STATUS "OpenCV_LIBS_DIR: ${OpenCV_LIBS_DIR}")
message(STATUS "OpenCV_INCLUDES: ${OpenCV_INCLUDES}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Link directories
target_link_directories(${PLUGIN_NAME} PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

# Add interface link libraries (dependencies that users of this library will need)
target_link_libraries(${PLUGIN_NAME}
    CUDA::cudart
    CUDA::cuda_driver
    ${DX12_LIBS}
    ${OpenCV_LIBS}
    bosdyn_client_static
    bosdyn_api_static
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
    ${TORCH_LIBRARIES}
    ${ONNX_LIB_DIR}/onnxruntime.lib
)

# Add interface compile definitions for Windows
if(WIN32)
    add_compile_definitions(TARGET ${PLUGIN_NAME}
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
        _WINDLL
    )
    # Add entry point specification
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        LINK_FLAGS "/DLL"
    )
endif()

# Option to disable automatic DLL copying
option(COPY_DLLS "Automatically copy required DLLs to executable directory" ON)

# Copy required DLLs next to the built binaries after build
if(COPY_DLLS AND WIN32)
    # DLLs of imported targets (vcpkg: protobuf, gRPC, OpenSSL, abseil, ...) are
    # enumerated by CMake itself via $<TARGET_RUNTIME_DLLS> — no hardcoded DLL
    # names or vcpkg paths. OpenCV / LibTorch / ONNX Runtime are linked as raw
    # libs (not imported targets), and torch/ONNX also load some DLLs dynamically
    # at runtime (cuDNN, CUDA execution provider), so their DLLs are collected
    # from their lib dirs explicitly.
    set(OPENCV_DLLS "${CMAKE_SOURCE_DIR}/extern/opencv/x64/vc16/bin/opencv_world4110.dll")
    file(GLOB LIBTORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB ONNX_DLLS "${ONNX_LIB_DIR}/*.dll")

    # $<TARGET_RUNTIME_DLLS> misses OpenSSL/ZLIB: they enter through CMake
    # find-modules (via gRPC), whose UNKNOWN-type imported targets the genex
    # cannot enumerate. Locate the vcpkg bin dir from protobuf's imported DLL
    # (no path guessing) and take every DLL vcpkg built for this manifest;
    # overlap with the genex is harmless for copy_if_different/install.
    get_target_property(_protobuf_dll protobuf::libprotobuf IMPORTED_LOCATION_RELEASE)
    if(NOT _protobuf_dll)
        get_target_property(_protobuf_dll protobuf::libprotobuf IMPORTED_LOCATION)
    endif()
    get_filename_component(VCPKG_BIN_DIR "${_protobuf_dll}" DIRECTORY)
    file(GLOB VCPKG_DLLS "${VCPKG_BIN_DIR}/*.dll")

    set(EXTERN_RUNTIME_DLLS ${VCPKG_DLLS} ${OPENCV_DLLS} ${LIBTORCH_DLLS} ${ONNX_DLLS})

    add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${PLUGIN_NAME}>
            ${EXTERN_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:${PLUGIN_NAME}>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLLs next to ${PLUGIN_NAME}"
    )
endif() # COPY_DLLS

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

# INSTALL
include(GNUInstallDirs)

# Install the main library
install(TARGETS ${PLUGIN_NAME}
    EXPORT SpotObserverTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SpotObserver
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install required DLLs on Windows (same set as the post-build copy above).
if(WIN32 AND COPY_DLLS)
    install(FILES
        $<TARGET_RUNTIME_DLLS:${PLUGIN_NAME}>
        ${EXTERN_RUNTIME_DLLS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Export targets for find_package support
install(EXPORT SpotObserverTargets
    FILE SpotObserverTargets.cmake
    NAMESPACE SpotObserver::
    DESTINATION ${CMAKE_INSTALL_BINDIR}/cmake/SpotObserver
)
