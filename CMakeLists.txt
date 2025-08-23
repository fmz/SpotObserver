cmake_minimum_required(VERSION 3.28)
project(SpotCameraStream LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build the test executable" ON)
set(PLUGIN_NAME "SpotObserverLib")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

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

set(USE_SYSTEM_NVTX ON)
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Libtorch not found. Please set CMAKE_PREFIX_PATH to the libtorch directory (e.g., extern/libtorch) containing the Torch cmake files.")
endif()
message(STATUS "Found libtorch: ${TORCH_LIBRARIES}")
message(STATUS "Torch install prefix: ${TORCH_INSTALL_PREFIX}")

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

# Unity Native-plugin API includes TODO: cleanup
set(UnityPluginAPI_INCLUDE "C:/Program Files/Unity/Hub/Editor/6000.2.0f1/Editor/Data/PluginAPI")

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

# Copy required DLLs to executable directory after build
if(COPY_DLLS)
    # Find vcpkg DLL directory dynamically
    if(DEFINED CMAKE_TOOLCHAIN_FILE)
        get_filename_component(VCPKG_ROOT "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
        get_filename_component(VCPKG_ROOT "${VCPKG_ROOT}" DIRECTORY)
        get_filename_component(VCPKG_ROOT "${VCPKG_ROOT}" DIRECTORY)
        set(VCPKG_DLL_DIR "${VCPKG_ROOT}/vcpkg_installed/x64-windows/bin")
    else()
        # Fallback: try to find vcpkg in common locations
        set(VCPKG_DLL_DIR "")
        foreach(path "$ENV{VCPKG_ROOT}/vcpkg_installed/x64-windows/bin" 
                      "${CMAKE_SOURCE_DIR}/../vcpkg/vcpkg_installed/x64-windows/bin"
                      "${CMAKE_SOURCE_DIR}/../../vcpkg/vcpkg_installed/x64-windows/bin")
            if(EXISTS "${path}")
                set(VCPKG_DLL_DIR "${path}")
                break()
            endif()
        endforeach()
    endif()

    set(OPENCV_DLL_DIR "${CMAKE_SOURCE_DIR}/extern/opencv/x64/vc16/bin")

    # List of required DLLs from vcpkg
    set(VCPKG_DLLS
        "re2.dll"
        "libprotobuf.dll"
        "libprotobuf-lite.dll"
        "abseil_dll.dll"
        "cares.dll"
        "libcrypto-3-x64.dll"
        "libssl-3-x64.dll"
        "zlib1.dll"
    )

    # List of required OpenCV DLLs
    set(OPENCV_DLLS
        "opencv_world4110.dll"
    )

    set(PLUGIN_DLLS
        ${CMAKE_CURRENT_BINARY_DIR}/${PLUGIN_NAME}.dll
    )

    set(MLLIB_DLLS)

    # Use GLOB to find all DLLs in the libtorch/lib directory
    file(GLOB LIBTORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    message(STATUS "Found LibTorch DLLs via GLOB: ${LIBTORCH_DLLS}")
    list(APPEND MLLIB_DLLS ${LIBTORCH_DLLS})

    # Find ONNX Runtime DLLs
    file(GLOB ONNX_DLLS "${ONNX_LIB_DIR}/*.dll")
    message(STATUS "Found ONNX Runtime DLLs via GLOB: ${ONNX_DLLS}")
    list(APPEND MLLIB_DLLS ${ONNX_DLLS})

    # Copy vcpkg DLLs (only if vcpkg directory was found)
    if(VCPKG_DLL_DIR AND EXISTS "${VCPKG_DLL_DIR}")
        message(STATUS "Found vcpkg DLL directory: ${VCPKG_DLL_DIR}")
        foreach(dll ${VCPKG_DLLS})
            if(EXISTS "${VCPKG_DLL_DIR}/${dll}")
                add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:${PLUGIN_NAME}>
                    COMMENT "Copying ${dll}"
                )
            else()
                message(WARNING "vcpkg DLL not found: ${VCPKG_DLL_DIR}/${dll}")
            endif()
        endforeach()
    else()
        message(WARNING "vcpkg DLL directory not found. DLLs will need to be copied manually.")
    endif()

    # Copy OpenCV DLLs (only if OpenCV directory exists)
    if(EXISTS "${OPENCV_DLL_DIR}")
        message(STATUS "Found OpenCV DLL directory: ${OPENCV_DLL_DIR}")
        foreach(dll ${OPENCV_DLLS})
            if(EXISTS "${OPENCV_DLL_DIR}/${dll}")
                add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:${PLUGIN_NAME}>
                    COMMENT "Copying ${dll}"
                )
            else()
                message(WARNING "OpenCV DLL not found: ${OPENCV_DLL_DIR}/${dll}")
            endif()
        endforeach()
    else()
        message(WARNING "OpenCV DLL directory not found: ${OPENCV_DLL_DIR}")
    endif()

    # Copy ML library DLLs (Libtorch, ONNX Runtime, etc.)
    foreach(dll ${MLLIB_DLLS})
        if(EXISTS "${dll}")
            add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${PLUGIN_NAME}>
                COMMENT "Copying ${dll}"
            )
        else()
            message(WARNING "ML library DLL not found: ${dll}")
        endif()
    endforeach()
endif() # COPY_DLLS

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

## Output directories
#set_target_properties(${PLUGIN_NAME}
#    PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#)
#
## INSTALL
#install(TARGETS ${PLUGIN_NAME}
#    RUNTIME DESTINATION "Plugins"
#    LIBRARY DESTINATION "Plugins"
#    ARCHIVE DESTINATION "Plugins"
#)