cmake_minimum_required(VERSION 3.28)
project(SpotCameraStream LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
set(SPOT_SDK_ROOT "extern/spot-sdk-install")
set(OpenCV_INCLUDES "extern/opencv/include")
set(OpenCV_LIBS_DIR "extern/opencv/x64/vc16/lib")
set(OpenCV_LIBS "opencv_world4110")

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDES}
    ${SPOT_SDK_ROOT}/include
)

# Link directories
link_directories(
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

# Create executable
add_executable(spot_camera_stream src/spot-camera-stream.cpp)

# Link libraries
target_link_libraries(spot_camera_stream 
    ${OpenCV_LIBS}
    bosdyn_client_static
    bosdyn_api_static
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
)

if(WIN32)
    add_compile_definitions(TARGET ${PLUGIN_NAME}
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1          # std::format / <format> needs this on MSVC
        _WINDLL                    # Ensure DllMain is called
    )

    # Add entry point specification
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        LINK_FLAGS "/DLL"
    )
endif()

target_compile_options(spot_camera_stream PRIVATE
    -Wall
)

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
    "opencv_videoio_ffmpeg4110_64.dll"
    "opencv_videoio_msmf4110_64.dll"
)

# Copy vcpkg DLLs (only if vcpkg directory was found)
if(VCPKG_DLL_DIR AND EXISTS "${VCPKG_DLL_DIR}")
    message(STATUS "Found vcpkg DLL directory: ${VCPKG_DLL_DIR}")
    foreach(dll ${VCPKG_DLLS})
        if(EXISTS "${VCPKG_DLL_DIR}/${dll}")
            add_custom_command(TARGET spot_camera_stream POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_DLL_DIR}/${dll}"
                $<TARGET_FILE_DIR:spot_camera_stream>
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
            add_custom_command(TARGET spot_camera_stream POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_DLL_DIR}/${dll}"
                $<TARGET_FILE_DIR:spot_camera_stream>
                COMMENT "Copying ${dll}"
            )
        else()
            message(WARNING "OpenCV DLL not found: ${OPENCV_DLL_DIR}/${dll}")
        endif()
    endforeach()
else()
    message(WARNING "OpenCV DLL directory not found: ${OPENCV_DLL_DIR}")
endif()

endif() # COPY_DLLS

# Install target
install(TARGETS spot_camera_stream
    RUNTIME DESTINATION bin
)