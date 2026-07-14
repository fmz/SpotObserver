# Tests CMakeLists.txt for SpotObserver

# Standalone camera streaming test executable
add_executable(spot_camera_stream_test
    spot-camera-stream-multithreaded.cpp
)
#
## Link only the dependencies actually needed for the camera streaming test
#target_link_libraries(spot_camera_stream_test
#    ${OpenCV_LIBS}
#    bosdyn_client_static
#    bosdyn_api_static
#    protobuf::libprotobuf
#    gRPC::grpc++
#    gRPC::grpc
#    Threads::Threads
#)

# Apply Windows compile definitions if needed
if(WIN32)
    target_compile_definitions(spot_camera_stream_test PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
    )
endif()

# Apply compile options
target_compile_options(spot_camera_stream_test PRIVATE
    -Wall
)

# Link directories
target_link_directories(spot_camera_stream_test PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

target_link_libraries(spot_camera_stream_test
    ${OpenCV_LIBS}
    bosdyn_client_static
    bosdyn_api_static
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
)

# Runtime DLLs are copied once into the shared top-level runtime directory
# (see CMAKE_RUNTIME_OUTPUT_DIRECTORY in the top-level CMakeLists.txt), where
# this executable also lands — no per-test copying needed.

if(INSTALL_TESTS)
    install(TARGETS spot_camera_stream_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel
    )
endif()