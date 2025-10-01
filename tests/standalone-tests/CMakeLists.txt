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

# Copy only the DLLs needed for this test
if(COPY_DLLS AND WIN32)
    # Copy vcpkg DLLs (for gRPC/protobuf)
    if(VCPKG_DLL_DIR AND EXISTS "${VCPKG_DLL_DIR}")
        foreach(dll ${VCPKG_DLLS})
            if(EXISTS "${VCPKG_DLL_DIR}/${dll}")
                add_custom_command(TARGET spot_camera_stream_test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:spot_camera_stream_test>
                    COMMENT "Copying ${dll} for test"
                )
            endif()
        endforeach()
    endif()
    
    # Copy OpenCV DLLs
    if(EXISTS "${OPENCV_DLL_DIR}")
        foreach(dll ${OPENCV_DLLS})
            if(EXISTS "${OPENCV_DLL_DIR}/${dll}")
                add_custom_command(TARGET spot_camera_stream_test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:spot_camera_stream_test>
                    COMMENT "Copying ${dll} for test"
                )
            endif()
        endforeach()
    endif()
endif()

# Install test executable if requested
if(INSTALL_TESTS)
    install(TARGETS spot_camera_stream_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/tests
        CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel
    )
endif()