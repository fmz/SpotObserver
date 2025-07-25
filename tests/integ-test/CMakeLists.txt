# Tests CMakeLists.txt for SpotObserver

# Standalone camera streaming test executable
add_executable(integ-test
    integ-test.cpp
)

# Apply Windows compile definitions if needed
if(WIN32)
    target_compile_definitions(integ-test PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
    )
endif()

# Apply compile options
target_compile_options(integ-test PRIVATE
    -Wall
)

# Link directories
target_link_directories(integ-test PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

target_link_libraries(integ-test
    ${PLUGIN_NAME}
    CUDA::cudart
)

# Copy only the DLLs needed for this test
if(COPY_DLLS AND WIN32)
    # Copy vcpkg DLLs (for gRPC/protobuf)
    if(VCPKG_DLL_DIR AND EXISTS "${VCPKG_DLL_DIR}")
        foreach(dll ${VCPKG_DLLS})
            if(EXISTS "${VCPKG_DLL_DIR}/${dll}")
                add_custom_command(TARGET integ-test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:integ-test>
                    COMMENT "Copying ${dll} for test"
                )
            endif()
        endforeach()
    endif()

    # Copy OpenCV DLLs
    if(EXISTS "${OPENCV_DLL_DIR}")
        foreach(dll ${OPENCV_DLLS})
            if(EXISTS "${OPENCV_DLL_DIR}/${dll}")
                add_custom_command(TARGET integ-test POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:integ-test>
                    COMMENT "Copying ${dll} for test"
                )
            endif()
        endforeach()
    endif()
    # Copy the main library DLL
    foreach(dll ${PLUGIN_DLLS})
        if(EXISTS ${dll})
            add_custom_command(TARGET integ-test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:integ-test>
                COMMENT "Copying ${dll} for test"
            )
        endif()
    endforeach()
endif()