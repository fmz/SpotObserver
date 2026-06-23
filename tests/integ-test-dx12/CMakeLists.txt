# Tests CMakeLists.txt for SpotObserver

# Standalone camera streaming test executable
add_executable(integ-test-dx12
    integ-test-dx12.cpp
)

add_executable(interop-upload-test
    interop-upload-test.cpp
)

# Apply Windows compile definitions if needed
if(WIN32)
    target_compile_definitions(integ-test-dx12 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
    )
    target_compile_definitions(interop-upload-test PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
        SOB_ENABLE_TEST_HOOKS
    )
endif()

# Apply compile options
target_compile_options(integ-test-dx12 PRIVATE
    -Wall
)
target_compile_options(interop-upload-test PRIVATE
    -Wall
)

# Link directories
target_link_directories(integ-test-dx12 PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)
target_link_directories(interop-upload-test PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

target_link_libraries(integ-test-dx12
    ${PLUGIN_NAME}
    CUDA::cudart
)
target_link_libraries(interop-upload-test
    ${PLUGIN_NAME}
    CUDA::cudart
    CUDA::cuda_driver
    ${DX12_LIBS}
)

function(copy_spotobserver_test_dlls target_name)
    if(NOT COPY_DLLS OR NOT WIN32)
        return()
    endif()

    if(VCPKG_DLL_DIR AND EXISTS "${VCPKG_DLL_DIR}")
        foreach(dll ${VCPKG_DLLS})
            if(EXISTS "${VCPKG_DLL_DIR}/${dll}")
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:${target_name}>
                    COMMENT "Copying ${dll} for ${target_name}"
                )
            endif()
        endforeach()
    endif()

    if(EXISTS "${OPENCV_DLL_DIR}")
        foreach(dll ${OPENCV_DLLS})
            if(EXISTS "${OPENCV_DLL_DIR}/${dll}")
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${OPENCV_DLL_DIR}/${dll}"
                    $<TARGET_FILE_DIR:${target_name}>
                    COMMENT "Copying ${dll} for ${target_name}"
                )
            endif()
        endforeach()
    endif()

    foreach(dll ${MLLIB_DLLS})
        if(EXISTS "${dll}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying ${dll} for ${target_name}"
            )
        endif()
    endforeach()

    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${PLUGIN_NAME}>
        $<TARGET_FILE_DIR:${target_name}>
        COMMENT "Copying SpotObserver plugin for ${target_name}"
    )
endfunction()

copy_spotobserver_test_dlls(integ-test-dx12)
copy_spotobserver_test_dlls(interop-upload-test)

add_test(NAME interop-upload-test COMMAND interop-upload-test)

if(INSTALL_TESTS)
    install(TARGETS integ-test-dx12
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel
    )
    install(TARGETS interop-upload-test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel
    )
endif()
