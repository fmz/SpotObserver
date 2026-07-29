# Tests CMakeLists.txt for SpotObserver

# Standalone camera streaming test executable
add_executable(integ-test-dx12
    integ-test-dx12.cpp
)

# Apply Windows compile definitions if needed
if(WIN32)
    target_compile_definitions(integ-test-dx12 PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _HAS_EXCEPTIONS=1
    )
endif()

# Apply compile options
target_compile_options(integ-test-dx12 PRIVATE
    -Wall
)

# Link directories
target_link_directories(integ-test-dx12 PRIVATE
    ${SPOT_SDK_ROOT}/lib
    ${OpenCV_LIBS_DIR}
)

target_link_libraries(integ-test-dx12
    ${PLUGIN_NAME}
    CUDA::cudart
)

# Runtime DLLs are copied once into the shared top-level runtime directory
# (see CMAKE_RUNTIME_OUTPUT_DIRECTORY in the top-level CMakeLists.txt), where
# this executable also lands — no per-test copying needed.

if(INSTALL_TESTS)
    install(TARGETS integ-test-dx12
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        CONFIGURATIONS Release Debug RelWithDebInfo MinSizeRel
    )
endif()