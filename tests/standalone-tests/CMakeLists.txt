# Tests CMakeLists.txt for SpotObserver

# Standalone camera streaming test executable
add_executable(spot_camera_stream_test
    spot-camera-stream-multithreaded.cpp
)

# Standalone four-pcs correctness/timing check. Only depends on Eigen (no
# bosdyn/OpenCV/CUDA needed), unlike the camera streaming test above.
add_executable(four_pcs_test
    four-pcs-test.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs.cpp
)
target_include_directories(four_pcs_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(four_pcs_test
    Eigen3::Eigen
)

# GPU-vs-CPU correctness check for four-pcs-gpu.cu -- needs CUDA (cudaMalloc/
# cudaMemcpy, plus the compiled four-pcs-gpu.cu kernels themselves) in
# addition to Eigen3.
add_executable(four_pcs_gpu_test
    four-pcs-gpu-test.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu.cu
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu-diagonal.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu-multiseed.cpp
)
target_include_directories(four_pcs_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(four_pcs_gpu_test
    Eigen3::Eigen
    CUDA::cudart
)

# Seed/parameter sweep tool for four-pcs-gpu.cu -- same deps as four_pcs_gpu_test.
add_executable(four_pcs_gpu_param_sweep
    four-pcs-gpu-param-sweep.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu.cu
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu-diagonal.cpp
)
target_include_directories(four_pcs_gpu_param_sweep PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(four_pcs_gpu_param_sweep
    Eigen3::Eigen
    CUDA::cudart
)

# Wall-constrained gravity-aligned coarse alignment (wall-align-gpu.cu) on
# real exported capture data -- the deterministic alternative to the 4PCS
# coarse stage. Needs the four-pcs GPU sources (plane fit, spatial grid, and
# the icpGPU used for per-candidate refinement).
add_executable(wall_align_gpu_test
    wall-align-gpu-test.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs.cpp
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu.cu
    ${CMAKE_SOURCE_DIR}/src/four-pcs-gpu-diagonal.cpp
    ${CMAKE_SOURCE_DIR}/src/wall-align-gpu.cu
)
target_include_directories(wall_align_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
)
target_link_libraries(wall_align_gpu_test
    Eigen3::Eigen
    CUDA::cudart
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