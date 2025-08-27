#include "utils.h"
#include "cuda_kernels.cuh"

#include <vector>
#include <cmath>
#include <limits>
#include <torch/torch.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Configuration constants
#define TILE_SIZE 32
#define WARP_SIZE 32
#define MAX_K_NEIGHBORS 4
#define SHARED_BORDER 4
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

namespace SOb {

// Rotation angles enum for template parameter
enum class Rotation : int {
    NONE = 0,
    CW_90 = 90,
    CW_180 = 180,
    CW_270 = 270
};

// Helper to compute rotated dimensions
template<Rotation R>
__device__ __forceinline__ void getRotatedCoords(
    int x, int y, int width, int height,
    int& out_x, int& out_y, int& out_w, int& out_h)
{
    if constexpr (R == Rotation::CW_90) {
        out_x = height - 1 - y;
        out_y = x;
        out_w = height;
        out_h = width;
    } else if constexpr (R == Rotation::CW_180) {
        out_x = width - 1 - x;
        out_y = height - 1 - y;
        out_w = width;
        out_h = height;
    } else if constexpr (R == Rotation::CW_270) {
        out_x = y;
        out_y = width - 1 - x;
        out_w = height;
        out_h = width;
    } else { // NONE
        out_x = x;
        out_y = y;
        out_w = width;
        out_h = height;
    }
}

template<Rotation R>
__global__ void rotateDepth_fast_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int dst_x, dst_y, dst_w, dst_h;
    getRotatedCoords<R>(x, y, width, height, dst_x, dst_y, dst_w, dst_h);

    dst[dst_y * dst_w + dst_x] = src[y * width + x];
}


// Optimized downscaling kernel using shared memory, separable convolutions, and vectorized accesses
template <int TILE_SZ, int SCALE_FACTOR, bool ROTATE_90_CW = false>
__global__ void downscale_optimized_kernel(
    const float* input_image,
    float* output_image,
    int original_width,
    int original_height
) {
    // Shared memory: input tile + intermediate results for separable convolution
    __shared__ float s_input[TILE_SZ * SCALE_FACTOR][TILE_SZ * SCALE_FACTOR + 1];
    __shared__ float s_horizontal[TILE_SZ][TILE_SZ * SCALE_FACTOR + 1];

    // Handle rotation: if rotating 90 CW, output dimensions are swapped
    const int new_width = ROTATE_90_CW ? (original_height / SCALE_FACTOR) : (original_width / SCALE_FACTOR);
    const int new_height = ROTATE_90_CW ? (original_width / SCALE_FACTOR) : (original_height / SCALE_FACTOR);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int out_x = blockIdx.x * TILE_SZ + tx;
    const int out_y = blockIdx.y * TILE_SZ + ty;

    const int in_tile_x = blockIdx.x * TILE_SZ * SCALE_FACTOR;
    const int in_tile_y = blockIdx.y * TILE_SZ * SCALE_FACTOR;

    // Phase 1: Cooperative loading of input data with vectorized accesses
    for (int load_y = ty; load_y < TILE_SZ * SCALE_FACTOR; load_y += blockDim.y) {
        int src_y = in_tile_y + load_y;
        if (src_y < original_height) {
            // Try vectorized loads when possible
            for (int load_x = tx * 4; load_x < TILE_SZ * SCALE_FACTOR; load_x += blockDim.x * 4) {
                int src_x = in_tile_x + load_x;

                // Vectorized load of 4 floats if aligned and within bounds
                if ((src_x & 3) == 0 && src_x + 3 < original_width) {
                    int src_idx = src_y * original_width + src_x;
                    float4 data = *reinterpret_cast<const float4*>(&input_image[src_idx]);

                    if (load_x < TILE_SZ * SCALE_FACTOR) s_input[load_y][load_x] = data.x;
                    if (load_x + 1 < TILE_SZ * SCALE_FACTOR) s_input[load_y][load_x + 1] = data.y;
                    if (load_x + 2 < TILE_SZ * SCALE_FACTOR) s_input[load_y][load_x + 2] = data.z;
                    if (load_x + 3 < TILE_SZ * SCALE_FACTOR) s_input[load_y][load_x + 3] = data.w;
                } else {
                    // Scalar fallback
                    for (int i = 0; i < 4 && load_x + i < TILE_SZ * SCALE_FACTOR; i++) {
                        int curr_x = src_x + i;
                        if (curr_x < original_width) {
                            s_input[load_y][load_x + i] = input_image[src_y * original_width + curr_x];
                        } else {
                            s_input[load_y][load_x + i] = 0.0f;
                        }
                    }
                }
            }
        } else {
            // Fill with zeros for out-of-bounds
            for (int load_x = tx; load_x < TILE_SZ * SCALE_FACTOR; load_x += blockDim.x) {
                s_input[load_y][load_x] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Phase 2: Separable convolution - horizontal pass
    if (ty < TILE_SZ && tx < TILE_SZ) {
        for (int row = 0; row < SCALE_FACTOR; row++) {
            int src_row = ty * SCALE_FACTOR + row;
            float sum = 0.0f;
            int valid_count = 0;

            // Horizontal averaging with unrolled loop for common scale factors
            #pragma unroll
            for (int k = 0; k < SCALE_FACTOR; k++) {
                float val = s_input[src_row][tx * SCALE_FACTOR + k];
                if (val > 0.01f) {
                    sum += val;
                    valid_count++;
                }
            }

            s_horizontal[ty][tx * SCALE_FACTOR + row] = (valid_count > 0) ? sum / valid_count : 0.0f;
        }
    }

    __syncthreads();

    // Phase 3: Separable convolution - vertical pass and write back
    if (ty < TILE_SZ && tx < TILE_SZ) {
        // First check if we're within the input processing bounds
        const int input_downscaled_width = original_width / SCALE_FACTOR;
        const int input_downscaled_height = original_height / SCALE_FACTOR;

        if (out_x < input_downscaled_width && out_y < input_downscaled_height) {
            float sum = 0.0f;
            int valid_count = 0;

            // Vertical averaging
            #pragma unroll
            for (int k = 0; k < SCALE_FACTOR; k++) {
                float val = s_horizontal[ty][tx * SCALE_FACTOR + k];
                if (val > 0.01f) {
                    sum += val;
                    valid_count++;
                }
            }

            // Write result to output buffer with optional rotation
            int final_x, final_y;
            if constexpr (ROTATE_90_CW) {
                final_x = new_width - 1 - out_y;
                final_y = out_x;
            } else {
                final_x = out_x;
                final_y = out_y;
            }

            // Comprehensive bounds checking for output write
            if (final_x >= 0 && final_x < new_width && final_y >= 0 && final_y < new_height) {
                int out_idx = final_y * new_width + final_x;
                output_image[out_idx] = (valid_count > 0) ? sum / valid_count : 0.0f;
            }
        }
    }
}

// Fast reciprocal approximation
__device__ __forceinline__ float fast_rcp(float x) {
    return __frcp_rn(x);
}

// Extract valid mask
static __global__ void extract_valid_mask_kernel(
    const float* depth_image,
    uint8_t* valid_mask,
    int size,
    float min_depth,
    float max_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float depth = depth_image[idx];
        valid_mask[idx] = (depth >= min_depth && depth <= max_depth) ? 1 : 0;
    }
}

// Optimized first pass using shared memory tiles
__global__ void fill_depth_pass1_shared(
    const float* depth_image,
    const uint8_t* valid_mask,
    float* temp_depth,
    uint8_t* update_mask,
    int32_t width,
    int32_t height,
    int32_t search_radius
) {
    // Shared memory for depth tile and valid mask
    __shared__ float s_depth[BLOCK_DIM_Y + 2*SHARED_BORDER][BLOCK_DIM_X + 2*SHARED_BORDER];
    __shared__ uint8_t s_valid[BLOCK_DIM_Y + 2*SHARED_BORDER][BLOCK_DIM_X + 2*SHARED_BORDER];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_DIM_X + tx;
    const int y = blockIdx.y * BLOCK_DIM_Y + ty;

    // Load tile with borders into shared memory
    const int shared_x = tx + SHARED_BORDER;
    const int shared_y = ty + SHARED_BORDER;

    // Load main tile
    if (x < width && y < height) {
        int idx = y * width + x;
        s_depth[shared_y][shared_x] = depth_image[idx];
        s_valid[shared_y][shared_x] = valid_mask[idx];
    } else {
        s_depth[shared_y][shared_x] = 0.0f;
        s_valid[shared_y][shared_x] = 0;
    }

    // Load borders (boundary threads load extra data)
    if (tx < SHARED_BORDER) {
        // Left border
        int bx = blockIdx.x * BLOCK_DIM_X - SHARED_BORDER + tx;
        int idx = y * width + bx;
        if (bx >= 0 && bx < width && y < height) {
            s_depth[shared_y][tx] = depth_image[idx];
            s_valid[shared_y][tx] = valid_mask[idx];
        } else {
            s_depth[shared_y][tx] = 0.0f;
            s_valid[shared_y][tx] = 0;
        }

        // Right border
        bx = blockIdx.x * BLOCK_DIM_X + BLOCK_DIM_X + tx;
        idx = y * width + bx;
        if (bx < width && y < height) {
            s_depth[shared_y][BLOCK_DIM_X + SHARED_BORDER + tx] = depth_image[idx];
            s_valid[shared_y][BLOCK_DIM_X + SHARED_BORDER + tx] = valid_mask[idx];
        } else {
            s_depth[shared_y][BLOCK_DIM_X + SHARED_BORDER + tx] = 0.0f;
            s_valid[shared_y][BLOCK_DIM_X + SHARED_BORDER + tx] = 0;
        }
    }

    if (ty < SHARED_BORDER) {
        // Top border
        int by = blockIdx.y * BLOCK_DIM_Y - SHARED_BORDER + ty;
        int idx = by * width + x;
        if (x < width && by >= 0 && by < height) {
            s_depth[ty][shared_x] = depth_image[idx];
            s_valid[ty][shared_x] = valid_mask[idx];
        } else {
            s_depth[ty][shared_x] = 0.0f;
            s_valid[ty][shared_x] = 0;
        }

        // Bottom border
        by = blockIdx.y * BLOCK_DIM_Y + BLOCK_DIM_Y + ty;
        idx = by * width + x;
        if (x < width && by < height) {
            s_depth[BLOCK_DIM_Y + SHARED_BORDER + ty][shared_x] = depth_image[idx];
            s_valid[BLOCK_DIM_Y + SHARED_BORDER + ty][shared_x] = valid_mask[idx];
        } else {
            s_depth[BLOCK_DIM_Y + SHARED_BORDER + ty][shared_x] = 0.0f;
            s_valid[BLOCK_DIM_Y + SHARED_BORDER + ty][shared_x] = 0;
        }
    }

    __syncthreads();

    // Process pixels using shared memory
    if (x < width && y < height) {
        const int idx = y * width + x;

        // Skip if already valid
        if (s_valid[shared_y][shared_x] == 1) {
            temp_depth[idx] = s_depth[shared_y][shared_x];
            update_mask[idx] = 0;
            return;
        }

        // Search for nearest neighbors in shared memory tile
        float sum_depth = 0.0f;
        float sum_weights = 0.0f;
        int count = 0;

        #pragma unroll
        for (int dy = -search_radius; dy <= search_radius; dy++) {
            #pragma unroll
            for (int dx = -search_radius; dx <= search_radius; dx++) {
                int sx = shared_x + dx;
                int sy = shared_y + dy;

                // Check bounds
                if (sx >= 0 && sx < BLOCK_DIM_X + 2*SHARED_BORDER &&
                    sy >= 0 && sy < BLOCK_DIM_Y + 2*SHARED_BORDER) {

                    if (s_valid[sy][sx] == 1) {
                        float d = s_depth[sy][sx];
                        float dist_sq = float(dx*dx + dy*dy);
                        float weight = __expf(-dist_sq * 0.1f); // Gaussian weight

                        sum_depth += d * weight;
                        sum_weights += weight;
                        count++;

                        if (count >= MAX_K_NEIGHBORS) break;
                    }
                }
            }
            if (count >= MAX_K_NEIGHBORS) break;
        }

        if (sum_weights > 0.0f) {
            temp_depth[idx] = sum_depth * fast_rcp(sum_weights);
            update_mask[idx] = 1;
        } else {
            temp_depth[idx] = depth_image[idx];
            update_mask[idx] = 0;
        }
    }
}

// Hierarchical search for larger radius (second pass)
__global__ void fill_depth_pass2_hierarchical(
    float* depth_image,
    const uint8_t* valid_mask,
    const uint8_t* update_mask,
    int32_t width,
    int32_t height,
    int32_t max_radius
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Skip if already updated in pass 1 or originally valid
    if (valid_mask[idx] == 1 || update_mask[idx] == 1) return;

    // Hierarchical search with early termination
    float best_depths[MAX_K_NEIGHBORS];
    float best_dists[MAX_K_NEIGHBORS];
    int num_found = 0;

    // Initialize with large distances
    #pragma unroll
    for (int i = 0; i < MAX_K_NEIGHBORS; i++) {
        best_dists[i] = 1e10f;
    }

    // Search in expanding rings with exponential step size
    for (int radius = 4; radius <= max_radius && num_found < MAX_K_NEIGHBORS; radius *= 2) {
        int step = max(1, radius / 8);

        // Search ring perimeter
        for (int i = -radius; i <= radius; i += step) {
            // Top and bottom edges
            int nx1 = x + i;
            int ny1 = y - radius;
            int ny2 = y + radius;

            if (nx1 >= 0 && nx1 < width) {
                if (ny1 >= 0) {
                    int nidx = ny1 * width + nx1;
                    if (valid_mask[nidx] == 1) {
                        float d = depth_image[nidx];
                        float dist = sqrtf(float(i*i + radius*radius));

                        // Insert into sorted list
                        for (int k = 0; k < MAX_K_NEIGHBORS; k++) {
                            if (dist < best_dists[k]) {
                                // Shift elements
                                for (int j = MAX_K_NEIGHBORS-1; j > k; j--) {
                                    best_dists[j] = best_dists[j-1];
                                    best_depths[j] = best_depths[j-1];
                                }
                                best_dists[k] = dist;
                                best_depths[k] = d;
                                if (num_found < MAX_K_NEIGHBORS) num_found++;
                                break;
                            }
                        }
                    }
                }

                if (ny2 < height) {
                    int nidx = ny2 * width + nx1;
                    if (valid_mask[nidx] == 1) {
                        float d = depth_image[nidx];
                        float dist = sqrtf(float(i*i + radius*radius));

                        for (int k = 0; k < MAX_K_NEIGHBORS; k++) {
                            if (dist < best_dists[k]) {
                                for (int j = MAX_K_NEIGHBORS-1; j > k; j--) {
                                    best_dists[j] = best_dists[j-1];
                                    best_depths[j] = best_depths[j-1];
                                }
                                best_dists[k] = dist;
                                best_depths[k] = d;
                                if (num_found < MAX_K_NEIGHBORS) num_found++;
                                break;
                            }
                        }
                    }
                }
            }

            // Left and right edges (avoid corners)
            if (i > -radius && i < radius) {
                int ny = y + i;
                int nx1 = x - radius;
                int nx2 = x + radius;

                if (ny >= 0 && ny < height) {
                    if (nx1 >= 0) {
                        int nidx = ny * width + nx1;
                        if (valid_mask[nidx] == 1) {
                            float d = depth_image[nidx];
                            float dist = sqrtf(float(radius*radius + i*i));

                            for (int k = 0; k < MAX_K_NEIGHBORS; k++) {
                                if (dist < best_dists[k]) {
                                    for (int j = MAX_K_NEIGHBORS-1; j > k; j--) {
                                        best_dists[j] = best_dists[j-1];
                                        best_depths[j] = best_depths[j-1];
                                    }
                                    best_dists[k] = dist;
                                    best_depths[k] = d;
                                    if (num_found < MAX_K_NEIGHBORS) num_found++;
                                    break;
                                }
                            }
                        }
                    }

                    if (nx2 < width) {
                        int nidx = ny * width + nx2;
                        if (valid_mask[nidx] == 1) {
                            float d = depth_image[nidx];
                            float dist = sqrtf(float(radius*radius + i*i));

                            for (int k = 0; k < MAX_K_NEIGHBORS; k++) {
                                if (dist < best_dists[k]) {
                                    for (int j = MAX_K_NEIGHBORS-1; j > k; j--) {
                                        best_dists[j] = best_dists[j-1];
                                        best_depths[j] = best_depths[j-1];
                                    }
                                    best_dists[k] = dist;
                                    best_depths[k] = d;
                                    if (num_found < MAX_K_NEIGHBORS) num_found++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute weighted average
    if (num_found > 0) {
        float sum_depth = 0.0f;
        float sum_weights = 0.0f;

        int k_use = min(8, num_found);
        #pragma unroll
        for (int i = 0; i < k_use; i++) {
            float weight = __expf(-best_dists[i] * 0.05f);
            sum_depth += best_depths[i] * weight;
            sum_weights += weight;
        }

        depth_image[idx] = sum_depth * fast_rcp(sum_weights);
    }
}

// Update valid mask kernel
static __global__ void update_valid_mask_kernel(
    uint8_t* valid_mask,
    const uint8_t* update_mask,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (update_mask[idx] == 1) {
            valid_mask[idx] = 1;
        }
    }
}

size_t depth_preprocessor_get_workspace_size(int width, int height) {
    // Calculate size needed for depth preprocessing
    size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    return pixel_count * (sizeof(float) + sizeof(uint8_t) * 4); // depth + update mask + valid mask
}

// Main preprocessing function
cudaError_t preprocess_depth_image(
    float* depth_image,
    int width,
    int height,
    int downscale_factor,
    uint8_t* workspace,
    bool rotate_90_cw
) {
    const float min_depth = 0.01f;
    const float max_depth = 100.0f;
    const int small_radius = 4;
    const int large_radius = 50;

    // Allocate temporary buffers if needed
    size_t image_size = width * height;

    // Carve up workspace
    float*   d_temp_depth  = reinterpret_cast<float*>(workspace);
    uint8_t* d_update_mask = reinterpret_cast<uint8_t*>(workspace + image_size * sizeof(float));
    uint8_t* d_valid_mask  = reinterpret_cast<uint8_t*>(workspace + image_size * (sizeof(float) + sizeof(uint8_t)));

    // Extract valid mask
    int blockSize = 256;
    int numBlocks = (image_size + blockSize - 1) / blockSize;

    extract_valid_mask_kernel<<<numBlocks, blockSize>>>(
        depth_image, d_valid_mask, image_size, min_depth, max_depth
    );

    // Pass 1: Small radius search using shared memory
    dim3 block1(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid1((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
               (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    fill_depth_pass1_shared<<<grid1, block1>>>(
        depth_image, d_valid_mask, d_temp_depth, d_update_mask,
        width, height, small_radius
    );

    // Update valid mask based on what was filled
    update_valid_mask_kernel<<<numBlocks, blockSize>>>(
        d_valid_mask, d_update_mask, image_size
    );

    // Pass 2: Large radius hierarchical search
    dim3 block2(32, 8);
    dim3 grid2((width + block2.x - 1) / block2.x,
               (height + block2.y - 1) / block2.y);

    fill_depth_pass2_hierarchical<<<grid2, block2>>>(
        d_temp_depth, d_valid_mask, d_update_mask,
        width, height, large_radius
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Apply downscaling
    if (downscale_factor > 1) {
        // Verify it's a power of 2
        if ((downscale_factor & (downscale_factor - 1)) != 0) {
            return cudaErrorInvalidValue; // Must be power of 2
        }

        // Helper lambda for launching downscale kernels
        auto launch_downscale_kernel = [&](int tile_size, auto kernel_func) {
            dim3 block(tile_size, tile_size);
            dim3 grid((width / downscale_factor + tile_size - 1) / tile_size,
                      (height / downscale_factor + tile_size - 1) / tile_size);
            kernel_func<<<grid, block>>>(d_temp_depth, depth_image, width, height);
        };

        // Launch appropriate kernel based on scale factor
        switch (downscale_factor) {
            case 2:
                if (rotate_90_cw) launch_downscale_kernel(16, downscale_optimized_kernel<16, 2, true>);
                else launch_downscale_kernel(16, downscale_optimized_kernel<16, 2, false>);
                break;
            case 4:
                if (rotate_90_cw) launch_downscale_kernel(16, downscale_optimized_kernel<16, 4, true>);
                else launch_downscale_kernel(16, downscale_optimized_kernel<16, 4, false>);
                break;
            case 8:
                if (rotate_90_cw) launch_downscale_kernel(8, downscale_optimized_kernel<8, 8, true>);
                else launch_downscale_kernel(8, downscale_optimized_kernel<8, 8, false>);
                break;
            case 16:
                if (rotate_90_cw) launch_downscale_kernel(4, downscale_optimized_kernel<4, 16, true>);
                else launch_downscale_kernel(4, downscale_optimized_kernel<4, 16, false>);
                break;
            case 32:
                if (rotate_90_cw) launch_downscale_kernel(2, downscale_optimized_kernel<2, 32, true>);
                else launch_downscale_kernel(2, downscale_optimized_kernel<2, 32, false>);
                break;
            default:
                return cudaErrorInvalidValue; // Unsupported scale factor
        }

        err = cudaGetLastError();
    } else {
        if (rotate_90_cw) {
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                      (height + TILE_SIZE - 1) / TILE_SIZE);
            rotateDepth_fast_kernel<Rotation::CW_90><<<grid, block>>>(d_temp_depth, depth_image, width, height);
            err = cudaGetLastError();
        } else {
            err = cudaMemcpyAsync(depth_image, d_temp_depth, image_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    return cudaGetLastError();
}

cudaError_t postprocess_depth_image(
    float* depth_image,
    int width,
    int height,
    float* workspace,
    bool rotate_90_ccw
) {
    if (!rotate_90_ccw) {
        return cudaSuccess; // No rotation needed
    }
    
    // Apply 270 CW rotation
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);
    
    // Rotate to workspace buffer
    rotateDepth_fast_kernel<Rotation::CW_270><<<grid, block>>>(
        depth_image, workspace, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Copy rotated result back
    size_t image_size = width * height * sizeof(float);
    err = cudaMemcpyAsync(depth_image, workspace, image_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;

    err = cudaGetLastError();
    
    return err;
}

struct int2_ { int x, y; };

__device__ __forceinline__
float sqr(float x) { return x * x; }

__global__
void init_seeds_kernel(
    const float* __restrict__ depth,
    int2_* __restrict__ seeds,
    int W, int H,
    float thres
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    float d = depth[idx];
    if (d > thres) {
        seeds[idx].x = x;
        seeds[idx].y = y;
    } else {
        seeds[idx].x = -1;
        seeds[idx].y = -1;
    }
}

__device__ __forceinline__
bool is_valid_seed(const int2_ s) {
    return s.x >= 0 && s.y >= 0;
}

__device__ __forceinline__
float dist2_to_seed(int x, int y, const int2_ s) {
    float dx = float(s.x - x);
    float dy = float(s.y - y);
    return dx*dx + dy*dy;
}

// Jump Flooding pass: examine 8 directions at stride 'step' and keep the closest seed
__global__
void jfa_pass_kernel(const int2_* __restrict__ seeds_in,
                     int2_* __restrict__ seeds_out,
                     int W, int H,
                     int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    int2_ best = seeds_in[idx];
    float best_d2 = is_valid_seed(best) ? dist2_to_seed(x, y, best) : CUDART_INF_F;

    // 8 directions + current pixel
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx * step;
            int ny = y + dy * step;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            int nidx = ny * W + nx;
            int2_ cand = seeds_in[nidx];
            if (!is_valid_seed(cand)) continue;
            float d2 = dist2_to_seed(x, y, cand);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = cand;
            }
        }
    }

    seeds_out[idx] = best;
}

__global__
void finalize_fill_kernel(const int2_* __restrict__ seeds,
                          const float* __restrict__ depth_in,
                          float* __restrict__ depth_out,
                          int W, int H,
                          float thres)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    float d = depth_out[idx];
    if (d > thres) return; // already valid

    int2_ s = seeds[idx];
    if (is_valid_seed(s)) {
        int sidx = s.y * W + s.x;
        depth_out[idx] = depth_in[sidx];
    }
}

// Host utility: next power-of-two floor for step init
static inline int highest_power_of_two_le(int v) {
    int p = 1;
    while ((p << 1) <= v) p <<= 1;
    return p;
}

size_t depth_preprocessor2_get_workspace_size(int width, int height) {
    // Calculate size needed for depth preprocessing
    size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    return pixel_count * (sizeof(float) + sizeof(int2_) * 2); // depth + update mask + valid mask
}

// In-place fill of invalid pixels (< threshold) with the nearest valid pixel value.
cudaError_t preprocess_depth_image2(
    float* depth_image,
    int width,
    int height,
    int downscale_factor,
    uint8_t* workspace,
    bool rotate_90_cw
) {
    const float threshold = 0.01f;
    const int extra_refine_passes = 2;


    const int W = width;
    const int H = height;
    const size_t N = static_cast<size_t>(W) * static_cast<size_t>(H);

    // Carve up workspace
    int2_* seeds_a = reinterpret_cast<int2_*>(workspace);
    int2_* seeds_b = reinterpret_cast<int2_*>(workspace + N * sizeof(int2_));
    float* d_out   = reinterpret_cast<float*>(workspace + 2 * N * sizeof(int2_));

    cudaError_t err = cudaSuccess;

    // Copy original depth to output buffer (preserve valid pixels)
    err = cudaMemcpyAsync(d_out, depth_image, N * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return err;

    { // kernel launch scope – prevents goto from bypassing initializations
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x,
                  (H + block.y - 1) / block.y);

        // Initialize seeds from valid pixels
        init_seeds_kernel<<<grid, block>>>(depth_image, seeds_a, W, H, threshold);
        err = cudaGetLastError();
        if (err != cudaSuccess)  return err;

        // Jump Flooding iterations
        int max_dim = (W > H) ? W : H;
        int step = highest_power_of_two_le(max_dim);
        bool ping = true;

        while (step >= 1) {
            const int2_* in_ptr  = ping ? seeds_a : seeds_b;
            int2_*       out_ptr = ping ? seeds_b : seeds_a;
            jfa_pass_kernel<<<grid, block>>>(in_ptr, out_ptr, W, H, step);
            err = cudaGetLastError();
            if (err != cudaSuccess)  return err;
            ping = !ping;
            step >>= 1;
        }

        // Optional refinement passes at step=1
        for (int i = 0; i < extra_refine_passes; ++i) {
            const int2_* in_ptr  = ping ? seeds_a : seeds_b;
            int2_*       out_ptr = ping ? seeds_b : seeds_a;
            jfa_pass_kernel<<<grid, block>>>(in_ptr, out_ptr, W, H, 1);
            err = cudaGetLastError();
            if (err != cudaSuccess) return err;
            ping = !ping;
        }

        // Finalize: copy nearest valid pixel values into invalid locations
        const int2_* final_seeds = ping ? seeds_a : seeds_b;
        finalize_fill_kernel<<<grid, block>>>(final_seeds, depth_image, d_out, W, H, threshold);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Apply downscaling
    if (downscale_factor > 1) {
        // Verify it's a power of 2
        if ((downscale_factor & (downscale_factor - 1)) != 0) {
            return cudaErrorInvalidValue; // Must be power of 2
        }
        
        err = cudaSuccess;
        
        // Helper lambda for launching downscale kernels
        auto launch_downscale_kernel = [&](int tile_size, auto kernel_func) {
            dim3 block(tile_size, tile_size);
            dim3 grid((width / downscale_factor + tile_size - 1) / tile_size,
                      (height / downscale_factor + tile_size - 1) / tile_size);
            kernel_func<<<grid, block>>>(d_out, depth_image, width, height);
        };

        // Launch appropriate kernel based on scale factor
        switch (downscale_factor) {
            case 2:
                if (rotate_90_cw) launch_downscale_kernel(16, downscale_optimized_kernel<16, 2, true>);
                else launch_downscale_kernel(16, downscale_optimized_kernel<16, 2, false>);
                break;
            case 4:
                if (rotate_90_cw) launch_downscale_kernel(16, downscale_optimized_kernel<16, 4, true>);
                else launch_downscale_kernel(16, downscale_optimized_kernel<16, 4, false>);
                break;
            case 8:
                if (rotate_90_cw) launch_downscale_kernel(8, downscale_optimized_kernel<8, 8, true>);
                else launch_downscale_kernel(8, downscale_optimized_kernel<8, 8, false>);
                break;
            case 16:
                if (rotate_90_cw) launch_downscale_kernel(4, downscale_optimized_kernel<4, 16, true>);
                else launch_downscale_kernel(4, downscale_optimized_kernel<4, 16, false>);
                break;
            case 32:
                if (rotate_90_cw) launch_downscale_kernel(2, downscale_optimized_kernel<2, 32, true>);
                else launch_downscale_kernel(2, downscale_optimized_kernel<2, 32, false>);
                break;
            default:
                return cudaErrorInvalidValue; // Unsupported scale factor
        }

        err = cudaGetLastError();
    } else {
        if (rotate_90_cw) {
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                      (height + TILE_SIZE - 1) / TILE_SIZE);
            rotateDepth_fast_kernel<Rotation::CW_90><<<grid, block>>>(d_out, depth_image, width, height);
            err = cudaGetLastError();
        } else {
            err = cudaMemcpyAsync(depth_image, d_out, N * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    return err;
}

// Dummy kernel for testing purposes (keeping original)
__global__ void setOutputToOnes(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0.01f) {
            output[idx] = input[idx];
        } else {
            output[idx] = 1.0f;
        }
    }
}

// For debugging purposes
cudaError_t testCudaKernel(float* d_input, float* d_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    setOutputToOnes<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    return cudaGetLastError();
}


// Optimized RGB rotation kernel using shared memory
template<Rotation R>
__global__ void rotateRGB_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int width, int height)
{
    __shared__ uint8_t tile[TILE_SIZE][TILE_SIZE][3];

    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int src_x = tile_x + thread_x;
    int src_y = tile_y + thread_y;

    // Load tile into shared memory (coalesced read)
    if (src_x < width && src_y < height) {
        int src_idx = (src_y * width + src_x) * 3;
        tile[thread_y][thread_x][0] = src[src_idx];
        tile[thread_y][thread_x][1] = src[src_idx + 1];
        tile[thread_y][thread_x][2] = src[src_idx + 2];
    }

    __syncthreads();

    // Compute destination coordinates for this pixel
    int dst_x, dst_y, dst_w, dst_h;
    getRotatedCoords<R>(src_x, src_y, width, height, dst_x, dst_y, dst_w, dst_h);

    // Write to destination (may not be coalesced, but cached in L2)
    if (src_x < width && src_y < height) {
        int dst_idx = (dst_y * dst_w + dst_x) * 3;
        dst[dst_idx] = tile[thread_y][thread_x][0];
        dst[dst_idx + 1] = tile[thread_y][thread_x][1];
        dst[dst_idx + 2] = tile[thread_y][thread_x][2];
    }
}

// Optimized depth rotation kernel
template<Rotation R>
__global__ void rotateDepth_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width, int height)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int src_x = tile_x + thread_x;
    int src_y = tile_y + thread_y;

    // Load tile into shared memory (coalesced read)
    if (src_x < width && src_y < height) {
        tile[thread_y][thread_x] = src[src_y * width + src_x];
    }

    __syncthreads();

    // Compute destination coordinates
    int dst_x, dst_y, dst_w, dst_h;
    getRotatedCoords<R>(src_x, src_y, width, height, dst_x, dst_y, dst_w, dst_h);

    // Write to destination
    if (src_x < width && src_y < height) {
        dst[dst_y * dst_w + dst_x] = tile[thread_y][thread_x];
    }
}

// Helper class to manage rotation operations
class CudaImageRotator {
private:
    cudaStream_t stream_;

    template<Rotation R>
    void launchRGBRotation(const uint8_t* src, uint8_t* dst, int width, int height) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                  (height + TILE_SIZE - 1) / TILE_SIZE);
        rotateRGB_kernel<R><<<grid, block, 0, stream_>>>(src, dst, width, height);
    }

    template<Rotation R>
    void launchDepthRotation(const float* src, float* dst, int width, int height) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
                  (height + TILE_SIZE - 1) / TILE_SIZE);
        rotateDepth_kernel<R><<<grid, block, 0, stream_>>>(src, dst, width, height);
    }

public:
    CudaImageRotator(cudaStream_t stream = 0) : stream_(stream) {}

    // Main rotation function for RGB images
    cudaError_t rotateRGB(const uint8_t* src, uint8_t* dst,
                          int width, int height, int rotation_degrees) {
        switch (rotation_degrees) {
            case 0:
                return cudaMemcpyAsync(dst, src, width * height * 3,
                                     cudaMemcpyDeviceToDevice, stream_);
            case 90:
                launchRGBRotation<Rotation::CW_90>(src, dst, width, height);
                break;
            case 180:
                launchRGBRotation<Rotation::CW_180>(src, dst, width, height);
                break;
            case 270:
                launchRGBRotation<Rotation::CW_270>(src, dst, width, height);
                break;
            default:
                return cudaErrorInvalidValue;
        }
        return cudaGetLastError();
    }

    // Main rotation function for depth images
    cudaError_t rotateDepth(const float* src, float* dst,
                           int width, int height, int rotation_degrees) {
        switch (rotation_degrees) {
            case 0:
                return cudaMemcpyAsync(dst, src, width * height * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream_);
            case 90:
                launchDepthRotation<Rotation::CW_90>(src, dst, width, height);
                break;
            case 180:
                launchDepthRotation<Rotation::CW_180>(src, dst, width, height);
                break;
            case 270:
                launchDepthRotation<Rotation::CW_270>(src, dst, width, height);
                break;
            default:
                return cudaErrorInvalidValue;
        }
        return cudaGetLastError();
    }

    // Get output dimensions after rotation
    static void getRotatedDimensions(int width, int height, int rotation_degrees,
                                    int& out_width, int& out_height) {
        if (rotation_degrees == 90 || rotation_degrees == 270) {
            out_width = height;
            out_height = width;
        } else {
            out_width = width;
            out_height = height;
        }
    }
};

// Alternative: Ultra-fast version without shared memory for modern GPUs (Ampere+)
// These rely more on L2 cache and have better performance on newer architectures
template<Rotation R>
__global__ void rotateRGB_fast_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int dst_x, dst_y, dst_w, dst_h;
    getRotatedCoords<R>(x, y, width, height, dst_x, dst_y, dst_w, dst_h);

    int src_idx = (y * width + x) * 3;
    int dst_idx = (dst_y * dst_w + dst_x) * 3;

    // Use vector loads for better throughput
    uchar3 pixel = *reinterpret_cast<const uchar3*>(&src[src_idx]);
    *reinterpret_cast<uchar3*>(&dst[dst_idx]) = pixel;
}

// Needed for preprocessing RGB images from 4-channel HWC uint8_t to 3-channel float CHW
// Batched NHWC uint8 (RGB or RGBA) -> NCHW float (RGB) conversion with optional 90-deg CW rotation
template<bool ROTATE_90_CW = false>
__global__ void nhwc_to_nchw_rgb_float_kernel(
    const uint8_t* __restrict__ in,   // [N,H,W,C] C=3 or 4
    float* __restrict__ out,          // [N,3,H,W] or [N,3,W,H] if rotated
    int N, int H, int W, int C,
    float scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int pixels_per_img = H * W;
    const int total_pixels = N * pixels_per_img;

    // Calculate output dimensions based on rotation
    const int out_H = ROTATE_90_CW ? W : H;
    const int out_W = ROTATE_90_CW ? H : W;
    const int out_pixels_per_img = out_H * out_W;

    // Fast path only when C==4 and input pointer 4-byte aligned
    if (C == 4 && ((reinterpret_cast<uintptr_t>(in) & 0x3) == 0)) {
        const uchar4* __restrict__ vin = reinterpret_cast<const uchar4*>(in);
        for (int i = tid; i < total_pixels; i += stride) {
            uchar4 px = vin[i];
            int n = i / pixels_per_img;
            int p = i - n * pixels_per_img;          // pixel index within image
            int y = p / W;
            int x = p - y * W;
            
            // Calculate output coordinates with optional rotation
            int out_x, out_y;
            if constexpr (ROTATE_90_CW) {
                out_x = H - 1 - y;
                out_y = x;
            } else {
                out_x = x;
                out_y = y;
            }
            
            int out_p = out_y * out_W + out_x;
            int base = n * 3 * out_pixels_per_img + out_p;   // position of R plane element
            out[base]                        = px.x * scale;
            out[base +     out_pixels_per_img] = px.y * scale;
            out[base + 2 * out_pixels_per_img] = px.z * scale;
        }
    } else {
        // Generic path (C==3 or C==4, handles unaligned)
        for (int i = tid; i < total_pixels; i += stride) {
            int n = i / pixels_per_img;
            int p = i - n * pixels_per_img;
            int y = p / W;
            int x = p - y * W;
            int in_offset = ((n * H + y) * W + x) * C;
            
            // Calculate output coordinates with optional rotation
            int out_x, out_y;
            if constexpr (ROTATE_90_CW) {
                out_x = H - 1 - y;
                out_y = x;
            } else {
                out_x = x;
                out_y = y;
            }
            
            int out_p = out_y * out_W + out_x;
            int base = n * 3 * out_pixels_per_img + out_p;
            uint8_t r = in[in_offset + 0];
            uint8_t g = in[in_offset + 1];
            uint8_t b = in[in_offset + 2];
            out[base]                        = r * scale;
            out[base +     out_pixels_per_img] = g * scale;
            out[base + 2 * out_pixels_per_img] = b * scale;
        }
    }
}

void convert_uint8_img_to_float_img(
    const uint8_t* d_in,  // [N,H,W,4] or [N,H,W,3]
    float* d_out,         // [N,3,H,W] or [N,3,W,H] if rotated
    int N, int H, int W, int C,
    bool normalize,
    cudaStream_t stream,
    bool rotate_90_cw
) {
    if (C != 3 && C != 4) {
        throw std::invalid_argument("Input channel count must be 3 or 4");
    }
    int total_pixels = N * H * W;
    if (total_pixels == 0) return;

    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    float scale = normalize ? (1.0f / 255.0f) : 1.0f;
    
    if (rotate_90_cw) {
        nhwc_to_nchw_rgb_float_kernel<true><<<blocks, threads, 0, stream>>>(
            d_in, d_out, N, H, W, C, scale);
    } else {
        nhwc_to_nchw_rgb_float_kernel<false><<<blocks, threads, 0, stream>>>(
            d_in, d_out, N, H, W, C, scale);
    }

    checkCudaError(cudaGetLastError(), "Failed to launch convert_uint8_to_float_3ch kernel");
}

}