#include <cuda_runtime.h>
#include "cuda_kernels.cuh"

// Configuration constants
#define TILE_SIZE 32
#define WARP_SIZE 32
#define MAX_K_NEIGHBORS 16
#define SHARED_BORDER 4
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

namespace SOb {

// Persistent memory
static float* d_temp_depth = nullptr;
static uint8_t* d_update_mask = nullptr;
static uint8_t* d_valid_mask_persistent = nullptr;
static size_t allocated_size = 0;

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

// Main preprocessing function
cudaError_t preprocess_depth_image(
    float* depth_image,
    int width,
    int height
) {
    const float min_depth = 0.01f;
    const float max_depth = 100.0f;
    const int small_radius = 4;
    const int large_radius = 50;
    
    // Allocate temporary buffers if needed
    size_t image_size = width * height;
    if (!d_temp_depth || allocated_size < image_size) {
        if (d_temp_depth) {
            cudaFree(d_temp_depth);
            cudaFree(d_update_mask);
            cudaFree(d_valid_mask_persistent);
        }
        cudaMalloc(&d_temp_depth, image_size * sizeof(float));
        cudaMalloc(&d_update_mask, image_size * sizeof(uint8_t));
        cudaMalloc(&d_valid_mask_persistent, image_size * sizeof(uint8_t));
        allocated_size = image_size;
    }
    
    // Extract valid mask
    int blockSize = 256;
    int numBlocks = (image_size + blockSize - 1) / blockSize;
    
    extract_valid_mask_kernel<<<numBlocks, blockSize>>>(
        depth_image, d_valid_mask_persistent, image_size, min_depth, max_depth
    );
    
    // Pass 1: Small radius search using shared memory
    dim3 block1(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid1((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
               (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    fill_depth_pass1_shared<<<grid1, block1>>>(
        depth_image, d_valid_mask_persistent, d_temp_depth, d_update_mask,
        width, height, small_radius
    );
    
    // Copy results back
    cudaMemcpy(depth_image, d_temp_depth, image_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Update valid mask based on what was filled
    update_valid_mask_kernel<<<numBlocks, blockSize>>>(
        d_valid_mask_persistent, d_update_mask, image_size
    );
    
    // Pass 2: Large radius hierarchical search
    dim3 block2(32, 8);
    dim3 grid2((width + block2.x - 1) / block2.x,
               (height + block2.y - 1) / block2.y);
    
    fill_depth_pass2_hierarchical<<<grid2, block2>>>(
        depth_image, d_valid_mask_persistent, d_update_mask,
        width, height, large_radius
    );
    
    return cudaGetLastError();
}

// Cleanup function
void cleanup_depth_preprocessor() {
    if (d_temp_depth) {
        cudaFree(d_temp_depth);
        d_temp_depth = nullptr;
    }
    if (d_update_mask) {
        cudaFree(d_update_mask);
        d_update_mask = nullptr;
    }
    if (d_valid_mask_persistent) {
        cudaFree(d_valid_mask_persistent);
        d_valid_mask_persistent = nullptr;
    }
    allocated_size = 0;
}

// For debugging purposes
cudaError_t testCudaKernel(float* d_input, float* d_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    setOutputToOnes<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    return cudaGetLastError();
}

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
        out_x = y;
        out_y = width - 1 - x;
        out_w = height;
        out_h = width;
    } else if constexpr (R == Rotation::CW_180) {
        out_x = width - 1 - x;
        out_y = height - 1 - y;
        out_w = width;
        out_h = height;
    } else if constexpr (R == Rotation::CW_270) {
        out_x = height - 1 - y;
        out_y = x;
        out_w = height;
        out_h = width;
    } else { // NONE
        out_x = x;
        out_y = y;
        out_w = width;
        out_h = height;
    }
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

}