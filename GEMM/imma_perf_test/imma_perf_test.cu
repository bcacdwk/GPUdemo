#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <mma.h>
#include <helper_cuda.h>
#include <helper_functions.h>
using namespace std;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

using namespace nvcuda;

// *******************************************************************
// KERNEL 2: Optimized WMMA GEMM with shared memory
// *******************************************************************

// Constants for the optimized kernel
#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK   8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define CHUNK_K 8
#define CHUNK_LINE_BYTES          (CHUNK_K * K * sizeof(uint8_t))
#define WARP_COPY_BYTES           (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES     (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)
#define GLOBAL_MEM_STRIDE N_GLOBAL
#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)
#define SKEW_UINT8 32

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B, const int *C, int *D, int M_GLOBAL_PARAM, int N_GLOBAL_PARAM, int K_GLOBAL_PARAM, int alpha, int beta)
{
    // This kernel is complex and will be adapted in a later step
    // For now, we just copy it to have it in the file.
    extern __shared__ uint8_t shmem[];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int *)shmem + (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    int *shmem_warp_stream_ptr = (int *)shmem + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    if (beta != 0) {
        beta /= alpha;
    }


    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / (N_GLOBAL_PARAM / N)) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % (N_GLOBAL_PARAM / N);

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= (M_GLOBAL_PARAM / M)) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx                 = (block_tile_i + warpId) * M * N_GLOBAL_PARAM + block_tile_j * N;
        const int   *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            if ( (gmem_idx + (N_GLOBAL_PARAM * i) + (laneId*4)) < (M_GLOBAL_PARAM * N_GLOBAL_PARAM) ) {
                typedef int4 copy_t;

                *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((copy_t *)(src_gmem_warp_stream_ptr + N_GLOBAL_PARAM * i) + laneId);
            }
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const uint8_t *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL_PARAM] + M * K_GLOBAL_PARAM * (warpId % 4) * 2)
                                               : (&B[block_tile_j * N * K_GLOBAL_PARAM] + N * K_GLOBAL_PARAM * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < (K_GLOBAL_PARAM / K); tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL_PARAM)
                           + (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                // Copy 16 bytes at once in each lane.
                *((int4 *)(shmem + shmem_idx * (CHUNK_K * K + SKEW_UINT8)) + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *)((uint8_t *)lane_ptr + K_GLOBAL_PARAM * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t         shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const uint8_t *tile_ptr    = shmem + shmem_idx_a * (CHUNK_K * K + SKEW_UINT8) + k_step * K;

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b      = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const uint8_t *tile_ptr = shmem + shmem_idx_b * (CHUNK_K * K + SKEW_UINT8) + k_step * K;

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
             if ( (gmem_idx + (N_GLOBAL_PARAM * i) + (laneId*4)) < (M_GLOBAL_PARAM * N_GLOBAL_PARAM) ) {
                *((int4 *)(dst_gmem_warp_stream_ptr + N_GLOBAL_PARAM * i) + laneId) =
                    *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
             }
        }

        __syncthreads();
    }
}
// *******************************************************************
// KERNEL 1: Simple WMMA GEMM without shared memory optimization
// *******************************************************************
// Performs an MxNxK GEMM (D = alpha * A * B + beta * C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm_imma kernel. It
// is designed for demonstration purposes only to show the CUDA WMMA API use without
// relying on availability of the shared memory.
__global__ void simple_wmma_gemm_imma(const uint8_t *a,
                                      const uint8_t *b,
                                      const int     *c,
                                      int           *d,
                                      int            m_ld,
                                      int            n_ld,
                                      int            k_ld,
                                      int            alpha,
                                      int            beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld; // A is row-major: M x K
    int ldb = k_ld; // B is col-major: K x N (stored as N x K)
    int ldc = n_ld; // C is row-major: M x N

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int>                   acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int>                   c_frag;

    wmma::fill_fragment(acc_frag, 0);

    // Loop over k
    for (int i = 0; i < k_ld; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;

        int bRow = warpN * 16;
        int bCol = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cRow = warpM * 16;
    int cCol = warpN * 16;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}

// Enum to select which kernel to run
enum GemmKernelType {
    SIMPLE,
    OPTIMIZED
};

// Unified performance test function for GEMM
void run_gemm_test(int M_size, int N_size, int K_size, GemmKernelType kernel_type, std::ofstream &csv_file) {
    std::string kernel_name = (kernel_type == SIMPLE) ? "simple_wmma_gemm" : "optimized_wmma_gemm";
    
    printf("\n=== Testing Kernel: %s, M=%d, N=%d, K=%d ===\n", kernel_name.c_str(), M_size, N_size, K_size);

    // Check WMMA basic requirements: M, N, K must be multiples of 16
    if (M_size % 16 != 0 || N_size % 16 != 0 || K_size % 16 != 0) {
        printf("Skipping kernel: Matrix dimensions must be multiples of 16 for WMMA.\n");
        csv_file << kernel_name << "," << M_size << "," << N_size << "," << K_size << ",N/A,N/A,N/A\n";
        return;
    }

    // Additional check for optimized kernel: requires 128-alignment for current implementation
    if (kernel_type == OPTIMIZED && (M_size % 128 != 0 || N_size % 128 != 0)) {
        printf("Skipping optimized kernel: Current implementation requires M,N to be multiples of 128.\n");
        csv_file << kernel_name << "," << M_size << "," << N_size << "," << K_size << ",N/A,N/A,N/A\n";
        return;
    }

    size_t bytes_a = (size_t)M_size * K_size * sizeof(uint8_t);
    size_t bytes_b = (size_t)K_size * N_size * sizeof(uint8_t);
    size_t bytes_c = (size_t)M_size * N_size * sizeof(int);
    size_t bytes_d = (size_t)M_size * N_size * sizeof(int);

    // Allocate host memory
    uint8_t *h_a = new uint8_t[bytes_a];
    uint8_t *h_b = new uint8_t[bytes_b];
    int *h_c = new int[bytes_c];

    // Initialize host matrices
    for (int i = 0; i < M_size * K_size; i++) h_a[i] = (uint8_t)(rand() % 3);
    for (int i = 0; i < K_size * N_size; i++) h_b[i] = (uint8_t)(rand() % 3);
    for (int i = 0; i < M_size * N_size; i++) h_c[i] = (rand() % 3);

    // Allocate device memory
    uint8_t *d_a, *d_b;
    int *d_c, *d_d;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));
    CUDA_CHECK(cudaMalloc(&d_d, bytes_d));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_d, 0, bytes_d));

    int alpha = 1;
    int beta = 1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    if (kernel_type == SIMPLE) {
        dim3 gridDim((M_size + 15) / 16, (N_size + 15) / 16);
        dim3 blockDim(32, 1); // Minimal block to launch
        simple_wmma_gemm_imma<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, M_size, N_size, K_size, alpha, beta);
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        // Calculate shared memory size for optimized kernel
        enum {
            SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                           M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
        };
        
        compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(d_a, d_b, d_c, d_d, M_size, N_size, K_size, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Performance measurement
    int profile_iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iters; ++i) {
        if (kernel_type == SIMPLE) {
            dim3 gridDim((M_size + 15) / 16, (N_size + 15) / 16);
            dim3 blockDim(32, 1);
            simple_wmma_gemm_imma<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, M_size, N_size, K_size, alpha, beta);
        } else {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            
            // Calculate shared memory size for optimized kernel
            enum {
                SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                               M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
            };
            
            compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(d_a, d_b, d_c, d_d, M_size, N_size, K_size, alpha, beta);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_kernel_time = milliseconds / profile_iters;

    double tops = (((double)M_size * N_size * K_size * 2) / (avg_kernel_time / 1000.0)) / 1e12;
    double bandwidth = ((bytes_a + bytes_b + bytes_c + bytes_d) / (avg_kernel_time / 1000.0)) / 1e9;

    printf("Avg Kernel Time: %.3f ms\n", avg_kernel_time);
    printf("Performance:     %.2f TOPS\n", tops);
    printf("Bandwidth:       %.2f GB/s\n", bandwidth);

    // Write to CSV
    csv_file << kernel_name << "," << M_size << "," << N_size << "," << K_size << "," << avg_kernel_time << "," << tops << "," << bandwidth << "\n";

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
    printf("=== IMMA Tensor Core GEMM Performance Test ===\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM72) architecture or higher.
    if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
        printf("immaTensorCoreGemm requires SM 7.2 or higher to use Tensor Cores.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    // Test logic will be added here.
    // Define the matrix sizes to test
    std::vector<int> sizes = {512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 8192, 16384};
    
    std::ofstream csv_file("imma_perf_results.csv");
    csv_file << "Kernel,M,N,K,Latency(ms),TOPS,Bandwidth(GB/s)\n";

    // Test all sizes
    for (int size : sizes) {
        run_gemm_test(size, size, size, SIMPLE, csv_file);
        run_gemm_test(size, size, size, OPTIMIZED, csv_file);
    }

    csv_file.close();
    printf("\nPerformance test finished. Results saved to imma_perf_results.csv\n");

    return 0;
}
