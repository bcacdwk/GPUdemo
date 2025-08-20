#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>  // for min function

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel函数
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = x[index] + y[index];
    }
}

// 性能测试函数
void runPerformanceTest(int N, int blockSize, int gridSize, const char* configName) {
    printf("\n=== %s ===\n", configName);
    
    size_t bytes = N * sizeof(float);
    printf("Vector size: %d elements\n", N);
    printf("Data type: float\n");
    printf("Memory usage: %.2f MB\n", bytes / (1024.0 * 1024.0));
    printf("Block size: %d threads\n", blockSize);
    printf("Grid size: %d blocks\n", gridSize);
    
    // 分配内存
    float *h_x, *h_y;
    h_x = new float[N];
    h_y = new float[N];
    
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    
    // 数据传输
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    
    // 创建事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up iterations
    int warmup_iters = 2;
    printf("Warm-up iters: %d\n", warmup_iters);
    for (int i = 0; i < warmup_iters; i++) {
        add<<<gridSize, blockSize>>>(N, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 性能测试
    int profile_iters = 100;
    printf("Profile iters: %d\n", profile_iters);
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < profile_iters; i++) {
        add<<<gridSize, blockSize>>>(N, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // 计算时间
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_kernel_time = milliseconds / profile_iters;
    
    printf("Avg kernel time: %.6f ms\n", avg_kernel_time);
    
    // 计算吞吐量
    // 每次kernel执行: 读取N个float + 读取N个float + 写入N个float = 3N个float
    float total_bytes = 3.0f * N * sizeof(float);
    float throughput_gb_s = (total_bytes / (avg_kernel_time / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    printf("Throughput: %.2f GB/s\n", throughput_gb_s);
    
    // 计算GFLOPS (每个元素一次加法运算)
    float total_ops = (float)N;
    float gflops = (total_ops / (avg_kernel_time / 1000.0)) / 1e9;
    printf("Compute perf: %.2f GFLOP/s\n", gflops);
    
    // 验证结果
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    
    bool passed = true;
    // 正确的期望值计算：初始值2.0 + 执行次数 * 每次加的值1.0
    int total_executions = warmup_iters + profile_iters;
    float expected = 2.0f + (float)total_executions * 1.0f;
    
    printf("Expected value: %.1f (initial 2.0 + %d executions * 1.0)\n", expected, total_executions);
    
    for (int i = 0; i < min(100, N); i++) { // 检查前100个元素
        if (fabsf(h_y[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: got %.6f, expected %.6f\n", i, h_y[i], expected);
            passed = false;
            break;
        }
    }
    
    if (passed) {
        printf("Sample values: h_y[0]=%.1f, h_y[99]=%.1f\n", h_y[0], h_y[99]);
    }
    
    printf("Verification: %s\n", passed ? "Passed" : "Failed");
    
    // 清理内存
    delete[] h_x;
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(void)
{
    int N = 1 << 20; // 1M elements，与图像中一致
    
    printf("=== GPU Performance Test ===\n");
    
    // 测试不同配置
    runPerformanceTest(N, 256, (N + 255) / 256, "<<<256, 256>>>");
    runPerformanceTest(N, 1, N, "<<<1,1>>>");
    
    // GPU满载测试
    int optimal_threads = 256;
    int optimal_blocks = (N + optimal_threads - 1) / optimal_threads;
    runPerformanceTest(N, optimal_threads, optimal_blocks, "GPU 满载");
    
    return 0;
}
