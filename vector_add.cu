#include <stdio.h>
#include <stdlib.h> // For malloc/free
#include <math.h>   // For fabs
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA 核函数定义：向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 定义向量长度
    int n = 1024;
    size_t size = n * sizeof(float);
    int i; // 声明循环变量
    
    // 声明所有变量
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int threadsPerBlock = 256;
    int blocksPerGrid;
    int success = 1;

    // 1. 分配主机内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return 1;
    }

    // 2. 初始化主机数据
    for (i = 0; i < n; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // 3. 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // 4. 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 5. 设置线程块和网格维度
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 6. 执行核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // 等待核函数执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. 将结果从设备复制到主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 8. 验证结果
    for (i = 0; i < 10; ++i) { // 只检查前10个结果作为示例
        printf("h_a[%d]=%.1f, h_b[%d]=%.1f, h_c[%d]=%.1f\n", i, h_a[i], i, h_b[i], i, h_c[i]);
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            printf("Verification FAILED at index %d!\n", i);
            success = 0;
        }
    }
    if(success) {
        printf("Verification PASSED!\n");
    }

    // 9. 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}