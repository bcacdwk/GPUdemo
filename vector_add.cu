#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函数定义：向量加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    // 分配主机内存并初始化
    int n = 1024; // 定义向量长度，可根据需要修改
    size_t size = n * sizeof(float); // 计算所需内存大小
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 设置线程块和网格维度
    // CUDA 1.0 要求每个线程块的线程数不超过512
    int threadsPerBlock = 256;
    int blocksPerGrid = (1024 + threadsPerBlock - 1) / threadsPerBlock;

    // 执行核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, 1024);

    // 将结果从设备复制到主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
}
