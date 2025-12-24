/*
 * cuSPARSELt H100 压缩权重被Matmul破坏 Bug 复现测试
 * 
 * 目的：证明cusparseLtMatmul会破坏压缩权重缓冲区
 * 
 * 测试方法：
 *   1. 压缩权重
 *   2. 保存压缩权重的拷贝
 *   3. 执行matmul
 *   4. 比较压缩权重是否被修改
 * 
 * 编译运行:
 *   nvcc -o test_weight_corruption test_weight_corruption.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_weight_corruption
 */

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

using AB_t = int8_t;
using C_t  = int;

#define CHECK_CUDA(func) {                                                  \
    cudaError_t status = (func);                                            \
    if (status != cudaSuccess) {                                            \
        printf("CUDA Error at line %d: %s\n", __LINE__,                     \
               cudaGetErrorString(status));                                 \
        exit(1);                                                            \
    }                                                                       \
}

#define CHECK_CUSPARSE(func) {                                              \
    cusparseStatus_t status = (func);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        printf("cuSPARSE Error at line %d: %s\n", __LINE__,                 \
               cusparseLtGetErrorString(status));                           \
        exit(1);                                                            \
    }                                                                       \
}

void test_weight_corruption(int alg_id, int M, int N, int K) {
    printf("\n=== 测试 alg_id=%d 的压缩权重是否被matmul破坏 ===\n", alg_id);
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    
    // 生成测试数据
    std::srand(42);
    std::vector<AB_t> hW(N * K);
    std::vector<AB_t> hA(M * K);
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = std::rand() % 256 - 128;
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = std::rand() % 256 - 128;
    
    AB_t *dW, *dA;
    C_t *dC;
    CHECK_CUDA(cudaMalloc(&dW, W_size));
    CHECK_CUDA(cudaMalloc(&dA, A_size));
    CHECK_CUDA(cudaMalloc(&dC, C_size));
    CHECK_CUDA(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
    
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));
    
    cusparseLtMatDescriptor_t matW, matA, matC;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matW,
        N, K, K, 16, CUDA_R_8I, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matA,
        M, K, K, 16, CUDA_R_8I, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC,
        N, M, M, 16, CUDA_R_32I, CUSPARSE_ORDER_ROW));
    
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &matW, &matA, &matC, &matC, CUSPARSE_COMPUTE_32I));
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle, &matmul,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW)));
    
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dW, dW,
        CUSPARSELT_PRUNE_SPMMA_TILE, nullptr));
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel,
        &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
    
    // 获取压缩尺寸
    size_t comp_size, comp_buf_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
        &comp_size, &comp_buf_size));
    printf("压缩权重大小: %zu bytes\n", comp_size);
    
    // 分配压缩权重
    AB_t* dW_compressed;
    CHECK_CUDA(cudaMalloc(&dW_compressed, comp_size));
    void* buf = nullptr;
    if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
    
    // 压缩
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dW,
        dW_compressed, buf, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (buf) cudaFree(buf);
    
    // 保存压缩权重的拷贝
    std::vector<AB_t> compressed_before(comp_size);
    CHECK_CUDA(cudaMemcpy(compressed_before.data(), dW_compressed, comp_size, 
                          cudaMemcpyDeviceToHost));
    printf("压缩后前8字节: %d %d %d %d %d %d %d %d\n",
           compressed_before[0], compressed_before[1], compressed_before[2], 
           compressed_before[3], compressed_before[4], compressed_before[5],
           compressed_before[6], compressed_before[7]);
    
    // 执行第一次matmul
    size_t ws_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &ws_size));
    void* d_ws = nullptr;
    if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
    
    CHECK_CUDA(cudaMemset(dC, 0, C_size));
    float alpha = 1.0f, beta = 0.0f;
    
    printf("\n执行第1次matmul...\n");
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
        dW_compressed, dA, &beta, dC, dC, d_ws, nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 获取第一次结果
    std::vector<C_t> result1(N * M);
    CHECK_CUDA(cudaMemcpy(result1.data(), dC, C_size, cudaMemcpyDeviceToHost));
    printf("第1次结果前5值: %d, %d, %d, %d, %d\n",
           result1[0], result1[1], result1[2], result1[3], result1[4]);
    
    // 检查压缩权重是否被修改
    std::vector<AB_t> compressed_after1(comp_size);
    CHECK_CUDA(cudaMemcpy(compressed_after1.data(), dW_compressed, comp_size, 
                          cudaMemcpyDeviceToHost));
    
    size_t diff_count = 0;
    size_t first_diff = 0;
    for (size_t i = 0; i < comp_size; ++i) {
        if (compressed_before[i] != compressed_after1[i]) {
            if (diff_count == 0) first_diff = i;
            diff_count++;
        }
    }
    
    if (diff_count == 0) {
        printf("第1次matmul后: 压缩权重未被修改 ✓\n");
    } else {
        printf("第1次matmul后: 压缩权重被修改! 差异%zu字节, 首个位置%zu\n", 
               diff_count, first_diff);
        printf("  位置%zu: 前=%d, 后=%d\n", first_diff, 
               compressed_before[first_diff], compressed_after1[first_diff]);
    }
    
    // 执行第二次matmul（用同样的压缩权重）
    printf("\n执行第2次matmul（用同样的压缩权重）...\n");
    CHECK_CUDA(cudaMemset(dC, 0, C_size));
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
        dW_compressed, dA, &beta, dC, dC, d_ws, nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<C_t> result2(N * M);
    CHECK_CUDA(cudaMemcpy(result2.data(), dC, C_size, cudaMemcpyDeviceToHost));
    printf("第2次结果前5值: %d, %d, %d, %d, %d\n",
           result2[0], result2[1], result2[2], result2[3], result2[4]);
    
    // 比较两次结果
    size_t match = 0;
    for (size_t i = 0; i < result1.size(); ++i) {
        if (result1[i] == result2[i]) match++;
    }
    
    if (match == result1.size()) {
        printf("两次计算结果: 完全相同 ✓\n");
    } else {
        printf("两次计算结果: 不同! 匹配%zu/%zu\n", match, result1.size());
    }
    
    // 检查第二次后的压缩权重
    std::vector<AB_t> compressed_after2(comp_size);
    CHECK_CUDA(cudaMemcpy(compressed_after2.data(), dW_compressed, comp_size, 
                          cudaMemcpyDeviceToHost));
    
    diff_count = 0;
    for (size_t i = 0; i < comp_size; ++i) {
        if (compressed_after1[i] != compressed_after2[i]) diff_count++;
    }
    if (diff_count > 0) {
        printf("第2次matmul后: 压缩权重又被修改! 差异%zu字节\n", diff_count);
    }
    
    // 清理
    if (d_ws) cudaFree(d_ws);
    cudaFree(dW_compressed);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("cuSPARSELt H100 压缩权重被Matmul破坏 Bug 复现测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("========================================================\n");
    
    int M = 16, N = 3840, K = 2560;
    printf("测试维度: M=%d, N=%d, K=%d\n", M, N, K);
    
    // 分别测试不同的算法ID
    test_weight_corruption(0, M, N, K);
    test_weight_corruption(1, M, N, K);
    test_weight_corruption(2, M, N, K);
    
    printf("\n========================================================\n");
    printf("测试完成\n");
    printf("========================================================\n");
    
    return 0;
}
