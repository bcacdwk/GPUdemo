/*
 * cuSPARSELt H100 多算法ID共享上下文 Bug 复现测试
 * 
 * 目的：证明在同一个cuSPARSELt handle下，多个算法ID的压缩权重
 *       在连续计算时会产生不一致的结果
 * 
 * 测试场景：
 *   场景A: 完全隔离 - 每个alg_id独立reset+init（应该一致）
 *   场景B: 共享handle - 同一个handle下先全部压缩，再全部计算（可能不一致）
 *   场景C: 共享handle+交替 - 压缩后立即计算（对比）
 * 
 * 编译运行:
 *   nvcc -o test_alg0_shared test_alg0_shared.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_alg0_shared
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

void compare_results(const char* label, int id1, const std::vector<C_t>& r1,
                     int id2, const std::vector<C_t>& r2) {
    size_t match = 0, total = r1.size();
    size_t first_diff = 0;
    bool found = false;
    for (size_t i = 0; i < total; ++i) {
        if (r1[i] == r2[i]) match++;
        else if (!found) { first_diff = i; found = true; }
    }
    
    printf("  [%s] id=%d vs id=%d: ", label, id1, id2);
    if (match == total) {
        printf("完全相同 (%zu/%zu)\n", match, total);
    } else {
        printf("不同! 匹配%zu/%zu, 首差位置%zu, 值:%d vs %d\n",
               match, total, first_diff, r1[first_diff], r2[first_diff]);
    }
}

// 场景B: 共享handle，先全部压缩，再全部计算（模拟原测试）
void test_shared_compress_then_compute(int M, int N, int K,
                                        const std::vector<AB_t>& hW,
                                        const std::vector<AB_t>& hA) {
    printf("\n=== 场景B: 共享handle，先全部压缩，再全部计算 ===\n");
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    size_t C_elems = N * M;
    
    AB_t *dW, *dA;
    C_t *dC;
    int *d_valid;
    CHECK_CUDA(cudaMalloc(&dW, W_size));
    CHECK_CUDA(cudaMalloc(&dA, A_size));
    CHECK_CUDA(cudaMalloc(&dC, C_size));
    CHECK_CUDA(cudaMalloc(&d_valid, sizeof(int)));
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
    
    const int num_test_ids = 3;
    int test_ids[] = {0, 1, 2};
    
    // 为每个算法ID创建独立的plan和压缩权重
    cusparseLtMatmulAlgSelection_t alg_sels[num_test_ids];
    cusparseLtMatmulPlan_t plans[num_test_ids];
    AB_t* compressed[num_test_ids];
    
    printf("步骤1: 依次压缩所有算法ID的权重\n");
    for (int i = 0; i < num_test_ids; ++i) {
        int alg_id = test_ids[i];
        
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sels[i],
            &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sels[i],
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plans[i], &matmul, &alg_sels[i]));
        
        size_t comp_size, comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plans[i],
            &comp_size, &comp_buf_size));
        CHECK_CUDA(cudaMalloc(&compressed[i], comp_size));
        
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plans[i], dW,
            compressed[i], buf, nullptr));
        if (buf) cudaFree(buf);
        
        printf("  alg_id=%d 压缩完成\n", alg_id);
    }
    
    // 依次计算
    printf("步骤2: 依次计算每个算法ID\n");
    std::vector<std::vector<C_t>> results(num_test_ids);
    
    for (int i = 0; i < num_test_ids; ++i) {
        int alg_id = test_ids[i];
        
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plans[i], &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plans[i], &alpha,
            compressed[i], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        results[i].resize(C_elems);
        CHECK_CUDA(cudaMemcpy(results[i].data(), dC, C_size, cudaMemcpyDeviceToHost));
        
        printf("  alg_id=%d 计算完成: 前3值 = %d, %d, %d\n",
               alg_id, results[i][0], results[i][1], results[i][2]);
    }
    
    // 比较结果
    printf("步骤3: 比较结果\n");
    compare_results("B", 0, results[0], 1, results[1]);
    compare_results("B", 1, results[1], 2, results[2]);
    
    // 清理
    for (int i = 0; i < num_test_ids; ++i) {
        cudaFree(compressed[i]);
        cusparseLtMatmulPlanDestroy(&plans[i]);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sels[i]);
    }
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC); cudaFree(d_valid);
}

// 场景C: 压缩后立即计算
void test_compress_and_compute_immediate(int M, int N, int K,
                                          const std::vector<AB_t>& hW,
                                          const std::vector<AB_t>& hA) {
    printf("\n=== 场景C: 共享handle，压缩后立即计算 ===\n");
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    size_t C_elems = N * M;
    
    AB_t *dW, *dA;
    C_t *dC;
    int *d_valid;
    CHECK_CUDA(cudaMalloc(&dW, W_size));
    CHECK_CUDA(cudaMalloc(&dA, A_size));
    CHECK_CUDA(cudaMalloc(&dC, C_size));
    CHECK_CUDA(cudaMalloc(&d_valid, sizeof(int)));
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
    
    const int num_test_ids = 3;
    int test_ids[] = {0, 1, 2};
    std::vector<std::vector<C_t>> results(num_test_ids);
    
    for (int i = 0; i < num_test_ids; ++i) {
        int alg_id = test_ids[i];
        
        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseLtMatmulPlan_t plan;
        
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel,
            &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
        
        size_t comp_size, comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
            &comp_size, &comp_buf_size));
        
        AB_t* comp_weight;
        CHECK_CUDA(cudaMalloc(&comp_weight, comp_size));
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dW,
            comp_weight, buf, nullptr));
        if (buf) cudaFree(buf);
        
        // 立即计算
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
            comp_weight, dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        results[i].resize(C_elems);
        CHECK_CUDA(cudaMemcpy(results[i].data(), dC, C_size, cudaMemcpyDeviceToHost));
        
        printf("  alg_id=%d: 压缩+计算完成, 前3值 = %d, %d, %d\n",
               alg_id, results[i][0], results[i][1], results[i][2]);
        
        cudaFree(comp_weight);
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    }
    
    printf("比较结果:\n");
    compare_results("C", 0, results[0], 1, results[1]);
    compare_results("C", 1, results[1], 2, results[2]);
    
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC); cudaFree(d_valid);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("cuSPARSELt H100 多算法ID共享上下文 Bug 复现测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("========================================================\n");
    
    int M = 16, N = 3840, K = 2560;
    printf("\n测试维度: M=%d, N=%d, K=%d\n", M, N, K);
    
    std::srand(42);
    std::vector<AB_t> hW(N * K);
    std::vector<AB_t> hA(M * K);
    for (size_t i = 0; i < hW.size(); ++i) hW[i] = std::rand() % 256 - 128;
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = std::rand() % 256 - 128;
    
    test_shared_compress_then_compute(M, N, K, hW, hA);
    test_compress_and_compute_immediate(M, N, K, hW, hA);
    
    printf("\n========================================================\n");
    printf("测试完成\n");
    printf("========================================================\n");
    
    return 0;
}
