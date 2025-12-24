/*
 * cuSPARSELt H100 多Plan共存时压缩权重被破坏测试
 * 
 * 目的：证明在多个plan共存的情况下，执行matmul会破坏压缩权重
 * 
 * 测试方法：
 *   1. 创建多个plan (plan0, plan1, plan2)
 *   2. 分别压缩得到 compressed[0], compressed[1], compressed[2]
 *   3. 保存所有压缩权重的拷贝 (backup)
 *   4. 执行 matmul(plan0, compressed[0])
 *   5. 检查所有compressed是否被修改
 *   6. 执行 matmul(plan1, compressed[1])
 *   7. 再次检查所有compressed是否被修改
 * 
 * 编译运行:
 *   nvcc -o test_weight_corruption_multiplan test_weight_corruption_multiplan.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_weight_corruption_multiplan
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

// 比较两个buffer，返回差异字节数和首个差异位置
size_t compare_buffers(const std::vector<AB_t>& a, const std::vector<AB_t>& b, 
                       size_t* first_diff = nullptr) {
    size_t diff_count = 0;
    bool found = false;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) {
            if (!found && first_diff) {
                *first_diff = i;
                found = true;
            }
            diff_count++;
        }
    }
    return diff_count;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("cuSPARSELt H100 多Plan共存时压缩权重被破坏测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    
    const int M = 16, N = 3840, K = 2560;
    const int NUM_PLANS = 3;
    int alg_ids[NUM_PLANS] = {0, 1, 2};
    
    printf("测试维度: M=%d, N=%d, K=%d\n", M, N, K);
    printf("测试算法ID: %d, %d, %d\n", alg_ids[0], alg_ids[1], alg_ids[2]);
    printf("========================================================\n");
    
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
    
    // 创建描述符
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
    
    // ============================================================
    // 步骤1: 创建所有plan
    // ============================================================
    printf("\n[步骤1] 创建所有plan...\n");
    
    cusparseLtMatmulAlgSelection_t alg_sel[NUM_PLANS];
    cusparseLtMatmulPlan_t plan[NUM_PLANS];
    
    for (int i = 0; i < NUM_PLANS; i++) {
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel[i],
            &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel[i],
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[i], sizeof(alg_ids[i])));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan[i], &matmul, &alg_sel[i]));
        printf("  plan[%d] (alg=%d) 创建完成\n", i, alg_ids[i]);
    }
    
    // ============================================================
    // 步骤2: 压缩所有权重
    // ============================================================
    printf("\n[步骤2] 压缩所有权重...\n");
    
    AB_t* dW_compressed[NUM_PLANS];
    size_t comp_sizes[NUM_PLANS];
    
    for (int i = 0; i < NUM_PLANS; i++) {
        size_t comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i],
            &comp_sizes[i], &comp_buf_size));
        CHECK_CUDA(cudaMalloc(&dW_compressed[i], comp_sizes[i]));
        
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW,
            dW_compressed[i], buf, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        if (buf) cudaFree(buf);
        
        printf("  compressed[%d]: %zu bytes\n", i, comp_sizes[i]);
    }
    
    // ============================================================
    // 步骤3: 保存所有压缩权重的备份
    // ============================================================
    printf("\n[步骤3] 保存压缩权重备份...\n");
    
    std::vector<std::vector<AB_t>> backup(NUM_PLANS);
    for (int i = 0; i < NUM_PLANS; i++) {
        backup[i].resize(comp_sizes[i]);
        CHECK_CUDA(cudaMemcpy(backup[i].data(), dW_compressed[i], 
                              comp_sizes[i], cudaMemcpyDeviceToHost));
        printf("  backup[%d]: 前8字节 = %d %d %d %d %d %d %d %d\n", i,
               backup[i][0], backup[i][1], backup[i][2], backup[i][3],
               backup[i][4], backup[i][5], backup[i][6], backup[i][7]);
    }
    
    // ============================================================
    // 步骤4: 依次执行matmul并检查权重变化
    // ============================================================
    printf("\n[步骤4] 依次执行matmul并检查压缩权重变化...\n");
    
    float alpha = 1.0f, beta = 0.0f;
    
    for (int exec_idx = 0; exec_idx < NUM_PLANS; exec_idx++) {
        printf("\n--- 执行 matmul(plan[%d], compressed[%d]) ---\n", exec_idx, exec_idx);
        
        // 获取workspace
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[exec_idx], &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        // 执行matmul
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[exec_idx], &alpha,
            dW_compressed[exec_idx], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        // 获取计算结果的前几个值
        std::vector<C_t> result(N * M);
        CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("  计算结果前5值: %d, %d, %d, %d, %d\n",
               result[0], result[1], result[2], result[3], result[4]);
        
        // 检查所有压缩权重是否被修改
        printf("  检查压缩权重变化:\n");
        for (int check_idx = 0; check_idx < NUM_PLANS; check_idx++) {
            std::vector<AB_t> current(comp_sizes[check_idx]);
            CHECK_CUDA(cudaMemcpy(current.data(), dW_compressed[check_idx],
                                  comp_sizes[check_idx], cudaMemcpyDeviceToHost));
            
            size_t first_diff = 0;
            size_t diff_count = compare_buffers(backup[check_idx], current, &first_diff);
            
            if (diff_count == 0) {
                printf("    compressed[%d]: 未变化 ✓\n", check_idx);
            } else {
                printf("    compressed[%d]: 被修改! 差异%zu字节, 首差位置%zu\n",
                       check_idx, diff_count, first_diff);
                printf("      位置%zu: 原=%d, 现=%d\n", first_diff,
                       backup[check_idx][first_diff], current[first_diff]);
                
                // 更新backup为当前值，以便跟踪后续变化
                backup[check_idx] = current;
            }
        }
    }
    
    // ============================================================
    // 步骤5: 重新加载原始备份，测试不同执行顺序
    // ============================================================
    printf("\n========================================================\n");
    printf("[步骤5] 重新测试：先执行plan[1]，再执行plan[0]\n");
    printf("========================================================\n");
    
    // 重新压缩所有权重
    printf("\n重新压缩所有权重...\n");
    for (int i = 0; i < NUM_PLANS; i++) {
        size_t comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i],
            &comp_sizes[i], &comp_buf_size));
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW,
            dW_compressed[i], buf, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        if (buf) cudaFree(buf);
    }
    
    // 保存新的备份
    for (int i = 0; i < NUM_PLANS; i++) {
        CHECK_CUDA(cudaMemcpy(backup[i].data(), dW_compressed[i], 
                              comp_sizes[i], cudaMemcpyDeviceToHost));
    }
    
    // 反序执行: 1 -> 0 -> 2
    int reverse_order[NUM_PLANS] = {1, 0, 2};
    
    for (int idx = 0; idx < NUM_PLANS; idx++) {
        int exec_idx = reverse_order[idx];
        printf("\n--- 执行 matmul(plan[%d], compressed[%d]) ---\n", exec_idx, exec_idx);
        
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[exec_idx], &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[exec_idx], &alpha,
            dW_compressed[exec_idx], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        std::vector<C_t> result(N * M);
        CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("  计算结果前5值: %d, %d, %d, %d, %d\n",
               result[0], result[1], result[2], result[3], result[4]);
        
        printf("  检查压缩权重变化:\n");
        for (int check_idx = 0; check_idx < NUM_PLANS; check_idx++) {
            std::vector<AB_t> current(comp_sizes[check_idx]);
            CHECK_CUDA(cudaMemcpy(current.data(), dW_compressed[check_idx],
                                  comp_sizes[check_idx], cudaMemcpyDeviceToHost));
            
            size_t first_diff = 0;
            size_t diff_count = compare_buffers(backup[check_idx], current, &first_diff);
            
            if (diff_count == 0) {
                printf("    compressed[%d]: 未变化 ✓\n", check_idx);
            } else {
                printf("    compressed[%d]: 被修改! 差异%zu字节, 首差位置%zu\n",
                       check_idx, diff_count, first_diff);
                backup[check_idx] = current;
            }
        }
    }
    
    // ============================================================
    // 清理
    // ============================================================
    for (int i = 0; i < NUM_PLANS; i++) {
        cudaFree(dW_compressed[i]);
        cusparseLtMatmulPlanDestroy(&plan[i]);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel[i]);
    }
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC);
    
    printf("\n========================================================\n");
    printf("测试完成\n");
    printf("========================================================\n");
    
    return 0;
}
