/*
 * cuSPARSELt H100 Bug 解耦测试
 * 
 * 目的：区分两个可能的bug来源
 *   假设A：alg_id=0 自身有bug，单独反复执行会破坏weight
 *   假设B：多plan共存导致污染，单独使用时正常
 * 
 * 测试场景：
 *   场景1：单独使用 alg_id=0，反复执行matmul，检查weight和结果
 *   场景2：单独使用 alg_id=1，反复执行matmul，检查weight和结果（对照组）
 *   场景3：多plan共存，但只用plan[0]计算，检查weight
 *   场景4：多plan共存，先用plan[1]计算，再用plan[0]计算
 * 
 * 编译运行:
 *   nvcc -o test_bug_decouple test_bug_decouple.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_bug_decouple
 */

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

size_t compare_buffers(const std::vector<AB_t>& a, const std::vector<AB_t>& b) {
    size_t diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i]) diff++;
    return diff;
}

size_t compare_results(const std::vector<C_t>& a, const std::vector<C_t>& b) {
    size_t diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i]) diff++;
    return diff;
}

const int M = 16, N = 3840, K = 2560;

// ============================================================
// 场景1/2：单独使用一个算法ID，反复执行
// ============================================================
void test_single_alg_repeated(int alg_id) {
    printf("\n");
    printf("########################################################\n");
    printf("# 场景%d: 单独使用 alg_id=%d，反复执行3次\n", alg_id == 0 ? 1 : 2, alg_id);
    printf("########################################################\n");
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    
    std::srand(42);
    std::vector<AB_t> hW(N * K), hA(M * K);
    for (auto& v : hW) v = std::rand() % 256 - 128;
    for (auto& v : hA) v = std::rand() % 256 - 128;
    
    AB_t *dW, *dA; C_t *dC;
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
    
    // 只创建一个plan
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel,
        &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
    
    // 压缩
    size_t comp_size, comp_buf_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan, &comp_size, &comp_buf_size));
    AB_t* dW_comp;
    CHECK_CUDA(cudaMalloc(&dW_comp, comp_size));
    void* buf = nullptr;
    if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dW, dW_comp, buf, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (buf) cudaFree(buf);
    
    printf("压缩完成: %zu bytes\n", comp_size);
    
    // 保存原始压缩权重
    std::vector<AB_t> original_comp(comp_size);
    CHECK_CUDA(cudaMemcpy(original_comp.data(), dW_comp, comp_size, cudaMemcpyDeviceToHost));
    printf("原始压缩权重前8字节: %d %d %d %d %d %d %d %d\n",
           original_comp[0], original_comp[1], original_comp[2], original_comp[3],
           original_comp[4], original_comp[5], original_comp[6], original_comp[7]);
    
    // 准备workspace
    size_t ws_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &ws_size));
    void* d_ws = nullptr;
    if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
    
    float alpha = 1.0f, beta = 0.0f;
    std::vector<std::vector<C_t>> results(3);
    
    // 反复执行3次
    for (int iter = 0; iter < 3; iter++) {
        printf("\n--- 第 %d 次执行 ---\n", iter + 1);
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
            dW_comp, dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 获取结果
        results[iter].resize(N * M);
        CHECK_CUDA(cudaMemcpy(results[iter].data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("计算结果前5值: %d, %d, %d, %d, %d\n",
               results[iter][0], results[iter][1], results[iter][2],
               results[iter][3], results[iter][4]);
        
        // 检查压缩权重
        std::vector<AB_t> current_comp(comp_size);
        CHECK_CUDA(cudaMemcpy(current_comp.data(), dW_comp, comp_size, cudaMemcpyDeviceToHost));
        
        size_t weight_diff = compare_buffers(original_comp, current_comp);
        if (weight_diff == 0) {
            printf("压缩权重: 未变化 ✓\n");
        } else {
            printf("压缩权重: 被修改! 差异 %zu / %zu 字节 (%.1f%%)\n",
                   weight_diff, comp_size, 100.0 * weight_diff / comp_size);
        }
        
        // 比较结果
        if (iter > 0) {
            size_t result_diff = compare_results(results[0], results[iter]);
            if (result_diff == 0) {
                printf("与第1次结果: 完全相同 ✓\n");
            } else {
                printf("与第1次结果: 不同! 差异 %zu / %zu\n", result_diff, results[0].size());
            }
        }
    }
    
    // 清理
    if (d_ws) cudaFree(d_ws);
    cudaFree(dW_comp);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC);
}

// ============================================================
// 场景3：多plan共存，但只用plan[0]计算
// ============================================================
void test_multi_plan_only_use_plan0() {
    printf("\n");
    printf("########################################################\n");
    printf("# 场景3: 多plan共存 (plan0,1,2)，但只用plan[0]计算\n");
    printf("########################################################\n");
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    
    std::srand(42);
    std::vector<AB_t> hW(N * K), hA(M * K);
    for (auto& v : hW) v = std::rand() % 256 - 128;
    for (auto& v : hA) v = std::rand() % 256 - 128;
    
    AB_t *dW, *dA; C_t *dC;
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
    
    // 创建3个plan
    const int NUM_PLANS = 3;
    int alg_ids[NUM_PLANS] = {0, 1, 2};
    cusparseLtMatmulAlgSelection_t alg_sel[NUM_PLANS];
    cusparseLtMatmulPlan_t plan[NUM_PLANS];
    AB_t* dW_comp[NUM_PLANS];
    size_t comp_size;
    
    printf("创建3个plan并压缩...\n");
    for (int i = 0; i < NUM_PLANS; i++) {
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel[i],
            &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel[i],
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[i], sizeof(alg_ids[i])));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan[i], &matmul, &alg_sel[i]));
        
        size_t comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i], &comp_size, &comp_buf_size));
        CHECK_CUDA(cudaMalloc(&dW_comp[i], comp_size));
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW, dW_comp[i], buf, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        if (buf) cudaFree(buf);
        printf("  plan[%d] 创建并压缩完成\n", i);
    }
    
    // 保存所有压缩权重
    std::vector<std::vector<AB_t>> backup(NUM_PLANS);
    for (int i = 0; i < NUM_PLANS; i++) {
        backup[i].resize(comp_size);
        CHECK_CUDA(cudaMemcpy(backup[i].data(), dW_comp[i], comp_size, cudaMemcpyDeviceToHost));
    }
    
    printf("\n只用 plan[0] 执行3次matmul:\n");
    
    size_t ws_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[0], &ws_size));
    void* d_ws = nullptr;
    if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
    
    float alpha = 1.0f, beta = 0.0f;
    
    for (int iter = 0; iter < 3; iter++) {
        printf("\n--- 第 %d 次执行 plan[0] ---\n", iter + 1);
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[0], &alpha,
            dW_comp[0], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        std::vector<C_t> result(N * M);
        CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("计算结果前5值: %d, %d, %d, %d, %d\n",
               result[0], result[1], result[2], result[3], result[4]);
        
        // 检查所有压缩权重
        for (int i = 0; i < NUM_PLANS; i++) {
            std::vector<AB_t> current(comp_size);
            CHECK_CUDA(cudaMemcpy(current.data(), dW_comp[i], comp_size, cudaMemcpyDeviceToHost));
            size_t diff = compare_buffers(backup[i], current);
            if (diff == 0) {
                printf("  compressed[%d]: 未变化 ✓\n", i);
            } else {
                printf("  compressed[%d]: 被修改! 差异 %zu 字节\n", i, diff);
                backup[i] = current;  // 更新backup
            }
        }
    }
    
    // 清理
    if (d_ws) cudaFree(d_ws);
    for (int i = 0; i < NUM_PLANS; i++) {
        cudaFree(dW_comp[i]);
        cusparseLtMatmulPlanDestroy(&plan[i]);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel[i]);
    }
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dC);
}

// ============================================================
// 场景4：多plan共存，先用plan[1]计算，再用plan[0]
// ============================================================
void test_multi_plan_use_1_then_0() {
    printf("\n");
    printf("########################################################\n");
    printf("# 场景4: 多plan共存，先用plan[1]，再用plan[0]\n");
    printf("########################################################\n");
    
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    
    std::srand(42);
    std::vector<AB_t> hW(N * K), hA(M * K);
    for (auto& v : hW) v = std::rand() % 256 - 128;
    for (auto& v : hA) v = std::rand() % 256 - 128;
    
    AB_t *dW, *dA; C_t *dC;
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
    
    // 创建2个plan
    const int NUM_PLANS = 2;
    int alg_ids[NUM_PLANS] = {0, 1};
    cusparseLtMatmulAlgSelection_t alg_sel[NUM_PLANS];
    cusparseLtMatmulPlan_t plan[NUM_PLANS];
    AB_t* dW_comp[NUM_PLANS];
    size_t comp_size;
    
    printf("创建 plan[0] 和 plan[1] 并压缩...\n");
    for (int i = 0; i < NUM_PLANS; i++) {
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel[i],
            &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel[i],
            CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[i], sizeof(alg_ids[i])));
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan[i], &matmul, &alg_sel[i]));
        
        size_t comp_buf_size;
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i], &comp_size, &comp_buf_size));
        CHECK_CUDA(cudaMalloc(&dW_comp[i], comp_size));
        void* buf = nullptr;
        if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
        CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW, dW_comp[i], buf, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());
        if (buf) cudaFree(buf);
    }
    
    // 保存压缩权重
    std::vector<std::vector<AB_t>> backup(NUM_PLANS);
    for (int i = 0; i < NUM_PLANS; i++) {
        backup[i].resize(comp_size);
        CHECK_CUDA(cudaMemcpy(backup[i].data(), dW_comp[i], comp_size, cudaMemcpyDeviceToHost));
    }
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 先用 plan[1]
    printf("\n--- 先执行 plan[1] ---\n");
    {
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[1], &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[1], &alpha,
            dW_comp[1], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        std::vector<C_t> result(N * M);
        CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("plan[1] 结果前5值: %d, %d, %d, %d, %d\n",
               result[0], result[1], result[2], result[3], result[4]);
        
        for (int i = 0; i < NUM_PLANS; i++) {
            std::vector<AB_t> current(comp_size);
            CHECK_CUDA(cudaMemcpy(current.data(), dW_comp[i], comp_size, cudaMemcpyDeviceToHost));
            size_t diff = compare_buffers(backup[i], current);
            printf("  compressed[%d]: %s\n", i, diff == 0 ? "未变化 ✓" : "被修改!");
            if (diff > 0) backup[i] = current;
        }
    }
    
    // 再用 plan[0]
    printf("\n--- 再执行 plan[0] ---\n");
    {
        size_t ws_size;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[0], &ws_size));
        void* d_ws = nullptr;
        if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
        
        CHECK_CUDA(cudaMemset(dC, 0, C_size));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[0], &alpha,
            dW_comp[0], dA, &beta, dC, dC, d_ws, nullptr, 0));
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (d_ws) cudaFree(d_ws);
        
        std::vector<C_t> result(N * M);
        CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
        printf("plan[0] 结果前5值: %d, %d, %d, %d, %d\n",
               result[0], result[1], result[2], result[3], result[4]);
        
        for (int i = 0; i < NUM_PLANS; i++) {
            std::vector<AB_t> current(comp_size);
            CHECK_CUDA(cudaMemcpy(current.data(), dW_comp[i], comp_size, cudaMemcpyDeviceToHost));
            size_t diff = compare_buffers(backup[i], current);
            printf("  compressed[%d]: %s\n", i, diff == 0 ? "未变化 ✓" : "被修改!");
        }
    }
    
    // 清理
    for (int i = 0; i < NUM_PLANS; i++) {
        cudaFree(dW_comp[i]);
        cusparseLtMatmulPlanDestroy(&plan[i]);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel[i]);
    }
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
    printf("cuSPARSELt H100 Bug 解耦测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("维度: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================================\n");
    
    // 场景1: 单独使用 alg_id=0
    test_single_alg_repeated(0);
    
    // 场景2: 单独使用 alg_id=1（对照组）
    test_single_alg_repeated(1);
    
    // 场景3: 多plan共存，只用plan[0]
    test_multi_plan_only_use_plan0();
    
    // 场景4: 多plan共存，先用plan[1]再用plan[0]
    test_multi_plan_use_1_then_0();
    
    printf("\n========================================================\n");
    printf("测试完成 - 结果分析\n");
    printf("========================================================\n");
    printf("如果场景1有问题，场景2没问题 → alg_id=0自身有bug\n");
    printf("如果场景1/2都没问题，场景3/4有问题 → 多plan共存导致污染\n");
    printf("========================================================\n");
    
    return 0;
}
