/*
 * cuSPARSELt H100 多Plan共存导致首次计算结果错误 Bug 复现
 * 
 * 核心发现：当多个matmul plan同时存在时，第一个执行的会得到不同的结果
 * 
 * 测试场景：
 *   场景D: 创建plan0, plan1, plan2 -> 压缩0,1,2 -> 计算0,1,2 -> 比较
 *   场景E: 创建plan0, plan1, plan2 -> 压缩0,1,2 -> 计算2,1,0 -> 比较（反序）
 *   场景F: 只创建一个plan，分三次测试
 * 
 * 编译运行:
 *   nvcc -o test_multi_plan test_multi_plan.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_multi_plan
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

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("cuSPARSELt H100 多Plan共存 Bug 复现测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    
    const int M = 16, N = 3840, K = 2560;
    const int NUM_ALG = 3;  // 测试3个算法ID
    int alg_ids[NUM_ALG] = {0, 1, 2};
    
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

    // ========================================================
    // 场景D: 创建所有plan -> 压缩所有 -> 按顺序计算
    // ========================================================
    printf("\n=== 场景D: 创建所有plan -> 压缩所有 -> 按顺序计算 ===\n");
    {
        CHECK_CUDA(cudaDeviceReset());
        CHECK_CUDA(cudaSetDevice(0));
        
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
        
        // 创建所有plan
        cusparseLtMatmulAlgSelection_t alg_sel[NUM_ALG];
        cusparseLtMatmulPlan_t plan[NUM_ALG];
        
        printf("创建所有plan...\n");
        for (int i = 0; i < NUM_ALG; i++) {
            CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel[i],
                &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
            CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel[i],
                CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[i], sizeof(alg_ids[i])));
            CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan[i], &matmul, &alg_sel[i]));
            printf("  plan[%d] (alg=%d) 创建完成\n", i, alg_ids[i]);
        }
        
        // 压缩所有权重
        AB_t* dW_compressed[NUM_ALG];
        printf("压缩所有权重...\n");
        for (int i = 0; i < NUM_ALG; i++) {
            size_t comp_size, comp_buf_size;
            CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i],
                &comp_size, &comp_buf_size));
            CHECK_CUDA(cudaMalloc(&dW_compressed[i], comp_size));
            void* buf = nullptr;
            if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
            CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW,
                dW_compressed[i], buf, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());
            if (buf) cudaFree(buf);
            printf("  dW_compressed[%d] 压缩完成\n", i);
        }
        
        // 按顺序计算 (0 -> 1 -> 2)
        std::vector<std::vector<C_t>> results(NUM_ALG, std::vector<C_t>(N * M));
        printf("按顺序计算 (0 -> 1 -> 2)...\n");
        for (int i = 0; i < NUM_ALG; i++) {
            size_t ws_size;
            CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[i], &ws_size));
            void* d_ws = nullptr;
            if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
            
            CHECK_CUDA(cudaMemset(dC, 0, C_size));
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[i], &alpha,
                dW_compressed[i], dA, &beta, dC, dC, d_ws, nullptr, 0));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            CHECK_CUDA(cudaMemcpy(results[i].data(), dC, C_size, cudaMemcpyDeviceToHost));
            if (d_ws) cudaFree(d_ws);
            printf("  计算[%d] (alg=%d) 完成, 前5值: %d, %d, %d, %d, %d\n",
                   i, alg_ids[i], results[i][0], results[i][1], 
                   results[i][2], results[i][3], results[i][4]);
        }
        
        // 比较结果
        printf("比较结果:\n");
        for (int i = 0; i < NUM_ALG - 1; i++) {
            size_t match = 0;
            size_t first_diff = 0;
            for (size_t j = 0; j < results[i].size(); j++) {
                if (results[i][j] == results[i+1][j]) match++;
                else if (first_diff == 0) first_diff = j;
            }
            if (match == results[i].size()) {
                printf("  [D] alg=%d vs alg=%d: 完全相同 ✓\n", alg_ids[i], alg_ids[i+1]);
            } else {
                printf("  [D] alg=%d vs alg=%d: 不同! 匹配%zu/%zu, 首差位置%zu\n",
                       alg_ids[i], alg_ids[i+1], match, results[i].size(), first_diff);
                printf("      位置%zu: %d vs %d\n", first_diff,
                       results[i][first_diff], results[i+1][first_diff]);
            }
        }
        
        // 清理
        for (int i = 0; i < NUM_ALG; i++) {
            cudaFree(dW_compressed[i]);
            cusparseLtMatmulPlanDestroy(&plan[i]);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel[i]);
        }
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dC);
    }

    // ========================================================
    // 场景E: 创建所有plan -> 压缩所有 -> 反序计算
    // ========================================================
    printf("\n=== 场景E: 创建所有plan -> 压缩所有 -> 反序计算 (2->1->0) ===\n");
    {
        CHECK_CUDA(cudaDeviceReset());
        CHECK_CUDA(cudaSetDevice(0));
        
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
        
        cusparseLtMatmulAlgSelection_t alg_sel[NUM_ALG];
        cusparseLtMatmulPlan_t plan[NUM_ALG];
        
        printf("创建所有plan...\n");
        for (int i = 0; i < NUM_ALG; i++) {
            CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel[i],
                &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
            CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel[i],
                CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[i], sizeof(alg_ids[i])));
            CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan[i], &matmul, &alg_sel[i]));
        }
        
        AB_t* dW_compressed[NUM_ALG];
        printf("压缩所有权重...\n");
        for (int i = 0; i < NUM_ALG; i++) {
            size_t comp_size, comp_buf_size;
            CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan[i],
                &comp_size, &comp_buf_size));
            CHECK_CUDA(cudaMalloc(&dW_compressed[i], comp_size));
            void* buf = nullptr;
            if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
            CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan[i], dW,
                dW_compressed[i], buf, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());
            if (buf) cudaFree(buf);
        }
        
        // 反序计算 (2 -> 1 -> 0)
        int order[NUM_ALG] = {2, 1, 0};
        std::vector<std::vector<C_t>> results(NUM_ALG, std::vector<C_t>(N * M));
        printf("反序计算 (2 -> 1 -> 0)...\n");
        for (int idx = 0; idx < NUM_ALG; idx++) {
            int i = order[idx];
            size_t ws_size;
            CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan[i], &ws_size));
            void* d_ws = nullptr;
            if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
            
            CHECK_CUDA(cudaMemset(dC, 0, C_size));
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan[i], &alpha,
                dW_compressed[i], dA, &beta, dC, dC, d_ws, nullptr, 0));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            CHECK_CUDA(cudaMemcpy(results[i].data(), dC, C_size, cudaMemcpyDeviceToHost));
            if (d_ws) cudaFree(d_ws);
            printf("  计算[%d] (alg=%d) 完成, 前5值: %d, %d, %d, %d, %d\n",
                   i, alg_ids[i], results[i][0], results[i][1], 
                   results[i][2], results[i][3], results[i][4]);
        }
        
        printf("比较结果:\n");
        for (int i = 0; i < NUM_ALG - 1; i++) {
            size_t match = 0;
            size_t first_diff = 0;
            for (size_t j = 0; j < results[i].size(); j++) {
                if (results[i][j] == results[i+1][j]) match++;
                else if (first_diff == 0) first_diff = j;
            }
            if (match == results[i].size()) {
                printf("  [E] alg=%d vs alg=%d: 完全相同 ✓\n", alg_ids[i], alg_ids[i+1]);
            } else {
                printf("  [E] alg=%d vs alg=%d: 不同! 匹配%zu/%zu\n",
                       alg_ids[i], alg_ids[i+1], match, results[i].size());
            }
        }
        
        for (int i = 0; i < NUM_ALG; i++) {
            cudaFree(dW_compressed[i]);
            cusparseLtMatmulPlanDestroy(&plan[i]);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel[i]);
        }
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dC);
    }

    // ========================================================
    // 场景F: 单独创建每个plan，一个一个测试（对照组）
    // ========================================================
    printf("\n=== 场景F: 单独创建每个plan，一个一个测试（对照组）===\n");
    {
        std::vector<std::vector<C_t>> results(NUM_ALG, std::vector<C_t>(N * M));
        
        for (int test_idx = 0; test_idx < NUM_ALG; test_idx++) {
            CHECK_CUDA(cudaDeviceReset());
            CHECK_CUDA(cudaSetDevice(0));
            
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
                CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_ids[test_idx], sizeof(alg_ids[test_idx])));
            CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
            
            size_t comp_size, comp_buf_size;
            CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
                &comp_size, &comp_buf_size));
            AB_t* dW_compressed;
            CHECK_CUDA(cudaMalloc(&dW_compressed, comp_size));
            void* buf = nullptr;
            if (comp_buf_size > 0) CHECK_CUDA(cudaMalloc(&buf, comp_buf_size));
            CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dW,
                dW_compressed, buf, nullptr));
            CHECK_CUDA(cudaDeviceSynchronize());
            if (buf) cudaFree(buf);
            
            size_t ws_size;
            CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &ws_size));
            void* d_ws = nullptr;
            if (ws_size > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_size));
            
            CHECK_CUDA(cudaMemset(dC, 0, C_size));
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
                dW_compressed, dA, &beta, dC, dC, d_ws, nullptr, 0));
            CHECK_CUDA(cudaDeviceSynchronize());
            
            CHECK_CUDA(cudaMemcpy(results[test_idx].data(), dC, C_size, cudaMemcpyDeviceToHost));
            printf("  [F] 独立测试 alg=%d 完成, 前5值: %d, %d, %d, %d, %d\n",
                   alg_ids[test_idx], results[test_idx][0], results[test_idx][1], 
                   results[test_idx][2], results[test_idx][3], results[test_idx][4]);
            
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
        
        printf("比较独立测试结果:\n");
        for (int i = 0; i < NUM_ALG - 1; i++) {
            size_t match = 0;
            for (size_t j = 0; j < results[i].size(); j++) {
                if (results[i][j] == results[i+1][j]) match++;
            }
            if (match == results[i].size()) {
                printf("  [F] alg=%d vs alg=%d: 完全相同 ✓\n", alg_ids[i], alg_ids[i+1]);
            } else {
                printf("  [F] alg=%d vs alg=%d: 不同! 匹配%zu/%zu\n",
                       alg_ids[i], alg_ids[i+1], match, results[i].size());
            }
        }
    }

    printf("\n========================================================\n");
    printf("测试完成\n");
    printf("========================================================\n");
    printf("\n结论:\n");
    printf("- 场景D/E: 如果多plan共存时结果不同 -> 这是BUG\n");
    printf("- 场景F: 独立测试应该完全相同 -> 正确行为\n");
    printf("========================================================\n");
    
    return 0;
}
