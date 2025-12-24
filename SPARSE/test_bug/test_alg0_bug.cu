/*
 * cuSPARSELt H100 算法ID=0 Bug 复现测试
 * 
 * 目的：证明在H100上，算法ID=0计算的golden结果与其他算法ID不同
 * 
 * 测试方法：
 *   1. 对同一个输入，分别用不同的算法ID压缩+计算
 *   2. 每个算法ID独立创建handle、plan、压缩权重（完全隔离）
 *   3. 比较结果：ID=0 vs ID=1, ID=1 vs ID=2
 * 
 * 编译运行:
 *   nvcc -o test_alg0_bug test_alg0_bug.cu -lcusparseLt
 *   CUDA_VISIBLE_DEVICES=1 ./test_alg0_bug
 */

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

// 使用INT8输入，INT32输出
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

// 用指定的算法ID计算一次矩阵乘法，返回结果
// 每次调用都是完全独立的：新建handle、plan、压缩权重
std::vector<C_t> compute_with_alg_id(
    int alg_id,
    int M, int N, int K,
    const std::vector<AB_t>& hW,  // sparse weight [N,K]
    const std::vector<AB_t>& hA   // dense activation [M,K]
) {
    printf("\n=== 计算 alg_id=%d ===\n", alg_id);
    
    // 重置GPU状态，确保干净环境
    CHECK_CUDA(cudaDeviceReset());
    CHECK_CUDA(cudaSetDevice(0));
    
    size_t W_size = N * K * sizeof(AB_t);
    size_t A_size = M * K * sizeof(AB_t);
    size_t C_size = N * M * sizeof(C_t);
    size_t C_elems = N * M;
    
    // 分配设备内存
    AB_t *dW, *dA;
    C_t *dC;
    int *d_valid;
    CHECK_CUDA(cudaMalloc(&dW, W_size));
    CHECK_CUDA(cudaMalloc(&dA, A_size));
    CHECK_CUDA(cudaMalloc(&dC, C_size));
    CHECK_CUDA(cudaMalloc(&d_valid, sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, C_size));
    
    // 初始化cuSPARSELt
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));
    
    // 创建矩阵描述符
    cusparseLtMatDescriptor_t matW, matA, matC;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matW,
        N, K, K, 16, CUDA_R_8I, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matA,
        M, K, K, 16, CUDA_R_8I, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC,
        N, M, M, 16, CUDA_R_32I, CUSPARSE_ORDER_ROW));
    
    // 创建matmul描述符: C = W * A^T
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &matW, &matA, &matC, &matC, CUSPARSE_COMPUTE_32I));
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle, &matmul,
        CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW)));
    
    // 剪枝
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dW, dW,
        CUSPARSELT_PRUNE_SPMMA_TILE, nullptr));
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dW, d_valid, nullptr));
    
    int is_valid = 0;
    CHECK_CUDA(cudaMemcpy(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (is_valid != 0) {
        printf("  剪枝检查失败!\n");
        exit(1);
    }
    
    // 创建算法选择，设置指定的alg_id
    cusparseLtMatmulAlgSelection_t alg_sel;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel,
        &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
    
    // 创建plan
    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));
    
    // 获取压缩尺寸并压缩
    size_t compressed_size, compress_buffer_size;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
        &compressed_size, &compress_buffer_size));
    
    AB_t* dW_compressed;
    void* compress_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dW_compressed, compressed_size));
    if (compress_buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&compress_buffer, compress_buffer_size));
    }
    
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dW,
        dW_compressed, compress_buffer, nullptr));
    
    printf("  压缩完成: compressed_size=%zu\n", compressed_size);
    
    // 获取workspace并计算
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }
    
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha,
        dW_compressed, dA, &beta, dC, dC, d_workspace, nullptr, 0));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果
    std::vector<C_t> result(C_elems);
    CHECK_CUDA(cudaMemcpy(result.data(), dC, C_size, cudaMemcpyDeviceToHost));
    
    printf("  计算完成: 前5个值 = %d, %d, %d, %d, %d\n",
           result[0], result[1], result[2], result[3], result[4]);
    
    // 清理
    if (d_workspace) cudaFree(d_workspace);
    if (compress_buffer) cudaFree(compress_buffer);
    cudaFree(dW_compressed);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);
    cudaFree(dW);
    cudaFree(dA);
    cudaFree(dC);
    cudaFree(d_valid);
    
    return result;
}

// 比较两个结果
void compare_results(const char* name1, int id1, const std::vector<C_t>& r1,
                     const char* name2, int id2, const std::vector<C_t>& r2) {
    if (r1.size() != r2.size()) {
        printf("\n[比较] %s(id=%d) vs %s(id=%d): 大小不同!\n", name1, id1, name2, id2);
        return;
    }
    
    size_t match = 0, total = r1.size();
    size_t first_diff_idx = 0;
    bool found_diff = false;
    
    for (size_t i = 0; i < total; ++i) {
        if (r1[i] == r2[i]) {
            match++;
        } else if (!found_diff) {
            first_diff_idx = i;
            found_diff = true;
        }
    }
    
    printf("\n========================================\n");
    printf("[比较] alg_id=%d vs alg_id=%d\n", id1, id2);
    printf("========================================\n");
    
    if (match == total) {
        printf("  结果: 完全相同 (%zu/%zu)\n", match, total);
    } else {
        printf("  结果: 不同! 匹配 %zu/%zu (%.2f%%)\n", 
               match, total, 100.0 * match / total);
        printf("  首个差异位置: %zu\n", first_diff_idx);
        printf("    id=%d 的值: %d\n", id1, r1[first_diff_idx]);
        printf("    id=%d 的值: %d\n", id2, r2[first_diff_idx]);
        
        // 统计差异分布
        int diff_count_small = 0, diff_count_large = 0;
        for (size_t i = 0; i < total; ++i) {
            if (r1[i] != r2[i]) {
                int diff = abs(r1[i] - r2[i]);
                if (diff < 1000) diff_count_small++;
                else diff_count_large++;
            }
        }
        printf("  差异分布: 小差异(<1000)=%d, 大差异(>=1000)=%d\n",
               diff_count_small, diff_count_large);
    }
}

void run_test(int M, int N, int K, const char* test_name) {
    printf("\n");
    printf("########################################################\n");
    printf("# 测试: %s\n", test_name);
    printf("# 维度: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# 计算: R[%d,%d] = W[%d,%d] * A[%d,%d]^T\n", N, M, N, K, M, K);
    printf("########################################################\n");
    
    // 生成固定的随机数据
    std::srand(42);
    std::vector<AB_t> hW(N * K);
    std::vector<AB_t> hA(M * K);
    
    for (size_t i = 0; i < hW.size(); ++i) {
        hW[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }
    for (size_t i = 0; i < hA.size(); ++i) {
        hA[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }
    
    // 分别用 alg_id=0, 1, 2 计算
    auto result_0 = compute_with_alg_id(0, M, N, K, hW, hA);
    auto result_1 = compute_with_alg_id(1, M, N, K, hW, hA);
    auto result_2 = compute_with_alg_id(2, M, N, K, hW, hA);
    
    // 比较结果
    compare_results("alg", 0, result_0, "alg", 1, result_1);
    compare_results("alg", 1, result_1, "alg", 2, result_2);
    compare_results("alg", 0, result_0, "alg", 2, result_2);
}

int main() {
    // 获取GPU信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================================\n");
    printf("cuSPARSELt H100 算法ID=0 Bug 复现测试\n");
    printf("========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("目的: 证明alg_id=0的计算结果与其他ID不同\n");
    printf("方法: 每个alg_id完全独立计算(含cudaDeviceReset)\n");
    printf("========================================================\n");
    
    // 测试不同的维度配置
    run_test(16, 3840, 2560, "Wqkv (small batch)");
    run_test(16, 2560, 2560, "Wo (small batch)");
    run_test(256, 3840, 2560, "Wqkv (medium batch)");
    
    printf("\n========================================================\n");
    printf("测试完成\n");
    printf("========================================================\n");
    
    return 0;
}
