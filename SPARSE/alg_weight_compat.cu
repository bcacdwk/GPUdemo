/*
算法-权重兼容性测试  ：完整的交叉测试（两阶段方法）

核心改进：
  - 阶段1：遍历所有ID，计算对角线golden结果（weight_id == alg_id）
  - 阶段2：遍历所有(weight_id, alg_id)组合，与对应golden比较
  - 每次压缩一个权重，测试完所有算法后立即释放，避免状态污染

测试逻辑：
  1. 阶段1（收集Golden）：
     - for weight_id in 0..max:
         压缩权重 -> 用相同alg_id计算 -> 保存golden -> 释放资源
  2. 阶段2（交叉测试）：
     - for weight_id in 0..max:
         压缩权重 -> for alg_id in 0..max: 计算并与golden[weight_id]比较 -> 释放资源
  3. 比较所有golden之间的一致性

编译运行:
nvcc -o alg_weight_compat alg_weight_compat.cu -lcusparseLt && ./alg_weight_compat
*/

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

using AB_t      = int8_t;
using C_t       = int;
using COMPUTE_t = int;

template <typename value_t>
struct cuda_type { };

template <>
struct cuda_type<int8_t> {
    static constexpr cudaDataType value = CUDA_R_8I;
};

template <>
struct cuda_type<int> {
    static constexpr cudaDataType value = CUDA_R_32I;
};

template <typename value_t>
struct cusparse_compute_type { };

template <>
struct cusparse_compute_type<int> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32I;
};

// (N, K) pairs 配置
struct NKPair {
    int n;
    int k;
    std::string name;
};

// 获取GPU名称
std::string getGpuName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string fullName = prop.name;
    
    std::string shortName = fullName;
    size_t nvidia_pos = fullName.find("NVIDIA ");
    if (nvidia_pos != std::string::npos) {
        shortName = fullName.substr(nvidia_pos + 7);
    }
    
    size_t end_pos = shortName.find_first_of(" -");
    if (end_pos != std::string::npos) {
        shortName = shortName.substr(0, end_pos);
    }
    
    if (shortName.empty()) {
        shortName = fullName;
        for (char& c : shortName) {
            if (c == ' ' || c == '-' || c == '/') c = '_';
        }
    }
    
    return shortName;
}

// 创建目录
bool createDirectory(const std::string& path) {
    size_t pos = 0;
    std::string dir;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        dir = path.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

#define CHECK_CUDA(func)                                                        \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        std::printf("CUDA API 错误 (行 %d): %s (%d)\n",                          \
                   __LINE__, cudaGetErrorString(status), status);               \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        std::printf("cuSPARSE API 错误 (行 %d): %s (%d)\n",                      \
                   __LINE__, cusparseLtGetErrorString(status), status);         \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

// 比较两个结果矩阵
bool compare_results(const std::vector<C_t>& a, const std::vector<C_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

size_t compare_results_detailed(const std::vector<C_t>& a, const std::vector<C_t>& b, 
                                 size_t* first_diff_idx = nullptr) {
    if (a.size() != b.size()) return 0;
    size_t match_count = 0;
    bool found_first_diff = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) {
            match_count++;
        } else if (!found_first_diff && first_diff_idx) {
            *first_diff_idx = i;
            found_first_diff = true;
        }
    }
    return match_count;
}

// 对单个 (m, n, k) 配置执行兼容性测试
int runCompatTest(int m, int n, int k, const std::string& csvPath, bool verbose = true) {
    std::srand(42);  // 固定种子

    if (verbose) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "测试维度: M=" << m << ", N=" << n << ", K=" << k << std::endl;
        std::cout << "计算: R[" << n << "," << m << "] = W[" << n << "," << k 
                  << "] * A^T[" << m << "," << k << "]" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // 矩阵配置
    auto orderA        = CUSPARSE_ORDER_ROW;
    auto orderB        = CUSPARSE_ORDER_ROW;
    auto orderC        = CUSPARSE_ORDER_ROW;
    auto opA           = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto opB           = CUSPARSE_OPERATION_TRANSPOSE;
    auto type_AB       = cuda_type<AB_t>::value;
    auto type_C        = cuda_type<C_t>::value;
    auto compute_type  = cusparse_compute_type<COMPUTE_t>::value;
    unsigned alignment = 16;

    int num_A_rows = n;
    int num_A_cols = k;
    int num_B_rows = m;
    int num_B_cols = k;
    int num_C_rows = n;
    int num_C_cols = m;

    int lda = num_A_cols;
    int ldb = num_B_cols;
    int ldc = num_C_cols;

    size_t A_elems = static_cast<size_t>(n) * k;
    size_t B_elems = static_cast<size_t>(m) * k;
    size_t C_elems = static_cast<size_t>(n) * m;

    size_t A_size = A_elems * sizeof(AB_t);
    size_t B_size = B_elems * sizeof(AB_t);
    size_t C_size = C_elems * sizeof(C_t);

    // 初始化主机数据
    std::vector<AB_t> hA(A_elems);
    std::vector<AB_t> hB(B_elems);
    std::vector<C_t>  hC_zero(C_elems, 0);

    for (size_t i = 0; i < A_elems; ++i) {
        hA[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }
    for (size_t i = 0; i < B_elems; ++i) {
        hB[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    // 分配设备内存
    AB_t *dA = nullptr, *dB = nullptr;
    C_t *dC = nullptr;
    int *d_valid = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA), A_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dB), B_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dC), C_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_valid), sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size, cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;

    // 初始化 cuSPARSELt
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));

    // 创建矩阵描述符
    cusparseLtMatDescriptor_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA,
                                                       num_A_rows, num_A_cols,
                                                       lda, alignment, type_AB,
                                                       orderA, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB,
                                                  num_B_rows, num_B_cols,
                                                  ldb, alignment, type_AB,
                                                  orderB));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC,
                                                  num_C_rows, num_C_cols,
                                                  ldc, alignment, type_C,
                                                  orderC));

    // 创建matmul描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                                   opA, opB, &matA, &matB,
                                                   &matC, &matC, compute_type));
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle, &matmul,
                                                     CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
                                                     &dA, sizeof(dA)));

    // 剪枝
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                        CUSPARSELT_PRUNE_SPMMA_TILE, stream));
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, d_valid, stream));

    int is_valid = 0;
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (is_valid != 0) {
        std::cerr << "错误：矩阵剪枝失败！" << std::endl;
        return EXIT_FAILURE;
    }

    // 获取算法ID上限
    int max_alg_id = -1;
    {
        cusparseLtMatmulAlgSelection_t alg_sel_tmp;
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel_tmp,
                                 &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel_tmp,
                                 CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                                 &max_alg_id, sizeof(max_alg_id)));
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp));
    }

    std::cout << "算法ID范围: 0 ~ " << max_alg_id << " (共 " << (max_alg_id + 1) << " 个算法)" << std::endl;

    const int num_algs = max_alg_id + 1;

    // 结果存储
    std::vector<std::vector<C_t>> golden_results(num_algs);
    std::vector<bool> golden_valid(num_algs, false);
    std::vector<std::vector<int>> compat_matrix(num_algs, std::vector<int>(num_algs, -1));
    std::vector<C_t> test_result(C_elems);

    // ============================================================
    // 阶段1：收集所有Golden结果（对角线）
    // ============================================================
    std::cout << "[阶段1] 收集Golden..." << std::flush;

    for (int id = 0; id < num_algs; ++id) {
        // 创建算法选择和plan
        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseLtMatmulPlan_t plan;
        AB_t* compressed_weight = nullptr;

        cusparseStatus_t status = cusparseLtMatmulAlgSelectionInit(
            &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            continue;
        }

        status = cusparseLtMatmulAlgSetAttribute(
            &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &id, sizeof(id));
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            continue;
        }

        status = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            continue;
        }

        // 获取压缩尺寸并压缩
        size_t compressed_size = 0, compressed_buffer_size = 0;
        status = cusparseLtSpMMACompressedSize(&handle, &plan, 
                                               &compressed_size, &compressed_buffer_size);
        if (status != CUSPARSE_STATUS_SUCCESS || compressed_size == 0) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            continue;
        }

        cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(&compressed_weight), 
                                              compressed_size);
        if (cuda_status != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            continue;
        }

        void* compress_buffer = nullptr;
        if (compressed_buffer_size > 0) {
            cudaMalloc(&compress_buffer, compressed_buffer_size);
        }

        status = cusparseLtSpMMACompress(&handle, &plan, dA, 
                                          compressed_weight, compress_buffer, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (compress_buffer) cudaFree(compress_buffer);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(compressed_weight);
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            continue;
        }

        // 获取workspace并执行计算
        size_t workspace_size = 0;
        cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
        void* d_workspace = nullptr;
        if (workspace_size > 0) cudaMalloc(&d_workspace, workspace_size);

        CHECK_CUDA(cudaMemcpy(dC, hC_zero.data(), C_size, cudaMemcpyHostToDevice));

        status = cusparseLtMatmul(
            &handle, &plan, &alpha, compressed_weight, dB,
            &beta, dC, dC, d_workspace, nullptr, 0);
        
        CHECK_CUDA(cudaDeviceSynchronize());

        if (d_workspace) cudaFree(d_workspace);

        if (status == CUSPARSE_STATUS_SUCCESS) {
            golden_results[id].resize(C_elems);
            CHECK_CUDA(cudaMemcpy(golden_results[id].data(), dC, C_size, cudaMemcpyDeviceToHost));
            golden_valid[id] = true;
            compat_matrix[id][id] = 1;  // 对角线自己和自己兼容
        } else {
            std::cout << "\n  [警告] ID " << id << ": 计算失败 (" 
                      << cusparseLtGetErrorString(status) << ")" << std::flush;
        }

        // 清理本次迭代的资源
        cudaFree(compressed_weight);
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    }

    // 统计有效golden数量
    int valid_count = 0;
    for (int i = 0; i < num_algs; ++i) {
        if (golden_valid[i]) valid_count++;
    }
    std::cout << " 完成 (" << valid_count << "/" << num_algs << ")" << std::endl;

    // 显示golden参考值
    int first_valid = -1;
    for (int i = 0; i < num_algs; ++i) {
        if (golden_valid[i]) { first_valid = i; break; }
    }
    if (first_valid >= 0) {
        std::cout << "  Golden参考值(ID" << first_valid << "): " 
                  << golden_results[first_valid][0] << ", " 
                  << golden_results[first_valid][1] << ", " 
                  << golden_results[first_valid][2] << std::endl;
    }

    if (valid_count == 0) {
        std::cerr << "错误：没有有效的Golden结果！" << std::endl;
        return EXIT_FAILURE;
    }

    // ============================================================
    // 阶段2：交叉测试（非对角线）
    // ============================================================
    std::cout << "[阶段2] 交叉测试..." << std::flush;
    int cross_test_errors = 0;

    for (int weight_id = 0; weight_id < num_algs; ++weight_id) {
        if (!golden_valid[weight_id]) continue;  // 跳过无效的权重ID

        // 使用 weight_id 创建plan并压缩权重
        cusparseLtMatmulAlgSelection_t weight_alg_sel;
        cusparseLtMatmulPlan_t weight_plan;
        AB_t* compressed_weight = nullptr;

        cusparseStatus_t status = cusparseLtMatmulAlgSelectionInit(
            &handle, &weight_alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (status != CUSPARSE_STATUS_SUCCESS) continue;

        status = cusparseLtMatmulAlgSetAttribute(
            &handle, &weight_alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &weight_id, sizeof(weight_id));
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&weight_alg_sel);
            continue;
        }

        status = cusparseLtMatmulPlanInit(&handle, &weight_plan, &matmul, &weight_alg_sel);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&weight_alg_sel);
            continue;
        }

        size_t compressed_size = 0, compressed_buffer_size = 0;
        cusparseLtSpMMACompressedSize(&handle, &weight_plan, 
                                       &compressed_size, &compressed_buffer_size);
        
        cudaMalloc(reinterpret_cast<void**>(&compressed_weight), compressed_size);

        void* compress_buffer = nullptr;
        if (compressed_buffer_size > 0) cudaMalloc(&compress_buffer, compressed_buffer_size);

        status = cusparseLtSpMMACompress(&handle, &weight_plan, dA, 
                                          compressed_weight, compress_buffer, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        if (compress_buffer) cudaFree(compress_buffer);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(compressed_weight);
            cusparseLtMatmulPlanDestroy(&weight_plan);
            cusparseLtMatmulAlgSelectionDestroy(&weight_alg_sel);
            continue;
        }

        // 遍历所有算法ID（跳过对角线）
        for (int alg_id = 0; alg_id < num_algs; ++alg_id) {
            if (alg_id == weight_id) continue;  // 跳过对角线（已在阶段1完成）
            if (!golden_valid[alg_id]) continue;  // 跳过无效的算法ID

            // 创建 alg_id 对应的plan
            cusparseLtMatmulAlgSelection_t alg_sel;
            cusparseLtMatmulPlan_t alg_plan;

            status = cusparseLtMatmulAlgSelectionInit(
                &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                compat_matrix[weight_id][alg_id] = -1;
                continue;
            }

            status = cusparseLtMatmulAlgSetAttribute(
                &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id));
            if (status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                compat_matrix[weight_id][alg_id] = -1;
                continue;
            }

            status = cusparseLtMatmulPlanInit(&handle, &alg_plan, &matmul, &alg_sel);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                compat_matrix[weight_id][alg_id] = -1;
                continue;
            }

            // 获取workspace
            size_t workspace_size = 0;
            cusparseLtMatmulGetWorkspace(&handle, &alg_plan, &workspace_size);
            void* d_workspace = nullptr;
            if (workspace_size > 0) cudaMalloc(&d_workspace, workspace_size);

            // 执行计算
            CHECK_CUDA(cudaMemcpy(dC, hC_zero.data(), C_size, cudaMemcpyHostToDevice));

            status = cusparseLtMatmul(
                &handle, &alg_plan, &alpha, compressed_weight, dB,
                &beta, dC, dC, d_workspace, nullptr, 0);
            
            CHECK_CUDA(cudaDeviceSynchronize());

            if (d_workspace) cudaFree(d_workspace);

            if (status == CUSPARSE_STATUS_SUCCESS) {
                CHECK_CUDA(cudaMemcpy(test_result.data(), dC, C_size, cudaMemcpyDeviceToHost));
                
                // 与 golden[weight_id] 比较（相同权重应产生相同结果）
                bool match = compare_results(test_result, golden_results[weight_id]);
                compat_matrix[weight_id][alg_id] = match ? 1 : 0;
                
                if (!match) {
                    cross_test_errors++;
                    size_t first_diff = 0;
                    size_t match_cnt = compare_results_detailed(test_result, 
                        golden_results[weight_id], &first_diff);
                    std::cout << "\n  [不匹配] W" << weight_id << "+A" << alg_id 
                              << ": " << match_cnt << "/" << C_elems 
                              << " 首差@" << first_diff << std::flush;
                }
            } else {
                compat_matrix[weight_id][alg_id] = -1;
                cross_test_errors++;
                std::cout << "\n  [失败] W" << weight_id << "+A" << alg_id 
                          << ": " << cusparseLtGetErrorString(status) << std::flush;
            }

            cusparseLtMatmulPlanDestroy(&alg_plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        }

        // 清理当前权重的资源
        cudaFree(compressed_weight);
        cusparseLtMatmulPlanDestroy(&weight_plan);
        cusparseLtMatmulAlgSelectionDestroy(&weight_alg_sel);
    }
    std::cout << " 完成" << (cross_test_errors > 0 ? "" : " (全部通过)") << std::endl;

    // ============================================================
    // Golden一致性检查（只输出不一致的）
    // ============================================================
    int base_golden = -1;
    for (int i = 0; i < num_algs; ++i) {
        if (golden_valid[i]) { base_golden = i; break; }
    }

    int golden_match_count = 0;
    int golden_total_count = 0;
    int golden_mismatch_count = 0;
    
    for (int i = 0; i < num_algs; ++i) {
        if (!golden_valid[i]) continue;
        golden_total_count++;
        
        if (i == base_golden) {
            golden_match_count++;
            continue;
        }

        bool match = compare_results(golden_results[base_golden], golden_results[i]);
        if (match) {
            golden_match_count++;
        } else {
            golden_mismatch_count++;
            size_t first_diff = 0;
            size_t match_cnt = compare_results_detailed(golden_results[base_golden], 
                                                         golden_results[i], &first_diff);
            std::cout << "  [Golden不一致] golden[" << base_golden << "] != golden[" << i << "] "
                      << "(匹配: " << match_cnt << "/" << C_elems 
                      << ", 首差@" << first_diff << ")" << std::endl;
        }
    }
    
    std::cout << "[Golden一致性] " << golden_match_count << "/" << golden_total_count;
    if (golden_mismatch_count > 0) {
        std::cout << " (" << golden_mismatch_count << "个不一致!)";
    }
    std::cout << std::endl;

    // ============================================================
    // 输出兼容性矩阵
    // ============================================================

    // 打印到控制台
    std::cout << "\n兼容性矩阵 (行=权重ID, 列=算法ID):" << std::endl;
    std::cout << "      ";
    for (int a = 0; a < num_algs; ++a) {
        if (golden_valid[a]) {
            if (a < 10) std::cout << " ";
            std::cout << "A" << a << " ";
        }
    }
    std::cout << std::endl;

    for (int w = 0; w < num_algs; ++w) {
        if (!golden_valid[w]) continue;
        std::cout << "W";
        if (w < 10) std::cout << " ";
        std::cout << w << ":  ";
        for (int a = 0; a < num_algs; ++a) {
            if (!golden_valid[a]) continue;
            int val = compat_matrix[w][a];
            if (val == 1) std::cout << " 1 ";
            else if (val == 0) std::cout << " 0 ";
            else std::cout << " - ";
        }
        std::cout << std::endl;
    }

    // 输出到CSV
    std::ofstream csv(csvPath);
    if (!csv.is_open()) {
        std::cerr << "无法创建结果文件 " << csvPath << std::endl;
        return EXIT_FAILURE;
    }

    csv << "WeightID";
    for (int a = 0; a < num_algs; ++a) {
        if (golden_valid[a]) csv << ",A" << a;
    }
    csv << "\n";

    for (int w = 0; w < num_algs; ++w) {
        if (!golden_valid[w]) continue;
        csv << "W" << w;
        for (int a = 0; a < num_algs; ++a) {
            if (!golden_valid[a]) continue;
            csv << "," << compat_matrix[w][a];
        }
        csv << "\n";
    }
    csv.close();

    // 统计
    int diagonal_match = 0, off_diagonal_match = 0, total_tests = 0;
    for (int w = 0; w < num_algs; ++w) {
        if (!golden_valid[w]) continue;
        for (int a = 0; a < num_algs; ++a) {
            if (!golden_valid[a]) continue;
            if (compat_matrix[w][a] < 0) continue;
            
            total_tests++;
            if (w == a) {
                if (compat_matrix[w][a] == 1) diagonal_match++;
            } else {
                if (compat_matrix[w][a] == 1) off_diagonal_match++;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "统计结果:" << std::endl;
    std::cout << "  有效算法数: " << valid_count << " / " << num_algs << std::endl;
    std::cout << "  对角线匹配(应全为1): " << diagonal_match << " / " << valid_count << std::endl;
    std::cout << "  非对角线匹配(可混用): " << off_diagonal_match << " / " 
              << (total_tests - valid_count) << std::endl;
    std::cout << "  结果已写入: " << csvPath << std::endl;
    std::cout << "========================================" << std::endl;

    // 清理
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtDestroy(&handle);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(d_valid);

    return EXIT_SUCCESS;
}

int main() {
    // 配置参数
    std::vector<int> m_list = {16, 256, 2048};
    
    std::vector<NKPair> nk_pairs = {
        {3840, 2560, "Wqkv"},
        {2560, 2560, "Wo"},
        {13824, 2560, "W13"},
        {2560, 6912, "W2"}
    };
    
    std::string gpuName = getGpuName();
    std::cout << "========================================" << std::endl;
    std::cout << "算法-权重兼容性测试 (两阶段方法)" << std::endl;
    std::cout << "检测到GPU: " << gpuName << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::string baseDir = "alg_weight_compat_matrix";
    std::string outputDir = baseDir + "/" + gpuName;
    
    createDirectory(outputDir);
    std::cout << "输出目录: " << outputDir << std::endl;
    
    std::cout << "\n测试配置:" << std::endl;
    std::cout << "  M值列表: [";
    for (size_t i = 0; i < m_list.size(); ++i) {
        std::cout << m_list[i];
        if (i < m_list.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  (N, K) pairs:" << std::endl;
    for (const auto& nk : nk_pairs) {
        std::cout << "    " << nk.name << ": (" << nk.n << ", " << nk.k << ")" << std::endl;
    }
    
    int total_tests = m_list.size() * nk_pairs.size();
    std::cout << "  总测试数: " << total_tests << std::endl;
    
    int test_idx = 0;
    int success_count = 0;
    int fail_count = 0;
    
    for (int m : m_list) {
        for (const auto& nk : nk_pairs) {
            test_idx++;
            
            std::cout << "\n########################################" << std::endl;
            std::cout << "测试 [" << test_idx << "/" << total_tests << "]: " 
                      << nk.name << " M=" << m << ", N=" << nk.n << ", K=" << nk.k << std::endl;
            std::cout << "########################################" << std::endl;
            
            std::string csvFileName = "m_" + std::to_string(m) + 
                                      "_n_" + std::to_string(nk.n) + 
                                      "_k_" + std::to_string(nk.k) + ".csv";
            std::string csvPath = outputDir + "/" + csvFileName;
            
            int result = runCompatTest(m, nk.n, nk.k, csvPath);
            
            if (result == EXIT_SUCCESS) {
                success_count++;
                std::cout << "测试 [" << test_idx << "] 完成: " << csvFileName << std::endl;
            } else {
                fail_count++;
                std::cout << "测试 [" << test_idx << "] 失败: " << csvFileName << std::endl;
                cudaDeviceReset();
                std::cout << "已重置CUDA设备" << std::endl;
            }
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "全部测试完成!" << std::endl;
    std::cout << "  成功: " << success_count << " / " << total_tests << std::endl;
    std::cout << "  失败: " << fail_count << " / " << total_tests << std::endl;
    std::cout << "  输出目录: " << outputDir << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (fail_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
