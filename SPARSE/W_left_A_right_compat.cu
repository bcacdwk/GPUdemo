/*
算法-权重兼容性测试：验证不同算法ID压缩的权重能否混用
测试逻辑：
  1. 固定维度 M=1024, N=3840, K=2560，计算 R[N,M] = W[N,K] * A^T[M,K]
  2. 对每个算法ID分别压缩一份权重，得到 num_algs 份压缩权重
  3. 先用 算法i + 权重i (一一对应) 计算得到 golden[i]
  4. 然后对每个权重j，遍历所有算法i，用算法i计算权重j的结果，与golden[j]比较
  5. 输出兼容性矩阵：matrix[alg_i][weight_j] = 1 表示算法i可以使用权重j

编译运行:
nvcc -o W_left_A_right_compat W_left_A_right_compat.cu -lcusparseLt && ./W_left_A_right_compat
*/

const char* kCsvFileName = "alg_weight_compat_matrix.csv";

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

#define CHECK_CUDA(func)                                                        \
{                                                                               \
	cudaError_t status = (func);                                                \
	if (status != cudaSuccess) {                                                \
		std::printf("CUDA API failed at line %d: %s (%d)\n",                    \
				   __LINE__, cudaGetErrorString(status), status);               \
		return EXIT_FAILURE;                                                    \
	}                                                                           \
}

#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
	cusparseStatus_t status = (func);                                           \
	if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
		std::printf("cuSPARSE API failed at line %d: %s (%d)\n",                \
				   __LINE__, cusparseLtGetErrorString(status), status);         \
		return EXIT_FAILURE;                                                    \
	}                                                                           \
}

// 比较两个结果矩阵是否完全相同（INT32精确比较，无舍入误差）
// 返回：匹配元素数量
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

bool compare_results(const std::vector<C_t>& a, const std::vector<C_t>& b) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); ++i) {
		if (a[i] != b[i]) return false;
	}
	return true;
}

int main() {
	std::srand(42);  // 固定种子，保证可复现

	// 固定维度：R[N,M] = W[N,K] * A^T[M,K]
	const int m = 1024;  // dense A 的行数，结果矩阵的列数
	const int n = 3840;  // sparse W 的行数，结果矩阵的行数
	const int k = 2560;  // 共享维度

	std::cout << "========================================" << std::endl;
	std::cout << "算法-权重兼容性测试" << std::endl;
	std::cout << "维度: M=" << m << ", N=" << n << ", K=" << k << std::endl;
	std::cout << "计算: R[" << n << "," << m << "] = W[" << n << "," << k 
			  << "] * A^T[" << m << "," << k << "]" << std::endl;
	std::cout << "========================================" << std::endl;

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

	// 计算维度
	int num_A_rows = n;  // sparse W: N rows
	int num_A_cols = k;  // sparse W: K cols
	int num_B_rows = m;  // dense A^T: M rows (transposed)
	int num_B_cols = k;  // dense A^T: K cols (transposed)
	int num_C_rows = n;  // result R: N rows
	int num_C_cols = m;  // result R: M cols

	int lda = num_A_cols;  // row-major
	int ldb = num_B_cols;  // row-major
	int ldc = num_C_cols;  // row-major

	size_t A_elems = static_cast<size_t>(n) * k;
	size_t B_elems = static_cast<size_t>(m) * k;
	size_t C_elems = static_cast<size_t>(n) * m;

	size_t A_size = A_elems * sizeof(AB_t);
	size_t B_size = B_elems * sizeof(AB_t);
	size_t C_size = C_elems * sizeof(C_t);

	// 分配并初始化主机数据
	std::vector<AB_t> hA(A_elems);  // sparse W [N,K]
	std::vector<AB_t> hB(B_elems);  // dense A [M,K]
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

	// 创建矩阵乘法描述符
	cusparseLtMatmulDescriptor_t matmul;
	CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
												   opA, opB, &matA, &matB,
												   &matC, &matC, compute_type));
	CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle, &matmul,
													 CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
													 &dA, sizeof(dA)));

	// 剪枝稀疏矩阵
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

	// ============================================================
	// 阶段1：为每个算法ID创建plan并压缩权重
	// ============================================================
	std::cout << "\n[阶段1] 为每个算法创建压缩权重..." << std::endl;

	std::vector<cusparseLtMatmulAlgSelection_t> alg_sels(num_algs);
	std::vector<cusparseLtMatmulPlan_t> plans(num_algs);
	std::vector<AB_t*> compressed_weights(num_algs, nullptr);
	std::vector<bool> alg_valid(num_algs, false);

	for (int alg_id = 0; alg_id < num_algs; ++alg_id) {
		// 初始化算法选择
		cusparseStatus_t status = cusparseLtMatmulAlgSelectionInit(
			&handle, &alg_sels[alg_id], &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			std::cout << "  算法ID " << alg_id << ": 初始化失败，跳过" << std::endl;
			continue;
		}

		// 设置算法ID
		status = cusparseLtMatmulAlgSetAttribute(
			&handle, &alg_sels[alg_id], CUSPARSELT_MATMUL_ALG_CONFIG_ID,
			&alg_id, sizeof(alg_id));
		if (status != CUSPARSE_STATUS_SUCCESS) {
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
			std::cout << "  算法ID " << alg_id << ": 设置失败，跳过" << std::endl;
			continue;
		}

		// 初始化plan
		status = cusparseLtMatmulPlanInit(&handle, &plans[alg_id], &matmul, &alg_sels[alg_id]);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
			std::cout << "  算法ID " << alg_id << ": Plan初始化失败，跳过" << std::endl;
			continue;
		}

		// 获取压缩尺寸
		size_t compressed_size = 0, compressed_buffer_size = 0;
		status = cusparseLtSpMMACompressedSize(&handle, &plans[alg_id], 
											   &compressed_size, &compressed_buffer_size);
		if (status != CUSPARSE_STATUS_SUCCESS || compressed_size == 0) {
			cusparseLtMatmulPlanDestroy(&plans[alg_id]);
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
			std::cout << "  算法ID " << alg_id << ": 获取压缩尺寸失败，跳过" << std::endl;
			continue;
		}

		// 分配压缩权重内存
		cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(&compressed_weights[alg_id]), 
											  compressed_size);
		if (cuda_status != cudaSuccess) {
			cusparseLtMatmulPlanDestroy(&plans[alg_id]);
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
			std::cout << "  算法ID " << alg_id << ": 分配内存失败，跳过" << std::endl;
			continue;
		}

		// 分配压缩缓冲区（如果需要）
		void* compress_buffer = nullptr;
		if (compressed_buffer_size > 0) {
			cuda_status = cudaMalloc(&compress_buffer, compressed_buffer_size);
			if (cuda_status != cudaSuccess) {
				cudaFree(compressed_weights[alg_id]);
				compressed_weights[alg_id] = nullptr;
				cusparseLtMatmulPlanDestroy(&plans[alg_id]);
				cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
				std::cout << "  算法ID " << alg_id << ": 分配缓冲区失败，跳过" << std::endl;
				continue;
			}
		}

		// 压缩权重
		status = cusparseLtSpMMACompress(&handle, &plans[alg_id], dA, 
										  compressed_weights[alg_id], compress_buffer, stream);
		if (compress_buffer) cudaFree(compress_buffer);

		if (status != CUSPARSE_STATUS_SUCCESS) {
			cudaFree(compressed_weights[alg_id]);
			compressed_weights[alg_id] = nullptr;
			cusparseLtMatmulPlanDestroy(&plans[alg_id]);
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[alg_id]);
			std::cout << "  算法ID " << alg_id << ": 压缩失败，跳过" << std::endl;
			continue;
		}

		alg_valid[alg_id] = true;
		std::cout << "  算法ID " << alg_id << ": 压缩成功" << std::endl;
	}

	// 统计有效算法数量
	int valid_count = 0;
	for (int i = 0; i < num_algs; ++i) {
		if (alg_valid[i]) valid_count++;
	}
	std::cout << "有效算法数量: " << valid_count << " / " << num_algs << std::endl;

	if (valid_count == 0) {
		std::cerr << "错误：没有有效的算法！" << std::endl;
		return EXIT_FAILURE;
	}

	// ============================================================
	// 阶段2：计算 golden 结果（算法i + 权重i）
	// ============================================================
	std::cout << "\n[阶段2] 计算golden结果（算法i配权重i）..." << std::endl;

	std::vector<std::vector<C_t>> golden_results(num_algs);

	for (int alg_id = 0; alg_id < num_algs; ++alg_id) {
		if (!alg_valid[alg_id]) continue;

		// 获取workspace大小
		size_t workspace_size = 0;
		cusparseLtMatmulGetWorkspace(&handle, &plans[alg_id], &workspace_size);
		void* d_workspace = nullptr;
		if (workspace_size > 0) {
			cudaMalloc(&d_workspace, workspace_size);
		}

		// 清零输出矩阵
		CHECK_CUDA(cudaMemcpy(dC, hC_zero.data(), C_size, cudaMemcpyHostToDevice));

		// 执行计算
		cusparseStatus_t status = cusparseLtMatmul(
			&handle, &plans[alg_id], &alpha, compressed_weights[alg_id], dB,
			&beta, dC, dC, d_workspace, nullptr, 0);
		
		CHECK_CUDA(cudaDeviceSynchronize());  // 确保计算完成

		if (d_workspace) cudaFree(d_workspace);

		if (status != CUSPARSE_STATUS_SUCCESS) {
			std::cout << "  算法ID " << alg_id << ": 计算失败 (" 
					  << cusparseLtGetErrorString(status) << ")" << std::endl;
			alg_valid[alg_id] = false;
			continue;
		}

		// 拷贝结果回主机
		golden_results[alg_id].resize(C_elems);
		CHECK_CUDA(cudaMemcpy(golden_results[alg_id].data(), dC, C_size, cudaMemcpyDeviceToHost));
		
		// 打印前几个值用于调试
		std::cout << "  算法ID " << alg_id << ": golden计算完成, 前3个值: " 
				  << golden_results[alg_id][0] << ", " 
				  << golden_results[alg_id][1] << ", " 
				  << golden_results[alg_id][2] << std::endl;
	}
	
	// 检查所有golden结果是否相同
	std::cout << "\n[调试] 检查golden结果一致性..." << std::endl;
	int first_valid = -1;
	for (int i = 0; i < num_algs; ++i) {
		if (alg_valid[i]) { first_valid = i; break; }
	}
	if (first_valid >= 0) {
		for (int i = first_valid + 1; i < num_algs; ++i) {
			if (!alg_valid[i]) continue;
			size_t first_diff = 0;
			size_t match = compare_results_detailed(golden_results[first_valid], 
													 golden_results[i], &first_diff);
			if (match == C_elems) {
				std::cout << "  golden[" << first_valid << "] == golden[" << i << "]" << std::endl;
			} else {
				std::cout << "  golden[" << first_valid << "] != golden[" << i << "] "
						  << "(匹配: " << match << "/" << C_elems << ", 首个差异位置: " << first_diff
						  << ", 值: " << golden_results[first_valid][first_diff] 
						  << " vs " << golden_results[i][first_diff] << ")" << std::endl;
			}
		}
	}

	// ============================================================
	// 阶段3：交叉测试，生成兼容性矩阵
	// ============================================================
	std::cout << "\n[阶段3] 交叉测试（算法i配权重j）..." << std::endl;

	// compat_matrix[alg_i][weight_j] = 1 表示算法i可以使用权重j的压缩结果
	std::vector<std::vector<int>> compat_matrix(num_algs, std::vector<int>(num_algs, -1));
	// -1: 未测试或无效, 0: 不兼容, 1: 兼容

	std::vector<C_t> test_result(C_elems);

	for (int weight_id = 0; weight_id < num_algs; ++weight_id) {
		if (!alg_valid[weight_id]) continue;

		std::cout << "  测试权重ID " << weight_id << " ..." << std::endl;

		for (int alg_id = 0; alg_id < num_algs; ++alg_id) {
			if (!alg_valid[alg_id]) continue;

			// 获取workspace
			size_t workspace_size = 0;
			cusparseLtMatmulGetWorkspace(&handle, &plans[alg_id], &workspace_size);
			void* d_workspace = nullptr;
			if (workspace_size > 0) {
				cudaMalloc(&d_workspace, workspace_size);
			}

			// 清零输出矩阵
			CHECK_CUDA(cudaMemcpy(dC, hC_zero.data(), C_size, cudaMemcpyHostToDevice));

			// 用算法alg_id计算权重weight_id的压缩结果
			cusparseStatus_t status = cusparseLtMatmul(
				&handle, &plans[alg_id], &alpha, compressed_weights[weight_id], dB,
				&beta, dC, dC, d_workspace, nullptr, 0);
			
			CHECK_CUDA(cudaDeviceSynchronize());  // 确保计算完成

			if (d_workspace) cudaFree(d_workspace);

			if (status != CUSPARSE_STATUS_SUCCESS) {
				compat_matrix[alg_id][weight_id] = 0;
				if (alg_id == weight_id) {
					std::cout << "    [警告] 对角线 A" << alg_id << "+W" << weight_id 
							  << " 计算失败: " << cusparseLtGetErrorString(status) << std::endl;
				}
				continue;
			}

			// 拷贝结果
			CHECK_CUDA(cudaMemcpy(test_result.data(), dC, C_size, cudaMemcpyDeviceToHost));

			// 与 golden[weight_id] 比较
			bool match = compare_results(test_result, golden_results[weight_id]);
			compat_matrix[alg_id][weight_id] = match ? 1 : 0;
			
			// 对角线如果不匹配，打印调试信息
			if (alg_id == weight_id && !match) {
				size_t first_diff = 0;
				size_t match_count = compare_results_detailed(test_result, golden_results[weight_id], &first_diff);
				std::cout << "    [异常] 对角线 A" << alg_id << "+W" << weight_id 
						  << " 不匹配! 匹配: " << match_count << "/" << C_elems
						  << ", 首个差异位置: " << first_diff
						  << ", 测试值: " << test_result[first_diff]
						  << ", golden值: " << golden_results[weight_id][first_diff] << std::endl;
			}
		}
	}

	// ============================================================
	// 阶段4：输出结果
	// ============================================================
	std::cout << "\n[阶段4] 输出兼容性矩阵..." << std::endl;

	// 打印到控制台
	std::cout << "\n兼容性矩阵 (行=算法ID, 列=权重ID):" << std::endl;
	std::cout << "     ";
	for (int j = 0; j < num_algs; ++j) {
		if (alg_valid[j]) std::cout << "W" << j << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < num_algs; ++i) {
		if (!alg_valid[i]) continue;
		std::cout << "A" << i << ":  ";
		for (int j = 0; j < num_algs; ++j) {
			if (!alg_valid[j]) continue;
			if (compat_matrix[i][j] == 1) {
				std::cout << " 1  ";
			} else if (compat_matrix[i][j] == 0) {
				std::cout << " 0  ";
			} else {
				std::cout << " -  ";
			}
		}
		std::cout << std::endl;
	}

	// 输出到CSV
	std::ofstream csv(kCsvFileName);
	if (!csv.is_open()) {
		std::cerr << "无法创建结果文件 " << kCsvFileName << std::endl;
		return EXIT_FAILURE;
	}

	// CSV表头：AlgID, W0, W1, W2, ...
	csv << "AlgID";
	for (int j = 0; j < num_algs; ++j) {
		if (alg_valid[j]) csv << ",W" << j;
	}
	csv << "\n";

	// CSV数据
	for (int i = 0; i < num_algs; ++i) {
		if (!alg_valid[i]) continue;
		csv << "A" << i;
		for (int j = 0; j < num_algs; ++j) {
			if (!alg_valid[j]) continue;
			csv << "," << (compat_matrix[i][j] == 1 ? 1 : 0);
		}
		csv << "\n";
	}
	csv.close();

	// 统计分析
	int diagonal_match = 0, off_diagonal_match = 0, total_tests = 0;
	for (int i = 0; i < num_algs; ++i) {
		if (!alg_valid[i]) continue;
		for (int j = 0; j < num_algs; ++j) {
			if (!alg_valid[j]) continue;
			if (compat_matrix[i][j] == 1) {
				if (i == j) diagonal_match++;
				else off_diagonal_match++;
			}
			total_tests++;
		}
	}

	std::cout << "\n========================================" << std::endl;
	std::cout << "统计结果:" << std::endl;
	std::cout << "  对角线匹配(应该全为1): " << diagonal_match << " / " << valid_count << std::endl;
	std::cout << "  非对角线匹配(权重可混用): " << off_diagonal_match << " / " 
			  << (total_tests - valid_count) << std::endl;
	std::cout << "  结果已写入: " << kCsvFileName << std::endl;
	std::cout << "========================================" << std::endl;

	// ============================================================
	// 清理资源
	// ============================================================
	for (int i = 0; i < num_algs; ++i) {
		if (alg_valid[i]) {
			if (compressed_weights[i]) cudaFree(compressed_weights[i]);
			cusparseLtMatmulPlanDestroy(&plans[i]);
			cusparseLtMatmulAlgSelectionDestroy(&alg_sels[i]);
		}
	}

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
