// 稀疏在左：自动遍历所有算法ID并记录性能
// $ nvcc -o sparse_L_search_all sparse_L_search_all.cu -lcusparseLt && ./sparse_L_search_all
const char* kCsvFileName = "L_TN_CC_C.csv"; // 可修改的结果文件名

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <sstream>

#define INT8 1001

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
		std::printf("CUDA API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",  \
				   __LINE__, cudaGetErrorString(status), status);               \
		return EXIT_FAILURE;                                                    \
	}                                                                           \
}

#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
	cusparseStatus_t status = (func);                                           \
	if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
		std::printf("cuSPARSE API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",\
				   __LINE__, cusparseLtGetErrorString(status), status);         \
		return EXIT_FAILURE;                                                    \
	}                                                                           \
}

int main() {
	std::srand(static_cast<unsigned>(time(nullptr)));

	std::vector<int> m_values = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
	//std::vector<int> m_values = {256, 512};

	std::vector<std::pair<int, int>> nk_pairs = {
		{2560, 2560},
		{3840, 2560},
		{13824, 2560},
		{2560, 6912}
	};

	const int num_runs = 10;

	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	std::ofstream csv(kCsvFileName);
	if (!csv.is_open()) {
		std::cerr << "无法创建结果文件 " << kCsvFileName << std::endl;
		return EXIT_FAILURE;
	}
	csv << "M,N,K,AlgorithmID,AverageTimeMs,ThroughputTOPS\n";

	std::vector<std::string> summary_rows;

	for (int m : m_values) {
		for (const auto &nk : nk_pairs) {
			int n = nk.first;
			int k = nk.second;

			std::cout << "正在测试矩阵维度: M=" << m << ", N=" << n << ", K=" << k << std::endl;

			size_t free_mem = 0, total_mem = 0;
			CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
			std::cout << "GPU内存状态: 可用 " << free_mem / (1024 * 1024)
					  << " MB, 总计 " << total_mem / (1024 * 1024) << " MB" << std::endl;
			//CUSPARSE_ORDER_ROW; CUSPARSE_ORDER_COL
			//CUSPARSE_OPERATION_NON_TRANSPOSE; CUSPARSE_OPERATION_TRANSPOSE
			auto orderA        = CUSPARSE_ORDER_COL;
			auto orderB        = CUSPARSE_ORDER_COL;
			auto orderC        = CUSPARSE_ORDER_COL;
			auto opA           = CUSPARSE_OPERATION_TRANSPOSE;
			auto opB           = CUSPARSE_OPERATION_NON_TRANSPOSE;
			auto type_AB       = cuda_type<AB_t>::value;
			auto type_C        = cuda_type<C_t>::value;
			auto compute_type  = cusparse_compute_type<COMPUTE_t>::value;
			unsigned alignment = 16;

			bool isA_rowmajor   = (orderA == CUSPARSE_ORDER_ROW);
			bool isB_rowmajor   = (orderB == CUSPARSE_ORDER_ROW);
			bool isC_rowmajor   = (orderC == CUSPARSE_ORDER_ROW);
			bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
			bool isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);

			int num_A_rows = isA_transposed ? k : m;
			int num_A_cols = isA_transposed ? m : k;
			int num_B_rows = isB_transposed ? n : k;
			int num_B_cols = isB_transposed ? k : n;
			int num_C_rows = m;
			int num_C_cols = n;

			int lda = isA_rowmajor ? num_A_cols : num_A_rows;
			int ldb = isB_rowmajor ? num_B_cols : num_B_rows;
			int ldc = isC_rowmajor ? num_C_cols : num_C_rows;

			int A_height = isA_rowmajor ? num_A_rows : num_A_cols;
			int B_height = isB_rowmajor ? num_B_rows : num_B_cols;
			int C_height = isC_rowmajor ? num_C_rows : num_C_cols;

			size_t A_elems = static_cast<size_t>(A_height) * lda;
			size_t B_elems = static_cast<size_t>(B_height) * ldb;
			size_t C_elems = static_cast<size_t>(C_height) * ldc;

			size_t A_size = A_elems * sizeof(AB_t);
			size_t B_size = B_elems * sizeof(AB_t);
			size_t C_size = C_elems * sizeof(C_t);

			std::vector<AB_t> hA(A_elems);
			std::vector<AB_t> hB(B_elems);
			std::vector<C_t>  hC(C_elems, static_cast<C_t>(0));

			for (size_t i = 0; i < static_cast<size_t>(m) * k; ++i) {
				hA[i] = static_cast<AB_t>(std::rand() % 256 - 128);
			}
			for (size_t i = 0; i < static_cast<size_t>(k) * n; ++i) {
				hB[i] = static_cast<AB_t>(std::rand() % 256 - 128);
			}

			float alpha = 1.0f;
			float beta  = 0.0f;

			AB_t *dA = nullptr, *dB = nullptr;
			C_t *dC = nullptr, *dD = nullptr;
			int *d_valid = nullptr;

			CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA), A_size));
			CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dB), B_size));
			CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dC), C_size));
			CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_valid), sizeof(int)));

			dD = dC;

			CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size, cudaMemcpyHostToDevice));
			CHECK_CUDA(cudaMemcpy(dC, hC.data(), C_size, cudaMemcpyHostToDevice));

			cusparseLtHandle_t           handle;
			cusparseLtMatDescriptor_t    matA, matB, matC;
			cusparseLtMatmulDescriptor_t matmul;
			cudaStream_t                 stream = nullptr;

			CHECK_CUSPARSE(cusparseLtInit(&handle));

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

			CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
														   opA, opB, &matA, &matB,
														   &matC, &matC, compute_type));

			CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle, &matmul,
															 CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
															 &dA, sizeof(dA)));

			CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
												CUSPARSELT_PRUNE_SPMMA_TILE, stream));

			CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
									  d_valid, stream));

			int is_valid = 0;
			CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
			CHECK_CUDA(cudaStreamSynchronize(stream));

			if (is_valid != 0) {
				std::printf("错误：矩阵剪枝失败，终止本次测试。\n");
				cusparseLtMatDescriptorDestroy(&matA);
				cusparseLtMatDescriptorDestroy(&matB);
				cusparseLtMatDescriptorDestroy(&matC);
				cusparseLtDestroy(&handle);
				cudaFree(dA);
				cudaFree(dB);
				cudaFree(dC);
				cudaFree(d_valid);
				continue;
			}

			// 查询算法ID上限，避免越界导致的异常
			int max_alg_id = -1;
			{
				cusparseLtMatmulAlgSelection_t alg_sel_tmp;
				CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel_tmp,
										 &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
				CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel_tmp,
										 CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
										 &max_alg_id, sizeof(max_alg_id)));
				CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp));
                std::cout << "该维度下的算法ID上限为: " << max_alg_id << std::endl;
			}

			if (max_alg_id < 0) {
				std::cout << "未能获取有效的算法ID上限，跳过该维度组合。" << std::endl;
				cusparseLtMatDescriptorDestroy(&matA);
				cusparseLtMatDescriptorDestroy(&matB);
				cusparseLtMatDescriptorDestroy(&matC);
				cusparseLtDestroy(&handle);
				cudaFree(dA);
				cudaFree(dB);
				cudaFree(dC);
				cudaFree(d_valid);
				continue;
			}

			double best_throughput_1 = -1.0;
			double best_throughput_2 = -1.0;
			float best_time_1 = 0.0f;
			float best_time_2 = 0.0f;
			int best_alg_1 = -1;
			int best_alg_2 = -1;

			for (int alg_id = 0; alg_id <= max_alg_id; ++alg_id) {
				AB_t *dA_compressed_local = nullptr;
				void *dA_compressedBuffer_local = nullptr;
				void *d_workspace_local = nullptr;
				bool record_valid = true;
				bool selection_created = false;
				bool plan_created = false;

				cusparseLtMatmulAlgSelection_t alg_sel;
				cusparseStatus_t sel_status = cusparseLtMatmulAlgSelectionInit(
					&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
				if (sel_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "初始化算法选择对象失败: "
						  << cusparseLtGetErrorString(sel_status) << std::endl;
					break;
				}
				selection_created = true;

				cusparseLtMatmulPlan_t plan;

				auto cleanup = [&]() {
					if (d_workspace_local) {
						cudaFree(d_workspace_local);
						d_workspace_local = nullptr;
					}
					if (dA_compressedBuffer_local) {
						cudaFree(dA_compressedBuffer_local);
						dA_compressedBuffer_local = nullptr;
					}
					if (dA_compressed_local) {
						cudaFree(dA_compressed_local);
						dA_compressed_local = nullptr;
					}
					if (plan_created) {
						cusparseLtMatmulPlanDestroy(&plan);
						plan_created = false;
					}
					if (selection_created) {
						cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
						selection_created = false;
					}
				};

				cusparseStatus_t attr_status = cusparseLtMatmulAlgSetAttribute(
					&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
					&alg_id, sizeof(alg_id));
				if (attr_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "设置算法ID " << alg_id << " 失败: "
						  << cusparseLtGetErrorString(attr_status)
						  << "，跳过该算法。" << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				cusparseStatus_t plan_status = cusparseLtMatmulPlanInit(
					&handle, &plan, &matmul, &alg_sel);
				if (plan_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "初始化算法ID " << alg_id << " 的执行计划失败: "
						  << cusparseLtGetErrorString(plan_status)
						  << "，跳过该算法。" << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}
				plan_created = true;

				size_t compressed_size = 0;
				size_t compressed_buffer_size = 0;
				cusparseStatus_t size_status = cusparseLtSpMMACompressedSize(
					&handle, &plan, &compressed_size, &compressed_buffer_size);
				if (size_status != CUSPARSE_STATUS_SUCCESS || compressed_size == 0) {
					std::cout << "算法ID " << alg_id << " 无法获取有效的压缩尺寸，跳过。" << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void **>(&dA_compressed_local), compressed_size);
				if (cuda_status != cudaSuccess) {
					std::cout << "算法ID " << alg_id << " 分配压缩矩阵失败: "
						  << cudaGetErrorString(cuda_status) << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				if (compressed_buffer_size > 0) {
					cuda_status = cudaMalloc(&dA_compressedBuffer_local, compressed_buffer_size);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 分配压缩缓冲区失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						cleanup();
						continue;
					}
				}

				cusparseStatus_t compress_status = cusparseLtSpMMACompress(
					&handle, &plan, dA, dA_compressed_local,
					dA_compressedBuffer_local, stream);
				if (compress_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "算法ID " << alg_id << " 的稀疏矩阵压缩失败: "
						  << cusparseLtGetErrorString(compress_status) << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				size_t workspace_size = 0;
				cusparseStatus_t workspace_status = cusparseLtMatmulGetWorkspace(
					&handle, &plan, &workspace_size);
				if (workspace_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "查询算法ID " << alg_id << " 的工作空间失败: "
						  << cusparseLtGetErrorString(workspace_status) << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				if (workspace_size > 0) {
					cuda_status = cudaMalloc(&d_workspace_local, workspace_size);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 分配工作空间失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						cleanup();
						continue;
					}
				}

				cuda_status = cudaMemcpy(dC, hC.data(), C_size, cudaMemcpyHostToDevice);
				if (cuda_status != cudaSuccess) {
					std::cout << "算法ID " << alg_id << " 拷贝矩阵C到设备失败: "
						  << cudaGetErrorString(cuda_status) << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				cusparseStatus_t warmup_status = cusparseLtMatmul(
					&handle, &plan, &alpha, dA_compressed_local, dB,
					&beta, dC, dD, d_workspace_local, nullptr, 0);
				if (warmup_status != CUSPARSE_STATUS_SUCCESS) {
					std::cout << "算法ID " << alg_id << " 预热计算失败: "
						  << cusparseLtGetErrorString(warmup_status) << std::endl;
					record_valid = false;
					cleanup();
					continue;
				}

				float total_time = 0.0f;
				for (int run = 0; run < num_runs; ++run) {
					cuda_status = cudaMemcpy(dC, hC.data(), C_size, cudaMemcpyHostToDevice);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次拷贝C失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						break;
					}

					cuda_status = cudaEventRecord(start);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次记录起始事件失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						break;
					}

					cusparseStatus_t compute_status = cusparseLtMatmul(
						&handle, &plan, &alpha, dA_compressed_local, dB,
						&beta, dC, dD, d_workspace_local, nullptr, 0);
					if (compute_status != CUSPARSE_STATUS_SUCCESS) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次执行失败: "
							  << cusparseLtGetErrorString(compute_status) << std::endl;
						record_valid = false;
						break;
					}

					cuda_status = cudaEventRecord(stop);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次记录结束事件失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						break;
					}

					cuda_status = cudaEventSynchronize(stop);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次同步事件失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						break;
					}

					float milliseconds = 0.0f;
					cuda_status = cudaEventElapsedTime(&milliseconds, start, stop);
					if (cuda_status != cudaSuccess) {
						std::cout << "算法ID " << alg_id << " 第 " << run
							  << " 次获取耗时失败: "
							  << cudaGetErrorString(cuda_status) << std::endl;
						record_valid = false;
						break;
					}

					total_time += milliseconds;
				}

				if (record_valid && total_time > 0.0f) {
					float avg_time = total_time / num_runs;
					double ops = 2.0 * static_cast<double>(m) * n * k;
					double throughput = ops / (avg_time / 1000.0) / 1e12;

					csv << m << ',' << n << ',' << k << ',' << alg_id << ','
						<< avg_time << ',' << throughput << '\n';

					std::cout << "算法ID " << alg_id << " 平均耗时: " << avg_time
						  << " ms, 吞吐量: " << throughput << " TOPS" << std::endl;

					if (throughput > best_throughput_1) {
						best_throughput_2 = best_throughput_1;
						best_time_2 = best_time_1;
						best_alg_2 = best_alg_1;

						best_throughput_1 = throughput;
						best_time_1 = avg_time;
						best_alg_1 = alg_id;
					} else if (throughput > best_throughput_2) {
						best_throughput_2 = throughput;
						best_time_2 = avg_time;
						best_alg_2 = alg_id;
					}
				}

				cleanup();

				if (!record_valid) {
					std::cout << "算法ID " << alg_id << " 未获得有效结果。" << std::endl;
				}
			}

			if (best_alg_1 >= 0) {
				std::cout << "组合 M=" << m << ", N=" << n << ", K=" << k
					  << " 最快算法ID: " << best_alg_1
					  << " (耗时 " << best_time_1 << " ms, 吞吐 "
					  << best_throughput_1 << " TOPS)" << std::endl;
				if (best_alg_2 >= 0) {
					std::cout << "次快算法ID: " << best_alg_2
						  << " (耗时 " << best_time_2 << " ms, 吞吐 "
						  << best_throughput_2 << " TOPS)" << std::endl;
				} else {
					std::cout << "未找到次快算法，只有一个有效算法结果。" << std::endl;
				}
			} else {
				std::cout << "组合 M=" << m << ", N=" << n << ", K=" << k
					  << " 没有获得任何有效算法结果。" << std::endl;
			}

			std::ostringstream summary;
			summary << m << ',' << n << ',' << k << ",BestID=";
			if (best_alg_1 >= 0) {
				summary << best_alg_1 << ";SecondID=";
				if (best_alg_2 >= 0) {
					summary << best_alg_2;
				} else {
					summary << "None";
				}
				summary << ',' << best_throughput_1 << ',';
				summary << ((best_alg_2 >= 0) ? best_throughput_2 : 0.0);
			} else {
				summary << "None;SecondID=None," << 0.0 << ',' << 0.0;
			}
			summary_rows.push_back(summary.str());
			csv << '\n';

			cusparseLtMatDescriptorDestroy(&matA);
			cusparseLtMatDescriptorDestroy(&matB);
			cusparseLtMatDescriptorDestroy(&matC);
			cusparseLtDestroy(&handle);

			cudaFree(dA);
			cudaFree(dB);
			cudaFree(dC);
			cudaFree(d_valid);
		}
	}

	if (!summary_rows.empty()) {
		csv << "SummaryM,SummaryN,SummaryK,SummaryIDs,BestTOPS,SecondTOPS\n";
		for (const auto &row : summary_rows) {
			csv << row << '\n';
		}
	}

	CHECK_CUDA(cudaEventDestroy(start));
	CHECK_CUDA(cudaEventDestroy(stop));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA异步错误: " << cudaGetErrorString(err) << std::endl;
	}

	std::cout << "搜索完成，结果已写入 " << kCsvFileName << std::endl;

	return EXIT_SUCCESS;
}
