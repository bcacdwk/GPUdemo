// cuBLAS 稀疏基准：固定 TN 配置的性能采样
// $ nvcc -o cuBLAS_search_all_fp16 cuBLAS_search_all.cu -lcublas && ./cuBLAS_search_all_fp16
// $ nvcc -o cuBLAS_search_all_int8 cuBLAS_search_all.cu -lcublas -DUSE_INT8 && ./cuBLAS_search_all_int8
// nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=nsys_cublas ./cuBLAS_search_all_int8


const char* kCsvFileName = "cuBLAS_TN_int8.csv"; // 可修改的结果文件名

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#ifdef USE_INT8
using mt = char;
using rt = int;
using st = int;
cudaDataType Atype = CUDA_R_8I;
cudaDataType Btype = CUDA_R_8I;
cudaDataType Ctype = CUDA_R_32I;
cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
#include <cuda_fp16.h>
using mt = half;
using rt = half;
using st = half;
cudaDataType Atype = CUDA_R_16F;
cudaDataType Btype = CUDA_R_16F;
cudaDataType Ctype = CUDA_R_16F;
cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
#endif

#define CHECK_CUDA(func)                                                      \
{                                                                            \
    cudaError_t status = (func);                                             \
    if (status != cudaSuccess) {                                             \
        std::printf("CUDA API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",\
                   __LINE__, cudaGetErrorString(status), status);             \
        return EXIT_FAILURE;                                                  \
    }                                                                        \
}

#define CHECK_CUBLAS(func)                                                    \
{                                                                            \
    cublasStatus_t status = (func);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
        std::printf("cuBLAS API 调用失败，位置：第 %d 行，错误代码：%d\n",\
                   __LINE__, status);                                         \
        return EXIT_FAILURE;                                                  \
    }                                                                        \
}

int main() {
    std::srand(static_cast<unsigned>(time(nullptr)));

    //std::vector<int> m_values = {256, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 2048, 4096, 8192};
	std::vector<int> m_values = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};

    std::vector<std::pair<int, int>> nk_pairs = {
        {2560, 2560},
        {3840, 2560},
        {13824, 2560},
        {2560, 6912}
    };

    const int num_runs = 100;

    std::ofstream csv(kCsvFileName);
    if (!csv.is_open()) {
        std::cerr << "无法创建结果文件 " << kCsvFileName << std::endl;
        return EXIT_FAILURE;
    }
    csv << "M,N,K,AverageTimeMs,ThroughputTOPS\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int m : m_values) {
        for (const auto &nk : nk_pairs) {
            int n = nk.first;
            int k = nk.second;

            std::cout << "正在测试矩阵维度: M=" << m << ", N=" << n << ", K=" << k << std::endl;

            size_t sizeA = static_cast<size_t>(m) * k;
            size_t sizeB = static_cast<size_t>(n) * k;
            size_t sizeC = static_cast<size_t>(m) * n;

            std::vector<mt> hA(sizeA);
            std::vector<mt> hB(sizeB);
            std::vector<rt> hC(sizeC, static_cast<rt>(0));

            for (size_t i = 0; i < sizeA; ++i) {
#ifdef USE_INT8
                hA[i] = static_cast<mt>(std::rand() % 256 - 128);
#else
                hA[i] = static_cast<mt>(static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
#endif
            }
            for (size_t i = 0; i < sizeB; ++i) {
#ifdef USE_INT8
                hB[i] = static_cast<mt>(std::rand() % 256 - 128);
#else
                hB[i] = static_cast<mt>(static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
#endif
            }

            mt *dA = nullptr, *dB = nullptr;
            rt *dC = nullptr;

            CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA), sizeA * sizeof(mt)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dB), sizeB * sizeof(mt)));
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dC), sizeC * sizeof(rt)));

            CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(mt), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(mt), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeC * sizeof(rt), cudaMemcpyHostToDevice));

            st alpha = static_cast<st>(1);
            st beta  = static_cast<st>(0);

            CHECK_CUBLAS(cublasGemmEx(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      dA,
                                      Atype,
                                      k,
                                      dB,
                                      Btype,
                                      k,
                                      &beta,
                                      dC,
                                      Ctype,
                                      m,
                                      computeType,
                                      CUBLAS_GEMM_AUTOTUNE));

            float total_time = 0.0f;
            for (int run = 0; run < num_runs; ++run) {
                CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeC * sizeof(rt), cudaMemcpyHostToDevice));

                CHECK_CUDA(cudaEventRecord(start));

                CHECK_CUBLAS(cublasGemmEx(handle,
                                          CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          dA,
                                          Atype,
                                          k,
                                          dB,
                                          Btype,
                                          k,
                                          &beta,
                                          dC,
                                          Ctype,
                                          m,
                                          computeType,
                                          CUBLAS_GEMM_AUTOTUNE));

                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));

                float milliseconds = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
                total_time += milliseconds;
            }

            float avg_time = total_time / num_runs;
            double ops = 2.0 * static_cast<double>(m) * n * k;
            double throughput = ops / (avg_time / 1000.0) / 1e12;

            csv << m << ',' << n << ',' << k << ',' << avg_time << ',' << throughput << '\n';

            std::cout << "平均耗时: " << avg_time << " ms, 吞吐量: "
                      << throughput << " TOPS" << std::endl;

            CHECK_CUDA(cudaFree(dA));
            CHECK_CUDA(cudaFree(dB));
            CHECK_CUDA(cudaFree(dC));
        }
        // csv << '\n';
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA 异步错误: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "搜索完成，结果已写入 " << kCsvFileName << std::endl;

    return EXIT_SUCCESS;
}