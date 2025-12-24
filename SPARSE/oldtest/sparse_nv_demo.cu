// $ nvcc -o sparse_nv_demo sparse_nv_demo.cu -lcusparseLt && ./sparse_nv_demo

#include <cuda_runtime_api.h>
#include <cusparseLt.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

template <typename value_t>
struct cuda_type;

template <>
struct cuda_type<int8_t> {
    static constexpr cudaDataType value = CUDA_R_8I;
};

template <>
struct cuda_type<int> {
    static constexpr cudaDataType value = CUDA_R_32I;
};

template <>
struct cuda_type<float> {
    static constexpr cudaDataType value = CUDA_R_32F;
};

template <typename value_t>
struct cusparse_compute_type;

template <>
struct cusparse_compute_type<int> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32I;
};

template <>
struct cusparse_compute_type<float> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32F;
};

#define CHECK_CUDA(func)                                                                 \
    do {                                                                                 \
        cudaError_t status = (func);                                                     \
        if (status != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",                      \
                         static_cast<int>(status), cudaGetErrorString(status),           \
                         __FILE__, __LINE__);                                            \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

#define CHECK_CUSPARSE(func)                                                             \
    do {                                                                                 \
        cusparseStatus_t status = (func);                                                \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuSPARSELt error %d (%s) at %s:%d\n",                \
                         static_cast<int>(status), cusparseLtGetErrorString(status),     \
                         __FILE__, __LINE__);                                            \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

template <typename AB_t, typename C_t, typename D_t, typename COMPUTE_t>
float benchmark_cusparselt(const char  trans_A,
                           const char  trans_B,
                           const char  layout_A,
                           const char  layout_B,
                           const char  layout_C,
                           const bool  is_A_sparse,
                           const int   m,
                           const int   n,
                           const int   k,
                           const float alpha,
                           AB_t*       d_A,
                           const int   lda,
                           AB_t*       d_B,
                           const int   ldb,
                           const float beta,
                           C_t*        d_C,
                           const int   ldc,
                           D_t*        d_D) {
    assert(trans_A == 'N' || trans_A == 'T');
    assert(trans_B == 'N' || trans_B == 'T');
    assert(layout_A == 'R' || layout_A == 'C');
    assert(layout_B == 'R' || layout_B == 'C');
    assert(layout_C == 'R' || layout_C == 'C');

    constexpr unsigned alignment = 16;

    const auto type_AB      = cuda_type<AB_t>::value;
    const auto type_C       = cuda_type<C_t>::value;
    const auto type_D       = cuda_type<D_t>::value;
    const auto compute_type = cusparse_compute_type<COMPUTE_t>::value;
    const bool is_B_sparse  = !is_A_sparse;

    const auto opA    = (trans_A == 'T') ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto opB    = (trans_B == 'T') ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto orderA = (layout_A == 'C') ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
    const auto orderB = (layout_B == 'C') ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
    const auto orderC = (layout_C == 'C') ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;

    const int num_A_rows = (trans_A == 'N') ? m : k;
    const int num_A_cols = (trans_A == 'N') ? k : m;
    const int num_B_rows = (trans_B == 'N') ? k : n;
    const int num_B_cols = (trans_B == 'N') ? n : k;
    const int num_C_rows = m;
    const int num_C_cols = n;
    const int num_D_rows = m;
    const int num_D_cols = n;

    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC, matD;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    CHECK_CUSPARSE(cusparseLtInit(&handle));

    if (is_A_sparse) {
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle,
                                                          &matA,
                                                          num_A_rows,
                                                          num_A_cols,
                                                          lda,
                                                          alignment,
                                                          type_AB,
                                                          orderA,
                                                          CUSPARSELT_SPARSITY_50_PERCENT));

        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                     &matB,
                                                     num_B_rows,
                                                     num_B_cols,
                                                     ldb,
                                                     alignment,
                                                     type_AB,
                                                     orderB));
    } else {
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle,
                                                          &matB,
                                                          num_B_rows,
                                                          num_B_cols,
                                                          ldb,
                                                          alignment,
                                                          type_AB,
                                                          orderB,
                                                          CUSPARSELT_SPARSITY_50_PERCENT));

        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                     &matA,
                                                     num_A_rows,
                                                     num_A_cols,
                                                     lda,
                                                     alignment,
                                                     type_AB,
                                                     orderA));
    }

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                 &matC,
                                                 num_C_rows,
                                                 num_C_cols,
                                                 ldc,
                                                 alignment,
                                                 type_C,
                                                 orderC));

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                 &matD,
                                                 num_D_rows,
                                                 num_D_cols,
                                                 ldc,
                                                 alignment,
                                                 type_D,
                                                 orderC));

    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle,
                                                  &matmul,
                                                  opA,
                                                  opB,
                                                  &matA,
                                                  &matB,
                                                  &matC,
                                                  &matD,
                                                  compute_type));

    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle,
                                                    &alg_sel,
                                                    &matmul,
                                                    CUSPARSELT_MATMUL_ALG_DEFAULT));

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

    AB_t* d_sparse = is_A_sparse ? d_A : d_B;
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle,
                                                    &matmul,
                                                    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER,
                                                    &d_sparse,
                                                    sizeof(d_sparse)));

    int* d_valid = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_valid), sizeof(int)));
    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle,
                                        &matmul,
                                        d_sparse,
                                        d_sparse,
                                        CUSPARSELT_PRUNE_SPMMA_TILE,
                                        stream));
    int is_valid = 0;
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, d_sparse, d_valid, stream));
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (is_valid != 0) {
        std::fprintf(stderr, "Prune check failed, matrix does not satisfy 2:4 sparsity.\n");
        CHECK_CUDA(cudaFree(d_valid));
        std::exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaFree(d_valid));

    size_t compressed_size        = 0;
    size_t compressed_buffer_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle,
                                                 &plan,
                                                 &compressed_size,
                                                 &compressed_buffer_size));

    AB_t* d_compressed          = nullptr;
    void* d_compress_buffer     = nullptr;
    void* d_workspace           = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_compressed), compressed_size));
    CHECK_CUDA(cudaMalloc(&d_compress_buffer, compressed_buffer_size));

    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle,
                                           &plan,
                                           d_sparse,
                                           d_compressed,
                                           d_compress_buffer,
                                           stream));

    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    int search_iters = 10;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle,
                                                   &alg_sel,
                                                   CUSPARSELT_MATMUL_SEARCH_ITERATIONS,
                                                   &search_iters,
                                                   sizeof(search_iters)));

    CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle,
                                          &plan,
                                          &alpha,
                                          is_A_sparse ? d_compressed : d_A,
                                          is_B_sparse ? d_compressed : d_B,
                                          &beta,
                                          d_C,
                                          d_D,
                                          nullptr,
                                          streams,
                                          num_streams));

    {
        int alg_id          = 0;
        int split_k         = 0;
        int split_k_mode    = 0;
        int split_k_buffers = 0;
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle,
                                                       &alg_sel,
                                                       CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                                       &alg_id,
                                                       sizeof(alg_id)));
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle,
                                                       &alg_sel,
                                                       CUSPARSELT_MATMUL_SPLIT_K,
                                                       &split_k,
                                                       sizeof(split_k)));
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle,
                                                       &alg_sel,
                                                       CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                                       &split_k_mode,
                                                       sizeof(split_k_mode)));
        CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle,
                                                       &alg_sel,
                                                       CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                                       &split_k_buffers,
                                                       sizeof(split_k_buffers)));
        std::printf("[cuSPARSELt] best alg=%d split_k=%d mode=%d buffers=%d\n",
                    alg_id, split_k, split_k_mode, split_k_buffers);
    }

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

    size_t workspace_size = 0;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    }

    auto cleanup = [&]() {
        if (d_workspace) {
            CHECK_CUDA(cudaFree(d_workspace));
        }
        if (d_compress_buffer) {
            CHECK_CUDA(cudaFree(d_compress_buffer));
        }
        if (d_compressed) {
            CHECK_CUDA(cudaFree(d_compressed));
        }
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
        CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matD));
        CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
        CHECK_CUSPARSE(cusparseLtDestroy(&handle));
    };

    CHECK_CUSPARSE(cusparseLtMatmul(&handle,
                                    &plan,
                                    &alpha,
                                    is_A_sparse ? d_compressed : d_A,
                                    is_B_sparse ? d_compressed : d_B,
                                    &beta,
                                    d_C,
                                    d_D,
                                    d_workspace,
                                    streams,
                                    num_streams));
    CHECK_CUDA(cudaDeviceSynchronize());

    const int  num_runs = 10;
    float      total_ms = 0.0f;
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int run = 0; run < num_runs; ++run) {
        CHECK_CUDA(cudaEventRecord(start, stream));
        CHECK_CUSPARSE(cusparseLtMatmul(&handle,
                                        &plan,
                                        &alpha,
                                        is_A_sparse ? d_compressed : d_A,
                                        is_B_sparse ? d_compressed : d_B,
                                        &beta,
                                        d_C,
                                        d_D,
                                        d_workspace,
                                        streams,
                                        num_streams));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        total_ms += elapsed;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg_ms = total_ms / static_cast<float>(num_runs);
    double dense_ops   = 2.0 * static_cast<double>(m) * n * k;
    double sparse_ops  = dense_ops * 0.5; // 50% structured sparsity
    double tops        = sparse_ops / (avg_ms / 1e3) / 1e12;
    std::printf("[Timing] m=%d n=%d k=%d avg=%.3f ms throughput=%.3f TOPS\n",
                m, n, k, avg_ms, tops);

    cleanup();
    return avg_ms;
}

template <typename T>
T fetch_element(const T* data, int row, int col, char layout, int ld) {
    if (layout == 'R') {
        return data[row * ld + col];
    }
    return data[col * ld + row];
}

int main() {
    std::cout << "=== cuSPARSELt Structured Sparse Demo ===" << std::endl;

    std::vector<int> dimensions = {512, 1024};

    // Matrix multiplication parameters
    const float      alpha      = 1.0f;
    const float      beta       = 0.0f;
    const char       trans_A    = 'N';
    const char       trans_B    = 'N';
    const char       layout_A   = 'C';
    const char       layout_B   = 'R';
    const char       layout_C   = 'R';
    const bool       is_A_sparse = false; // toggle if you want sparse B instead

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-1, 1);

    for (int dim : dimensions) {
        const int m = dim;
        const int n = dim;
        const int k = dim;

        using AB_t       = int8_t;
        using C_t        = int;
        using D_t        = int;
        using COMPUTE_t  = int;

        const int lda = (layout_A == 'R') ? k : m;
        const int ldb = (layout_B == 'R') ? n : k;
        const int ldc = (layout_C == 'R') ? n : m;

    const size_t matrix_A_rows = (layout_A == 'R') ? m : k;
    const size_t matrix_B_rows = (layout_B == 'R') ? k : n;
    const size_t matrix_C_rows = (layout_C == 'R') ? m : n;
    const size_t size_A_bytes  = static_cast<size_t>(lda) * matrix_A_rows * sizeof(AB_t);
    const size_t size_B_bytes  = static_cast<size_t>(ldb) * matrix_B_rows * sizeof(AB_t);
    const size_t size_C_bytes  = static_cast<size_t>(ldc) * matrix_C_rows * sizeof(C_t);
    const size_t size_D_bytes  = static_cast<size_t>(ldc) * matrix_C_rows * sizeof(D_t);

        std::vector<AB_t> h_A(m * k);
        std::vector<AB_t> h_B(k * n);
        std::vector<C_t>  h_C(m * n, 0);
        std::vector<D_t>  h_D(m * n, 0);

        for (auto& val : h_A) {
            val = static_cast<AB_t>(dist(rng));
        }
        for (auto& val : h_B) {
            val = static_cast<AB_t>(dist(rng));
        }

        AB_t* d_A = nullptr;
        AB_t* d_B = nullptr;
        C_t*  d_C = nullptr;
        D_t*  d_D = nullptr;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A), size_A_bytes));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B), size_B_bytes));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_C), size_C_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_D), size_D_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), size_C_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_D, h_D.data(), size_D_bytes, cudaMemcpyHostToDevice));

        std::printf("\n[Run] dim=%d (sparse %s)\n", dim, is_A_sparse ? "A" : "B");
        benchmark_cusparselt<AB_t, C_t, D_t, COMPUTE_t>(trans_A,
                                                        trans_B,
                                                        layout_A,
                                                        layout_B,
                                                        layout_C,
                                                        is_A_sparse,
                                                        m,
                                                        n,
                                                        k,
                                                        alpha,
                                                        d_A,
                                                        lda,
                                                        d_B,
                                                        ldb,
                                                        beta,
                                                        d_C,
                                                        ldc,
                                                        d_D);

        CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, size_D_bytes, cudaMemcpyDeviceToHost));
        if (is_A_sparse) {
            CHECK_CUDA(cudaMemcpy(h_A.data(), d_A, size_A_bytes, cudaMemcpyDeviceToHost));
        } else {
            CHECK_CUDA(cudaMemcpy(h_B.data(), d_B, size_B_bytes, cudaMemcpyDeviceToHost));
        }

        if (dim <= 512) {
            bool correct = true;
            for (int row = 0; row < m && correct; ++row) {
                for (int col = 0; col < n && correct; ++col) {
                    COMPUTE_t acc = 0;
                    for (int kk = 0; kk < k; ++kk) {
                        AB_t a_val = (trans_A == 'N')
                                             ? fetch_element(h_A.data(), row, kk, layout_A, lda)
                                             : fetch_element(h_A.data(), kk, row, layout_A, lda);
                        AB_t b_val = (trans_B == 'N')
                                             ? fetch_element(h_B.data(), kk, col, layout_B, ldb)
                                             : fetch_element(h_B.data(), col, kk, layout_B, ldb);
                        acc += static_cast<COMPUTE_t>(a_val) * static_cast<COMPUTE_t>(b_val);
                    }
                    C_t expected = static_cast<C_t>(alpha * static_cast<float>(acc));
                    C_t got      = fetch_element(h_D.data(), row, col, layout_C, ldc);
                    if (expected != got) {
                        std::fprintf(stderr,
                                     "Mismatch at (%d,%d): expected %d got %d\n",
                                     row,
                                     col,
                                     expected,
                                     got);
                        correct = false;
                    }
                }
            }
            if (correct) {
                std::cout << "[Check] CPU vs GPU match." << std::endl;
            } else {
                std::cout << "[Check] Validation failed." << std::endl;
            }
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_D));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "\n=== Demo finished ===" << std::endl;
    return 0;
}
