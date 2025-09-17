/******************************************************************************
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************/

/**
 * cuSPARSELt 结构化稀疏矩阵乘法示例程序
 * 
 * 本程序演示如何使用 cuSPARSELt 库进行高效的结构化稀疏矩阵乘法运算 C = α * A * B + β * C
 * 其中 A 是结构化稀疏矩阵（50% 稀疏度），B 和 C 是稠密矩阵
 * 
 * 编译和运行步骤：
 * mkdir build
 * cd build
 * cmake -DCUSPARSELT_PATH=/usr -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc ..
 * make
 */

// ======================= 头文件包含 =======================
#include <cuda_runtime_api.h> // CUDA 运行时 API：cudaMalloc, cudaMemcpy 等
#include <cusparseLt.h>       // cuSPARSELt 库头文件：结构化稀疏矩阵乘法
#include <cstdio>             // C 标准输入输出：printf
#include <cstdlib>            // C 标准库：std::rand 随机数生成
                              
#include <cuda_fp8.h>         // CUDA FP8 数据类型支持

// ======================= 数据类型选择宏定义 =======================
// 定义三种数据类型的标识符，用于条件编译
#define FP16 1000   // 半精度浮点数（16位）
#define INT8 1001   // 8位有符号整数  
#define FP8  1002   // 8位浮点数（较新的数据类型）

/*
 * 选择矩阵 A 和 B 的数据类型
 * 可以通过修改这里来测试不同精度的性能
 */
#define AB_TYPE FP16    // 当前使用 FP16 精度
// #define AB_TYPE FP8  // 可选：使用 FP8 精度（需要较新的 GPU 支持）
// #define AB_TYPE INT8 // 可选：使用 INT8 精度（整数运算）

// ======================= 条件编译：数据类型定义 =======================
// 根据选择的数据类型，定义对应的类型别名和 CUDA 数据类型
#if AB_TYPE == FP8
    // FP8 配置：使用 8 位浮点数进行矩阵运算
    using AB_t         = __nv_fp8_e4m3;  // 矩阵 A, B 使用 FP8 E4M3 格式（4位指数，3位尾数）
    using C_t          = __half;         // 结果矩阵 C 使用 FP16（更高精度存储结果）
    using COMPUTE_t    = float;          // 计算过程使用 FP32（保证计算精度）
#elif AB_TYPE == FP16
    // FP16 配置：使用 16 位半精度浮点数（最常用的配置）
    using AB_t         = __half;         // 矩阵 A, B 使用半精度浮点数
    using C_t          = __half;         // 结果矩阵 C 也使用半精度
    using COMPUTE_t    = float;          // 计算过程使用 FP32（避免精度损失）
#elif AB_TYPE == INT8
    // INT8 配置：使用 8 位整数进行矩阵运算（适用于量化模型）
    using AB_t         = int8_t;         // 矩阵 A, B 使用 8 位有符号整数
    using C_t          = int8_t;         // 结果矩阵 C 可以是 int8_t, __half, __nv_bfloat16 等
    using COMPUTE_t    = int;            // 计算过程使用 32 位整数
#endif
                              
// ======================= 类型映射模板：C++ 类型 → CUDA 数据类型 =======================
// 这些模板结构体将 C++ 数据类型映射为对应的 CUDA 数据类型枚举
// cuSPARSELt API 需要明确的 cudaDataType 来识别数据格式

template <typename value_t>
struct cuda_type { }; // 基础模板（未特化，不应直接使用）

// 特化：半精度浮点数 __half → CUDA_R_16F
template <>
struct cuda_type <__half> {
    static constexpr cudaDataType value = CUDA_R_16F;
};

// 特化：Brain Float 16 __nv_bfloat16 → CUDA_R_16BF  
template <>
struct cuda_type <__nv_bfloat16> {
    static constexpr cudaDataType value = CUDA_R_16BF;
};

// 特化：FP8 E4M3 格式 __nv_fp8_e4m3 → CUDA_R_8F_E4M3
template <>
struct cuda_type <__nv_fp8_e4m3> {
    static constexpr cudaDataType value = CUDA_R_8F_E4M3;
};

// 特化：8位有符号整数 int8_t → CUDA_R_8I
template <>
struct cuda_type <int8_t> {
    static constexpr cudaDataType value = CUDA_R_8I;
};

// 特化：32位有符号整数 int → CUDA_R_32I
template <>
struct cuda_type <int> {
    static constexpr cudaDataType value = CUDA_R_32I;
};

// ======================= 计算类型映射模板 =======================
// 将 C++ 计算类型映射为 cuSPARSE 计算类型枚举
// 这决定了矩阵乘法内部计算使用的精度

template <typename value_t>
struct cusparse_compute_type {  }; // 基础模板

// 特化：float 计算类型 → CUSPARSE_COMPUTE_32F（32位浮点运算）
template <>
struct cusparse_compute_type<float> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32F;
};

// 特化：int 计算类型 → CUSPARSE_COMPUTE_32I（32位整数运算）
template <>
struct cusparse_compute_type<int> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32I;
};

// ======================= 错误检查宏定义 =======================
// 这两个宏用于简化 CUDA 和 cuSPARSE API 的错误检查，避免重复的错误处理代码

// CUDA API 错误检查宏
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",   \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// cuSPARSE API 错误检查宏
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n", \
               __LINE__, cusparseLtGetErrorString(status), status);            \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// 定义不支持的设备退出码
constexpr int EXIT_UNSUPPORTED = 2;


// ======================= 主函数开始 =======================
int main(void) {
    // -------------------- 第一步：GPU 兼容性检查 --------------------
    // cuSPARSELt 只支持特定计算能力的 GPU，需要先检查当前设备
    int major_cc, minor_cc; // 存储计算能力版本号
    
    // 获取 GPU 计算能力的主版本号（如 8.0 中的 8）
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    // 获取 GPU 计算能力的副版本号（如 8.0 中的 0）                                   
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    
    // 检查是否为 cuSPARSELt 支持的 GPU 型号
    // 支持的计算能力：8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.1, 11.0, 12.0, 12.1
    if (!(major_cc == 8 && minor_cc == 0) &&   // A100
        !(major_cc == 8 && minor_cc == 6) &&   // RTX 30 系列
        !(major_cc == 8 && minor_cc == 7) &&   // A10, A16 等
        !(major_cc == 8 && minor_cc == 9) &&   // RTX 40 系列
        !(major_cc == 9 && minor_cc == 0) &&   // H100
        !(major_cc == 10 && minor_cc == 0) &&  
        !(major_cc == 10 && minor_cc == 1) &&  
        !(major_cc == 11 && minor_cc == 0) &&  
        !(major_cc == 12 && minor_cc == 0) &&  
        !(major_cc == 12 && minor_cc == 1)) {
        std::printf("\ncuSPARSELt 仅支持计算能力为 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.1, 11.0, 12.0, 12.1 的 GPU\n"
                    "当前设备计算能力：%d.%d\n\n", major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // -------------------- 第二步：问题参数定义 --------------------
    // 定义矩阵乘法的基本参数：C = α * A * B + β * C
    // 使用相对较小的尺寸以便快速测试（生产环境中通常使用更大的矩阵）
    constexpr int m            = 32;    // 矩阵 A 的行数，矩阵 C 的行数
    constexpr int n            = 32;    // 矩阵 B 的列数，矩阵 C 的列数
    constexpr int k            = 64;    // 矩阵 A 的列数，矩阵 B 的行数（内积维度）

    // 矩阵存储和操作参数配置
    auto     order          = CUSPARSE_ORDER_ROW;           // 行主序存储（C/C++ 默认）
    auto     opA            = CUSPARSE_OPERATION_NON_TRANSPOSE; // A 矩阵不转置
    auto     opB            = CUSPARSE_OPERATION_TRANSPOSE;     // B 矩阵转置（常见的优化策略）
    auto     type_AB        = cuda_type<AB_t>::value;       // A, B 矩阵的 CUDA 数据类型
    auto     type_C         = cuda_type<C_t>::value;        // C 矩阵的 CUDA 数据类型
    auto     compute_type   = cusparse_compute_type<COMPUTE_t>::value; // 计算精度
    bool     matmul_search  = true;                         // 是否启用算法搜索优化
    
    // 根据转置操作计算实际的矩阵布局
    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);      // 是否行主序
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE); // A 是否转置
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE); // B 是否转置
    // 考虑转置操作后的实际矩阵维度
    // 如果 A 转置：[m,k] → [k,m]，如果 B 转置：[k,n] → [n,k]
    auto     num_A_rows     = (isA_transposed) ? k : m;     // A 矩阵实际行数
    auto     num_A_cols     = (isA_transposed) ? m : k;     // A 矩阵实际列数
    auto     num_B_rows     = (isB_transposed) ? n : k;     // B 矩阵实际行数
    auto     num_B_cols     = (isB_transposed) ? k : n;     // B 矩阵实际列数
    auto     num_C_rows     = m;                            // C 矩阵行数（固定）
    auto     num_C_cols     = n;                            // C 矩阵列数（固定）
    
    unsigned alignment      = 16;                           // 内存对齐要求（字节）
    
    // 计算 leading dimension（矩阵存储时每行的跨度）
    // 行主序：lda = 列数，列主序：lda = 行数
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows; // A 的 leading dimension
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows; // B 的 leading dimension  
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows; // C 的 leading dimension
    
    // 计算实际需要分配的矩阵高度（考虑存储顺序）
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    
    // 计算各矩阵需要的内存大小（字节）
    auto     A_size         = A_height * lda * sizeof(AB_t);
    auto     B_size         = B_height * ldb * sizeof(AB_t);
    auto     C_size         = C_height * ldc * sizeof(C_t);

    // -------------------- 第三步：主机内存分配和初始化 --------------------
    // 在主机（CPU）内存中分配矩阵存储空间
    auto     hA             = new AB_t[A_size / sizeof(AB_t)]; // 主机端 A 矩阵
    auto     hB             = new AB_t[B_size / sizeof(AB_t)]; // 主机端 B 矩阵  
    auto     hC             = new C_t[C_size / sizeof(C_t)];   // 主机端 C 矩阵

    // 使用随机数初始化矩阵 A（将被稀疏化）
    // 值域：-2 到 +2，这样的小值有利于数值稳定性
    for (int i = 0; i < m * k; i++) 
        hA[i] = static_cast<AB_t>(static_cast<float>(std::rand() % 5 - 2)); // -2 ~ 2

    // 使用随机数初始化矩阵 B（稠密矩阵）
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<AB_t>(static_cast<float>(std::rand() % 5 - 2));

    // 使用随机数初始化矩阵 C（用于测试 β * C 项）
    for (int i = 0; i < m * n; i++)
        hC[i] = static_cast<C_t>(static_cast<float>(std::rand() % 5 - 2));

    // 定义标量乘数：C = α * A * B + β * C
    float alpha = 1.0f; // α = 1，矩阵乘积的权重
    float beta  = 1.0f; // β = 1，原矩阵 C 的权重（相当于累加操作）

    // -------------------- 第四步：设备内存管理 --------------------
    // 在 GPU 设备内存中分配空间存储矩阵数据
    
    AB_t* dA, *dB, *dA_compressed;  // 设备端矩阵指针：原始A，B矩阵，压缩后的A矩阵
    C_t* dC, *dD;                   // 设备端结果矩阵：输入C，输出D（可以是同一个）
    int*   d_valid;                 // 设备端标志位：用于验证稀疏化是否正确

    // 分配 GPU 内存
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )      // 为矩阵 A 分配 GPU 内存
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )      // 为矩阵 B 分配 GPU 内存  
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )      // 为矩阵 C 分配 GPU 内存
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) ) // 为验证标志分配 GPU 内存
    dD = dC;  // 结果矩阵 D 与输入矩阵 C 使用相同内存（就地操作）

    // 将主机数据拷贝到 GPU
    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) ) // 拷贝 A 矩阵
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) ) // 拷贝 B 矩阵
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) ) // 拷贝 C 矩阵
    // -------------------- 第五步：cuSPARSELt 对象初始化 --------------------
    // 创建 cuSPARSELt 所需的各种描述符和句柄
    
    cusparseLtHandle_t             handle;   // cuSPARSELt 库句柄（类似上下文）
    cusparseLtMatDescriptor_t      matA, matB, matC; // 三个矩阵的描述符
    cusparseLtMatmulDescriptor_t   matmul;   // 矩阵乘法操作描述符
    cusparseLtMatmulAlgSelection_t alg_sel;  // 算法选择描述符
    cusparseLtMatmulPlan_t         plan;     // 执行计划描述符
    cudaStream_t                   stream = nullptr; // CUDA 流（使用默认流）

    // 初始化 cuSPARSELt 句柄
    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    // -------------------- 第六步：矩阵描述符初始化 --------------------
    // 为三个矩阵分别创建描述符，告诉 cuSPARSELt 每个矩阵的属性
    
    // 初始化矩阵 A 的描述符（结构化稀疏矩阵，50% 稀疏度）
    // 结构化稀疏意味着稀疏模式是规律的，可以被硬件加速
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,  // 句柄，描述符，行数
                                            num_A_cols, lda, alignment, // 列数，leading dimension，对齐
                                            type_AB, order,             // 数据类型，存储顺序
                                            CUSPARSELT_SPARSITY_50_PERCENT) ) // 50% 稀疏度

    // 初始化矩阵 B 的描述符（稠密矩阵）
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,  // 句柄，描述符，行数
                                            num_B_cols, ldb, alignment, // 列数，leading dimension，对齐  
                                            type_AB, order) )           // 数据类型，存储顺序
                                            
    // 初始化矩阵 C 的描述符（稠密矩阵，存储结果）                                        
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,  // 句柄，描述符，行数
                                            num_C_cols, ldc, alignment, // 列数，leading dimension，对齐
                                            type_C, order) )            // 数据类型，存储顺序

    // -------------------- 第七步：矩阵乘法操作配置 --------------------
    // 配置矩阵乘法的具体参数和执行策略
    
    // 初始化矩阵乘法描述符，定义操作 C = α * op(A) * op(B) + β * C
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,     // 句柄，描述符，A和B的操作
                                            &matA, &matB, &matC, &matC,     // 输入矩阵A,B和输出矩阵C（输入和输出C相同）
                                            compute_type) )                 // 计算精度类型

    // 初始化算法选择，选择默认的矩阵乘法算法
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,     // 句柄，算法选择，矩阵乘法描述符
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) ) // 使用默认算法

    // 初始化执行计划，将所有配置组合成可执行的计划
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel))

    // 设置稀疏矩阵指针，告诉 cuSPARSELt 哪个是稀疏矩阵
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle,
                                                    &matmul,                           // 矩阵乘法描述符
                                                    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, // 设置稀疏矩阵指针属性
                                                    &dA,                               // 矩阵 A 的设备指针
                                                    sizeof(dA)));                      // 指针大小

    // -------------------- 第八步：矩阵稀疏化和验证 --------------------
    // 将随机初始化的矩阵 A 转换为符合 cuSPARSELt 要求的结构化稀疏矩阵
    
    // 对矩阵 A 进行就地剪枝（pruning），生成结构化稀疏模式
    // TILE 模式：按照 2:4 结构化稀疏模式剪枝（每4个元素中保留2个最大的）
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,          // 输入和输出都是 dA（就地操作）
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) ) // 使用 TILE 剪枝模式
    
    // 验证剪枝后的矩阵是否符合结构化稀疏要求
    // 如果不符合，后续的矩阵乘法将产生错误结果
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,         // 检查剪枝后的矩阵 A
                                              d_valid, stream) )            // 结果存储在 d_valid 中
    
    // 将验证结果从 GPU 拷贝到 CPU
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),           // 异步拷贝验证结果
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )                            // 等待拷贝完成
    
    // 检查剪枝是否成功
    if (is_valid != 0) {
        std::printf("错误：矩阵剪枝失败！矩阵不符合结构化稀疏要求，"
                    "cuSPARSELt 将无法提供正确的计算结果\n");
        return EXIT_FAILURE;
    }
    // -------------------- 第九步：稀疏矩阵压缩 --------------------
    // 将结构化稀疏矩阵压缩为更紧凑的格式，提高存储效率和计算性能
    
    size_t compressed_size, compressed_buffer_size;  // 压缩后的大小和临时缓冲区大小
    void*  dA_compressedBuffer;                      // 压缩操作的临时缓冲区
    
    // 查询压缩后矩阵所需的内存大小
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,           // 基于执行计划查询
                                                  &compressed_size,         // 压缩矩阵大小
                                                  &compressed_buffer_size) ) // 临时缓冲区大小
    
    // 分配压缩矩阵和临时缓冲区的 GPU 内存                                              
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )      // 压缩后的矩阵 A
    CHECK_CUDA( cudaMalloc((void**) &dA_compressedBuffer,                   // 压缩临时缓冲区
                           compressed_buffer_size) )

    // 执行矩阵压缩操作
    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, // 从原始 A 压缩到 dA_compressed
                                            dA_compressedBuffer,stream) )      // 使用临时缓冲区

    // -------------------- 第十步：算法搜索优化（可选） --------------------
    // cuSPARSELt 支持多种算法实现，可以搜索最优算法以获得最佳性能
    
    int           num_streams = 0;      // 不使用多流并行
    cudaStream_t* streams     = nullptr; // 流数组为空

    if (matmul_search) {
        // 执行算法搜索，尝试不同的实现并选择最快的
        // 注意：搜索过程会修改 dC，所以之后需要重新初始化 dC
        CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,      // 句柄，计划，α
                                               dA_compressed, dB, &beta,    // 压缩的A，矩阵B，β  
                                               dC, dD, nullptr,             // 输入C，输出D，无偏置
                                               streams, num_streams) )      // 流配置
        
        // 重置 dC，因为搜索过程中 dC 的值已被修改
        // 为了后续的正确性检查，需要恢复原始值
        CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    }

    // -------------------- 第十一步：执行矩阵乘法 --------------------
    // 现在所有准备工作已完成，执行实际的稀疏矩阵乘法运算
    
    size_t workspace_size; // 矩阵乘法所需的工作空间大小

    // 查询矩阵乘法运算所需的临时工作空间大小
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))
    
    // 分配工作空间内存
    void* d_workspace;
    CHECK_CUDA( cudaMalloc((void**) &d_workspace, workspace_size) )
    
    // 执行矩阵乘法：C = α * A * B + β * C
    // 这是整个程序的核心计算步骤
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, // α, 压缩的A, 矩阵B
                                     &beta, dC, dD, d_workspace, streams,       // β, 输入C, 输出D, 工作空间, 流
                                     num_streams) )                             // 流数量
    // -------------------- 第十二步：资源清理 --------------------
    // 销毁所有 cuSPARSELt 对象，释放相关资源
    
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )        // 销毁矩阵 A 描述符
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )        // 销毁矩阵 B 描述符
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )        // 销毁矩阵 C 描述符
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionDestroy(&alg_sel) ) // 销毁算法选择对象
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )           // 销毁执行计划
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )                   // 销毁 cuSPARSELt 句柄

    // -------------------- 第十三步：结果验证 --------------------
    // 将 GPU 计算结果与 CPU 计算的参考结果进行比较，验证正确性
    
    // 首先获取剪枝后的矩阵 A（因为 A 在 GPU 上被修改了）
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )

    // 确定矩阵的内存布局（用于正确访问矩阵元素）
    bool A_std_layout = (is_rowmajor != isA_transposed); // A 是否为标准布局
    bool B_std_layout = (is_rowmajor != isB_transposed); // B 是否为标准布局

    // 在 CPU 上计算参考结果进行对比
    C_t* hC_result = new C_t[C_height * ldc]; // 分配 CPU 计算结果的存储空间

    // 三重循环执行矩阵乘法：C[i][j] = Σ A[i][k] * B[k][j]
    for (int i = 0; i < m; i++) {           // 遍历结果矩阵的行
        for (int j = 0; j < n; j++) {       // 遍历结果矩阵的列
            COMPUTE_t sum  = static_cast<COMPUTE_t>(0); // 累加器，使用高精度避免误差
            
            // 内积计算：对 k 维度求和
            for (int k1 = 0; k1 < k; k1++) {
                // 根据内存布局计算 A[i][k] 和 B[k][j] 的线性索引
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda; // A 的位置
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb; // B 的位置
                sum      += static_cast<COMPUTE_t>(hA[posA]) *            // A[i][k]
                            static_cast<COMPUTE_t>(hB[posB]);             // B[k][j]
            }
            
            // 计算最终结果：C = α * A * B + β * C
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;  // C 的位置
            hC_result[posC] = static_cast<C_t>(alpha * sum + beta * static_cast<float>(hC[posC]));
        }
    }

    // 获取 GPU 计算的结果
    CHECK_CUDA( cudaMemcpy(hC, dD, C_size, cudaMemcpyDeviceToHost) )

    // 逐元素比较 GPU 结果和 CPU 参考结果
    int correct = 1; // 正确性标志
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = hC[pos];     // GPU 计算结果
            auto host_value   = hC_result[pos]; // CPU 参考结果
            
            // 比较结果（注意：直接浮点比较在生产环境中不够robust，应使用容差比较）
            if (device_value != host_value) {
                std::printf("结果不匹配 位置(%d, %d): CPU = %3.0f, GPU = %3.0f\n",
                            i, j, static_cast<float>(host_value), static_cast<float>(device_value));
                correct = 0;
                break;
            }
        }
    }

    // 输出测试结果
    if (correct) {
        std::printf("矩阵乘法测试 通过 ✓\n");
    }
    else {
        std::printf("矩阵乘法测试 失败 ✗ (计算结果错误)\n");
    }

    // -------------------- 第十四步：最终资源清理 --------------------
    
    // 释放主机（CPU）内存
    delete[] hA;        // 释放主机端矩阵 A
    delete[] hB;        // 释放主机端矩阵 B  
    delete[] hC;        // 释放主机端矩阵 C
    delete[] hC_result; // 释放 CPU 计算结果存储
    
    // 释放设备（GPU）内存
    CHECK_CUDA( cudaFree(dA_compressed) )     // 释放压缩的矩阵 A
    CHECK_CUDA( cudaFree(dA) )                // 释放原始矩阵 A
    CHECK_CUDA( cudaFree(dB) )                // 释放矩阵 B
    CHECK_CUDA( cudaFree(dC) )                // 释放矩阵 C
    CHECK_CUDA( cudaFree(d_valid) )           // 释放验证标志
    CHECK_CUDA( cudaFree(d_workspace) )       // 释放工作空间
    CHECK_CUDA( cudaFree(dA_compressedBuffer) ) // 释放压缩临时缓冲区

    return EXIT_SUCCESS; // 程序成功结束
} // main 函数结束

/*
 * 程序总结：
 * 
 * 本程序展示了如何使用 cuSPARSELt 库进行高效的结构化稀疏矩阵乘法：
 * 
 * 1. **设备兼容性检查**：确保 GPU 支持 cuSPARSELt
 * 2. **矩阵初始化**：创建随机矩阵数据
 * 3. **内存管理**：在 CPU 和 GPU 之间传输数据
 * 4. **稀疏化处理**：将稠密矩阵转换为结构化稀疏矩阵
 * 5. **压缩优化**：压缩稀疏矩阵以提高性能
 * 6. **算法优化**：搜索最佳计算算法
 * 7. **矩阵运算**：执行 C = α * A * B + β * C
 * 8. **结果验证**：与 CPU 计算结果比较确保正确性
 * 9. **资源清理**：释放所有分配的内存和对象
 * 
 * 关键优势：
 * - **性能**：结构化稀疏矩阵乘法比稠密矩阵乘法更快
 * - **内存**：稀疏格式减少内存使用
 * - **精度**：支持多种数据类型（FP16, INT8, FP8）
 * - **灵活性**：支持不同的矩阵操作（转置、缩放等）
 */