// 编译命令:
// $ nvcc -o t2sp_int8 t2sp.cu -lcusparseLt -lcublas -DUSE_INT8  // 使用INT8模式
// $ nvcc -o t2sp_fp16 t2sp.cu -lcusparseLt -lcublas            // 使用FP16模式
// $ nsys nvprof --print-gpu-trace ./t2sp_int8                  // 性能分析

#include <cusparseLt.h> // 引入 cuSPARSELt 库的头文件，提供结构化稀疏矩阵操作功能
#include <cublas_v2.h>  // 引入 cuBLAS 库的头文件，用于参考和比较
#include <iostream>     // 标准输入输出流
#include <vector>       // 向量容器
#include <random>       // 随机数生成功能

// 根据编译标志选择不同的数据类型和计算精度
#ifdef USE_INT8
// INT8 模式：使用8位整数进行矩阵存储，32位整数进行累加计算
using mt = int8_t;  // 矩阵元素类型 (matrix type)：8位有符号整数
using rt = int32_t; // 结果元素类型 (result type)：32位有符号整数
using st = int32_t; // 标量类型 (scalar type)：用于alpha和beta
cudaDataType Atype = CUDA_R_8I;          // A和B矩阵的数据类型：8位有符号整数
cudaDataType Ctype = CUDA_R_32I;         // C矩阵的数据类型：32位有符号整数
cusparseComputeType computeType = CUSPARSE_COMPUTE_32I; // 累加计算类型：32位整数
#else
// FP16 模式：使用16位半精度浮点数进行矩阵存储和计算
#include <cuda_fp16.h> // 引入半精度浮点数支持
using mt = half;  // 矩阵元素类型：16位半精度浮点数
using rt = half;  // 结果元素类型：16位半精度浮点数
using st = float; // 标量类型：32位单精度浮点数（cuSPARSELt中半精度计算通常使用单精度标量）
cudaDataType Atype = CUDA_R_16F;         // A和B矩阵的数据类型：16位半精度浮点数
cudaDataType Ctype = CUDA_R_16F;         // C矩阵的数据类型：16位半精度浮点数
cusparseComputeType computeType = CUSPARSE_COMPUTE_16F; // 累加计算类型：16位半精度浮点数
#endif


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

/**
 * 随机初始化矩阵函数
 * 根据数据类型生成适当范围的随机值填充矩阵
 * 
 * @param matrix 指向矩阵数据的指针
 * @param size 矩阵元素总数
 */
template <typename T>
void initializeMatrix(T* matrix, size_t size) {
    std::random_device rd;  // 获取随机种子
    std::mt19937 gen(rd()); // 使用Mersenne Twister算法的随机数生成器
    
#ifdef USE_INT8
    // INT8模式：生成-5到5之间的随机整数，避免溢出风险
    std::uniform_int_distribution<> dis(-5, 5);
    for (size_t i = 0; i < size; i++) {
        matrix[i] = static_cast<T>(dis(gen));
    }
#else
    // FP16模式：生成-1.0到1.0之间的随机浮点数
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; i++) {
        matrix[i] = static_cast<T>(dis(gen));
    }
#endif
}

int main() {
    // 定义要测试的矩阵维度数组（所有维度均为方阵 M=N=K）
    // 从较小尺寸开始，逐步增大以观察性能变化
    std::vector<int> dimensions = {512, 1024, 2048, 4096, 8192};
    
    // 每个维度下重复执行的次数，用于获取稳定的性能数据
    const int num_runs = 100;
    
    // 步骤1: 初始化cuSPARSELt库句柄
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));
    
    // 步骤2: 创建CUDA事件对象，用于精确测量GPU执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // 步骤3: 遍历所有测试维度，分别进行测试
    for (int dim : dimensions) {
        // 设置当前测试的矩阵维度：M是A的行数，N是B的列数，K是A的列数和B的行数
        int m = dim; // 矩阵A的行数，矩阵C的行数
        int n = dim; // 矩阵B的列数，矩阵C的列数
        int k = dim; // 矩阵A的列数，矩阵B的行数
        
        std::cout << "正在测试矩阵维度: " << m << "x" << n << "x" << k << std::endl;
        
        // 步骤4: 在主机内存中分配并初始化矩阵
        std::vector<mt> h_A(m * k); // 矩阵A: m行k列
        std::vector<mt> h_B(k * n); // 矩阵B: k行n列
        
        // 使用随机值初始化两个矩阵
        initializeMatrix(h_A.data(), h_A.size());
        initializeMatrix(h_B.data(), h_B.size());
        
        // 步骤5: 在GPU设备内存中分配矩阵空间
        mt *d_A, *d_B;  // 输入矩阵
        rt *d_C;        // 输出矩阵
        int*   d_valid;            // 设备端标志位：用于验证稀疏化是否正确
        CHECK_CUDA(cudaMalloc(&d_A, sizeof(mt) * m * k)); // 为矩阵A分配设备内存
        CHECK_CUDA(cudaMalloc(&d_B, sizeof(mt) * k * n)); // 为矩阵B分配设备内存
        CHECK_CUDA(cudaMalloc(&d_C, sizeof(rt) * m * n)); // 为结果矩阵C分配设备内存
        CHECK_CUDA(cudaMalloc((void**) &d_valid, sizeof(int))); // 为验证标志分配 GPU 内存

        // 步骤6: 将矩阵数据从主机复制到设备
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeof(mt) * m * k, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeof(mt) * k * n, cudaMemcpyHostToDevice));
        
        // 步骤7: 设置矩阵乘法的标量参数alpha和beta
        // C = alpha * (A^T * B) + beta * C
#ifdef USE_INT8
        st alpha_val = 1; // 标量α设为1
        st beta_val = 0;  // 标量β设为0
#else
        st alpha_val = 1.0f;
        st beta_val = 0.0f;
#endif
        // cuSPARSELt需要使用指向主机内存的指针
        st *alpha = &alpha_val;
        st *beta = &beta_val;
        
        // 步骤8: 设置矩阵描述符，定义矩阵属性
        cusparseLtMatDescriptor_t matA, matB, matC;
        int alignment = 16; // 内存对齐要求，通常为16字节
        
        // 初始化矩阵A的描述符
        // 注意：cuSPARSELt使用列主序存储，主维度为行数
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle,           // cuSPARSELt句柄
            &matA,             // 要初始化的描述符
            m, k,              // 矩阵A的行数和列数
            m,                 // 主维度 (leadingDimension)：列主序中为行数
            alignment,         // 内存对齐要求
            Atype,             // 矩阵元素数据类型
            CUSPARSE_ORDER_COL // 存储顺序：列主序
        ));
        
        // 初始化矩阵B的描述符（此矩阵将被稀疏化）
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &matB, k, n,
            k,                 // 主维度：列主序中为行数
            alignment, Atype,
            CUSPARSE_ORDER_COL // 存储顺序：列主序
        ));
        
        // 初始化矩阵C的描述符（结果矩阵）
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &matC, m, n,
            m,                 // 主维度：列主序中为行数
            alignment, Ctype,
            CUSPARSE_ORDER_COL // 存储顺序：列主序
        ));
        
        // 步骤9: 创建稀疏矩阵描述符
        // cuSPARSELt使用结构化稀疏矩阵，不需要显式的稀疏化步骤
        cusparseLtMatDescriptor_t matB_sparse;
        cusparseLtMatmulDescriptor_t matmul;
        cudaStream_t    stream = nullptr; // CUDA 流（使用默认流）

        // 为结构化稀疏创建专门的描述符
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            &handle, &matB_sparse, k, n,
            k, // 主维度
            alignment, Atype,
            CUSPARSE_ORDER_COL,
            CUSPARSELT_SPARSITY_50_PERCENT // 50%稀疏度
        ));
        
        // 设置稀疏矩阵指针，告诉 cuSPARSELt 哪个是稀疏矩阵
        CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&handle,
                                                    &matmul,                           // 矩阵乘法描述符
                                                    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, // 设置稀疏矩阵指针属性
                                                    &matB_sparse,                               // 矩阵 A 的设备指针
                                                    sizeof(matB_sparse))); 

        // TILE 模式：按照 2:4 结构化稀疏模式剪枝（每4个元素中保留2个最大的）
        CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, d_B, d_B,          // 输入和输出都是 dA（就地操作）
                                            CUSPARSELT_PRUNE_SPMMA_TILE, stream) ) // 使用 TILE 剪枝模式
        
        // 验证剪枝后的矩阵是否符合结构化稀疏要求
        // 如果不符合，后续的矩阵乘法将产生错误结果
        CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, d_B,         // 检查剪枝后的矩阵 A
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
                                                
        // 剪枝和压缩矩阵B
        CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
            &handle, &matB_sparse, &compressedSize
        ));
        
        CHECK_CUSPARSE(cusparseLtSpMMACompress(
            &handle, &matB_sparse, d_B_compressed, d_buffer
        ));


        // 获取压缩后矩阵所需的内存大小
        size_t compressedSize, compressedBufferSize;
        CHECK_CUSPARSE(cusparseLtSpMMaBufSize(
            &handle, &matB_sparse, &compressedSize, &compressedBufferSize
        ));
        
        // 为压缩后的稀疏矩阵分配内存
        void* d_B_compressed;
        void* d_buffer;
        CHECK_CUDA(cudaMalloc(&d_B_compressed, compressedSize));
        if (compressedBufferSize > 0) {
            CHECK_CUDA(cudaMalloc(&d_buffer, compressedBufferSize));
        } else {
            d_buffer = nullptr;
        }

        
        // 步骤14: 创建矩阵乘法描述符，设置操作和数据类型
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            &handle,                       // cuSPARSELt句柄
            &matmul,                       // 要初始化的矩阵乘法描述符
            CUSPARSE_OPERATION_TRANSPOSE,  // A矩阵操作：转置
            CUSPARSE_OPERATION_NON_TRANSPOSE, // B矩阵操作：不转置
            &matA, &matB_sparse, &matC, &matC,    // 输入和输出矩阵描述符，使用稀疏矩阵描述符
            computeType                    // 计算类型
        ));
        
        // 步骤15: 创建矩阵乘法计划，优化执行路径
        cusparseLtMatmulPlan_t plan;
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, nullptr));
        
        // 步骤16: 确定矩阵乘法操作所需的工作区大小
        size_t workspaceSize = 0;
        CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize));
        
        // 步骤16: 如果需要，分配工作区内存
        void* workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
        }
        
        // 步骤17: 创建矩阵乘法计划，优化执行路径
        cusparseLtMatmulPlan_t plan;
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, nullptr));
        
        // 步骤18: 执行预热运行，初始化GPU状态并避免首次执行的额外开销
        CHECK_CUSPARSE(cusparseLtMatmul(
            &handle,        // cuSPARSELt句柄
            &plan,          // 矩阵乘法计划
            alpha,          // 标量α
            d_A,            // 矩阵A（密集格式）
            d_B_compressed, // 矩阵B（压缩的稀疏格式）
            beta,           // 标量β
            d_C,            // 结果矩阵C
            d_C,            // 输出矩阵C
            workspace,      // 工作区内存
            nullptr,        // CUDA流，nullptr表示使用默认流
            0               // 附加选项
        ));
        
        // 步骤19: 初始化计时变量
        float total_time = 0.0f;
        
        // 步骤20: 多次执行稀疏矩阵乘法以获取平均执行时间
        for (int i = 0; i < num_runs; ++i) {
            // 记录开始时间
            CHECK_CUDA(cudaEventRecord(start));
            
            // 执行结构化稀疏矩阵乘法: C = alpha * (A^T * B_sparse) + beta * C
            CHECK_CUSPARSE(cusparseLtMatmul(
                &handle, &plan, alpha, d_A, d_B_compressed, beta, d_C, d_C, workspace,
                nullptr, 0
            ));
            
            // 记录结束时间
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop)); // 等待GPU完成所有工作
            
            // 计算本次执行的耗时（毫秒）
            float milliseconds = 0;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            total_time += milliseconds;
        }
        
        // 步骤21: 计算平均执行时间
        float avg_time = total_time / num_runs;
        
        // 步骤22: 计算计算吞吐量
        // 矩阵乘法的理论操作数为 2*M*N*K（乘加操作）
        // 结构化稀疏（50%稀疏度）时，实际操作数约为 2*M*N*K*0.5
        double ops = 2.0 * m * n * k * 0.5;
        // 吞吐量 = 操作数 / 时间(秒)，结果单位为 TOPS (每秒万亿次操作)
        double throughput = ops / (avg_time / 1000.0) / 1e12;
        
        // 步骤23: 输出当前维度的测试结果
        std::cout << "维度: " << dim << "x" << dim << "x" << dim
                  << ", 平均耗时: " << avg_time << " 毫秒"
                  << ", 吞吐量(50%稀疏): " << throughput << " TOPS" << std::endl;
        
        // 步骤24: 释放本轮测试分配的资源
        CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_B_compressed));
        CHECK_CUDA(cudaFree(d_C));
        if (workspace) {
            CHECK_CUDA(cudaFree(workspace));
        }
        if (d_buffer) {
            CHECK_CUDA(cudaFree(d_buffer));
        }
    }
    
    // 步骤25: 释放全局资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseLtDestroy(&handle));
    
    // 步骤26: 检查是否有异步CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA异步错误: " << cudaGetErrorString(err) << std::endl;
    }
    
    std::cout << "测试完成！" << std::endl;
    return 0;
}