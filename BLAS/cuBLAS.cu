// $ nvcc -o t2int8 t2.cu -lcublas -DUSE_INT8 && ./t2int8
// $ nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true --stats=true --force-overwrite true --output=detail_res_t2int8 ./t2int8

#include <cublas_v2.h> // 引入 cuBLAS 库的头文件，这是使用 cuBLAS API 的必需文件
#include <iostream>      // 引入 C++ 标准输入输出流库，用于打印结果
#include <vector>        // 引入 C++ 标准向量库，用于存储待测试的矩阵维度

// 使用宏定义进行条件编译。如果在编译时添加了 -DUSE_INT8 标志，则会编译这部分代码
#ifdef USE_INT8
using mt = char; // 定义 mt (matrix type) 为 char 类型，即 8 位有符号整数 (INT8)
using rt = int;  // 定义 rt (result type) 为 int 类型，即 32 位有符号整数 (INT32)，用于存储计算结果
using st = int;  // 定义 st (scale type) 为 int 类型，用于 alpha 和 beta 标量
cudaDataType   Atype = CUDA_R_8I;       // A 和 B 矩阵的数据类型设置为 8 位有符号整数
cudaDataType   Ctype = CUDA_R_32I;      // C 矩阵的数据类型设置为 32 位有符号整数
cublasComputeType_t   computeType = CUBLAS_COMPUTE_32I; // 计算过程中的累加精度设置为 32 位整数
#else

// 如果没有定义 USE_INT8，则默认使用 FP16 (半精度浮点数)
#include <cuda_fp16.h> // 引入半精度浮点数的头文件
using mt = half; // 定义矩阵类型为 half
using rt = half; // 定义结果类型为 half
using st = half; // 定义标量类型为 half
cudaDataType   Atype = CUDA_R_16F;       // A, B 矩阵类型为 16 位浮点数
cudaDataType   Ctype = CUDA_R_16F;       // C 矩阵类型为 16 位浮点数
cublasComputeType_t   computeType = CUBLAS_COMPUTE_16F; // 计算过程累加精度为 16 位浮点数
#endif

int main(){
  // 定义一个向量，用于存储需要测试的多个矩阵维度 (M=N=K)
  std::vector<int> dimensions = {512, 1024, 2048, 4096, 8192, 12288, 16384};
  // 为每个维度下的 GEMM 运算重复执行的次数，以获取稳定和平均的性能数据
  const int num_runs = 100;

  cublasHandle_t h; // 声明一个 cuBLAS 句柄，它是后续所有 cuBLAS API 调用的上下文
  // 创建 cuBLAS 句柄
  cublasStatus_t stat = cublasCreate(&h);
  // 检查句柄是否创建成功
  if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "cuBLAS initialization failed" << std::endl;
      return 1;
  }

  // 创建两个 CUDA 事件，用于精确测量 GPU 执行时间
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 遍历所有需要测试的矩阵维度
  for (int dim : dimensions) {
    // 设置矩阵维度 M, N, K。这里我们测试 M=N=K=dim 的情况
    int m = dim;
    int n = dim;
    int k = dim;

    // 声明指向设备（GPU）内存的指针
    mt *A, *B;
    rt *C;
    // 在 GPU 上为矩阵 A, B, C 分配内存
    cudaMalloc(&A, sizeof(mt) * m * k);
    cudaMalloc(&B, sizeof(mt) * n * k);
    cudaMalloc(&C, sizeof(rt) * n * m);

    // 定义 GEMM 运算的标量 alpha 和 beta。公式为 C = alpha * op(A) * op(B) + beta * C
    st alpha = 1;
    st beta = 0;
    
    // 初始化总时间为 0
    float total_time = 0.0f;

    // 预热运行（Warm-up）：第一次调用 CUDA 函数通常会包含一些一次性的开销（如 JIT 编译、上下文初始化等）。
    // 进行一次预热运行可以确保我们后续计时的准确性，避免将这些开销计入性能测试。
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, Atype, k, B, Atype, k, &beta, C, Ctype, m, computeType, CUBLAS_GEMM_DEFAULT);

    // 开始循环计时，执行 num_runs 次以计算平均时间
    for (int i = 0; i < num_runs; ++i) {
      // 在 CUDA 流中插入一个开始计时的事件点
      cudaEventRecord(start);

      // 调用 cuBLAS 的通用矩阵乘法函数 (GEMM)
      stat = cublasGemmEx(h,             // cuBLAS 句柄
                               CUBLAS_OP_T,   // A 矩阵转置，A(m,k)_row = A(k,m)_col，因此lda=k，转置变为A'(m,k)_col 
                               CUBLAS_OP_N,   // B 矩阵不转置，B(n,k)_row = B(k,n)_col，因此ldb=k，因此注意这里的B在输入前其实是转置了一下，形状反一下
                               m,             // op(A) = A' 和 C 的行数 = m，这里是操作后的维度，A'是 mxk_col
                               n,             // op(B) = B 和 C 的列数 = n，这里是操作后的维度，B是 kxn_col
                               k,             // A' 的列数和 B 的行数 = k
                               &alpha,        // 标量 alpha
                               A,             // 指向矩阵 A 的指针
                               Atype,         // 矩阵 A 的数据类型
                               k,             // lda: A 的主维度，和转置无关，在传入地址组织是确定为 k
                               B,             // 指向矩阵 B 的指针
                               Atype,         // 矩阵 B 的数据类型
                               k,             // ldb: B 的主维度，和转置无关，在传入地址组织是确定为 k
                               &beta,         // 标量 beta
                               C,             // 指向矩阵 C 的指针
                               Ctype,         // 矩阵 C 的数据类型
                               m,             // ldc: C 的主维度，由于是列主序，一定是行数 m
                               computeType,   // 计算过程中的数据类型
                               CUBLAS_GEMM_AUTOTUNE); // 调优算法；CUBLAS_GEMM_DEFAULT 使用默认的 GEMM 算法
      
      // 在 CUDA 流中插入一个结束计时的事件点
      cudaEventRecord(stop);
      // 阻塞 CPU，直到 GPU 完成了 stop 事件之前的所有任务，确保计时准确
      cudaEventSynchronize(stop);

      // 检查 GEMM 操作是否成功
      if (stat != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "cublasGemmEx failed with status: " << stat << std::endl;
          break; // 如果失败则跳出循环
      }

      // 计算从 start 事件到 stop 事件之间的耗时（单位：毫秒）
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      // 累加总时间
      total_time += milliseconds;
    }

    // 释放之前在 GPU 上分配的内存，避免内存泄漏
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // 计算平均执行时间
    float avg_time = total_time / num_runs;
    // 计算吞吐量（对于整数运算，单位是 TOPS；对于浮点数，单位是 TFLOPS）
    // 一次 GEMM 的操作数约等于 2 * M * N * K
    double ops = 2.0 * m * n * k;
    // 吞吐量 = 总操作数 / 平均时间（秒）。avg_time 是毫秒，所以要除以 1000。结果再除以 1e12 换算成 Tera (T) 级别。
    double throughput = ops / (avg_time / 1000.0) / 1e12;

    // 打印当前维度的测试结果
    std::cout << "Dim: " << dim << "x" << dim << "x" << dim
              << ", Avg Time: " << avg_time << " ms"
              << ", Throughput: " << throughput << " TOPS" << std::endl;
  }

  // 销毁 CUDA 事件和 cuBLAS 句柄，释放相关资源
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(h);

  // 获取并检查在程序执行过程中是否发生了异步的 CUDA 错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
  }

  return 0; // 程序正常退出
}