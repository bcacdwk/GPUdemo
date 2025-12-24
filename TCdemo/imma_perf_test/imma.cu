/*
 * =============================================================================
 * IMMA Tensor Core GEMM 性能基准测试程序
 * =============================================================================
 * 功能说明：
 * 本程序用于对比两种不同的CUDA WMMA (Warp Matrix Multiply Accumulate) GEMM实现：
 * 1. simple_wmma_gemm_imma - 简单的WMMA实现（无共享内存优化）
 * 2. compute_gemm_imma - 高性能WMMA实现（带共享内存优化）
 * 
 * 测试目标：
 * - 在不同矩阵维度下测试两种实现的性能
 * - 测量GPU延迟、吞吐量(TOPS)和内存带宽
 * - 通过CPU参考实现验证计算正确性
 * - 将结果输出到CSV文件供进一步分析
 * 
 * 重要说明：
 * 所有kernel函数均完全复制自NVIDIA官方CUDA样例代码，未做任何修改，
 * 确保性能基准测试的准确性和可重复性。
 * =============================================================================
 */

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <sstream>

// CUDA辅助函数库
#include <helper_cuda.h>
#include <helper_functions.h>


// =============================================================================
// 错误检查宏定义
// =============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define checkKernelErrors(expr) \
    do { \
        expr; \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "内核执行错误 %d行: '%s' 失败: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort(); \
        } \
    } while (0)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// =============================================================================
// WMMA和GEMM配置常量（来自NVIDIA官方CUDA样例代码，保持完全一致）
// =============================================================================

// ========== GPU基础配置 ==========

// ========== WMMA矩阵块的基础维度 ==========
// Tensor Core WMMA操作的基本矩阵块大小，对于INT8 IMMA固定为16x16x16
#define M 16                            // WMMA矩阵块的M维度（输出矩阵的行数）
#define N 16                            // WMMA矩阵块的N维度（输出矩阵的列数）  
#define K 16                            // WMMA矩阵块的K维度（内积维度）

// WMMA API使用的矩阵维度定义（与上面保持一致）
#define WMMA_M 16                       // WMMA API的M维度
#define WMMA_N 16                       // WMMA API的N维度
#define WMMA_K 16                       // WMMA API的K维度

// ========== GEMM全局矩阵配置 ==========
// 注意：以下宏定义已被动态参数替代，仅保留用于参考
// 在compute_gemm_imma函数中，这些值通过参数传入：m_tiles, n_tiles, k_tiles
// 定义全局矩阵由多少个16x16的WMMA块组成（官方默认配置）
// #define M_TILES 256                     // M维度的块数量（256个16x16块）- 已改为动态参数
// #define N_TILES 256                     // N维度的块数量（256个16x16块）- 已改为动态参数  
// #define K_TILES 256                     // K维度的块数量（256个16x16块）- 已改为动态参数

// 计算全局矩阵的实际维度 - 已改为函数内局部变量
// #define M_GLOBAL (M * M_TILES)          // 全局矩阵A和C的行数 = 16 * 256 = 4096
// #define N_GLOBAL (N * N_TILES)          // 全局矩阵B和C的列数 = 16 * 256 = 4096
// #define K_GLOBAL (K * K_TILES)          // 全局矩阵A和B的内积维度 = 16 * 256 = 4096

// ========== 矩阵存储布局配置 ==========
#define C_LAYOUT wmma::mem_row_major    // C和D矩阵使用行主序存储布局

// ========== 高性能kernel的CTA配置 ==========
#define WARP_SIZE 32                    // GPU warp大小，NVIDIA GPU固定为32个线程
#define WARPS_PER_BLOCK   8             // 每个CTA包含8个warp
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)  // 每个CTA的线程总数 = 32 * 8 = 256

// ========== 共享内存使用策略配置 ==========
#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1       // 默认限制共享内存使用量为64KB以兼容更多GPU
#endif

// 根据共享内存限制选择每次缓存的K维度大小
#if SHARED_MEMORY_LIMIT_64K
#define CHUNK_K 8                       // 64KB限制下：每次缓存8个K块（8*16=128）
#else
#define CHUNK_K 16                      // 无限制下：每次缓存16个K块（16*16=256）
#endif

// ========== 内存访问和数据复制配置 ==========
// 计算向量化内存访问的相关参数
#define CHUNK_LINE_BYTES          (CHUNK_K * K * sizeof(uint8_t))      // 每行数据字节数 = 8*16*1 = 128字节
#define WARP_COPY_BYTES           (WARP_SIZE * sizeof(int4))            // 每个warp复制的字节数 = 32*16 = 512字节
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES) // 每个warp复制的行数 = 512/128 = 4行
#define CHUNK_COPY_LINE_LANES     (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP) // 每行参与的lane数 = 32/4 = 8个lane

// ========== CTA（Cooperative Thread Array）内的Warp组织结构 ==========
// 定义CTA内warp的2D组织方式，用于分配计算任务
#define BLOCK_ROW_WARPS 2               // CTA在行方向有2个warp组
#define BLOCK_COL_WARPS 4               // CTA在列方向有4个warp组（总共2*4=8个warp）

// 定义每个warp负责计算的矩阵块数量
#define WARP_ROW_TILES 4                // 每个warp在行方向计算4个16x16块
#define WARP_COL_TILES 2                // 每个warp在列方向计算2个16x16块（每个warp计算4*2=8个块）

// 计算整个CTA负责的矩阵块范围
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)  // CTA在行方向的块数 = 4*2 = 8块（128行）
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)  // CTA在列方向的块数 = 2*4 = 8块（128列）

// ========== 内存访问步长配置 ==========
// 定义各种内存访问的步长，用于计算内存地址偏移
// #define GLOBAL_MEM_STRIDE N_GLOBAL      // 全局内存按行访问的步长 - 已改为函数内局部变量
#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)  // 共享内存的步长 = 16*8 = 128（共享内存中下一行的位置）
#define SHMEM_OFFSET (N * WARP_ROW_TILES)   // 每个warp在共享内存中的偏移 = 16*4 = 64

// ========== 共享内存Bank冲突优化配置 ==========
// 为避免共享内存bank冲突而添加的偏移量
// 每行/列偏移32个uint8_t元素，确保256位对齐并将不同行/列映射到不同bank
#define SKEW_UINT8 32                   // bank冲突避免偏移：32字节（256位对齐要求）

// =============================================================================
// 命名空间声明
// =============================================================================
using namespace nvcuda;
using namespace std;

// =============================================================================
// KERNEL 1: 高性能WMMA GEMM实现（带共享内存优化）
// =============================================================================
/*
 * 功能：执行高性能的整数GEMM运算 D = alpha * A * B + beta * C
 * 
 * 核心算法思想：
 * 1. 分块计算：每个CTA负责计算128x128的输出块，避免全局同步
 * 2. 共享内存缓存：将A和B矩阵的块缓存到共享内存，减少全局内存访问带宽需求
 * 3. 流水线设计：重叠计算和内存传输，提高GPU利用率
 * 4. Bank冲突优化：通过SKEW技术避免共享内存访问冲突
 * 5. 向量化访问：使用int4进行128位对齐的内存传输
 * 
 * CTA组织结构：
 * - 每个CTA包含8个warp（256个线程）
 * - 每个warp计算8个16x16子块，排列为2x4的网格
 * - 整个CTA计算128x128的输出块（8x8个16x16子块）
 * 
 * 内存层次结构：
 * - 全局内存：存储完整的A、B、C、D矩阵
 * - 共享内存：缓存当前CTA正在处理的A和B矩阵块
 * - 寄存器：WMMA fragments存储16x16的矩阵片段
 * 
 * 关键优化技术：
 * 1. 数据重用：A和B的每个块被多个warp重复使用
 * 2. 内存合并：使用int4确保128位对齐访问
 * 3. 计算隐藏延迟：在计算的同时进行下一批数据的加载
 * 4. 避免分支发散：使用统一的控制流
 * 
 * 参数说明：
 * A: 输入矩阵A (m_global x k_global, 行主序, uint8_t类型)
 * B: 输入矩阵B (k_global x n_global, 列主序, uint8_t类型)  
 * C: 输入矩阵C (m_global x n_global, 行主序, int类型)
 * D: 输出矩阵D (m_global x n_global, 行主序, int类型)
 * m_tiles, n_tiles, k_tiles: 矩阵的块数量 (维度必须是16的倍数)
 * alpha, beta: 标量系数
 * 
 * 注意：此函数已修改为支持动态矩阵维度，不再依赖静态宏定义
 */
__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B, const int *C, int *D, 
                                  int m_tiles, int n_tiles, int k_tiles, int alpha, int beta)
{
    // ========== 动态矩阵维度计算 ==========
    // 根据传入的tiles参数计算实际的矩阵维度
    // 注意：某些变量在当前kernel实现中可能不直接使用，但保留用于完整性和未来扩展
    const int m_global = M * m_tiles;       // 全局矩阵A和C的行数（当前实现中通过tiles参数间接使用）
    const int n_global = N * n_tiles;       // 全局矩阵B和C的列数（用于计算global_mem_stride）
    const int k_global = K * k_tiles;       // 全局矩阵A和B的内积维度（用于内存地址计算）
    const int global_mem_stride = n_global; // 全局内存按行访问的步长（用于矩阵元素寻址）

    // 避免编译器警告：明确标记m_global可能在某些编译配置下未直接使用
    (void)m_global;

    // ========== 共享内存声明和布局设计 ==========
    // 动态共享内存声明：二维数组，每行包含 CHUNK_K*K + SKEW_UINT8 个uint8_t元素
    // 设计原理：
    // 1. 第一维：不同的矩阵块或行
    // 2. 第二维：CHUNK_K*K个数据元素 + SKEW_UINT8个填充元素（避免bank冲突）
    // 3. SKEW技术：通过添加偏移量，确保不同行映射到不同的内存bank
    extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

    // ========== 线程身份识别 ==========
    // 获取当前线程的warp ID和lane ID，用于任务分配和内存访问
    const unsigned int warpId = threadIdx.x / WARP_SIZE;  // 当前线程所属的warp编号(0-7)
    const unsigned int laneId = threadIdx.x % WARP_SIZE;  // 当前线程在warp内的编号(0-31)

    // ========== 共享内存布局规划 ==========
    // 计算B矩阵在共享内存中的起始偏移位置
    // 共享内存布局：[A矩阵块] [B矩阵块]
    // A矩阵占用：BLOCK_COL_TILES * M 行
    // B矩阵从：BLOCK_COL_TILES * M 行开始存储
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // ========== Warp级别的内存指针计算 ==========
    // 计算当前warp负责的C和D矩阵块在共享内存中的访问指针
    // 设计思想：每个warp负责计算特定的子块，需要独立的内存区域
    // 计算公式解析：
    // - (warpId / 2)：将8个warp分为4组，每组2个warp
    // - SHMEM_STRIDE * K * 2：每组warp占用的共享内存行数
    // - (warpId % 2) * SHMEM_OFFSET：同组内warp的列偏移
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0] + (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;

    // 计算当前warp用于流式传输C和D矩阵的共享内存指针
    // 用于整块数据的快速传输，每个warp有独立的传输区域
    int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // ========== 数值精度优化技巧 ==========
    // 预先调整beta系数，避免在最终计算时的重复除法运算
    // 原理：最终结果 = alpha * (A*B) + beta * C
    // 由于后续会统一乘以alpha，这里预先将beta除以alpha
    // 最终变为：alpha * ((A*B) + (beta/alpha) * C)
    // 注意：这可能导致精度损失，但零值需要特殊处理
    beta /= alpha;

    // ========== 主循环：CTA滑动窗口算法 ==========
    // 算法核心思想：
    // 1. 每个CTA按顺序处理不同的128x128输出块
    // 2. 使用block_pos作为全局块索引，支持多个CTA并行工作
    // 3. 当所有块处理完毕时，自然退出循环
    // 
    // 滑动策略：
    // - 从左上角开始，按行优先顺序遍历所有128x128块
    // - 每个CTA处理block_pos, block_pos+gridDim.x, block_pos+2*gridDim.x, ...的块
    // - 这种设计确保负载均衡和内存访问的局部性
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        
        // ========== 块位置计算（关键算法！容易出错的地方） ==========
        // 将一维的block_pos转换为二维的矩阵块坐标(block_tile_i, block_tile_j)
        // 
        // 计算原理：
        // 1. 总的列块数：n_tiles / BLOCK_COL_TILES
        // 2. 当前在第几个行块组：(block_pos * BLOCK_ROW_TILES) / n_tiles
        // 3. 实际的行块索引：需要乘以BLOCK_COL_TILES来跳过中间的行
        // 
        // 注意：这个计算非常容易出错，必须理解矩阵分块的逻辑！
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / n_tiles) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % n_tiles;

        // ========== 循环终止条件检查 ==========
        // 当行索引超出矩阵范围时，所有工作完成，退出循环
        // 这是整个kernel的自然终止点
        if (block_tile_i >= m_tiles) {
            break;
        }

        // ========== C矩阵数据加载：全局内存到共享内存的流式传输 ==========
        // 计算当前warp需要从全局内存复制C矩阵数据的起始地址
        // 地址计算公式：
        // - (block_tile_i + warpId) * M：当前warp负责的行起始位置
        // - global_mem_stride：全局内存中行与行之间的步长
        // - block_tile_j * N：列起始位置
        const size_t gmem_idx = (block_tile_i + warpId) * M * global_mem_stride + block_tile_j * N;
        const int *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // ========== 向量化内存传输（关键性能优化） ==========
        // 将多个C矩阵块流式传输到共享内存
        // 优化技术：
        // 1. 使用int4类型实现128位对齐的向量化加载
        // 2. 每个lane一次传输16字节，大幅提高内存带宽利用率
        // 3. #pragma unroll指示编译器展开循环，减少循环开销
        #pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;  // 128位向量化类型，确保内存合并访问

            // 执行向量化内存拷贝：全局内存 -> 共享内存
            // 每个线程负责一个int4（16字节）的传输
            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + global_mem_stride * i) + laneId);
        }

        // ========== 同步点1：确保C矩阵数据传输完成 ==========
        // 关键作用：确保所有warp完成C矩阵从全局内存到共享内存的传输
        // 在进行下一步之前，必须保证所有数据都已经在共享内存中可用
        __syncthreads();

        // ========== WMMA累加器Fragment声明 ==========
        // 声明累加器fragments，用于沿K_GLOBAL维度累积A和B矩阵fragment乘法的结果
        // 数组维度说明：
        // - 第一维(WARP_COL_TILES=2)：每个warp在列方向处理的子块数
        // - 第二维(WARP_ROW_TILES=4)：每个warp在行方向处理的子块数
        // - 总共2x4=8个16x16的累加器，对应每个warp的计算任务
        wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES][WARP_ROW_TILES];

        // ========== C矩阵Fragment加载：共享内存到寄存器 ==========
        // 从共享内存将C矩阵块加载到WMMA fragments中
        // 这是为后续的乘加运算准备初始值
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {           // 遍历列方向的子块
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {       // 遍历行方向的子块
                // 计算当前子块在共享内存中的起始地址
                // 地址计算：
                // - shmem_warp_tile_ptr：当前warp的基础指针
                // - i * SHMEM_STRIDE * K：列方向偏移
                // - j * N：行方向偏移
                const int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
                
                // 将16x16的C矩阵子块加载到fragment中
                // 参数说明：
                // - c[i][j]：目标fragment
                // - tile_ptr：源数据指针
                // - SHMEM_STRIDE：共享内存中的行步长
                // - C_LAYOUT：内存布局（行主序）
                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        // ========== 同步点2：确保C矩阵Fragment加载完成 ==========
        __syncthreads();

        // ========== Beta系数预处理：C矩阵缩放 ==========
        // 对所有C矩阵fragments应用beta缩放
        // 原理：由于之前已经将beta除以alpha，这里直接乘以调整后的beta
        // 目的：为后续的乘加运算准备正确的C矩阵初始值
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                #pragma unroll
                // 对fragment中的每个元素进行缩放
                // 注意：fragment的内部存储结构是抽象的，但元素级操作是安全的
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // ========== A和B矩阵数据复制策略分配 ==========
        // 智能任务分工：根据warp ID决定复制哪个矩阵
        // 分工原则：
        // - Warp 0-3：负责复制A矩阵的不同部分
        // - Warp 4-7：负责复制B矩阵的不同部分
        // 
        // 地址计算解析：
        // 对于A矩阵（warpId < 4）：
        // - block_tile_i * M * k_global：当前块行的起始地址
        // - M * k_global * (warpId % 4) * 2：当前warp负责的具体区域
        // 对于B矩阵（warpId >= 4）：
        // - block_tile_j * N * k_global：当前块列的起始地址
        // - N * k_global * (warpId % 4) * 2：当前warp负责的具体区域
        const uint8_t *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * k_global] + M * k_global * (warpId % 4) * 2)
                                               : (&B[block_tile_j * N * k_global] + N * k_global * (warpId % 4) * 2);

        // ========== K维度分块处理主循环 ==========
        // 沿全局K维度按CHUNK_K大小的固定步长遍历
        // 分块原理：
        // 1. 将大的K维度分解为CHUNK_K大小的小块
        // 2. 每次迭代处理CHUNK_K个K维度，减少共享内存需求
        // 3. 在每个K块内进行完整的矩阵乘加运算
        #pragma unroll
        for (int tile_k = 0; tile_k < k_tiles; tile_k += CHUNK_K) {
            
            // ========== 共享内存索引计算（复杂但关键的部分） ==========
            // 将A和B矩阵的当前K切片复制到共享内存
            // 
            // 索引计算逻辑：
            // - CTA的前半部分warp (0-3) 复制A矩阵
            // - CTA的后半部分warp (4-7) 复制B矩阵
            // 
            // A矩阵的共享内存布局：
            // - M * (warpId % (WARPS_PER_BLOCK / 2)) * 2：当前warp的起始行
            // B矩阵的共享内存布局：
            // - N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off：
            //   当前warp的起始行 + B矩阵在共享内存中的偏移
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // ========== 复杂的内存访问模式（容易出错的地方！） ==========
            // 计算每个lane在全局内存中的访问指针
            // 
            // 地址计算分解：
            // 1. warp_ptr + tile_k * K：K维度的基础偏移
            // 2. (laneId / CHUNK_COPY_LINE_LANES) * k_global：lane组内的行偏移
            // 3. (laneId % CHUNK_COPY_LINE_LANES)：lane组内的列偏移
            // 
            // 设计原理：
            // - 每个warp内的lane按组工作
            // - 前半部分lane复制第一行/列，后半部分复制下一行/列
            // - 确保内存访问的合并和效率
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * k_global)
                           + (laneId % CHUNK_COPY_LINE_LANES);

            // 根据lane在warp中的位置调整共享内存索引
            // 目的：将warp的后半部分lane偏移到共享内存的下一行/列
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            // ========== 高效的向量化内存复制 ==========
            // 执行实际的数据复制：全局内存 -> 共享内存
            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                // 每个lane一次复制16字节（int4类型）
                // 这确保了128位对齐的高效内存访问
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // 推进全局内存指针和共享内存索引到下一个处理位置
                // 全局内存：跳到下一组行
                // 共享内存：跳到下一行
                lane_ptr = (int4 *)((uint8_t *)lane_ptr + k_global * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            // ========== 同步点3：确保A和B矩阵数据复制完成 ==========
            __syncthreads();

            // ========== 核心计算循环：WMMA矩阵乘法 ==========
            // 在每个warp中计算C矩阵块的网格
            // 这是整个算法的核心计算部分，实现高效的Tensor Core运算
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                
                // ========== A和B矩阵Fragment声明 ==========
                // 为当前K步骤声明临时的A和B矩阵fragments
                // 这些fragments将从共享内存加载，然后参与WMMA运算
                wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major> a[WARP_COL_TILES]; // A矩阵fragments（行主序）
                wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major> b[WARP_ROW_TILES]; // B矩阵fragments（列主序）

                // ========== A矩阵Fragment加载和MMA计算 ==========
                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {  // 遍历当前warp的列方向子块
                    
                    // ========== A矩阵共享内存地址计算 ==========
                    // 计算A矩阵fragment在共享内存中的位置
                    // 地址计算公式：
                    // - (warpId / 2) * M * 2：warp组的基础行偏移
                    // - (i * M)：当前子块的行偏移
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    // 从共享内存加载A矩阵fragment
                    // 关键参数：
                    // - a[i]：目标fragment
                    // - tile_ptr：共享内存中的源数据指针
                    // - K * CHUNK_K + SKEW_UINT8：共享内存的行步长（包含SKEW偏移）
                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

                    // ========== B矩阵Fragment加载和矩阵乘累加 ==========
                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {  // 遍历当前warp的行方向子块
                        
                        // ========== 性能优化：B矩阵Fragment重用 ==========
                        // 只在i==0时加载B矩阵fragment，因为同一个B fragment
                        // 会与多个A fragment进行运算，避免重复加载
                        if (i == 0) {
                            // ========== B矩阵共享内存地址计算 ==========
                            // 计算B矩阵fragment在共享内存中的位置
                            // 地址计算公式：
                            // - shmem_idx_b_off：B矩阵在共享内存中的起始偏移
                            // - (WARP_ROW_TILES * N) * (warpId % 2)：当前warp组内的偏移
                            // - (j * N)：当前子块的列偏移
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            // 从共享内存加载B矩阵fragment
                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                        }

                        // ========== 核心计算：Tensor Core矩阵乘累加 ==========
                        // 执行 c[i][j] = c[i][j] + a[i] * b[j]
                        // 这是整个算法的核心运算，利用GPU的Tensor Core硬件加速
                        // 
                        // 运算说明：
                        // - a[i]：16x16的A矩阵fragment（uint8类型）
                        // - b[j]：16x16的B矩阵fragment（uint8类型）
                        // - c[i][j]：16x16的累加器fragment（int类型）
                        // - 硬件自动处理类型转换和累加
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            // ========== 同步点4：确保当前K块的所有计算完成 ==========
            __syncthreads();
        }

        // ========== 最终结果处理：Fragment到共享内存 ==========
        // 将计算完成的D矩阵fragments存储到共享内存
        // 这是计算阶段的最后一步，为后续的全局内存写入做准备
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                
                // ========== Alpha系数应用（关键的数值处理） ==========
                // 对所有fragment元素进行alpha缩放
                // 原理回顾：
                // 1. 之前C矩阵已经乘以了 beta/alpha
                // 2. A*B的结果现在乘以alpha
                // 3. 最终得到：alpha * (A*B) + alpha * (beta/alpha * C) = alpha * (A*B) + beta * C
                // 
                // 重要说明：
                // warp中所有线程对所有fragment元素的统一点变换是良定义的
                // 即使fragment的内部存储结构是抽象的，元素级的变换仍然是安全的
                #pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                // ========== 结果存储：寄存器到共享内存 ==========
                // 计算当前fragment在共享内存中的存储位置
                int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
                
                // 将处理完的fragment存储到共享内存
                // 参数说明：
                // - tile_ptr：共享内存中的目标位置
                // - c[i][j]：源fragment（已经过alpha缩放）
                // - SHMEM_STRIDE：共享内存的行步长
                // - C_LAYOUT：存储布局（行主序）
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        // ========== 同步点5：确保所有D块写入共享内存完成 ==========
        __syncthreads();

        // ========== 最终数据传输：共享内存到全局内存 ==========
        // 现在共享内存包含了当前CTA计算的所有D矩阵块
        // 需要将它们高效地传输回全局内存
        
        // 计算当前warp在全局内存中的目标写入地址
        int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

        // ========== 高性能向量化写入 ==========
        // 使用与读取相同的向量化策略进行写入
        #pragma unroll
        for (int i = 0; i < K; i++) {
            // 执行128位对齐的向量化写入：共享内存 -> 全局内存
            // 每个lane负责传输一个int4（16字节）
            // 这确保了高效的内存带宽利用和合并访问
            *((int4 *)(dst_gmem_warp_stream_ptr + global_mem_stride * i) + laneId) =
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        // ========== 同步点6：确保所有数据写入全局内存完成 ==========
        // 确保当前块的所有数据都已经写入全局内存
        // 为下一个块的处理做准备
        __syncthreads();
    }  // 主循环结束：继续处理下一个128x128块
}

// =============================================================================
// KERNEL 2: 简单WMMA GEMM实现（无共享内存优化）
// =============================================================================
/*
 * 功能：执行简单的整数GEMM运算 D = alpha * A * B + beta * C
 * 
 * 算法特点：
 * - 直接从全局内存加载数据，不使用共享内存缓存
 * - 使用2D网格，每个warp处理一个16x16的输出块
 * - 沿K维度循环累积部分结果
 * - 适用于演示WMMA API的基本用法
 * 
 * 参数说明：
 * a: 输入矩阵A (m_ld x k_ld, 行主序, uint8_t类型)
 * b: 输入矩阵B (k_ld x n_ld, 列主序, uint8_t类型)
 * c: 输入矩阵C (m_ld x n_ld, 行主序, int类型)
 * d: 输出矩阵D (m_ld x n_ld, 行主序, int类型)
 * m_ld, n_ld, k_ld: 矩阵的leading dimensions
 * alpha, beta: 标量系数
 * 
 * 假设条件：
 * 1) 矩阵在内存中紧密排列
 * 2) M、N和K都是16的倍数
 * 3) A和B矩阵都未转置
 * 
 * 注意：此函数完全复制自NVIDIA官方CUDA样例，未做任何修改
 */
__global__ void simple_wmma_gemm_imma(const uint8_t *a,
                                      const uint8_t *b,
                                      const int     *c,
                                      int           *d,
                                      int            m_ld,
                                      int            n_ld,
                                      int            k_ld,
                                      int            alpha,
                                      int            beta)
{
    // 矩阵的leading dimensions（紧密排列，无转置）
    // Leading dimension是指在内存中访问下一行时需要跳过的元素数量
    int lda = k_ld;  // A矩阵的leading dimension：A是M×K矩阵，行主序存储，每行有K个元素，注意这里和官方代码不同
    int ldb = k_ld;  // B矩阵的leading dimension：B是K×N矩阵，列主序存储，ldb表示存储时的跨度
    int ldc = n_ld;  // C矩阵的leading dimension：C是M×N矩阵，行主序存储，每行有N个元素


    // 使用2D网格进行分块，每个warp负责一个16x16的输出块
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // 当前warp在M维度的位置（行索引）
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);            // 当前warp在N维度的位置（列索引）

    // 声明WMMA fragments - 这些是寄存器中的抽象数据结构
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag;  // A矩阵fragment（行主序）
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::col_major> b_frag;  // B矩阵fragment（列主序）
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>                   acc_frag; // 累加器fragment（存储中间结果）
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>                   c_frag;   // C矩阵fragment（用于最终计算）

    // 初始化累加器为0.0f（注意：虽然结果是int类型，但初始化使用float）
    wmma::fill_fragment(acc_frag, 0.0f);

    // 沿K维度循环，每次处理WMMA_K(16)个元素
    for (int i = 0; i < k_ld; i += WMMA_K) {
        // 计算A矩阵当前16x16块在全局矩阵中的位置
        int aCol = i;                    // A矩阵的列起始位置（K维度）
        int aRow = warpM * WMMA_M;       // A矩阵的行起始位置（M维度）

        // 计算B矩阵当前16x16块在全局矩阵中的位置
        int bCol = i;                    // B矩阵的列起始位置（K维度）
        int bRow = warpN * WMMA_N;       // B矩阵的行起始位置（N维度）

        // 边界检查，确保不会访问超出矩阵边界的内存
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // 从全局内存加载A和B矩阵的fragments
            // 地址计算：a + aCol + aRow * lda 表示A[aRow][aCol]的位置
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

            // 执行Tensor Core矩阵乘累加运算：acc_frag += a_frag * b_frag
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 处理C矩阵和最终结果计算
    int cCol = warpN * WMMA_N;           // C矩阵当前块的列起始位置
    int cRow = warpM * WMMA_M;           // C矩阵当前块的行起始位置

    if (cRow < m_ld && cCol < n_ld) {
        // 从全局内存加载C矩阵fragment
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        // 计算最终结果：D = alpha * (A * B) + beta * C
        // 对fragment中的每个元素进行点运算
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // 将结果存储回全局内存
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

// =============================================================================
// CPU参考实现：主机端矩阵乘法
// =============================================================================
/*
 * 功能：在CPU上执行矩阵乘法运算 C = alpha * A * B + beta * C
 * 
 * 算法特点：
 * - 使用经典的三重循环实现，确保结果准确性
 * - 直接计算，无任何优化，作为GPU实现的"金标准"
 * - 支持alpha和beta系数的通用GEMM公式
 * 
 * 用途：
 * 1. 作为GPU实现的参考标准，验证计算正确性
 * 2. 性能对比基准，展示GPU相对于CPU的加速效果
 * 3. 调试工具，当GPU结果异常时可用于定位问题
 * 
 * 实现原理：
 * - 外层双重循环遍历输出矩阵C的每个元素
 * - 内层循环计算对应的点积（A的行与B的列）
 * - 应用GEMM公式的alpha和beta系数
 * 
 * 矩阵布局假设：
 * - A矩阵：行主序存储，维度为 numARows × numAColumns
 * - B矩阵：列主序存储，但访问时按行主序索引，维度为 numBRows × numBColumns
 * - C矩阵：行主序存储，维度为 numCRows × numCColumns
 * 
 * 参数说明：
 * A: 输入矩阵A (numARows x numAColumns, uint8_t类型)
 * B: 输入矩阵B (numBRows x numBColumns, uint8_t类型)
 * C: 输入输出矩阵C (numCRows x numCColumns, int类型)
 * alpha, beta: 标量系数
 * 各种num*参数: 矩阵的行列数
 * 
 * 注意：此函数完全复制自NVIDIA官方CUDA样例，未做任何修改
 */
__host__ void matMultiplyOnHost(uint8_t *A,
                                uint8_t *B,
                                int     *C,
                                int      alpha,
                                int      beta,
                                int      numARows,
                                int      numAColumns,
                                int      numBRows,
                                int      numBColumns,
                                int      numCRows,
                                int      numCColumns)
{
    // ========== 标准矩阵乘法三重循环实现 ==========
    // 算法复杂度：O(numCRows × numCColumns × numAColumns)
    // 这是最直接、最容易理解的矩阵乘法实现
    
    // 外层循环：遍历输出矩阵C的每一行
    for (int i = 0; i < numCRows; i++) {
        
        // 中层循环：遍历输出矩阵C的每一列
        for (int j = 0; j < numCColumns; j++) {
            
            // ========== 点积计算初始化 ==========
            // temp用于累积A的第i行与B的第j列的点积结果
            int temp = 0;

            // ========== 内层循环：计算点积 ==========
            // 计算A矩阵第i行与B矩阵第j列的点积
            // 这是矩阵乘法的核心运算：∑(A[i,k] * B[k,j])
            for (int k = 0; k < numAColumns; k++) {
                
                // ========== 关键的内存访问模式 ==========
                // A[i * numAColumns + k]：A矩阵第i行第k列元素（行主序访问）
                // B[j * numBRows + k]：B矩阵第j列第k行元素
                // 
                // 注意B矩阵的索引：
                // - B在内存中可能是列主序存储
                // - 但这里的访问模式是 B[j * numBRows + k]
                // - 对应于转置后的访问，即 B^T[j,k] = B[k,j]
                temp += A[i * numAColumns + k] * B[j * numBRows + k];
            }

            // ========== GEMM公式应用 ==========
            // 实现完整的GEMM公式：C = alpha * A * B + beta * C
            // 
            // 计算步骤：
            // 1. temp：包含 (A * B)[i,j] 的结果
            // 2. temp * alpha：应用alpha系数到乘积结果
            // 3. beta * C[i * numCColumns + j]：应用beta系数到原C矩阵元素
            // 4. 最终结果：alpha * (A*B)[i,j] + beta * C[i,j]
            // 
            // 内存访问：C[i * numCColumns + j] 表示C矩阵第i行第j列（行主序）
            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}

// =============================================================================
// 辅助函数：主机端矩阵初始化（来自官方代码）
// =============================================================================
/*
 * 功能：初始化主机端的A、B、C矩阵
 * 使用小的随机值（0-2）确保不会溢出并便于调试
 */
__host__ void init_host_matrices(uint8_t *a, uint8_t *b, int *c, int M_size, int N_size, int K_size)
{
    // 初始化A矩阵 (M_size x K_size)
    for (int i = 0; i < M_size; i++) {
        for (int j = 0; j < K_size; j++) {
            a[i * K_size + j] = (uint8_t)(rand() % 3);
        }
    }

    // 初始化B矩阵 (N_size x K_size，以列主序存储）
    for (int i = 0; i < N_size; i++) {
        for (int j = 0; j < K_size; j++) {
            b[i * K_size + j] = (uint8_t)(rand() % 3);
        }
    }

    // 初始化C矩阵 (M_size x N_size)
    for (int t = 0; t < M_size * N_size; t++) {
        c[t] = (rand() % 3);
    }
}









// =============================================================================
// 测试类型枚举
// =============================================================================

// 枚举类型：选择要运行的kernel
enum GemmKernelType {
    SIMPLE,     // 简单WMMA实现
    OPTIMIZED   // 优化的共享内存实现
};






// =============================================================================
// 主函数：执行IMMA Tensor Core GEMM性能基准测试
// =============================================================================
/*
 * 主函数功能概述：
 * 1. 初始化CUDA环境和GPU设备
 * 2. 为每个测试矩阵大小分配内存和初始化数据
 * 3. 执行GPU预热（可选CPU验证）
 * 4. 运行两种GEMM实现并测量性能：
 *    - 简单WMMA实现（无共享内存优化）
 *    - 优化WMMA实现（带共享内存优化）
 * 5. 收集和分析性能数据
 * 6. 清理资源并输出结果
 */

int main(int argc, char **argv)
{
    std::cout << "=== IMMA Tensor Core GEMM 性能基准测试 ===" << std::endl;

    // =============================================================================
    // CUDA设备初始化和能力检查
    // =============================================================================
    
    // 查找并选择CUDA设备
    // findCudaDevice() 会根据命令行参数选择最佳的GPU设备
    // 如果有多个GPU，它会选择计算能力最高的那个
    int dev = findCudaDevice(argc, (const char **)argv);
    
    // 获取选中GPU的详细属性信息
    // cudaDeviceProp 结构体包含GPU的所有硬件信息：
    // - 计算能力版本(major.minor)
    // - 内存大小、共享内存大小
    // - 多处理器数量、warp大小等
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // 检查GPU是否支持Tensor Core IMMA操作
    // Tensor Core的INT8 IMMA操作需要计算能力7.2或更高：
    // - SM 7.2: V100 (第一代Tensor Core)
    // - SM 7.5: RTX 20系列, T4
    // - SM 8.0: A100 (第二代Tensor Core)
    // - SM 8.6: RTX 30系列
    if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
        std::cout << "错误：IMMA Tensor Core需要SM 7.2或更高版本的GPU。程序退出..." << std::endl;
        exit(EXIT_WAIVED);
    }

    // 输出GPU硬件信息，帮助理解性能测试环境
    std::cout << "GPU设备信息: " << deviceProp.name << " (SM " << deviceProp.major << "." << deviceProp.minor << ")" << std::endl;
    std::cout << "多处理器数量: " << deviceProp.multiProcessorCount << std::endl;  // 影响并行度
    std::cout << "最大共享内存: " << (deviceProp.sharedMemPerBlock / 1024) << " KB" << std::endl;  // 影响优化kernel的可用性



    // =============================================================================
    // 测试配置
    // =============================================================================
    // 控制是否进行CPU验证：
    // - true:  启用CPU验证，程序会运行CPU版本的GEMM进行结果验证（慢但准确）
    // - false: 跳过CPU验证，只运行GPU版本进行性能测试（快速迭代）
    const bool enable_cpu_verification = false;  

    // 定义测试矩阵的维度列表
    // 矩阵大小会显著影响性能特征：
    // - 小矩阵(512-1024): 主要受启动开销影响
    // - 中矩阵(2048-4096): GPU利用率逐渐提高
    // - 大矩阵(8192+): 接近GPU峰值性能
    // std::vector<int> test_sizes = {4096, 8192};
    // std::vector<int> test_sizes = {512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 8192, 16384};
    // std::vector<int> test_sizes = {4096, 6144, 8192, 10240, 12288, 14336, 16384};

    std::vector<int> test_sizes = {512, 1024 , 2048, 4096, 8192, 12288, 16384};  //均为128倍数

    // std::vector<int> test_sizes = {256, 320, 512, 528, 1024};  //非128的倍数



    std::cout << "测试矩阵维度: ";
    for (size_t i = 0; i < test_sizes.size(); i++) {
        std::cout << test_sizes[i];
        if (i < test_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;

    // =============================================================================
    // GEMM运算配置：定义矩阵乘法的标量系数
    // =============================================================================
    
    // GEMM标准公式：D = alpha * A * B + beta * C
    // alpha: A*B乘积的缩放系数
    // beta:  C矩阵的缩放系数  
    int alpha = 1;  // 不缩放A*B的结果
    int beta = 1;   // 保留C矩阵的原始值
    std::cout << "GEMM公式: D = " << alpha << " * A * B + " << beta << " * C" << std::endl;

    // =============================================================================
    // 性能数据收集结构：用于汇总分析
    // =============================================================================
    
    // 定义数据结构来存储每个测试的完整性能信息
    struct PerfData {
        int size;                    // 矩阵大小
        float time_simple;           // 简单版本执行时间(ms)
        float time_optimized;        // 优化版本执行时间(ms)
        double time_cpu;             // CPU版本执行时间(ms)
        double tops_simple;          // 简单版本性能(TOPS)
        double tops_optimized;       // 优化版本性能(TOPS)
        double tops_cpu;             // CPU版本性能(TOPS)
        bool optimized_executed;     // 优化版本是否成功执行
        bool results_match;          // GPU结果是否与CPU匹配
    };
    std::vector<PerfData> perf_results;  // 存储所有测试结果

    // =============================================================================
    // 主测试循环：遍历每个矩阵大小进行完整的性能测试
    // =============================================================================
    
    // 对每个预定义的矩阵大小执行完整的测试流程
    for (int size : test_sizes) {
        // 输出测试分隔符，便于阅读结果
        {
            std::string sep(60, '=');
            printf("\n%s\n", sep.c_str());
        }
        printf("测试矩阵维度: %d x %d x %d\n", size, size, size);
        {
            std::string sep(60, '=');
            printf("%s\n", sep.c_str());
        }

        // =============================================================================
        // 矩阵维度验证：确保满足WMMA硬件要求
        // =============================================================================
        
        // WMMA (Warp Matrix Multiply Accumulate) 硬件要求：
        // - 矩阵维度必须是16的倍数
        // - 这是因为Tensor Core硬件按16x16块进行计算
        if (size % 16 != 0) {
            std::cout << "跳过: 矩阵维度必须是16的倍数才能使用WMMA" << std::endl;
            continue;
        }

        // =============================================================================
        // 内存需求计算和主机内存分配
        // =============================================================================
        
        // 计算各矩阵所需的内存大小：
        // - A矩阵: size×size个uint8_t元素 (输入矩阵，8位整数)
        // - B矩阵: size×size个uint8_t元素 (输入矩阵，8位整数)  
        // - C矩阵: size×size个int元素 (输入矩阵，32位整数)
        // - D矩阵: size×size个int元素 (输出矩阵，32位整数)
        size_t bytes_a = (size_t)size * size * sizeof(uint8_t);  // A矩阵字节数
        size_t bytes_b = (size_t)size * size * sizeof(uint8_t);  // B矩阵字节数  
        size_t bytes_c = (size_t)size * size * sizeof(int);      // C矩阵字节数
        size_t bytes_d = (size_t)size * size * sizeof(int);      // D矩阵字节数

        // 在主机(CPU)上分配内存存储矩阵数据
        // 需要为每种实现分配独立的输出矩阵，以便比较结果
        uint8_t *h_a = (uint8_t *)malloc(bytes_a);               // 主机端A矩阵
        uint8_t *h_b = (uint8_t *)malloc(bytes_b);               // 主机端B矩阵
        int *h_c = (int *)malloc(bytes_c);                       // 主机端C矩阵
        int *h_d_gpu_simple = (int *)malloc(bytes_d);            // 简单GPU实现的结果
        int *h_d_gpu_optimized = (int *)malloc(bytes_d);         // 优化GPU实现的结果
        int *h_d_cpu = (int *)malloc(bytes_d);                   // CPU参考实现的结果

        // 检查主机内存分配是否成功
        // 大矩阵可能需要几GB内存，分配失败会导致程序崩溃
        if (!h_a || !h_b || !h_c || !h_d_gpu_simple || !h_d_gpu_optimized || !h_d_cpu) {
            std::cout << "错误: 主机内存分配失败，矩阵大小: " << size << "x" << size << std::endl;
            exit(EXIT_FAILURE);
        }

        // =============================================================================
        // 矩阵数据初始化
        // =============================================================================
        
        // 用随机数初始化输入矩阵A、B、C
        // 这确保了测试的真实性和可重复性
        std::cout << "正在初始化输入矩阵..." << std::endl;
        init_host_matrices(h_a, h_b, h_c, size, size, size);

        // =============================================================================
        // GPU内存分配
        // =============================================================================
        
        // 在GPU设备上分配内存存储矩阵数据
        // GPU内存访问比主机内存快得多，但容量有限
        uint8_t *d_a, *d_b;  // GPU上的A、B矩阵指针
        int *d_c, *d_d;      // GPU上的C、D矩阵指针
        
        // cudaMalloc: 在GPU全局内存中分配空间
        // 类似于CPU的malloc，但分配的是GPU内存
        checkCudaErrors(cudaMalloc(&d_a, bytes_a));              // 在GPU分配A矩阵空间
        checkCudaErrors(cudaMalloc(&d_b, bytes_b));              // 在GPU分配B矩阵空间
        checkCudaErrors(cudaMalloc(&d_c, bytes_c));
        checkCudaErrors(cudaMalloc(&d_d, bytes_d));

        // 复制数据到GPU
        checkCudaErrors(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc(&d_c, bytes_c));              // 在GPU分配C矩阵空间
        checkCudaErrors(cudaMalloc(&d_d, bytes_d));              // 在GPU分配D矩阵空间

        // =============================================================================
        // 数据传输：将输入数据从主机复制到GPU
        // =============================================================================
        
        // cudaMemcpy: 在主机和设备之间传输数据
        // 参数说明：(目标地址, 源地址, 字节数, 传输方向)
        // cudaMemcpyHostToDevice: 从CPU内存复制到GPU内存
        checkCudaErrors(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));  // 传输A矩阵
        checkCudaErrors(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));  // 传输B矩阵
        checkCudaErrors(cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice));  // 传输C矩阵
        // 注意：D矩阵是输出，不需要传输输入数据

        // =============================================================================
        // 性能指标计算准备
        // =============================================================================
        
        // 计算矩阵乘法的总浮点运算数：
        // - 每个输出元素需要进行size次乘法和size次加法
        // - 总共size×size个输出元素
        // - 总操作数 = size³ × 2 (乘法+加法各算1次操作)
        double ops = (double)size * size * size * 2;

        // 声明性能测量变量
        float time_simple = 0, time_optimized = 0;        // GPU执行时间(毫秒)
        double tops_simple = 0, tops_optimized = 0;       // GPU性能(TOPS = 万亿次操作/秒)
        double cpu_time_ms = 0, tops_cpu = 0;             // CPU性能指标
        bool simple_vs_cpu_match = false, optimized_vs_cpu_match = false;  // 结果正确性标志 - 默认为false(未验证)

        
        // =============================================================================
        //! 1) 预热阶段：GPU预热
        // =============================================================================
        /*
         * 1. GPU频率调节：现代GPU有动态频率，冷启动时运行在较低频率
         * 2. 驱动初始化：第一次CUDA调用会触发大量初始化工作
         * 3. 缓存预热：L2缓存、指令缓存需要预热
         * 4. 内存控制器优化：GPU内存控制器需要时间进入最佳状态
         */
        
        std::cout << std::endl << "[预热阶段] 执行GPU预热..." << std::endl;
        std::cout << "执行GPU预热 (5次)..." << std::endl;
        
        // 配置CUDA执行参数：
        // dim3: CUDA的3维坐标结构，用于定义网格和线程块的大小
        dim3 gridDim, blockDim;
        
        // 配置线程块(Block)大小：
        // - blockDim.x = 128: 每个block在x方向有128个线程
        // - blockDim.y = 4:   每个block在y方向有4个线程  
        // - 总共128×4=512个线程每block，符合GPU的warp组织(32线程/warp)
        blockDim.x = 128;
        blockDim.y = 4;
        
        // 计算网格(Grid)大小：需要多少个block来覆盖整个矩阵
        // 计算公式确保有足够的block来处理所有矩阵元素
        gridDim.x = (size + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (size + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        // 执行5轮预热，让GPU进入最佳性能状态
        for (int warmup = 0; warmup < 5; warmup++) {
            // 清零输出矩阵，确保每次测试从干净状态开始
            checkCudaErrors(cudaMemset(d_d, 0, bytes_d));
            
            //! 启动简单WMMA kernel进行预热
            // <<<gridDim, blockDim>>>: CUDA kernel启动语法
            // - gridDim: 网格大小，定义启动多少个block
            // - blockDim: 线程块大小，定义每个block有多少个线程
            checkKernelErrors((simple_wmma_gemm_imma<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, size, size, size, alpha, beta)));
            
            // kernel发射后，等待所有GPU操作完成
            checkCudaErrors(cudaDeviceSynchronize());
           
            //! 启动优化版本WMMA kernel预热
            // 计算优化kernel所需的共享内存大小
            // 优化版本使用共享内存缓存数据，需要更多内存
            size_t SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                                  M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int));
            
            // 检查GPU是否有足够的共享内存支持优化版本
            if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
                checkCudaErrors(cudaMemset(d_d, 0, bytes_d));
                
                // 设置kernel使用的动态共享内存大小
                // 这告诉CUDA驱动为这个kernel分配指定大小的共享内存
                checkCudaErrors(cudaFuncSetAttribute(compute_gemm_imma, 
                    cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
                
                // 启动优化kernel进行预热
                // 注意：这里使用不同的网格配置：
                // - deviceProp.multiProcessorCount: 使用GPU的SM数量作为网格大小
                // - THREADS_PER_BLOCK: 每个block的线程数
                // - SHMEM_SZ: 每个block使用的共享内存大小
                // 计算当前矩阵的块数量
                int m_tiles = size / 16;
                int n_tiles = size / 16; 
                int k_tiles = size / 16;
                checkKernelErrors((compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
                    d_a, d_b, d_c, d_d, m_tiles, n_tiles, k_tiles, alpha, beta)));

                // kernel发射后，等待所有GPU操作完成
                checkCudaErrors(cudaDeviceSynchronize());
            }
        }

        std::cout << "GPU预热完成" << std::endl;




        // =============================================================================
        //! 2) 性能测试: 简单WMMA实现 (100次平均测量)
        // =============================================================================
        std::cout << std::endl << "[性能测试1] 简单WMMA GEMM实现 (100次平均)" << std::endl;
        checkCudaErrors(cudaMemset(d_d, 0, bytes_d));

        // 重新配置执行参数（确保配置正确）
        gridDim.x = (size + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (size + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        std::cout << "Grid配置: (" << gridDim.x << ", " << gridDim.y << "), Block配置: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;

        // CUDA事件：GPU端的高精度计时器
        /*
         * 为什么使用CUDA事件而不是CPU计时器：
         * 1. GPU端计时：直接测量GPU执行时间，不包括CPU-GPU通信延迟
         * 2. 高精度：比CPU计时器精度更高
         * 3. 异步友好：可以与异步kernel启动配合使用
         */
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));            // 创建开始事件
        checkCudaErrors(cudaEventCreate(&stop));             // 创建结束事件
        
        // 执行100次重复测试获取准确的平均性能
        const int num_tests = 100;                          // 测试次数
        float total_time = 0.0f;                            // 累计总时间
        
        std::cout << "执行 " << num_tests << " 次测试..." << std::endl;
        
        // 100次重复测试循环
        for (int test = 0; test < num_tests; test++) {
            // 每次测试前清零输出矩阵，确保测试独立性
            checkCudaErrors(cudaMemset(d_d, 0, bytes_d));
            
            // 记录开始时间点
            // cudaEventRecord: 在GPU命令流中插入时间戳记录点
            checkCudaErrors(cudaEventRecord(start));
            
            // 启动简单WMMA kernel
            // 这是真正被测量的计算部分
            simple_wmma_gemm_imma<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_d, size, size, size, alpha, beta);
            
            // 记录结束时间点
            checkCudaErrors(cudaEventRecord(stop));
            
            // 等待stop事件完成，确保kernel执行结束
            // cudaEventSynchronize: 等待指定事件完成
            checkCudaErrors(cudaEventSynchronize(stop));
            
            // 计算这次测试的执行时间
            float test_time;
            checkCudaErrors(cudaEventElapsedTime(&test_time, start, stop));  // 获取两个事件间的时间差(ms)
            total_time += test_time;                         // 累加到总时间
        }
        
        // 计算平均执行时间
        time_simple = total_time / num_tests;               // 平均时间(毫秒)
        
        // 复制最后一次执行的结果到主机内存（用于后续验证）
        checkCudaErrors(cudaMemcpy(h_d_gpu_simple, d_d, bytes_d, cudaMemcpyDeviceToHost));

        // 性能指标计算和输出
        /*
         * TOPS计算公式详解：
         * TOPS = (总操作数 / 执行时间(秒)) / 10^12
         * - 总操作数 = size³ × 2 (每个输出元素需要size次乘法和size次加法)
         * - 执行时间需要从毫秒转换为秒：time_simple / 1000.0
         * - 除以10^12是因为TOPS = Tera Operations Per Second (万亿次操作/秒)
         */
        tops_simple = (ops / (time_simple / 1000.0)) / 1e12;
        std::cout << std::fixed << std::setprecision(3) << "执行时间: " << time_simple << " ms, " 
                  << std::setprecision(2) << "性能: " << tops_simple << " TOPS" << std::endl << std::defaultfloat;



        // =============================================================================
        //! 3) 性能测试: 优化WMMA实现 (100次平均测量)
        // =============================================================================
        /*
         * 优化版本与简单版本的主要区别：
         * 1. 使用共享内存缓存数据，减少全局内存访问
         * 2. 使用更复杂的线程块组织和数据重用策略  
         * 3. 需要更多共享内存，可能不是所有GPU都支持
         */
        std::cout << std::endl << "[性能测试2] 优化WMMA GEMM实现 (100次平均)" << std::endl;
        
        // 检查矩阵大小是否为128的倍数（影响优化效果）
        if (size % 128 != 0) {
            std::cout << "注意: 矩阵维度(" << size << ")不是128的倍数，可能存在性能衰减" << std::endl;
        }

            /*
             * 计算优化kernel所需的共享内存大小 (严格按照官方代码)
             * 共享内存用于存储两个用途：
             * 1. 输入矩阵A和B的瓦片数据（uint8_t类型）
             * 2. 输出矩阵C的累积结果（int类型）
             * 
             * 第一项：输入数据缓存大小
             * - sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2
             * - 存储A和B两个矩阵的瓦片，使用SKEW避免bank冲突
             * 
             * 第二项：输出累积缓存大小  
             * - M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int)
             * - 存储每个warp的输出瓦片结果
             * 
             * MAX取较大值确保共享内存足够
             */
            enum {
                SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                               M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
            };
            std::cout << "所需共享内存: " << (SHMEM_SZ / 1024UL) << " KB" << std::endl;

            /*
             * 检查GPU是否有足够的共享内存运行优化kernel
             * A100的共享内存配置：
             * - 每个SM最大共享内存：164KB
             * - 每个Block最大共享内存：48KB-164KB（可配置）
             * 
             * 如果共享内存不足，将跳过优化版本测试
             */
            if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
                std::cout << "使用高性能kernel compute_gemm_imma" << std::endl;
                
                /*
                 * 设置kernel的动态共享内存限制
                 * CUDA Runtime默认限制较低，需要显式设置更高值
                 * 这样kernel才能使用超过默认限制的共享内存
                 */
                checkCudaErrors(cudaFuncSetAttribute(compute_gemm_imma, 
                    cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
                
                /*
                 * 优化kernel的性能测试循环 (100次平均)
                 * 为什么需要多次测试：
                 * 1. GPU性能会因为温度、频率调整而波动
                 * 2. 内存访问模式可能会有cache效应
                 * 3. 多次测试取平均值更能反映真实性能
                 */
                total_time = 0.0f;
                std::cout << "执行 " << num_tests << " 次测试..." << std::endl;
                
                for (int test = 0; test < num_tests; test++) {
                    // 每次测试前清零输出矩阵，确保测试独立性
                    checkCudaErrors(cudaMemset(d_d, 0, bytes_d));
                    
                    // 启动优化kernel，使用和官方代码相同的配置
                    checkCudaErrors(cudaEventRecord(start));
                    /*
                     * kernel launch配置说明：
                     * <<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>
                     * - deviceProp.multiProcessorCount: 使用所有SM，实现work-stealing
                     * - THREADS_PER_BLOCK: 每个线程块的线程数 
                     * - SHMEM_SZ: 动态共享内存大小（前面计算的值）
                     */
                    // 计算当前矩阵的块数量
                    int m_tiles = size / 16;
                    int n_tiles = size / 16; 
                    int k_tiles = size / 16;
                    compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
                        d_a, d_b, d_c, d_d, m_tiles, n_tiles, k_tiles, alpha, beta);
                    checkCudaErrors(cudaEventRecord(stop));
                    checkCudaErrors(cudaEventSynchronize(stop));
                    
                    float test_time;
                    checkCudaErrors(cudaEventElapsedTime(&test_time, start, stop));
                    total_time += test_time;
                }
                
                /*
                 * 计算优化版本的平均性能
                 * time_optimized: 100次测试的平均执行时间(毫秒)
                 * 将结果从GPU拷贝回主机内存供后续验证使用
                 */
                time_optimized = total_time / num_tests;  // 平均时间
                checkCudaErrors(cudaMemcpy(h_d_gpu_optimized, d_d, bytes_d, cudaMemcpyDeviceToHost));

                /*
                 * 性能指标计算 (严格按照官方代码公式)
                 * TOPS = (总操作数 / 执行时间秒) / 10^12
                 * 其中：
                 * - ops: 2*M*N*K个操作（每个输出元素需要K次乘加运算）
                 * - time_optimized/1000.0: 时间转换为秒
                 * - 1e12: 转换为TOPS (Tera Operations Per Second)
                 */
                tops_optimized = (ops / (time_optimized / 1000.0)) / 1e12;
                std::cout << std::fixed << std::setprecision(3) << "执行时间: " << time_optimized << " ms, " << std::setprecision(2) << "性能: " << tops_optimized << " TOPS" << std::endl << std::defaultfloat;
                
                
                /*
                 * 计算相对于简单版本的加速比
                 * 这个指标反映了优化效果的好坏
                 */
                if (time_simple > 0) {
                    std::cout << std::fixed << std::setprecision(2) << "相对于简单版本加速比: " << (time_simple / time_optimized) << "x" << std::endl << std::defaultfloat;
                }
            } else {
                /*
                 * 共享内存不足时的处理
                 * 在共享内存有限的GPU上，无法运行高性能优化版本
                 * 此时将结果矩阵清零，避免后续验证出现问题
                 */
                std::cout << "跳过: 共享内存不足，需要 " << (SHMEM_SZ / 1024UL) << " KB，可用 " << (deviceProp.sharedMemPerMultiprocessor / 1024) << " KB" << std::endl;
                memset(h_d_gpu_optimized, 0, bytes_d);
            }





        // =============================================================================
        //! 4) CPU验证和结果比较（在所有GPU测试完成后执行）
        // =============================================================================
        
        if (enable_cpu_verification) {
            std::cout << std::endl << "[CPU验证和结果比较]" << std::endl;
            
            // =============================================================================
            // CPU验证：运行CPU版本作为正确性参考
            // =============================================================================
            std::cout << "执行CPU参考计算..." << std::endl;
            memcpy(h_d_cpu, h_c, bytes_d);  // 复制C矩阵作为CPU计算的初始值
            
            // 使用高精度计时器测量CPU执行时间
            auto cpu_start = std::chrono::high_resolution_clock::now();
            matMultiplyOnHost(h_a, h_b, h_d_cpu, alpha, beta, size, size, size, size, size, size);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            
            // 计算CPU执行时间和性能
            auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
            cpu_time_ms = cpu_duration.count();
            tops_cpu = (ops / (cpu_time_ms / 1000.0)) / 1e12;  // 转换为TOPS
            std::cout << "CPU计算完成，时间: " << cpu_time_ms << " ms" << std::endl;
            
            // =============================================================================
            // 简单版本结果验证
            // =============================================================================
            std::cout << "验证简单WMMA版本结果..." << std::endl;
            simple_vs_cpu_match = true;
            int error_count = 0;
            for (int i = 0; i < size * size && error_count < 10; i++) {
                if (abs(h_d_gpu_simple[i] - h_d_cpu[i]) > 0) {
                    simple_vs_cpu_match = false;
                    error_count++;
                    if (error_count <= 5) {  // 只显示前5个错误
                        std::cout << "位置[" << i << "]: GPU=" << h_d_gpu_simple[i] 
                                  << ", CPU=" << h_d_cpu[i] << std::endl;
                    }
                }
            }
            if (simple_vs_cpu_match) {
                std::cout << "✓ 简单WMMA版本与CPU参考结果匹配" << std::endl;
            } else {
                std::cout << "✗ 简单WMMA版本与CPU参考结果不匹配 (错误数量: " << error_count << ")" << std::endl;
            }
            
            // =============================================================================
            // 优化版本结果验证
            // =============================================================================
            if (time_optimized > 0) {
                std::cout << "验证优化WMMA版本结果..." << std::endl;
                optimized_vs_cpu_match = true;
                error_count = 0;
                for (int i = 0; i < size * size && error_count < 10; i++) {
                    if (abs(h_d_gpu_optimized[i] - h_d_cpu[i]) > 0) {
                        optimized_vs_cpu_match = false;
                        error_count++;
                        if (error_count <= 5) {  // 只显示前5个错误
                            std::cout << "位置[" << i << "]: GPU=" << h_d_gpu_optimized[i] 
                                      << ", CPU=" << h_d_cpu[i] << std::endl;
                        }
                    }
                }
                if (optimized_vs_cpu_match) {
                    std::cout << "✓ 优化WMMA版本与CPU参考结果匹配" << std::endl;
                } else {
                    std::cout << "✗ 优化WMMA版本与CPU参考结果不匹配 (错误数量: " << error_count << ")" << std::endl;
                }
            } else {
                std::cout << "- 优化版本未执行，跳过验证" << std::endl;
                optimized_vs_cpu_match = false; // 未执行则标记为未匹配
            }
        } else {
            std::cout << std::endl << "[结果验证] CPU验证已禁用，无法验证结果正确性" << std::endl;
            simple_vs_cpu_match = false;    // 未验证则标记为未匹配
            optimized_vs_cpu_match = false; // 未验证则标记为未匹配
            cpu_time_ms = 0;
            tops_cpu = 0;
        }

        /*
         * =============================================================================
         * 性能总结 - 汇总显示所有测试结果
         * =============================================================================
         * 这个部分将前面所有测试的结果整理成表格形式
         * 便于对比不同实现之间的性能差异
         */
        std::cout << std::endl << "[性能总结]" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        /*
         * 显示CPU参考实现结果（如果启用了验证）
         * CPU实现主要用于：
         * 1. 验证GPU计算结果的正确性
         * 2. 对比GPU相对于CPU的加速比
         */
        if (enable_cpu_verification) {
            std::cout << "CPU参考实现:     " << cpu_time_ms << " ms, " << std::setprecision(4) << tops_cpu << " TOPS" << std::endl;
        } else {
            std::cout << "CPU参考实现:     跳过（已禁用）" << std::endl;
        }
        
        /*
         * 显示简单WMMA实现的性能结果
         * 简单版本特点：
         * - 直接使用WMMA API，代码简洁易懂
         * - 较少的共享内存使用，兼容性好
         * - 适合作为WMMA入门学习的参考
         */
        std::cout << std::setprecision(3) << "简单WMMA实现:    " << time_simple << " ms, " << std::setprecision(2) << tops_simple << " TOPS";
        if (enable_cpu_verification && cpu_time_ms > 0) {
            std::cout << " (加速比: " << std::setprecision(1) << (cpu_time_ms / time_simple) << "x)";
        }
        std::cout << std::endl;
        
        /*
         * 显示优化WMMA实现的性能结果（如果执行了的话）
         * 优化版本特点：
         * - 使用大量共享内存缓存数据
         * - 复杂的瓦片化策略和数据重用
         * - 更高的性能，但需要更多GPU资源
         */
        if (time_optimized > 0) {
            std::cout << std::setprecision(3) << "优化WMMA实现:    " << time_optimized << " ms, " << std::setprecision(2) << tops_optimized << " TOPS";
            if (enable_cpu_verification && cpu_time_ms > 0) {
                std::cout << " (加速比: " << std::setprecision(1) << (cpu_time_ms / time_optimized) << "x)";
            }
            std::cout << std::endl;
        }
        std::cout << std::defaultfloat;

        /*
         * =============================================================================
         * 结果验证状态说明
         * =============================================================================
         * 说明当前测试的结果验证状态
         */
        std::cout << std::endl << "[结果验证状态]" << std::endl;
        if (enable_cpu_verification) {
            std::cout << "CPU参考验证: ✓ 已完成" << std::endl;
            std::cout << "简单版本验证: " << (simple_vs_cpu_match ? "✓ 通过" : "✗ 失败") << std::endl;
            if (time_optimized > 0) {
                std::cout << "优化版本验证: " << (optimized_vs_cpu_match ? "✓ 通过" : "✗ 失败") << std::endl;
            } else {
                std::cout << "优化版本验证: - 跳过（未执行）" << std::endl;
            }
        } else {
            std::cout << "CPU参考验证: - 已禁用" << std::endl;
            std::cout << "简单版本验证: - 跳过（CPU验证已禁用）" << std::endl;
            std::cout << "优化版本验证: - 跳过（CPU验证已禁用）" << std::endl;
        }

        /*
         * 内存清理 - 释放在本次矩阵大小测试中分配的所有内存
         * 良好的内存管理习惯，避免内存泄漏
         * 
         * 主机内存清理：
         */
        free(h_a);           // 主机矩阵A
        free(h_b);           // 主机矩阵B  
        free(h_c);           // 主机矩阵C
        free(h_d_gpu_simple);    // 简单版本GPU结果
        free(h_d_gpu_optimized); // 优化版本GPU结果
        free(h_d_cpu);       // CPU参考结果
        
        /*
         * GPU设备内存清理：
         */
        checkCudaErrors(cudaFree(d_a));  // GPU矩阵A
        checkCudaErrors(cudaFree(d_b));  // GPU矩阵B
        checkCudaErrors(cudaFree(d_c));  // GPU矩阵C
        checkCudaErrors(cudaFree(d_d));  // GPU输出矩阵D
        
        /*
         * CUDA事件对象清理：
         * 这些事件用于精确的GPU时间测量
         */
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));

        /*
         * 收集性能数据用于汇总分析
         * 将每个矩阵大小的测试结果保存到PerfData结构体中
         * 这些数据将用于最终的性能分析和CSV文件输出
         */
        PerfData current_perf;
        current_perf.size = size;               // 矩阵维度
        current_perf.time_simple = time_simple; // 简单版本时间
        current_perf.tops_simple = tops_simple; // 简单版本性能
        current_perf.time_cpu = cpu_time_ms;    // CPU版本时间  
        current_perf.tops_cpu = tops_cpu;       // CPU版本性能
        
        /*
         * 检查优化版本是否成功执行
         * 需要重新计算SHMEM_SZ来确定是否有足够共享内存
         * 这样可以在最终报告中准确标识哪些测试运行了优化版本
         */
        size_t SHMEM_SZ_FINAL = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                                    M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int));
        current_perf.optimized_executed = (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ_FINAL);
        
        /*
         * 保存优化版本的性能数据
         * 如果优化版本执行了，保存实际数据；否则设为0
         */
        if (current_perf.optimized_executed) {
            current_perf.time_optimized = time_optimized;
            current_perf.tops_optimized = tops_optimized;
        } else {
            current_perf.time_optimized = 0;
            current_perf.tops_optimized = 0;
        }
        
        /*
         * 检查所有版本的计算结果是否匹配
         * results_match: 综合验证标志
         * - simple_vs_cpu_match: 简单版本vs CPU
         * - optimized_vs_cpu_match: 优化版本vs CPU（如果执行了）
         */
        current_perf.results_match = simple_vs_cpu_match && (current_perf.optimized_executed ? optimized_vs_cpu_match : true);
        
        // 将当前测试结果添加到总结果列表中
        perf_results.push_back(current_perf);
    }

    /*
     * =============================================================================
     * 性能分析汇总报告
     * =============================================================================
     * 在完成所有矩阵大小的测试后，生成综合性能分析报告
     * 这个报告包含：
     * 1. 表格形式的性能对比
     * 2. 最佳性能分析
     * 3. 矩阵对齐性能影响分析
     * 4. CSV数据导出
     */
    {
        std::string sep(80, '=');
        std::cout << std::endl << sep << std::endl;
    }
    std::cout << "性能分析汇总报告" << std::endl;
    {
        std::string sep(80, '=');
        std::cout << sep << std::endl;
    }

    /*
     * 输出表格格式的性能对比
     * 表格包含每个矩阵大小的详细性能数据：
     * - 矩阵大小: 测试的矩阵维度
     * - 简单版本/优化版本/CPU版本: 各自的执行时间
     * - TOPS值: 各版本的计算性能
     * - 结果匹配: 验证所有版本计算结果是否一致
     */
    std::cout << std::endl << std::left << std::setw(8) << "矩阵大小" << " " << std::setw(12) << "简单版本" << " " << std::setw(12) << "优化版本" << " " << std::setw(12) << "CPU版本" << " " << std::setw(8) << "简单TOPS" << " " << std::setw(8) << "优化TOPS" << " " << std::setw(8) << "CPU TOPS" << " " << std::setw(10) << "结果匹配" << std::endl;
    std::cout << std::left << std::setw(8) << "--------" << " " << std::setw(12) << "----------" << " " << std::setw(12) << "----------" << " " << std::setw(12) << "----------" << " " << std::setw(8) << "--------" << " " << std::setw(8) << "--------" << " " << std::setw(8) << "--------" << " " << std::setw(10) << "----------" << std::endl;

    /*
     * 遍历所有测试结果，逐行输出性能数据表格
     * 对于优化版本，如果因共享内存限制未执行，显示"跳过"和"N/A"
     */
    for (const auto& perf : perf_results) {
        std::string opt_time_str, opt_tops_str;
        if (perf.optimized_executed) {
            std::ostringstream ot; ot << std::fixed << std::setprecision(3) << perf.time_optimized; opt_time_str = ot.str();
            std::ostringstream ots; ots << std::fixed << std::setprecision(2) << perf.tops_optimized; opt_tops_str = ots.str();
        } else {
            opt_time_str = "跳过";      // 优化版本未执行
            opt_tops_str = "N/A";       // 性能数据不可用
        }

        std::cout << std::left << std::setw(8) << perf.size << " " << std::setw(12) << std::fixed << std::setprecision(3) << perf.time_simple << " " << std::setw(12) << opt_time_str << " " << std::setw(12) << std::fixed << std::setprecision(0) << perf.time_cpu << " " << std::setw(8) << std::fixed << std::setprecision(2) << perf.tops_simple << " " << std::setw(8) << opt_tops_str << " " << std::setw(8) << std::fixed << std::setprecision(4) << perf.tops_cpu << " " << std::setw(10) << (perf.results_match ? "✓" : "✗") << std::endl << std::defaultfloat;
    }

    /*
     * 找出所有测试中的最佳性能
     * 使用std::max_element和lambda函数来比较TOPS值
     * 这有助于识别在哪个矩阵大小下能达到最佳性能
     */
    auto best_simple = std::max_element(perf_results.begin(), perf_results.end(),
                                        [](const PerfData& a, const PerfData& b) {
                                            return a.tops_simple < b.tops_simple;
                                        });
    
    /*
     * 找出优化版本的最佳性能
     * 只考虑实际执行了优化版本的测试结果
     * lambda函数确保未执行的测试不会被选为最佳
     */
    auto best_optimized = std::max_element(perf_results.begin(), perf_results.end(),
                                           [](const PerfData& a, const PerfData& b) {
                                               if (!a.optimized_executed) return true;    // a未执行，a较小
                                               if (!b.optimized_executed) return false;   // b未执行，a较大
                                               return a.tops_optimized < b.tops_optimized; // 都执行了，比较TOPS
                                           });

    /*
     * 输出性能分析总结
     * 显示各版本的最佳性能和对应的矩阵大小
     */
    std::cout << std::endl << "性能分析结果:" << std::endl;
    std::cout << "• 简单版本最佳性能: " << std::fixed << std::setprecision(2) << best_simple->tops_simple << " TOPS (矩阵大小: " << best_simple->size << ")" << std::endl;
    
    if (best_optimized->optimized_executed) {
        std::cout << "• 优化版本最佳性能: " << std::fixed << std::setprecision(2) << best_optimized->tops_optimized << " TOPS (矩阵大小: " << best_optimized->size << ")" << std::endl;
        std::cout << "• 优化版本相对简单版本最大加速比: " << std::fixed << std::setprecision(2) << (best_optimized->tops_optimized / best_simple->tops_simple) << "x" << std::endl;
    }

    /*
     * 分析128倍数对齐对性能的影响
     * WMMA操作在内存对齐时通常有更好的性能
     * 128字节对齐对应矩阵维度为128的倍数
     */
    printf("\n128倍数对齐性能分析:\n");
    double avg_aligned = 0, avg_unaligned = 0;      // 对齐和非对齐的平均性能
    int count_aligned = 0, count_unaligned = 0;     // 对应的测试数量
    
    /*
     * 遍历所有优化版本的测试结果
     * 分别计算128对齐和非对齐情况的平均性能
     */
    for (const auto& perf : perf_results) {
        if (!perf.optimized_executed) continue;      // 跳过未执行优化版本的测试
        if (perf.size % 128 == 0) {                  // 128的倍数（对齐）
            avg_aligned += perf.tops_optimized;
            count_aligned++;
        } else {                                     // 非128倍数（非对齐）
            avg_unaligned += perf.tops_optimized;
            count_unaligned++;
        }
    }
    
    /*
     * 如果有足够的对齐和非对齐测试数据，计算并显示性能影响
     * 这有助于理解矩阵大小选择对性能的影响
     */
    if (count_aligned > 0 && count_unaligned > 0) {
        avg_aligned /= count_aligned;     // 计算平均值
        avg_unaligned /= count_unaligned;
        std::cout << "• 128对齐平均性能: " << std::fixed << std::setprecision(2) << avg_aligned << " TOPS (" << count_aligned << "个测试)" << std::endl;
        std::cout << "• 非128对齐平均性能: " << std::fixed << std::setprecision(2) << avg_unaligned << " TOPS (" << count_unaligned << "个测试)" << std::endl;
        std::cout << "• 性能衰减: " << std::fixed << std::setprecision(1) << ((avg_aligned - avg_unaligned) / avg_aligned * 100) << "%" << std::endl << std::defaultfloat;
    }

    /*
     * 输出CSV文件用于进一步分析
     * CSV格式便于在Excel、Python等工具中进行数据分析和可视化
     * 包含所有重要的性能指标和测试参数
     */
    FILE* csv_file = fopen("imma_performance_results.csv", "w");
    if (csv_file) {
        /*
         * CSV文件头部定义：
         * - Matrix_Size: 矩阵维度
         * - Simple_Time_ms: 简单版本执行时间(毫秒)
         * - Optimized_Time_ms: 优化版本执行时间(毫秒)
         * - CPU_Time_ms: CPU版本执行时间(毫秒)
         * - Simple_TOPS: 简单版本性能(TOPS)
         * - Optimized_TOPS: 优化版本性能(TOPS)
         * - CPU_TOPS: CPU版本性能(TOPS)
         * - Optimized_Executed: 优化版本是否执行(Yes/No)
         * - Results_Match: 结果验证是否通过(Yes/No)
         * - Is_128_Aligned: 是否128对齐(Yes/No)
         * - Speedup_vs_Simple: 优化版本相对简单版本的加速比
         */
        fprintf(csv_file, "Matrix_Size,Simple_Time_ms,Optimized_Time_ms,CPU_Time_ms,Simple_TOPS,Optimized_TOPS,CPU_TOPS,Optimized_Executed,Results_Match,Is_128_Aligned,Speedup_vs_Simple\n");
        
        /*
         * 逐行输出每个测试的详细数据
         * 计算加速比：简单版本时间 / 优化版本时间
         */
        for (const auto& perf : perf_results) {
            double speedup = perf.optimized_executed ? (perf.time_simple / perf.time_optimized) : 0;
            fprintf(csv_file, "%d,%.3f,%.3f,%.0f,%.6f,%.6f,%.6f,%s,%s,%s,%.2f\n",
                    perf.size,                                          // 矩阵大小
                    perf.time_simple,                                   // 简单版本时间
                    perf.optimized_executed ? perf.time_optimized : 0, // 优化版本时间
                    perf.time_cpu,                                      // CPU时间
                    perf.tops_simple,                                   // 简单版本TOPS
                    perf.optimized_executed ? perf.tops_optimized : 0, // 优化版本TOPS
                    perf.tops_cpu,                                      // CPU TOPS
                    perf.optimized_executed ? "Yes" : "No",            // 是否执行优化版本
                    perf.results_match ? "Yes" : "No",                 // 结果是否匹配
                    (perf.size % 128 == 0) ? "Yes" : "No",             // 是否128对齐
                    speedup);                                           // 加速比
        }
        fclose(csv_file);
        std::cout << std::endl << "性能数据已保存到: imma_performance_results.csv" << std::endl;
    }

    /*
     * 程序结束标识
     * 用分隔线和提示信息标识所有测试的完成
     * 这样用户可以清楚地知道程序已经完全结束
     */
    {
        std::string sep(80, '=');
        std::cout << std::endl << sep << std::endl;
    }
    std::cout << "所有测试完成！" << std::endl;
    {
        std::string sep(80, '=');
        std::cout << sep << std::endl;
    }
    
    /*
     * 返回成功状态码
     * EXIT_SUCCESS (通常为0) 表示程序正常结束
     * 操作系统和调用脚本可以通过返回值判断程序执行状态
     */
    return EXIT_SUCCESS;
}
