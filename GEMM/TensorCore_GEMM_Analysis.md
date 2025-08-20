# NVIDIA TensorCore GEMM 代码深度分析

## 概述

这个文件夹包含了5个不同精度的NVIDIA TensorCore GEMM（通用矩阵乘法）实现，展示了如何使用CUDA的Warp Matrix Multiply and Accumulate (WMMA) API来充分利用Tensor Core硬件单元进行高性能矩阵计算。

### 5个实现版本对比

| 实现 | 数据类型 | Tile大小 | K维度 | 特点 |
|------|----------|----------|--------|------|
| **bf16TensorCoreGemm** | `__nv_bfloat16` | 16×16×16 | 16 | BF16混合精度，支持异步拷贝 |
| **cudaTensorCoreGemm** | `half` (FP16) | 16×16×16 | 16 | 传统FP16实现，基础版本 |  
| **tf32TensorCoreGemm** | `float` (TF32) | 16×16×8 | 8 | TF32格式，平衡精度和性能 |
| **dmmaTensorCoreGemm** | `double` | 8×8×4 | 4 | 双精度浮点，最高精度 |
| **immaTensorCoreGemm** | `int8_t` | 16×16×16 | 16 | 整数运算，最高吞吐量 |

---

## 核心算法架构

### 🧩 分块策略 (Tiling Strategy)

所有实现都采用了**分层分块**的设计：

```
全局矩阵 → CTA块 → Warp子块 → MMA Tile
8192×8192 → 128×128 → 64×32 → 16×16
```

#### 1. 全局级别分块
```cpp
#define M_TILES 512      // 总共512个M方向的tile
#define N_TILES 512      // 总共512个N方向的tile  
#define K_TILES 512      // 总共512个K方向的tile

#define M_GLOBAL (M * M_TILES)  // = 16 × 512 = 8192
#define N_GLOBAL (N * N_TILES)  // = 16 × 512 = 8192
#define K_GLOBAL (K * K_TILES)  // = 16 × 512 = 8192
```

#### 2. CTA (Cooperative Thread Array) 级别分块
```cpp
#define BLOCK_ROW_TILES 8    // 每个CTA处理8×8=64个tile
#define BLOCK_COL_TILES 8    // 对应128×128的矩阵块
#define WARPS_PER_BLOCK 8    // 每个CTA有8个warp
```

#### 3. Warp级别分块
```cpp
#define WARP_ROW_TILES 4     // 每个warp处理4×2=8个tile
#define WARP_COL_TILES 2     // 对应64×32的矩阵块
```

### 🏗️ 内存层次优化

#### 1. 共享内存管理
```cpp
// 共享内存声明 (以BF16为例)
extern __shared__ __nv_bfloat16 shmem[][CHUNK_K * K + SKEW_BF16];
```

**Bank Conflict优化 - SKEW机制：**
```cpp
#define SKEW_BF16 16    // BF16版本的skew
#define SKEW_HALF 8     // FP16版本的skew  
#define SKEW_FLOAT 8    // FP32版本的skew
#define SKEW_DOUBLE 4   // FP64版本的skew
```

**SKEW的作用：**
- **问题**：共享内存有32个bank，如果多个线程访问同一bank会产生冲突
- **解决**：通过在每行末尾添加padding，让连续行映射到不同bank
- **原理**：保持256位对齐要求的同时避免bank冲突

#### 2. 数据预取和流水线
```cpp
// 异步拷贝 (CUDA 11.0+)
cuda::memcpy_async(group, dest, src, sizeof(int4), pipeline);
pipeline.producer_commit();
pipeline.consumer_wait();
```

---

## 关键代码块详解

### 🔧 1. Warp和Thread标识
```cpp
// 在每个kernel中都有的标准模式
const unsigned int warpId = threadIdx.x / WARP_SIZE;  // warp在block中的ID
const unsigned int laneId = threadIdx.x % WARP_SIZE;  // thread在warp中的ID

// Block在Grid中的位置
const unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
```

**作用**：
- `warpId`：确定当前warp负责计算哪些tile
- `laneId`：确定线程在warp内的角色和数据访问模式
- `blockId`：确定当前block处理全局矩阵的哪个区域

### 🔧 2. WMMA Fragment声明
```cpp
// 以BF16为例
wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a[WARP_COL_TILES];
wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b[WARP_ROW_TILES];  
wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
```

**参数含义**：
- `matrix_a/matrix_b/accumulator`：fragment类型
- `M, N, K`：tile维度（16×16×16 或 8×8×4）
- `__nv_bfloat16/half/float/double`：数据类型
- `row_major/col_major`：内存布局

### 🔧 3. 核心计算循环
```cpp
// 外层循环：遍历K维度的块
for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
    
    // 1. 数据加载到共享内存
    // ...拷贝A和B的数据块到shmem...
    __syncthreads();
    
    // 内层循环：处理当前块内的K步长
    for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        
        // 2. 从共享内存加载到fragment
        wmma::load_matrix_sync(a[i], tile_ptr, ldm);
        wmma::load_matrix_sync(b[j], tile_ptr, ldm);
        
        // 3. 执行矩阵乘累加
        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
    }
    __syncthreads();
}
```

### 🔧 4. 结果输出和缩放
```cpp
// Alpha缩放
for (int t = 0; t < c[i][j].num_elements; t++)
    c[i][j].x[t] *= alpha;

// 存储到共享内存
wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);

// 从共享内存拷贝到全局内存
// ...vectorized copy using float4...
```

---

## 5个版本的核心差异

### 🎯 1. BF16 TensorCore GEMM
**特点**：
- **数据类型**：`__nv_bfloat16` (8位指数 + 7位尾数)
- **优势**：与FP32动态范围相同，但占用更少内存
- **Tile配置**：16×16×16
- **特殊特性**：支持异步内存拷贝

```cpp
// 关键类型定义
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;

// 异步拷贝支持
cuda::memcpy_async(group, dst, src, sizeof(int4), pipeline);
```

### 🎯 2. CUDA TensorCore GEMM (FP16)
**特点**：
- **数据类型**：`half` (5位指数 + 10位尾数)
- **优势**：传统的混合精度方案，广泛支持
- **Tile配置**：16×16×16
- **特殊特性**：最成熟的实现

```cpp
// 关键类型定义
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c; // 累加器用FP32
```

### 🎯 3. TF32 TensorCore GEMM
**特点**：
- **数据类型**：`float` 但以TF32格式执行 (8位指数 + 10位尾数)
- **优势**：无需修改FP32代码即可获得加速
- **Tile配置**：16×16×8 (注意K=8)
- **特殊特性**：自动截断FP32精度

```cpp
// 关键配置差异
#define K 8  // TF32的K维度是8，不是16

// 编译器自动使用TF32
wmma::fragment<wmma::matrix_a, 16, 16, 8, float, wmma::row_major> a;
```

### 🎯 4. Double Precision GEMM
**特点**：
- **数据类型**：`double` (11位指数 + 52位尾数)
- **优势**：最高精度，适合科学计算
- **Tile配置**：8×8×4 (更小的tile)
- **特殊特性**：需要更多寄存器和内存

```cpp
// 更小的tile配置
#define M 8
#define N 8  
#define K 4

// 更小的block配置
#define BLOCK_ROW_TILES 4  // 32×32 block instead of 128×128
#define BLOCK_COL_TILES 4
```

### 🎯 5. Integer MMA GEMM
**特点**：
- **数据类型**：`int8_t` 输入，`int32_t` 累加
- **优势**：最高吞吐量，适合推理
- **Tile配置**：16×16×16
- **特殊特性**：整数运算，需要量化处理

```cpp
// 整数类型配置
wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c;

// 可能需要后处理进行去量化
for (int t = 0; t < c[i][j].num_elements; t++)
    c[i][j].x[t] = c[i][j].x[t] * scale_a * scale_b + bias;
```

---

## 性能优化技术对比

### 📊 内存访问模式

| 版本 | 数据类型大小 | 向量化拷贝 | Bank Conflict处理 |
|------|-------------|------------|-------------------|
| BF16 | 2字节 | `int4` (16字节) | `SKEW_BF16 = 16` |
| FP16 | 2字节 | `int4` (16字节) | `SKEW_HALF = 8` |
| TF32 | 4字节 | `float4` (16字节) | `SKEW_FLOAT = 8` |
| FP64 | 8字节 | `double2` (16字节) | `SKEW_DOUBLE = 4` |
| INT8 | 1字节 | `int4` (16字节) | `SKEW_HALF = 8` |

### 📊 计算能力需求

| 版本 | 最低Compute Capability | Tensor Core代数 | 特殊要求 |
|------|----------------------|----------------|----------|
| BF16 | 8.0 (A100+) | 第3代 | CUDA 11.0+ |
| FP16 | 7.0 (V100+) | 第1代 | CUDA 9.0+ |
| TF32 | 8.0 (A100+) | 第3代 | 自动启用 |
| FP64 | 8.0 (A100+) | 第3代 | 大共享内存 |
| INT8 | 7.2 (T4+) | 第2代 | 量化支持 |

---

## 编译和使用指南

### 编译命令
```bash
# 针对A100 (compute capability 8.0)
nvcc -arch=compute_80 -code=sm_80 -O3 -o bf16gemm bf16TensorCoreGemm.cu

# 检查Tensor Core使用情况
ncu --metrics tensor_precision_fu_utilization.avg.pct_of_peak_sustained_active ./bf16gemm
```

### 性能分析要点
1. **Tensor Core利用率**：目标 >80%
2. **内存带宽利用率**：目标 >70%
3. **Occupancy**：目标 >75%
4. **Bank Conflicts**：应该接近0

### 常见优化检查点
- [ ] 矩阵维度是否是16的倍数（或对应tile大小的倍数）
- [ ] 共享内存使用是否超过限制
- [ ] 是否正确使用了异步拷贝
- [ ] Bank conflict是否已优化
- [ ] 数据布局是否符合要求（row-major vs col-major）

---

## 总结

这5个GEMM实现展示了NVIDIA Tensor Core在不同精度下的优化策略：

1. **BF16**：AI训练的首选，动态范围大
2. **FP16**：推理和训练的平衡选择
3. **TF32**：FP32代码的无缝加速
4. **FP64**：科学计算的精度保证
5. **INT8**：边缘推理的极致性能

每个版本都针对其特定的使用场景和硬件特性进行了深度优化，是学习GPU高性能计算的绝佳例子。
