# CUDA开发环境配置说明

这个项目提供了完整的CUDA开发环境配置，支持Windows和Linux双平台。

## 📁 文件结构

```
MSRA/
├── .vscode/                    # VS Code配置目录
│   ├── c_cpp_properties.json  # C++智能感知配置
│   ├── tasks.json             # 编译任务配置  
│   ├── launch.json            # 调试配置
│   ├── settings.json          # VS Code设置
│   └── extensions.json        # 推荐扩展
├── GPUdemo/                   # CUDA示例代码目录
│   ├── compile_cuda.bat       # Windows编译脚本（推荐）
│   ├── Compile-Cuda.ps1       # PowerShell编译脚本
│   ├── vector_add.cu          # CUDA向量加法示例
│   └── README.md              # 本文档
└── CUDAsampleSHY/             # CUDA官方示例
```

## 🚀 快速开始

### Windows用户（推荐）

1. **使用批处理脚本**（最简单）：
   ```cmd
   cd GPUdemo
   compile_cuda.bat vector_add.cu
   ```

2. **使用PowerShell脚本**：
   ```powershell
   cd GPUdemo
   .\Compile-Cuda.ps1 vector_add.cu
   ```

3. **在VS Code中**：
   - 打开.cu文件
   - 按 `Ctrl+Shift+P`
   - 选择 "Tasks: Run Task"
   - 选择 "CUDA: 编译并运行 (Windows推荐)"

### Linux用户

```bash
cd GPUdemo
nvcc -o vector_add vector_add.cu
./vector_add
```

## 🔧 配置文件说明

### VS Code配置文件

| 文件名 | 功能 | 关键配置 |
|--------|------|----------|
| `c_cpp_properties.json` | 智能感知配置 | CUDA头文件路径、编译器路径 |
| `tasks.json` | 编译任务 | 调用编译脚本或直接编译 |
| `settings.json` | VS Code设置 | 文件关联、错误提示配置 |
| `launch.json` | 调试配置 | CUDA调试器配置 |
| `extensions.json` | 推荐扩展 | C++和CUDA扩展推荐 |

### 编译脚本

| 脚本名 | 平台 | 功能 | 特点 |
|--------|------|------|------|
| `compile_cuda.bat` | Windows | 编译+运行 | 自动设置VS环境，详细输出 |
| `Compile-Cuda.ps1` | Windows | 编译+运行 | PowerShell版本，彩色输出 |

## 🛠️ 编译脚本使用方法

### compile_cuda.bat

```cmd
# 基本用法
compile_cuda.bat 源文件.cu

# 指定输出文件名
compile_cuda.bat 源文件.cu 自定义名称

# 指定编译选项
compile_cuda.bat 源文件.cu 输出名称 "-O3"
```

### Compile-Cuda.ps1

```powershell
# 基本用法
.\Compile-Cuda.ps1 源文件.cu

# 指定输出文件名
.\Compile-Cuda.ps1 源文件.cu -OutputName 自定义名称

# 指定编译选项
.\Compile-Cuda.ps1 源文件.cu -CompileFlags "-O3"
```

## 🐛 常见问题

### 1. 智能感知不工作（#include标红）
- 检查 `c_cpp_properties.json` 中的CUDA路径是否正确
- 确保安装了 "C/C++" 扩展
- 重启VS Code

### 2. 编译失败
- 确保安装了CUDA Toolkit和Visual Studio
- 检查环境变量PATH中是否包含CUDA路径
- 使用编译脚本确保环境正确设置

### 3. VS Code任务不工作
- 直接使用编译脚本（更稳定）
- 检查路径分隔符（Windows用\\，Linux用/）

## 📝 开发建议

1. **推荐使用编译脚本**：比VS Code任务更稳定
2. **文件编码**：保存为UTF-8编码避免中文注释问题
3. **代码风格**：使用C语言风格变量声明兼容性更好
4. **调试**：CUDA调试需要特殊设置，建议先用printf调试

## 🔄 版本兼容性

- **CUDA**: 支持10.0+版本（配置中使用通配符）
- **Visual Studio**: 支持2019/2022版本
- **Linux**: 支持Ubuntu 18.04+、CentOS 7+
- **VS Code**: 支持1.60+版本

## 📚 学习资源

- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [VS Code C++配置](https://code.visualstudio.com/docs/languages/cpp)
- [CUDA示例代码](https://github.com/NVIDIA/cuda-samples)
