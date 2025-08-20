#!/bin/bash

# NCU性能分析脚本
# 使用方法: ./run_ncu_analysis.sh <可执行文件名> [输出文件名]
# 例如: ./run_ncu_analysis.sh add my_analysis

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法: $0 <可执行文件名> [输出文件名]"
    echo "例如: $0 add"
    echo "例如: $0 add my_analysis"
    exit 1
fi

EXECUTABLE=$1
OUTPUT_NAME=${2:-"ncu_analysis_$(date +%Y%m%d_%H%M%S)"}

# 检查可执行文件是否存在
if [ ! -f "./$EXECUTABLE" ]; then
    echo "错误: 可执行文件 './$EXECUTABLE' 不存在"
    echo "请先编译你的CUDA程序"
    exit 1
fi

echo "🚀 开始NCU性能分析..."
echo "📁 可执行文件: $EXECUTABLE"
echo "📄 输出文件: ${OUTPUT_NAME}.csv"
echo "⏱️  开始时间: $(date)"
echo "----------------------------------------"

# 运行NCU分析，使用详细模式和CSV输出
sudo /usr/local/cuda-12.2/bin/ncu \
    --set detailed \
    --csv \
    --log-file "${OUTPUT_NAME}.csv" \
    --force-overwrite \
    ./$EXECUTABLE

# 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "✅ 分析完成!"
    echo "📄 CSV文件已保存到: ${OUTPUT_NAME}.csv"
    echo "📊 文件大小: $(ls -lh ${OUTPUT_NAME}.csv | awk '{print $5}')"
    echo ""
    echo "💡 提示:"
    echo "   - 使用 head -20 ${OUTPUT_NAME}.csv 查看前20行"
    echo "   - 使用 tail -20 ${OUTPUT_NAME}.csv 查看后20行" 
    echo "   - 可以用Excel或其他工具打开CSV文件进行分析"
else
    echo "❌ NCU分析失败"
    exit 1
fi
