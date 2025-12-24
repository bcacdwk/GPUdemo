#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体支持 - 使用系统默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def read_and_plot():
    """读取CSV数据并绘制图表"""
    # 读取CSV文件
    df = pd.read_csv('imma_performance_results.csv')
    
    # 过滤掉无效数据
    df = df[df['Optimized_Executed'] == 'Yes']
    
    print("数据概览:")
    print(df[['Matrix_Size', 'Simple_TOPS', 'Optimized_TOPS', 'Speedup_vs_Simple']])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 性能对比 (TOPS)
    ax1.plot(df['Matrix_Size'], df['Simple_TOPS'], 
             marker='o', linewidth=2, markersize=8, 
             label='Simple WMMA', color='blue', alpha=0.8)
    
    ax1.plot(df['Matrix_Size'], df['Optimized_TOPS'], 
             marker='s', linewidth=2, markersize=8, 
             label='Optimized WMMA', color='red', alpha=0.8)
    
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Performance (TOPS)', fontsize=12)
    ax1.set_title('IMMA Tensor Core GEMM Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    
    # 添加性能数值标注
    for i, row in df.iterrows():
        ax1.annotate(f'{row["Simple_TOPS"]:.1f}', 
                    (row['Matrix_Size'], row['Simple_TOPS']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax1.annotate(f'{row["Optimized_TOPS"]:.1f}', 
                    (row['Matrix_Size'], row['Optimized_TOPS']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # 图2: 加速比
    ax2.plot(df['Matrix_Size'], df['Speedup_vs_Simple'], 
             marker='^', linewidth=2, markersize=8, 
             label='Speedup (Optimized vs Simple)', color='green', alpha=0.8)
    
    # 添加1.0基准线
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Speedup (1.0x)')
    
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('Optimized vs Simple WMMA Speedup', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xscale('linear')
    
    # 添加加速比数值标注
    for i, row in df.iterrows():
        ax2.annotate(f'{row["Speedup_vs_Simple"]:.2f}x', 
                    (row['Matrix_Size'], row['Speedup_vs_Simple']), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('imma_performance_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印性能分析
    print("\n=== 性能分析摘要 ===")
    print(f"测试矩阵大小范围: {df['Matrix_Size'].min()} - {df['Matrix_Size'].max()}")
    print(f"Simple WMMA最高性能: {df['Simple_TOPS'].max():.2f} TOPS (矩阵大小: {df.loc[df['Simple_TOPS'].idxmax(), 'Matrix_Size']})")
    print(f"Optimized WMMA最高性能: {df['Optimized_TOPS'].max():.2f} TOPS (矩阵大小: {df.loc[df['Optimized_TOPS'].idxmax(), 'Matrix_Size']})")
    print(f"最大加速比: {df['Speedup_vs_Simple'].max():.2f}x (矩阵大小: {df.loc[df['Speedup_vs_Simple'].idxmax(), 'Matrix_Size']})")
    
    # 找到性能交叉点
    crossover = df[df['Speedup_vs_Simple'] > 1.0]
    if not crossover.empty:
        min_crossover = crossover['Matrix_Size'].min()
        print(f"优化版本开始超越简单版本的矩阵大小: {min_crossover}")
    else:
        print("在测试范围内，优化版本未超越简单版本")

if __name__ == "__main__":
    read_and_plot()
