#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 设置中文字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_data(filename):
    """读取CSV数据并处理"""
    df = pd.read_csv(filename)
    
    # 过滤掉N/A值
    df = df[df['Latency(ms)'] != 'N/A']
    
    # 转换数据类型
    df['Latency(ms)'] = pd.to_numeric(df['Latency(ms)'])
    df['TOPS'] = pd.to_numeric(df['TOPS'])
    df['Bandwidth(GB/s)'] = pd.to_numeric(df['Bandwidth(GB/s)'])
    
    return df

def create_performance_plot(df):
    """创建性能对比图"""
    # 分离simple和optimized数据
    simple_data = df[df['Kernel'] == 'simple_wmma_gemm'].copy()
    optimized_data = df[df['Kernel'] == 'optimized_wmma_gemm'].copy()
    
    # 按矩阵大小排序
    simple_data = simple_data.sort_values('M')
    optimized_data = optimized_data.sort_values('M')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # === 计算性能图 (TOPS) ===
    ax1.plot(simple_data['M'], simple_data['TOPS'], 
             marker='o', linewidth=2, markersize=6, 
             label='Simple WMMA', color='blue', alpha=0.8)
    
    ax1.plot(optimized_data['M'], optimized_data['TOPS'], 
             marker='s', linewidth=2, markersize=6, 
             label='Optimized WMMA', color='red', alpha=0.8)
    
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Performance (TOPS)', fontsize=12)
    ax1.set_title('IMMA TensorCore GEMM Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 设置x轴刻度
    ax1.set_xticks(simple_data['M'])
    ax1.set_xticklabels([str(x) for x in simple_data['M']])
    
    # === 带宽图 (GB/s) ===
    ax2.plot(simple_data['M'], simple_data['Bandwidth(GB/s)'], 
             marker='o', linewidth=2, markersize=6, 
             label='Simple WMMA', color='blue', alpha=0.8)
    
    ax2.plot(optimized_data['M'], optimized_data['Bandwidth(GB/s)'], 
             marker='s', linewidth=2, markersize=6, 
             label='Optimized WMMA', color='red', alpha=0.8)
    
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Memory Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('Memory Bandwidth Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # 设置x轴刻度
    ax2.set_xticks(simple_data['M'])
    ax2.set_xticklabels([str(x) for x in simple_data['M']])
    
    plt.tight_layout()
    return fig

def create_combined_plot(df):
    """创建双Y轴合并图"""
    # 分离simple和optimized数据
    simple_data = df[df['Kernel'] == 'simple_wmma_gemm'].copy()
    optimized_data = df[df['Kernel'] == 'optimized_wmma_gemm'].copy()
    
    # 按矩阵大小排序
    simple_data = simple_data.sort_values('M')
    optimized_data = optimized_data.sort_values('M')
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 左Y轴 - 计算性能 (TOPS)
    color1 = 'tab:blue'
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Performance (TOPS)', color=color1, fontsize=12)
    
    line1 = ax1.plot(simple_data['M'], simple_data['TOPS'], 
                     marker='o', linewidth=2, markersize=6, 
                     label='Simple WMMA (TOPS)', color='blue', linestyle='-')
    
    line2 = ax1.plot(optimized_data['M'], optimized_data['TOPS'], 
                     marker='s', linewidth=2, markersize=6, 
                     label='Optimized WMMA (TOPS)', color='red', linestyle='-')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 右Y轴 - 带宽 (GB/s)
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Memory Bandwidth (GB/s)', color=color2, fontsize=12)
    
    line3 = ax2.plot(simple_data['M'], simple_data['Bandwidth(GB/s)'], 
                     marker='^', linewidth=2, markersize=6, 
                     label='Simple WMMA (GB/s)', color='lightblue', linestyle='--', alpha=0.7)
    
    line4 = ax2.plot(optimized_data['M'], optimized_data['Bandwidth(GB/s)'], 
                     marker='D', linewidth=2, markersize=6, 
                     label='Optimized WMMA (GB/s)', color='orange', linestyle='--', alpha=0.7)
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    # 设置x轴刻度
    ax1.set_xticks(simple_data['M'])
    ax1.set_xticklabels([str(x) for x in simple_data['M']])
    
    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.title('IMMA TensorCore GEMM: Performance vs Bandwidth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_crossover_analysis(df):
    """分析性能交叉点"""
    simple_data = df[df['Kernel'] == 'simple_wmma_gemm'].copy()
    optimized_data = df[df['Kernel'] == 'optimized_wmma_gemm'].copy()
    
    # 按矩阵大小排序
    simple_data = simple_data.sort_values('M')
    optimized_data = optimized_data.sort_values('M')
    
    print("=== 性能交叉点分析 ===")
    print("矩阵大小\tSimple TOPS\tOptimized TOPS\t优化比例")
    print("-" * 50)
    
    for i, size in enumerate(simple_data['M']):
        if size in optimized_data['M'].values:
            simple_tops = simple_data[simple_data['M'] == size]['TOPS'].iloc[0]
            opt_tops = optimized_data[optimized_data['M'] == size]['TOPS'].iloc[0]
            ratio = opt_tops / simple_tops
            
            status = "✓" if ratio > 1.0 else "✗"
            print(f"{size}\t\t{simple_tops:.2f}\t\t{opt_tops:.2f}\t\t{ratio:.2f}x {status}")
    
    # 找到性能交叉点
    crossover_sizes = []
    for i, size in enumerate(simple_data['M']):
        if size in optimized_data['M'].values:
            simple_tops = simple_data[simple_data['M'] == size]['TOPS'].iloc[0]
            opt_tops = optimized_data[optimized_data['M'] == size]['TOPS'].iloc[0]
            if opt_tops > simple_tops:
                crossover_sizes.append(size)
    
    if crossover_sizes:
        print(f"\n优化kernel开始超越simple kernel的矩阵大小: {min(crossover_sizes)}")
    
    return crossover_sizes

def main():
    """主函数"""
    # 读取数据
    print("读取CSV数据...")
    df = read_csv_data('imma_perf_results.csv')
    print(f"成功读取 {len(df)} 条记录")
    
    # 创建性能分析
    crossover_sizes = create_crossover_analysis(df)
    
    # 创建分离图表
    print("\n创建分离图表...")
    fig1 = create_performance_plot(df)
    fig1.savefig('imma_performance_separate.png', dpi=300, bbox_inches='tight')
    print("保存图表: imma_performance_separate.png")
    
    # 创建合并图表
    print("创建合并图表...")
    fig2 = create_combined_plot(df)
    fig2.savefig('imma_performance_combined.png', dpi=300, bbox_inches='tight')
    print("保存图表: imma_performance_combined.png")
    
    # 显示图表
    plt.show()
    
    print(f"\n📊 图表生成完成！")
    print(f"🔍 关键发现: 优化kernel在矩阵大小 ≥ {min(crossover_sizes) if crossover_sizes else 'N/A'} 时开始显示优势")

if __name__ == "__main__":
    main()
