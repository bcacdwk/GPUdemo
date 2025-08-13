#!/usr/bin/env python3
"""
GPU监控脚本 - 实时显示GPU使用情况
使用方法: python3 gpu_monitor.py [刷新间隔秒数，默认2秒]
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        # 查询GPU基本信息
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,fan.speed',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_data = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                gpu_data.append({
                    'index': parts[0],
                    'name': parts[1],
                    'gpu_util': parts[2],
                    'mem_util': parts[3], 
                    'mem_used': parts[4],
                    'mem_total': parts[5],
                    'power': parts[6],
                    'temp': parts[7],
                    'fan': parts[8] if len(parts) > 8 and parts[8] != 'N/A' else 'N/A'
                })
        return gpu_data
        
    except subprocess.CalledProcessError as e:
        print(f"错误: 无法获取GPU信息 - {e}")
        return None
    except FileNotFoundError:
        print("错误: nvidia-smi 命令未找到，请确认NVIDIA驱动已正确安装")
        return None

def get_gpu_processes():
    """获取正在使用GPU的进程"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,gpu_uuid,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        processes = []
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line.strip() and 'No running' not in line:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 4:
                        processes.append({
                            'pid': parts[0],
                            'name': parts[1],
                            'gpu_uuid': parts[2],
                            'memory': parts[3]
                        })
        return processes
    except subprocess.CalledProcessError:
        return []

def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')

def format_bar(percentage, width=20):
    """创建进度条"""
    if percentage == 'N/A' or percentage == '':
        return '[' + ' ' * width + '] N/A'
    
    try:
        pct = float(percentage)
        filled = int(width * pct / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}] {pct:5.1f}%'
    except ValueError:
        return '[' + ' ' * width + '] N/A'

def display_gpu_status(gpu_data, processes):
    """显示GPU状态"""
    print("=" * 120)
    print(f"🖥️  GPU服务器监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    
    if not gpu_data:
        print("❌ 无法获取GPU信息")
        return
    
    # GPU状态表格
    print(f"{'ID':>2} {'名称':.<25} {'GPU利用率':<25} {'显存利用率':<25} {'显存使用':<15} {'功耗':<10} {'温度':<6}")
    print("-" * 120)
    
    total_gpu_util = 0
    total_mem_util = 0
    active_gpus = 0
    
    for gpu in gpu_data:
        gpu_util_bar = format_bar(gpu['gpu_util'])
        mem_util_bar = format_bar(gpu['mem_util'])
        
        # 统计活跃GPU
        try:
            if float(gpu['gpu_util']) > 0:
                active_gpus += 1
            total_gpu_util += float(gpu['gpu_util']) if gpu['gpu_util'] != 'N/A' else 0
            total_mem_util += float(gpu['mem_util']) if gpu['mem_util'] != 'N/A' else 0
        except ValueError:
            pass
        
        # 显存使用情况
        mem_info = f"{gpu['mem_used']:>5}MB/{gpu['mem_total']:>5}MB"
        
        # 功耗和温度
        power_info = f"{gpu['power']:>6}W" if gpu['power'] != 'N/A' else "   N/A"
        temp_info = f"{gpu['temp']:>3}°C" if gpu['temp'] != 'N/A' else " N/A"
        
        print(f"{gpu['index']:>2} {gpu['name'][:24]:.<25} {gpu_util_bar:<25} {mem_util_bar:<25} {mem_info:<15} {power_info:<10} {temp_info:<6}")
    
    print("-" * 120)
    
    # 整体统计
    avg_gpu_util = total_gpu_util / len(gpu_data) if gpu_data else 0
    avg_mem_util = total_mem_util / len(gpu_data) if gpu_data else 0
    
    print(f"📊 整体状态: {active_gpus}/{len(gpu_data)} GPU活跃 | 平均GPU利用率: {avg_gpu_util:.1f}% | 平均显存利用率: {avg_mem_util:.1f}%")
    
    # 正在运行的进程
    if processes:
        print(f"\n🔄 正在运行的GPU进程 ({len(processes)}个):")
        print(f"{'PID':>8} {'进程名':<30} {'显存占用':<10}")
        print("-" * 50)
        for proc in processes:
            print(f"{proc['pid']:>8} {proc['name'][:29]:<30} {proc['memory']:>8} MB")
    else:
        print(f"\n💤 当前没有运行GPU计算任务")
    
    print(f"\n💡 提示: 按 Ctrl+C 退出监控")

def main():
    # 获取刷新间隔参数
    refresh_interval = 2
    if len(sys.argv) > 1:
        try:
            refresh_interval = float(sys.argv[1])
        except ValueError:
            print(f"警告: 无效的刷新间隔 '{sys.argv[1]}'，使用默认值2秒")
    
    print(f"🚀 启动GPU监控，刷新间隔: {refresh_interval}秒")
    time.sleep(1)
    
    try:
        while True:
            clear_screen()
            gpu_data = get_gpu_info()
            processes = get_gpu_processes()
            display_gpu_status(gpu_data, processes)
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 监控已停止")
    except Exception as e:
        print(f"\n❌ 监控过程中出现错误: {e}")

if __name__ == "__main__":
    main()
