#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt 算法离线搜索 v1.0

架构说明：
=========
本工具采用 pybind11 扩展方式，而非 subprocess 调用：
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

分工合理性：
1. Python 控制外层 NK 循环，方便做进度条、异常捕获、断点续跑
2. C++ 控制内层 M 循环和算法枚举，避免跨进程通信开销
3. JSON 完全在 Python 端生成，避免 C++ 端的 JSON 处理脆弱性
4. torch.utils.cpp_extension.load 自动检测 GPU 架构，无需手动指定 -arch

显存优化：
- 预分配最大尺寸的张量，复用 buffer 避免反复 malloc/free
- 搜索结束后显式释放并调用 empty_cache()

运行示例:
CUDA_VISIBLE_DEVICES=0 python run_alg_search.py --dtype int8 --layout auto --verify --compile
CUDA_VISIBLE_DEVICES=0 python run_alg_search.py --dtype fp8e4m3 --layout auto --verify --compile

"""

import argparse
import ctypes
import ctypes.util
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.cpp_extension import load


# === CUDA 版本信息获取 ===

def get_nvidia_smi_cuda_version() -> str:
    """获取 nvidia-smi 显示的 CUDA Version（驱动支持的最高 CUDA 版本）"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 解析 "CUDA Version: 13.0" 这样的格式
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    # 格式通常是 "... CUDA Version: 13.0 ..."
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 备选方案：通过 cudaDriverGetVersion API
    try:
        driver_version = torch.cuda.driver_version()
        if driver_version:
            major = driver_version // 1000
            minor = (driver_version % 1000) // 10
            return f"{major}.{minor}"
    except Exception:
        pass
    
    return "unknown"


def get_cuda_runtime_version() -> str:
    """获取 CUDA runtime 版本（PyTorch 编译时使用的版本）"""
    try:
        # torch.version.cuda 是 PyTorch 编译时使用的 CUDA toolkit 版本
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


# === cuSPARSELt 动态库加载（必须在加载自定义 .so 之前完成）===
_CUSPARSELT_LOADED = False

def ensure_cusparselt_loaded() -> None:
    """优先加载系统或环境变量指定的 cuSPARSELt，避免符号冲突。"""
    global _CUSPARSELT_LOADED
    if _CUSPARSELT_LOADED:
        return

    preferred_paths = []
    env_path = os.environ.get("CUSPARSELT_PATH")
    if env_path:
        preferred_paths.append(env_path)

    arch = platform.machine()
    preferred_paths.extend(
        [
            "/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            "/usr/local/cuda/lib64/libcusparseLt.so.0",
        ]
    )
    found = ctypes.util.find_library("cusparseLt")
    if found:
        preferred_paths.append(found)

    for path in dict.fromkeys(preferred_paths):  # 去重但保持优先级
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            getattr(lib, "cusparseLtMatmulAlgSelectionDestroy")
            _CUSPARSELT_LOADED = True
            return
        except (OSError, AttributeError):
            continue

    raise OSError(
        "无法找到兼容的 libcusparseLt，请设置 CUSPARSELT_PATH 或安装 CUDA 12.9+。"
    )

# === 扩展编译/加载 ===

def load_extension(verbose: bool = True, force_compile: bool = False) -> object:
    """
    加载 CUDA 扩展。
    
    Args:
        verbose: 是否显示进度信息
        force_compile: 是否强制重新编译。默认 False 将复用已有 .so
    
    Returns:
        编译好的扩展模块
    """
    if verbose:
        print("[1/4] 加载 cuSPARSELt 库...", end=" ", flush=True)
    ensure_cusparselt_loaded()
    if verbose:
        print("✓", flush=True)
    
    src_path = Path(__file__).parent / "cusparselt_alg_search.cu"
    build_dir = Path(__file__).parent / "build_alg_ext"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有编译好的 .so
    so_pattern = build_dir / "cusparselt_alg_search_ext*.so"
    existing_so = list(build_dir.glob("cusparselt_alg_search_ext*.so"))
    
    if not force_compile and existing_so:
        # 直接加载已有的 .so
        if verbose:
            print(f"[2/4] 加载已编译的扩展...", end=" ", flush=True)
        import importlib.util
        so_file = existing_so[0]
        spec = importlib.util.spec_from_file_location("cusparselt_alg_search_ext", str(so_file))
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        if verbose:
            print(f"✓ ({so_file.name})", flush=True)
        return ext
    else:
        # 需要编译
        if force_compile and existing_so:
            if verbose:
                print("[2/4] 强制重新编译 CUDA 扩展...", end=" ", flush=True)
            # 删除旧的 .so 和编译缓存
            import shutil
            shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
        else:
            if verbose:
                print("[2/4] 编译 CUDA 扩展...", end=" ", flush=True)
        
        ext = load(
            name="cusparselt_alg_search_ext",
            sources=[str(src_path)],
            extra_cuda_cflags=["-O3"],
            # cuSPARSELt 可能需要 nvrtc 和 dl 库（某些发行版/静态链接场景）
            extra_ldflags=["-lcusparseLt", "-lnvrtc", "-ldl"],
            verbose=False,
            build_directory=str(build_dir),
            with_cuda=True,
        )
        if verbose:
            print("✓", flush=True)
        return ext

# === 架构检测 ===

def detect_arch() -> Tuple[str, int]:
    """
    检测 GPU 架构。
    
    返回:
        (arch_name, arch_id) 其中:
        - arch_name: "Ampere", "Hopper", "Blackwell" 等
        - arch_id: 0=Ampere, 1=Hopper, 2=Blackwell, 99=未知
    
    注意: torch.utils.cpp_extension.load 会自动检测 GPU 架构并选择合适的
    编译参数（-arch=sm_xx），无需手动指定。
    """
    prop = torch.cuda.get_device_properties(0)
    cc = (prop.major, prop.minor)
    if prop.major == 8:
        return "Ampere", 0
    if prop.major == 9:
        return "Hopper", 1
    if prop.major >= 10:
        return "Blackwell", 2
    return f"SM{prop.major}{prop.minor}", 99

# === 默认 NK/M 列表 ===

def default_nk_list() -> List[Tuple[int, int]]:
    return [
        (3840, 2560),  # Wqkv
        (2560, 2560),  # Wo
        (13824, 2560), # W13
        (2560, 6912),  # W2
    ]


def default_m_list() -> List[int]:
    # pow2 序列从 16 到 16384，覆盖 decode 和 prefill 的各种 batch size
    return [16, 64, 128, 512, 2048, 8192, 16384]


# === JSON 查询辅助函数 ===

def lookup_best_alg(json_data: Dict, N: int, K: int, M: int) -> Optional[int]:
    """
    从 JSON 数据中查询最佳算法 ID。
    
    查询逻辑：
    1. 用 (N, K) 在 nk_entries 中找到对应条目
    2. 在 m_thresholds 中找到 <= query_M 的最大值
    3. 返回该 M 对应的 alg_by_m[m][0]（最佳算法）
    
    Args:
        json_data: 加载的 JSON 数据
        N: 稀疏矩阵 W 的行数
        K: 共享维度
        M: 稠密矩阵 A 的行数（查询的 batch size）
    
    Returns:
        最佳算法 ID，如果找不到返回 None
    """
    nk_key = f"({N},{K})"
    nk_entries = json_data.get("nk_entries", {})
    
    if nk_key not in nk_entries:
        return None
    
    entry = nk_entries[nk_key]
    m_thresholds = entry.get("m_thresholds", [])
    alg_by_m = entry.get("alg_by_m", {})
    
    if not m_thresholds:
        return None
    
    # 找到 <= M 的最大阈值
    selected_m = None
    for threshold in m_thresholds:
        if threshold <= M:
            selected_m = threshold
        else:
            break
    
    if selected_m is None:
        # M 比所有阈值都小，使用最小的阈值
        selected_m = m_thresholds[0]
    
    m_key = str(selected_m)
    if m_key in alg_by_m:
        # 简化格式: alg_by_m[m_key] = [best_id, 2nd_id, 3rd_id]
        alg_list = alg_by_m[m_key]
        if isinstance(alg_list, list) and len(alg_list) > 0:
            return alg_list[0]
    
    return None


# === 运行一次 layout 的搜索 ===

def run_layout(ext, layout: str, dtype: str, nk_list: List[Tuple[int, int]], m_list: List[int],
               warmup: int, repeat: int, verify: bool, arch_id: int, verbose: bool = True) -> Dict:
    """
    运行一次 layout 配置的算法搜索。
    
    注意：
    - A100 的 NTRRrow 布局需要屏蔽某些算法 ID
    - FP8 在 A100 上不支持，需要提前拦截
    - 每个 NK 组合生成新的随机数据（显存优化在 v2.0 实现）
    """
    # === FP8 在 A100 上的防御 ===
    # A100 (Ampere, arch_id=0) 硬件不支持 FP8 Tensor Core
    if arch_id == 0 and dtype == "fp8e4m3":
        raise ValueError(
            f"FP8E4M3 不支持在 A100 (Ampere) 上运行。"
            f"A100 仅支持 INT8。请使用 --dtype int8"
        )
    
    results = []
    max_M = max(m_list)
    blacklist = [0, 1, 2, 3, 4] if (arch_id == 0 and layout == "NTRRrow") else []
    total_nk = len(nk_list)

    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据（每个 NK 新分配，简洁明了）
        max_M = max(m_list)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)

        # 先做 2:4 剪枝
        W_pruned = ext.prune_24(W, layout)

        # 调用搜索
        out = ext.search_topk(
            W_pruned,
            A,
            m_list,
            layout,
            dtype,
            warmup,
            repeat,
            verify,
            blacklist,
            3,
        )
        
        # 显示压缩算法ID和每个 M 的有效算法数
        compress_alg_id = out["compress_alg_id"]
        num_valid_per_m = out["num_valid_algs_per_M"].cpu().tolist()
        
        if verbose:
            print(f"      → 最大有效算法ID: {compress_alg_id}，正在通过 id={compress_alg_id} 进行压缩")
            # 显示每个 M 的有效算法数（取第一个作为代表，应该都一样）
            first_valid = num_valid_per_m[0] if num_valid_per_m else 0
            print(f"      → 每个 M 的有效算法数: {first_valid} ✓")
        
        results.append({
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "raw": out,
        })
        
        # 释放当前 NK 的张量
        del W, A, W_pruned
    
    torch.cuda.empty_cache()
    
    return {
        "layout": layout,
        "dtype": dtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
    }

# === 落盘工具 ===

def save_outputs(out_dir: Path, arch_name: str, arch_id: int, layout: str, dtype: str,
                 search_ret: Dict, warmup: int, repeat: int, verify: bool) -> None:
    """
    保存搜索结果到 CSV 和 JSON 文件。
    
    CSV 排序规则：先按 M 升序，M 相同时按 nk_list 传入顺序排序。
    
    JSON 格式设计用于两步查询：
    1. 先按 (N, K) 查找对应的 nk_entry
    2. 在 nk_entry 的 m_thresholds 中找到 <= 目标 M 的最大阈值，使用其 best_alg_id
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{arch_name}_{arch_id}_{layout}_{dtype.upper()}.csv"
    json_path = csv_path.with_suffix(".json")

    prop = torch.cuda.get_device_properties(0)
    
    # 获取 CUDA 版本信息
    cuda_driver_ver = get_nvidia_smi_cuda_version()  # nvidia-smi 显示的 CUDA 版本
    cuda_runtime_ver = get_cuda_runtime_version()    # PyTorch 编译时的 CUDA 版本
    
    meta = {
        "gpu_name": prop.name,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "arch_name": arch_name,
        "arch_id": arch_id,
        "layout": layout,
        "dtype": dtype,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": cuda_driver_ver,      # nvidia-smi 显示的版本，驱动支持的最高CUDA
        "cuda_version_runtime": cuda_runtime_ver,    # PyTorch编译时使用的CUDA toolkit版本
        "time": datetime.datetime.now().isoformat(),
        "M_list": search_ret["M_list"],
        "NK_list": search_ret["NK_list"],
    }

    # === CSV 生成（按 M 升序，M 相同时按 nk_list 顺序）===
    lines = []
    header_info = [
        f"# GPU: {prop.name}",
        f"# CC: {prop.major}.{prop.minor}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {cuda_driver_ver}, runtime: {cuda_runtime_ver}",
        f"# layout: {layout}, dtype: {dtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {search_ret['M_list']}",
        f"# NK_list: {search_ret['NK_list']}",
    ]
    lines.extend(header_info)
    lines.append("M,N,K,best_id1,lat_us1,tops1,best_id2,lat_us2,tops2,best_id3,lat_us3,tops3")

    # 收集所有数据行，用于排序
    csv_rows = []  # [(M, nk_idx, csv_line_str), ...]
    
    for nk_idx, res in enumerate(search_ret["results"]):
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        topk_lat = raw["topk_lat_us"].cpu()
        topk_tops = raw["topk_tops"].cpu()
        valid = raw["valid_mask"].cpu()

        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            lats = topk_lat[m_i]
            tops = topk_tops[m_i]
            vmask = valid[m_i]

            csv_values = [str(M), str(res["N"]), str(res["K"])]
            for k in range(3):
                if vmask[k]:
                    csv_values.extend([
                        str(int(algs[k].item())),
                        f"{float(lats[k].item()):.3f}",
                        f"{float(tops[k].item()):.6f}",
                    ])
                else:
                    csv_values.extend(["", "", ""])
            csv_rows.append((M, nk_idx, ",".join(csv_values)))

    # 排序：先按 M 升序，M 相同时按 nk_idx（即 nk_list 顺序）
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        lines.append(line)

    csv_path.write_text("\n".join(lines))

    # === JSON 生成（简化版：只保留 top3 算法 ID）===
    # 格式设计：
    # {
    #   "meta": {...},
    #   "nk_entries": {
    #     "(N,K)": {
    #       "m_thresholds": [m1, m2, ...],  # 升序排列的 M 值
    #       "alg_by_m": {
    #         "m1": [best_id, 2nd_id, 3rd_id],
    #         "m2": [...],
    #         ...
    #       }
    #     }
    #   }
    # }
    # 
    # 查询逻辑：
    # 1. 用 (N, K) 找到 nk_entry
    # 2. 在 m_thresholds 中找到 <= query_M 的最大值 m_key
    # 3. 返回 alg_by_m[m_key][0] 作为最佳算法
    
    nk_entries = {}
    
    for nk_idx, res in enumerate(search_ret["results"]):
        N, K = res["N"], res["K"]
        nk_key = f"({N},{K})"
        
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        valid = raw["valid_mask"].cpu()

        m_thresholds = []
        alg_by_m = {}
        
        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            vmask = valid[m_i]

            # 只有当有有效结果时才记录
            if vmask[0]:
                m_thresholds.append(M)
                # 简化格式：只记录 top3 的 alg_id
                top3_ids = [int(algs[k].item()) for k in range(3) if vmask[k]]
                alg_by_m[str(M)] = top3_ids
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }

    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")

# === 主流程 ===

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt 算法离线搜索 v1.0")
    p.add_argument("--dtype", default="int8", choices=["int8", "fp8e4m3"], help="数据类型")
    p.add_argument("--layout", default="auto", choices=["auto", "NTRRcol", "NTRRrow"], help="布局")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展（默认复用已有 .so）")
    p.add_argument("--out_dir", default=None, help="输出目录，默认 ./alg_search_results/<timestamp>")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    # === 显示配置信息 ===
    print("="*60)
    print("cuSPARSELt 算法离线搜索 v1.0")
    print("="*60)
    
    arch_name, arch_id = detect_arch()
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor}, {arch_name})")
    print(f"参数: dtype={args.dtype}, layout={args.layout}, warmup={args.warmup}, repeat={args.repeat}")
    if args.verify:
        print("注意: 已开启 verify 模式，会降低搜索速度")
    print()

    # 选择 layout 组合
    layout_list = []
    if args.layout == "auto":
        layout_list.append("NTRRcol")
        if arch_id == 0:
            layout_list.append("NTRRrow")
    else:
        layout_list.append(args.layout)

    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    ext = load_extension(verbose=True, force_compile=args.compile)

    nk_list = default_nk_list()
    m_list = default_m_list()
    
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()

    for layout_idx, layout in enumerate(layout_list):
        print(f"  [{layout_idx+1}/{len(layout_list)}] Layout: {layout}")
        
        ret = run_layout(
            ext,
            layout,
            args.dtype,
            nk_list,
            m_list,
            args.warmup,
            args.repeat,
            args.verify,
            arch_id,
            verbose=True,
        )
        save_outputs(
            out_dir,
            arch_name,
            arch_id,
            layout,
            args.dtype,
            ret,
            args.warmup,
            args.repeat,
            args.verify,
        )
        print()
    
    print(f"[4/4] 完成! 结果已保存到: {out_dir}")
    print("="*60)
if __name__ == "__main__":
    main()
