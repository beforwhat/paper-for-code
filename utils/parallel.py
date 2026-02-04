# -*- coding: utf-8 -*-
"""
并行计算模块
核心功能：
1. 多进程并行加速Shapley值（SA贡献度）计算（计算密集型）；
2. 多客户端并行训练（联邦学习核心场景，支持GPU分片）；
3. 通用并行任务调度（兼容多进程/多线程，自动适配任务类型）；
4. 进度监控、异常容错、结果统一汇总；
5. 设备适配：支持多GPU分配、CPU/GPU混合并行。
"""

import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import torch
import tqdm

# 全局配置（适配不同硬件）
DEFAULT_CONFIG = {
    "n_workers": max(1, multiprocessing.cpu_count() - 1),  # 默认进程数（CPU核心数-1）
    "gpu_ids": None,  # GPU ID列表（如[0,1,2]），None表示自动分配
    "timeout": 3600,  # 单任务超时时间（秒）
    "chunk_size": 1,  # 任务分片大小（Shapley计算用）
    "daemon": True,   # 守护进程（避免僵尸进程）
    "verbose": True   # 是否显示进度条
}

# 进程间通信的全局变量（用于GPU分配）
_shared_ctx = multiprocessing.Manager().dict()

def _init_worker(gpu_ids: Optional[List[int]] = None) -> None:
    """
    子进程初始化函数（设置GPU、禁用不必要的日志）
    """
    # 忽略SIGINT（让主进程处理中断）
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 分配GPU（按进程ID轮询）
    if gpu_ids and torch.cuda.is_available():
        worker_id = multiprocessing.current_process().pid % len(gpu_ids)
        gpu_id = gpu_ids[worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
    
    # 禁用子进程的tqdm输出（避免混乱）
    if not DEFAULT_CONFIG["verbose"]:
        import builtins
        builtins.print = lambda *args, **kwargs: None

class ParallelRunner:
    """
    通用并行任务管理器
    核心能力：
    - 自动选择多进程（计算密集型）/多线程（IO密集型）
    - 支持GPU分片分配
    - 进度监控、异常容错、结果汇总
    """
    def __init__(
        self,
        n_workers: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        use_threads: bool = False,  # True=多线程（IO密集），False=多进程（计算密集）
        timeout: int = DEFAULT_CONFIG["timeout"],
        verbose: bool = DEFAULT_CONFIG["verbose"]
    ):
        self.n_workers = n_workers or DEFAULT_CONFIG["n_workers"]
        self.gpu_ids = gpu_ids or DEFAULT_CONFIG["gpu_ids"]
        self.use_threads = use_threads
        self.timeout = timeout
        self.verbose = verbose
        
        # 日志（复用全局日志器）
        try:
            from utils import get_global_logger
            self.logger = get_global_logger()
        except ImportError:
            self.logger = None
        
        # 状态记录
        self.results = []
        self.failed_tasks = []
        self.total_tasks = 0
        self.completed_tasks = 0
    
    def _log(self, msg: str, level: str = "info") -> None:
        """内部日志函数"""
        if self.logger:
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
        elif self.verbose:
            print(f"[{level.upper()}] {msg}")
    
    def _update_progress(self, desc: str = "并行任务") -> tqdm.tqdm:
        """创建进度条"""
        if not self.verbose:
            return None
        return tqdm.tqdm(total=self.total_tasks, desc=desc, unit="task", dynamic_ncols=True)
    
    def run_parallel(
        self,
        func: Callable,
        tasks: List[Any],
        desc: str = "并行任务"
    ) -> Tuple[List[Any], List[Any]]:
        """
        执行并行任务
        Args:
            func: 任务函数（需可序列化，多进程下不能是类方法）
            tasks: 任务列表（每个元素是func的参数元组）
            desc: 进度条描述
        Returns:
            results: 成功任务的结果列表（按任务顺序）
            failed_tasks: 失败任务的信息列表
        """
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        self.results = [None] * self.total_tasks
        self.failed_tasks = []
        
        if self.total_tasks == 0:
            self._log("无并行任务需要执行", level="warning")
            return self.results, self.failed_tasks
        
        # 初始化执行器
        progress_bar = self._update_progress(desc)
        executor_cls = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": self.n_workers,
            "initializer": _init_worker if not self.use_threads else None,
            "initargs": (self.gpu_ids,) if not self.use_threads else ()
        }
        
        try:
            with executor_cls(**executor_kwargs) as executor:
                # 提交任务并记录索引（保证结果顺序）
                future_to_idx = {
                    executor.submit(func, *task): idx for idx, task in enumerate(tasks)
                }
                
                # 收集结果
                for future in as_completed(future_to_idx, timeout=self.timeout):
                    idx = future_to_idx[future]
                    try:
                        self.results[idx] = future.result()
                        self.completed_tasks += 1
                    except Exception as e:
                        self.failed_tasks.append({
                            "task_idx": idx,
                            "task_args": tasks[idx],
                            "error": str(e)
                        })
                        self._log(f"任务{idx}执行失败：{e}", level="error")
                    finally:
                        if progress_bar:
                            progress_bar.update(1)
            
            # 进度条收尾
            if progress_bar:
                progress_bar.close()
            
            # 日志汇总
            success_rate = (self.completed_tasks - len(self.failed_tasks)) / self.total_tasks * 100
            self._log(f"并行任务完成：总数={self.total_tasks}，成功={self.completed_tasks - len(self.failed_tasks)}，失败={len(self.failed_tasks)}，成功率={success_rate:.1f}%")
            
            if self.failed_tasks:
                self._log(f"失败任务详情：{self.failed_tasks}", level="warning")
        
        except KeyboardInterrupt:
            self._log("并行任务被用户中断", level="warning")
            executor.shutdown(wait=False, cancel_futures=True)
            if progress_bar:
                progress_bar.close()
            raise
        except Exception as e:
            self._log(f"并行执行器异常：{e}", level="error")
            if progress_bar:
                progress_bar.close()
            raise
        
        return self.results, self.failed_tasks

# ======================== 核心1：并行计算Shapley值（SA贡献度） ========================
def _shapley_calculate_worker(args: Tuple) -> Tuple[int, float]:
    """
    Shapley值计算子进程函数（需独立函数，不能是类方法）
    Args:
        args: (client_id, data, model, shapley_config)
    Returns:
        (client_id, shapley_value): 客户端ID和对应的Shapley值
    """
    try:
        client_id, data, model_state_dict, shapley_config = args
        
        # 重建模型
        model = shapley_config["model_cls"](**shapley_config.get("model_kwargs", {}))
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # 设备配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Shapley值计算核心逻辑（适配不同的Shapley计算方法）
        calculate_func = shapley_config["calculate_func"]
        shapley_value = calculate_func(
            client_id=client_id,
            data=data,
            model=model,
            **shapley_config.get("func_kwargs", {})
        )
        
        return (client_id, float(shapley_value))
    except Exception as e:
        raise RuntimeError(f"客户端{client_id}Shapley值计算失败：{str(e)}") from e

def parallel_shapley_calculate(
    client_data: Dict[int, Any],  # {client_id: 客户端数据}
    model: torch.nn.Module,
    calculate_func: Callable,
    shapley_config: Optional[Dict] = None,
    n_workers: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, float]:
    """
    并行计算多客户端Shapley值（SA贡献度）
    Args:
        client_data: 客户端ID到数据的映射
        model: 全局模型（用于计算Shapley值）
        calculate_func: 单个客户端Shapley值计算函数
        shapley_config: Shapley计算配置（model_cls/func_kwargs等）
        n_workers: 并行进程数
        gpu_ids: GPU ID列表
        verbose: 是否显示进度条
    Returns:
        shapley_values: {client_id: shapley_value}（仅成功计算的客户端）
    """
    # 初始化配置
    shapley_config = shapley_config or {}
    shapley_config["calculate_func"] = calculate_func
    shapley_config["model_cls"] = model.__class__
    shapley_config["model_kwargs"] = shapley_config.get("model_kwargs", {})
    
    # 准备任务列表（模型序列化为state_dict，避免进程间拷贝大对象）
    model_state_dict = model.state_dict()
    tasks = []
    for client_id, data in client_data.items():
        tasks.append((client_id, data, model_state_dict, shapley_config))
    
    # 执行并行计算
    runner = ParallelRunner(
        n_workers=n_workers,
        gpu_ids=gpu_ids,
        use_threads=False,  # Shapley计算是计算密集型，用多进程
        verbose=verbose
    )
    results, failed_tasks = runner.run_parallel(
        func=_shapley_calculate_worker,
        tasks=tasks,
        desc="并行计算Shapley值"
    )
    
    # 整理结果
    shapley_values = {}
    for res in results:
        if res is not None:
            client_id, value = res
            shapley_values[client_id] = value
    
    # 日志
    runner._log(f"Shapley值并行计算完成：成功{len(shapley_values)}个客户端，失败{len(failed_tasks)}个")
    return shapley_values

# ======================== 核心2：多客户端并行训练 ========================
def _client_train_worker(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """
    客户端训练子进程函数
    Args:
        args: (client_id, train_data, model_state_dict, train_config)
    Returns:
        (client_id, result): 客户端ID和训练结果（model_state_dict/metrics/loss等）
    """
    try:
        client_id, train_data, model_state_dict, train_config = args
        
        # 重建模型
        model = train_config["model_cls"](**train_config.get("model_kwargs", {}))
        model.load_state_dict(model_state_dict)
        
        # 设备配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # 初始化优化器
        optimizer = train_config["optimizer_cls"](
            model.parameters(),
            **train_config.get("optimizer_kwargs", {})
        )
        
        # 训练逻辑
        epochs = train_config.get("epochs", 1)
        loss_fn = train_config["loss_fn"]
        metrics = train_config.get("metrics", {})
        
        # 执行训练
        train_loss = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X, y in train_data:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss.append(epoch_loss / len(train_data))
        
        # 计算指标
        model.eval()
        metric_results = {}
        with torch.no_grad():
            for metric_name, metric_func in metrics.items():
                metric_results[metric_name] = metric_func(model, train_data, device)
        
        # 收集结果（仅返回state_dict，避免传递大模型）
        result = {
            "client_id": client_id,
            "model_state_dict": model.state_dict(),
            "train_loss": np.mean(train_loss),
            "metrics": metric_results,
            "epochs": epochs,
            "device": str(device)
        }
        
        return (client_id, result)
    except Exception as e:
        raise RuntimeError(f"客户端{client_id}训练失败：{str(e)}") from e

def parallel_client_training(
    client_datasets: Dict[int, Any],  # {client_id: 训练数据集}
    global_model: torch.nn.Module,
    train_config: Dict[str, Any],
    n_workers: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    多客户端并行训练
    Args:
        client_datasets: 客户端ID到训练数据集的映射
        global_model: 全局模型（初始模型）
        train_config: 训练配置（model_cls/optimizer_cls/loss_fn/epochs等）
        n_workers: 并行进程数
        gpu_ids: GPU ID列表
        verbose: 是否显示进度条
    Returns:
        client_results: {client_id: 训练结果}（仅成功训练的客户端）
    """
    # 验证训练配置
    required_config = ["model_cls", "optimizer_cls", "loss_fn"]
    for key in required_config:
        if key not in train_config:
            raise ValueError(f"训练配置缺少必要项：{key}")
    
    # 准备任务列表
    global_model_state = global_model.state_dict()
    tasks = []
    for client_id, dataset in client_datasets.items():
        tasks.append((client_id, dataset, global_model_state, train_config))
    
    # 执行并行训练
    runner = ParallelRunner(
        n_workers=n_workers,
        gpu_ids=gpu_ids,
        use_threads=False,  # 模型训练是计算密集型，用多进程
        verbose=verbose
    )
    results, failed_tasks = runner.run_parallel(
        func=_client_train_worker,
        tasks=tasks,
        desc="多客户端并行训练"
    )
    
    # 整理结果
    client_results = {}
    for res in results:
        if res is not None:
            client_id, result = res
            client_results[client_id] = result
    
    # 日志
    runner._log(f"多客户端并行训练完成：成功{len(client_results)}个客户端，失败{len(failed_tasks)}个")
    return client_results

# ======================== 通用并行工具函数 ========================
def auto_split_tasks(
    tasks: List[Any],
    n_workers: int = DEFAULT_CONFIG["n_workers"],
    chunk_size: int = DEFAULT_CONFIG["chunk_size"]
) -> List[List[Any]]:
    """
    自动分片任务（用于大任务集的并行处理）
    Args:
        tasks: 原始任务列表
        n_workers: 进程数
        chunk_size: 每个分片的任务数
    Returns:
        task_chunks: 分片后的任务列表
    """
    task_chunks = []
    for i in range(0, len(tasks), chunk_size * n_workers):
        chunk = tasks[i:i + chunk_size * n_workers]
        # 均分每个进程的任务数
        for j in range(n_workers):
            worker_chunk = chunk[j::n_workers]
            if worker_chunk:
                task_chunks.append(worker_chunk)
    return task_chunks

def get_available_gpus() -> List[int]:
    """
    获取可用的GPU ID列表（适配PyTorch）
    Returns:
        gpu_ids: 可用GPU ID列表（如[0,1]）
    """
    if not torch.cuda.is_available():
        return []
    gpu_ids = list(range(torch.cuda.device_count()))
    # 过滤不可用的GPU
    available_gpus = []
    for gpu_id in gpu_ids:
        try:
            torch.cuda.device(gpu_id)
            available_gpus.append(gpu_id)
        except Exception:
            continue
    return available_gpus

def parallel_map(
    func: Callable,
    iterable: List[Any],
    n_workers: Optional[int] = None,
    use_threads: bool = False,
    verbose: bool = True
) -> List[Any]:
    """
    简化版并行map函数（类似Python内置map，但并行执行）
    Args:
        func: 映射函数
        iterable: 可迭代对象
        n_workers: 进程数
        use_threads: 是否使用线程
        verbose: 是否显示进度条
    Returns:
        results: 映射结果列表（按输入顺序）
    """
    tasks = [(item,) for item in iterable]
    runner = ParallelRunner(
        n_workers=n_workers,
        use_threads=use_threads,
        verbose=verbose
    )
    results, _ = runner.run_parallel(func, tasks, desc="并行Map")
    return results

# ======================== 快捷函数（简化实验调用） ========================
def create_parallel_runner(
    n_workers: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    use_threads: bool = False
) -> ParallelRunner:
    """
    创建并行运行器（快捷函数）
    """
    return ParallelRunner(
        n_workers=n_workers,
        gpu_ids=gpu_ids,
        use_threads=use_threads
    )

def accelerate_shapley_calculation(
    client_data: Dict[int, Any],
    model: torch.nn.Module,
    calculate_func: Callable,
    auto_gpu: bool = True,
    n_workers: Optional[int] = None
) -> Dict[int, float]:
    """
    一键加速Shapley值计算（自动适配GPU）
    """
    gpu_ids = get_available_gpus() if auto_gpu else None
    if gpu_ids and n_workers is None:
        n_workers = len(gpu_ids)  # GPU数=进程数
    return parallel_shapley_calculate(
        client_data=client_data,
        model=model,
        calculate_func=calculate_func,
        gpu_ids=gpu_ids,
        n_workers=n_workers
    )

def accelerate_client_training(
    client_datasets: Dict[int, Any],
    global_model: torch.nn.Module,
    train_config: Dict[str, Any],
    auto_gpu: bool = True,
    n_workers: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """
    一键加速多客户端训练（自动适配GPU）
    """
    gpu_ids = get_available_gpus() if auto_gpu else None
    if gpu_ids and n_workers is None:
        n_workers = len(gpu_ids)
    return parallel_client_training(
        client_datasets=client_datasets,
        global_model=global_model,
        train_config=train_config,
        gpu_ids=gpu_ids,
        n_workers=n_workers
    )