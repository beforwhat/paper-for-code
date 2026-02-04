# -*- coding: utf-8 -*-
"""
指标计算模块
核心功能：
1. SA贡献度精准度验证：皮尔逊相关系数、MAE/RMSE（评估贡献度预测准确性）；
2. 公平性指标：基尼系数、综合公平性指数、方差/标准差等；
3. 鲁棒性指标：性能保持率、鲁棒性得分、性能波动系数等；
4. 效率指标：时间/资源/通信效率相关量化；
5. 隐私-效用指标（新增核心）：准确率保留率、DLG攻击成功率（PSNR）、隐私-效用平衡得分；
6. 统一封装MetricsCalculator类，支持一站式指标计算。
"""

import os
import time
import numpy as np
import psutil
import torch
from scipy.stats import pearsonr, variation
from skimage.metrics import peak_signal_noise_ratio  # 计算PSNR（DLG攻击评估）
from typing import Dict, List, Optional, Tuple, Union

# ======================== 核心：SA贡献度精准度验证指标 ========================
def calculate_pearson_corr(
    true_contributions: Union[np.ndarray, List[float]],
    pred_contributions: Union[np.ndarray, List[float]]
) -> float:
    """
    计算皮尔逊相关系数，验证SA贡献度的预测精准度
    （核心指标：越接近1表示预测贡献度与真实贡献度的线性相关性越强，精准度越高）
    Args:
        true_contributions: 真实贡献度数组（如人工标注/理论Shapley值）
        pred_contributions: SA算法预测的贡献度数组
    Returns:
        corr: 皮尔逊相关系数（-1~1），异常时返回0.0
    """
    try:
        # 转换为numpy数组并清理无效值
        true_arr = np.array(true_contributions, dtype=np.float64).flatten()
        pred_arr = np.array(pred_contributions, dtype=np.float64).flatten()
        
        # 过滤NaN/Inf值
        valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
        true_arr = true_arr[valid_mask]
        pred_arr = pred_arr[valid_mask]
        
        if len(true_arr) < 2 or len(pred_arr) < 2:
            return 0.0
        
        corr, _ = pearsonr(true_arr, pred_arr)
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0
# ======================== 基础性能指标（补充缺失） ========================
def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算分类任务的准确率（核心缺失函数，供privacy_utility.py导入）
    Args:
        outputs: 模型输出的logits (batch_size, num_classes)
        labels: 真实标签 (batch_size,)
    Returns:
        accuracy: 准确率（百分比，0-100）
    """
    try:
        with torch.no_grad():
            # 计算预测标签（取概率最大的类别）
            pred_labels = torch.argmax(outputs, dim=1)
            # 统计正确预测数
            correct = (pred_labels == labels).sum().item()
            # 计算准确率（转百分比）
            accuracy = (correct / len(labels)) * 100.0
        return float(accuracy)
    except Exception:
        return 0.0

def calculate_batch_accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    辅助函数：计算模型在整个数据集上的平均准确率（批量评估）
    Args:
        model: 待评估模型
        dataloader: 数据集加载器
        device: 计算设备（cuda/cpu）
    Returns:
        avg_accuracy: 平均准确率（百分比）
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_correct += (torch.argmax(output, dim=1) == target).sum().item()
            total_samples += len(data)
    return (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

# ======================== DLG攻击函数（补充缺失，供privacy_utility.py导入） ========================
def dlg_attack(
    victim_grad: list[torch.Tensor],
    target_model: torch.nn.Module,
    dummy_data: torch.Tensor,
    dummy_label: torch.Tensor,
    criterion,
    optimizer,
    iterations: int = 3000,
    device: torch.device = torch.device("cuda")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DLG梯度泄露攻击：从梯度恢复训练数据（隐私攻击核心函数）
    Args:
        victim_grad: 受害者的梯度列表（模型各层参数的梯度）
        target_model: 目标模型（与受害者模型结构一致）
        dummy_data: 初始化的虚拟数据（待优化）
        dummy_label: 初始化的虚拟标签（待优化）
        criterion: 损失函数（与训练一致）
        optimizer: 优化器（用于优化虚拟数据/标签）
        iterations: 攻击迭代次数
        device: 计算设备
    Returns:
        recovered_data: 恢复的虚拟数据（逼近原始数据）
        recovered_label: 恢复的虚拟标签
    """
    try:
        # 展平受害者梯度（便于计算MSE损失）
        victim_grad_flat = torch.cat([g.detach().flatten() for g in victim_grad])
        
        # 开启虚拟数据/标签的梯度追踪
        dummy_data.requires_grad = True
        dummy_label.requires_grad = True
        
        # 迭代优化虚拟数据，逼近原始数据
        for i in range(iterations):
            # 前向传播：计算虚拟数据的梯度
            target_model.zero_grad()
            dummy_output = target_model(dummy_data)
            dummy_loss = criterion(dummy_output, dummy_label)
            dummy_grad = torch.autograd.grad(dummy_loss, target_model.parameters(), create_graph=True)
            dummy_grad_flat = torch.cat([g.flatten() for g in dummy_grad])
            
            # 计算梯度损失（MSE）并反向传播
            grad_loss = torch.nn.functional.mse_loss(dummy_grad_flat, victim_grad_flat)
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()
            
            # 每500轮打印进度（可选）
            if (i + 1) % 500 == 0:
                print(f"DLG攻击迭代 {i+1}/{iterations} | 梯度损失：{grad_loss.item():.6f}")
        
        # 停止梯度追踪，返回结果
        return dummy_data.detach(), dummy_label.detach()
    except Exception as e:
        print(f"DLG攻击执行失败：{e}")
        return dummy_data, dummy_label
def calculate_contribution_error_metrics(
    true_contributions: Union[np.ndarray, List[float]],
    pred_contributions: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    计算SA贡献度预测的误差指标（辅助验证精准度）
    Args:
        true_contributions: 真实贡献度数组
        pred_contributions: 预测贡献度数组
    Returns:
        error_metrics: 误差指标字典（MAE/RMSE/MAPE）
    """
    try:
        true_arr = np.array(true_contributions, dtype=np.float64).flatten()
        pred_arr = np.array(pred_contributions, dtype=np.float64).flatten()
        
        # 过滤无效值
        valid_mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
        true_arr = true_arr[valid_mask]
        pred_arr = pred_arr[valid_mask]
        
        if len(true_arr) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        
        # 平均绝对误差（MAE）
        mae = np.mean(np.abs(true_arr - pred_arr))
        # 均方根误差（RMSE）
        rmse = np.sqrt(np.mean((true_arr - pred_arr) ** 2))
        # 平均绝对百分比误差（MAPE）（避免除0）
        mape = np.mean(np.abs((true_arr - pred_arr) / (true_arr + 1e-8))) * 100
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    except Exception:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

# ======================== 公平性指标（复用公平性验证实验） ========================
def calculate_gini_coefficient(values: Union[np.ndarray, List[float]]) -> float:
    """
    计算基尼系数（衡量客户端性能分布公平性，0~1，0=完全公平，1=完全不公平）
    Args:
        values: 客户端性能指标列表（如准确率、损失）
    Returns:
        gini: 基尼系数，异常时返回0.0
    """
    try:
        values = np.array(values, dtype=np.float64).flatten()
        if len(values) == 0 or np.all(values == values[0]):
            return 0.0
        
        values_sorted = np.sort(values)
        n = len(values_sorted)
        cumsum = np.cumsum(values_sorted)
        
        # 基尼系数核心公式
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return float(np.clip(gini, 0.0, 1.0))  # 限制在0~1范围内
    except Exception:
        return 0.0

def calculate_fairness_metrics(
    client_performances: Union[Dict[int, float], List[float]]
) -> Dict[str, float]:
    """
    计算多维度公平性指标（整合基尼系数、方差、极差等）
    Args:
        client_performances: 客户端性能字典 {client_id: perf} 或列表
    Returns:
        fairness_metrics: 公平性指标字典
    """
    try:
        # 统一转换为数组
        if isinstance(client_performances, dict):
            performances = np.array(list(client_performances.values()), dtype=np.float64)
        else:
            performances = np.array(client_performances, dtype=np.float64)
        
        performances = performances[np.isfinite(performances)]  # 过滤无效值
        if len(performances) == 0:
            return {
                "mean": 0.0, "std": 0.0, "var": 0.0, "cv": 0.0,
                "min": 0.0, "max": 0.0, "range": 0.0, "gini": 0.0,
                "fairness_index": 0.0
            }
        
        # 基础统计量
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        var_perf = np.var(performances)
        cv_perf = variation(performances) if mean_perf != 0 else 0.0  # 变异系数
        min_perf = np.min(performances)
        max_perf = np.max(performances)
        range_perf = max_perf - min_perf
        
        # 核心公平性指标
        gini = calculate_gini_coefficient(performances)
        
        # 综合公平性指数（0~1，越高越公平）
        fairness_index = (1 - gini) * (1 - cv_perf) * (min_perf / (mean_perf + 1e-8))
        fairness_index = np.clip(fairness_index, 0.0, 1.0)
        
        return {
            "mean": float(mean_perf),
            "std": float(std_perf),
            "var": float(var_perf),
            "cv": float(cv_perf),          # 变异系数（相对离散程度）
            "min": float(min_perf),
            "max": float(max_perf),
            "range": float(range_perf),    # 性能极差
            "gini": float(gini),           # 核心公平性指标
            "fairness_index": float(fairness_index)  # 综合公平性指数
        }
    except Exception as e:
        return {
            "mean": 0.0, "std": 0.0, "var": 0.0, "cv": 0.0,
            "min": 0.0, "max": 0.0, "range": 0.0, "gini": 0.0,
            "fairness_index": 0.0
        }

# ======================== 鲁棒性指标（复用效率鲁棒性验证实验） ========================
def calculate_robustness_metrics(
    baseline_perf: float,
    perturbed_perfs: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """
    计算鲁棒性指标（衡量算法在扰动场景下的稳定性）
    Args:
        baseline_perf: 基准场景下的性能（如准确率）
        perturbed_perfs: 扰动场景（规模/噪声/故障/异构）下的性能列表
    Returns:
        robustness_metrics: 鲁棒性指标字典
    """
    try:
        perturbed_arr = np.array(perturbed_perfs, dtype=np.float64).flatten()
        perturbed_arr = perturbed_arr[np.isfinite(perturbed_arr)]
        
        if baseline_perf == 0 or len(perturbed_arr) == 0:
            return {
                "baseline_perf": 0.0, "avg_retention_rate_pct": 0.0,
                "perf_std": 0.0, "perf_cv": 0.0, "robustness_score": 0.0
            }
        
        # 性能保持率（%）：扰动后性能 / 基准性能 * 100
        retention_rates = (perturbed_arr / baseline_perf) * 100
        avg_retention_rate = np.mean(retention_rates)
        
        # 性能波动指标
        perf_std = np.std(perturbed_arr)
        perf_mean = np.mean(perturbed_arr)
        perf_cv = perf_std / perf_mean if perf_mean != 0 else 0.0
        
        # 鲁棒性得分（0~1，越高越鲁棒）：综合保持率和波动
        robustness_score = (avg_retention_rate / 100) * (1 - perf_cv)
        robustness_score = np.clip(robustness_score, 0.0, 1.0)
        
        return {
            "baseline_perf": float(baseline_perf),
            "avg_retention_rate_pct": float(avg_retention_rate),
            "perf_std": float(perf_std),
            "perf_cv": float(perf_cv),          # 性能变异系数
            "robustness_score": float(robustness_score)  # 综合鲁棒性得分
        }
    except Exception:
        return {
            "baseline_perf": 0.0, "avg_retention_rate_pct": 0.0,
            "perf_std": 0.0, "perf_cv": 0.0, "robustness_score": 0.0
        }

# ======================== 效率指标（复用效率鲁棒性验证实验） ========================
def calculate_efficiency_metrics(
    start_time: float,
    end_time: float,
    client_params_sizes: Optional[List[List[int]]] = None,
    process: Optional[psutil.Process] = None
) -> Dict[str, float]:
    """
    计算效率指标（时间/资源/通信）
    Args:
        start_time: 训练开始时间戳
        end_time: 训练结束时间戳
        client_params_sizes: 每轮各客户端传输参数大小（字节），格式[[round1_c1, round1_c2], [round2_c1, ...]]
        process: psutil.Process对象（用于监控资源占用）
    Returns:
        efficiency_metrics: 效率指标字典
    """
    try:
        # 时间效率
        total_time = end_time - start_time
        num_rounds = len(client_params_sizes) if client_params_sizes else 0
        avg_round_time = total_time / num_rounds if num_rounds > 0 else 0.0
        
        # 资源效率（内存/CPU/GPU）
        memory_usage = 0.0
        cpu_usage = 0.0
        gpu_memory = 0.0
        
        if process is not None:
            memory_usage = process.memory_info().rss / (1024 * 1024)  # 转换为MB
            cpu_usage = process.cpu_percent()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # GPU显存（MB）
        
        # 通信效率
        total_comm_bytes = 0.0
        if client_params_sizes:
            total_comm_bytes = sum([sum(sizes) for sizes in client_params_sizes if sizes])
        total_comm_mb = total_comm_bytes / (1024 * 1024)  # 转换为MB
        avg_round_comm_mb = total_comm_mb / num_rounds if num_rounds > 0 else 0.0
        
        return {
            # 时间效率
            "total_time": float(total_time),
            "avg_round_time": float(avg_round_time),
            # 资源效率
            "memory_usage_mb": float(memory_usage),
            "cpu_usage_pct": float(cpu_usage),
            "gpu_memory_mb": float(gpu_memory),
            # 通信效率
            "total_comm_mb": float(total_comm_mb),
            "avg_round_comm_mb": float(avg_round_comm_mb)
        }
    except Exception:
        return {
            "total_time": 0.0, "avg_round_time": 0.0,
            "memory_usage_mb": 0.0, "cpu_usage_pct": 0.0, "gpu_memory_mb": 0.0,
            "total_comm_mb": 0.0, "avg_round_comm_mb": 0.0
        }

# ======================== 新增核心：隐私-效用指标 ========================
def calculate_accuracy_retention_rate(
    low_epsilon_accuracy: float,
    high_epsilon_accuracy: float
) -> float:
    """
    计算准确率保留率（核心隐私-效用平衡指标）
    公式：低ε（强隐私）准确率 / 高ε（弱隐私）准确率
    越高说明隐私约束下的效用损失越小，隐私-效用平衡越好
    Args:
        low_epsilon_accuracy: 低ε（如0.1）下的全局准确率
        high_epsilon_accuracy: 高ε（如20）下的全局准确率
    Returns:
        retention_rate: 准确率保留率（0~1），异常时返回0.0
    """
    try:
        if high_epsilon_accuracy <= 0 or low_epsilon_accuracy < 0:
            return 0.0
        retention_rate = low_epsilon_accuracy / high_epsilon_accuracy
        return float(np.clip(retention_rate, 0.0, 1.0))
    except Exception:
        return 0.0

def calculate_dlg_attack_success_rate(
    original_images: Union[np.ndarray, List[np.ndarray]],
    reconstructed_images: Union[np.ndarray, List[np.ndarray]],
    data_range: float = 255.0  # 图像像素值范围（0~255）
) -> float:
    """
    计算DLG梯度泄露攻击成功率（基于PSNR值转换）
    PSNR越高 → 重建图像越接近原始图像 → 攻击成功率越高
    转换规则：
    - PSNR ≥ 30dB → 攻击成功率 100%
    - PSNR ≤ 10dB → 攻击成功率 0%
    - 10~30dB → 线性映射为0%~100%
    Args:
        original_images: 原始图像数组（单张/批量），shape: [H,W,C] 或 [N,H,W,C]
        reconstructed_images: DLG攻击重建的图像数组
        data_range: 图像像素值范围（默认255）
    Returns:
        attack_success_rate: 攻击成功率（0~100，%），异常时返回100.0（最差情况）
    """
    try:
        # 统一转换为数组并调整维度
        orig_arr = np.array(original_images, dtype=np.float64)
        recon_arr = np.array(reconstructed_images, dtype=np.float64)
        
        # 处理批量图像（N,H,W,C）→ 展平为单张计算平均PSNR
        if len(orig_arr.shape) == 4:
            psnr_list = []
            for o, r in zip(orig_arr, recon_arr):
                if o.shape == r.shape and np.prod(o.shape) > 0:
                    psnr = peak_signal_noise_ratio(o, r, data_range=data_range)
                    psnr_list.append(psnr)
            if not psnr_list:
                return 100.0
            avg_psnr = np.mean(psnr_list)
        else:
            # 单张图像
            if orig_arr.shape != recon_arr.shape or np.prod(orig_arr.shape) == 0:
                return 100.0
            avg_psnr = peak_signal_noise_ratio(orig_arr, recon_arr, data_range=data_range)
        
        # PSNR转换为攻击成功率（线性映射）
        if avg_psnr >= 30.0:
            success_rate = 100.0
        elif avg_psnr <= 10.0:
            success_rate = 0.0
        else:
            success_rate = (avg_psnr - 10.0) / (30.0 - 10.0) * 100.0
        
        return float(np.clip(success_rate, 0.0, 100.0))
    except Exception:
        return 100.0  # 异常时默认攻击成功（最差隐私性）

def calculate_privacy_utility_balance_score(
    accuracy_retention_rate: float,
    dlg_attack_rate: float,
    privacy_weight: float = 0.5,  # 隐私权重（0~1）
    utility_weight: float = 0.5   # 效用权重（0~1）
) -> float:
    """
    计算综合隐私-效用平衡得分（0~1，越高越好）
    公式：(1 - 攻击率/100) * 隐私权重 + 准确率保留率 * 效用权重
    Args:
        accuracy_retention_rate: 准确率保留率（0~1）
        dlg_attack_rate: DLG攻击成功率（0~100，%）
        privacy_weight: 隐私维度权重
        utility_weight: 效用维度权重
    Returns:
        balance_score: 综合平衡得分（0~1）
    """
    try:
        # 归一化攻击率（0~1，越低越好）
        normalized_attack_rate = dlg_attack_rate / 100.0
        privacy_score = 1.0 - normalized_attack_rate
        
        # 加权求和
        total_weight = privacy_weight + utility_weight
        balance_score = (privacy_score * privacy_weight + accuracy_retention_rate * utility_weight) / total_weight
        
        return float(np.clip(balance_score, 0.0, 1.0))
    except Exception:
        return 0.0

def calculate_privacy_utility_metrics(
    accuracy_dict: Dict[float, float],  # {ε值: 对应准确率}
    original_images: Union[np.ndarray, List[np.ndarray]],
    reconstructed_images: Union[np.ndarray, List[np.ndarray]],
    low_epsilon: float = 0.1,
    high_epsilon: float = 20.0,
    delta: float = 1e-5
) -> Dict[str, float]:
    """
    一站式计算隐私-效用所有核心指标
    Args:
        accuracy_dict: 不同ε下的准确率字典，如{0.1:78.5, 20:91.5}
        original_images: 原始图像（DLG攻击评估）
        reconstructed_images: 重建图像（DLG攻击评估）
        low_epsilon: 低ε值（强隐私约束）
        high_epsilon: 高ε值（弱隐私约束）
        delta: 固定隐私参数δ
    Returns:
        privacy_utility_metrics: 隐私-效用指标字典
    """
    try:
        # 1. 基础准确率
        low_eps_acc = accuracy_dict.get(low_epsilon, 0.0)
        high_eps_acc = accuracy_dict.get(high_epsilon, 0.0)
        
        # 2. 核心指标
        retention_rate = calculate_accuracy_retention_rate(low_eps_acc, high_eps_acc)
        dlg_attack_rate = calculate_dlg_attack_success_rate(original_images, reconstructed_images)
        balance_score = calculate_privacy_utility_balance_score(retention_rate, dlg_attack_rate)
        
        return {
            # 隐私参数
            "delta": float(delta),
            "low_epsilon": float(low_epsilon),
            "high_epsilon": float(high_epsilon),
            # 基础准确率
            "low_epsilon_accuracy": float(low_eps_acc),
            "high_epsilon_accuracy": float(high_eps_acc),
            # 核心平衡指标
            "accuracy_retention_rate": float(retention_rate),
            "dlg_attack_success_rate_pct": float(dlg_attack_rate),
            "privacy_utility_balance_score": float(balance_score)
        }
    except Exception as e:
        return {
            "delta": float(delta),
            "low_epsilon": float(low_epsilon),
            "high_epsilon": float(high_epsilon),
            "low_epsilon_accuracy": 0.0,
            "high_epsilon_accuracy": 0.0,
            "accuracy_retention_rate": 0.0,
            "dlg_attack_success_rate_pct": 100.0,
            "privacy_utility_balance_score": 0.0
        }
def calculate_gradient_entropy(grads: List[np.ndarray]) -> float:
    """
    计算梯度分布的熵（衡量Non-IID程度：熵越小，Non-IID越严重）
    Args:
        grads: 梯度列表，格式为[[param1_grad, param2_grad,...], ...]（多批次/样本的梯度）
    Returns:
        entropy: 梯度分布的平均熵（0~+∞），异常时返回1.0（默认中等Non-IID）
    """
    try:
        # 展平所有梯度为一维数组（合并所有参数的梯度）
        flat_grads = []
        for batch_grads in grads:
            for param_grad in batch_grads:
                flat_grad = param_grad.flatten()
                flat_grads.extend(flat_grad)
        
        if len(flat_grads) == 0:
            return 1.0
        
        # 归一化梯度值到[0,1]区间
        grad_arr = np.array(flat_grads, dtype=np.float64)
        grad_arr = (grad_arr - grad_arr.min()) / (grad_arr.max() - grad_arr.min() + 1e-8)
        
        # 计算概率分布（直方图统计）
        hist, _ = np.histogram(grad_arr, bins=50, density=True)
        hist = hist[hist > 0]  # 过滤0概率区间（避免log(0)）
        
        # 计算香农熵（H = -Σp(x)log2(p(x))）
        entropy = -np.sum(hist * np.log2(hist)) / len(hist)
        return float(np.clip(entropy, 0.0, 10.0))  # 限制熵值范围
    except Exception:
        return 1.0  
# ======================== 统一指标计算类（新增隐私-效用支持） ========================
class MetricsCalculator:
    """
    一站式指标计算类，整合所有实验所需指标，简化调用流程
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.results = {}  # 存储计算后的所有指标
    
    # SA贡献度精准度
    def calculate_sa_contribution_metrics(
        self,
        true_contributions: Union[np.ndarray, List[float]],
        pred_contributions: Union[np.ndarray, List[float]],
        save_key: str = "sa_contribution"
    ) -> Dict[str, float]:
        """计算SA贡献度精准度指标并存储"""
        pearson_corr = calculate_pearson_corr(true_contributions, pred_contributions)
        error_metrics = calculate_contribution_error_metrics(true_contributions, pred_contributions)
        
        metrics = {
            "pearson_corr": pearson_corr,
            **error_metrics
        }
        self.results[save_key] = metrics
        return metrics
    
    # 公平性指标
    def calculate_fairness(
        self,
        client_performances: Union[Dict[int, float], List[float]],
        save_key: str = "fairness"
    ) -> Dict[str, float]:
        """计算公平性指标并存储"""
        metrics = calculate_fairness_metrics(client_performances)
        self.results[save_key] = metrics
        return metrics
    
    # 鲁棒性指标
    def calculate_robustness(
        self,
        baseline_perf: float,
        perturbed_perfs: Union[np.ndarray, List[float]],
        save_key: str = "robustness"
    ) -> Dict[str, float]:
        """计算鲁棒性指标并存储"""
        metrics = calculate_robustness_metrics(baseline_perf, perturbed_perfs)
        self.results[save_key] = metrics
        return metrics
    
    # 效率指标
    def calculate_efficiency(
        self,
        start_time: float,
        end_time: float,
        client_params_sizes: Optional[List[List[int]]] = None,
        save_key: str = "efficiency"
    ) -> Dict[str, float]:
        """计算效率指标并存储"""
        metrics = calculate_efficiency_metrics(start_time, end_time, client_params_sizes, self.process)
        self.results[save_key] = metrics
        return metrics
    
    # 新增：隐私-效用指标
    def calculate_privacy_utility(
        self,
        accuracy_dict: Dict[float, float],
        original_images: Union[np.ndarray, List[np.ndarray]],
        reconstructed_images: Union[np.ndarray, List[np.ndarray]],
        low_epsilon: float = 0.1,
        high_epsilon: float = 20.0,
        delta: float = 1e-5,
        save_key: str = "privacy_utility"
    ) -> Dict[str, float]:
        """计算隐私-效用指标并存储"""
        metrics = calculate_privacy_utility_metrics(
            accuracy_dict=accuracy_dict,
            original_images=original_images,
            reconstructed_images=reconstructed_images,
            low_epsilon=low_epsilon,
            high_epsilon=high_epsilon,
            delta=delta
        )
        self.results[save_key] = metrics
        return metrics
    
    # 批量计算所有指标（适用于完整实验）
    def calculate_all(
        self,
        sa_true: Optional[Union[np.ndarray, List[float]]] = None,
        sa_pred: Optional[Union[np.ndarray, List[float]]] = None,
        client_perfs: Optional[Union[Dict[int, float], List[float]]] = None,
        robust_baseline: Optional[float] = None,
        robust_perturbed: Optional[Union[np.ndarray, List[float]]] = None,
        eff_start: Optional[float] = None,
        eff_end: Optional[float] = None,
        eff_params: Optional[List[List[int]]] = None,
        privacy_accuracy_dict: Optional[Dict[float, float]] = None,
        privacy_original_imgs: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        privacy_recon_imgs: Optional[Union[np.ndarray, List[np.ndarray]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """批量计算所有需要的指标（含隐私-效用）"""
        # SA贡献度
        if sa_true is not None and sa_pred is not None:
            self.calculate_sa_contribution_metrics(sa_true, sa_pred)
        # 公平性
        if client_perfs is not None:
            self.calculate_fairness(client_perfs)
        # 鲁棒性
        if robust_baseline is not None and robust_perturbed is not None:
            self.calculate_robustness(robust_baseline, robust_perturbed)
        # 效率
        if eff_start is not None and eff_end is not None:
            self.calculate_efficiency(eff_start, eff_end, eff_params)
        # 隐私-效用
        if privacy_accuracy_dict is not None and privacy_original_imgs is not None and privacy_recon_imgs is not None:
            self.calculate_privacy_utility(privacy_accuracy_dict, privacy_original_imgs, privacy_recon_imgs)
        
        return self.results
    
    def get_all_results(self) -> Dict[str, Dict[str, float]]:
        """获取所有已计算的指标"""
        return self.results
    
    def reset(self):
        """重置指标存储"""
        self.results = {}

# ======================== 测试示例（可直接运行） ========================
if __name__ == "__main__":
    # 1. 初始化计算器
    calculator = MetricsCalculator()
    
    # 2. 测试隐私-效用指标计算
    # 模拟不同ε下的准确率
    accuracy_dict = {0.1: 78.5, 1.0: 85.2, 20.0: 91.5}
    # 模拟原始图像和重建图像（随机生成）
    original_imgs = np.random.randint(0, 255, size=(10, 32, 32, 3), dtype=np.uint8)
    reconstructed_imgs = original_imgs * 0.9 + np.random.randn(*original_imgs.shape) * 5  # 带噪声的重建
    
    # 计算隐私-效用指标
    privacy_metrics = calculator.calculate_privacy_utility(
        accuracy_dict=accuracy_dict,
        original_images=original_imgs,
        reconstructed_images=reconstructed_imgs,
        low_epsilon=0.1,
        high_epsilon=20.0
    )
    
    print("=== 隐私-效用指标计算结果 ===")
    for key, value in privacy_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 3. 测试批量计算所有指标
    calculator.reset()
    all_metrics = calculator.calculate_all(
        # SA贡献度
        sa_true=[0.8, 0.5, 0.3, 0.9],
        sa_pred=[0.75, 0.52, 0.28, 0.88],
        # 公平性
        client_perfs=[85.2, 78.9, 90.1, 82.5, 88.7],
        # 鲁棒性
        robust_baseline=90.0,
        robust_perturbed=[88.5, 87.2, 89.1, 86.8],
        # 效率
        eff_start=time.time() - 3600,
        eff_end=time.time(),
        eff_params=[[1024, 2048], [1024, 2048]],
        # 隐私-效用
        privacy_accuracy_dict=accuracy_dict,
        privacy_original_imgs=original_imgs,
        privacy_recon_imgs=reconstructed_imgs
    )
    
    print("\n=== 全量指标计算结果 ===")
    for metric_type, metrics in all_metrics.items():
        print(f"\n【{metric_type}】")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")