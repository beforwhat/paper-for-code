# core/dp/adaptive_clipping_dp.py
"""
自适应裁剪差分隐私优化器（AdaptiveClippingDP）
核心改进：
1. 保留核心：「本轮-上轮梯度差值」计算逻辑；
2. 新增精细化差值处理：梯度差值归一化 + 差值分级（低/中/高），不同级别差异化裁剪；
3. 新增自身时序辅助校准：基于历史梯度时序特征（滑动窗口均值/方差）校准裁剪阈值；
4. 新增稳定性约束：限制阈值变化率 + 滑动窗口平滑，避免阈值剧烈波动；
5. 兼容差分隐私核心流程：裁剪（Clipping） + 加噪（Adding Noise），保证DP隐私预算。
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy

# 项目内模块导入
from configs.config_loader import load_config

class AdaptiveClippingDP:
    """
    自适应裁剪差分隐私优化器
    核心方法：
    - adaptive_clip_and_add_noise()：核心入口，裁剪+加噪，返回带DP保护的梯度；
    - _calculate_gradient_diff()：保留核心，计算本轮-上轮梯度差值；
    - _refine_gradient_diff()：新增，精细化处理梯度差值（归一化+分级）；
    - _calibrate_threshold_by_temporal()：新增，自身时序辅助校准裁剪阈值；
    - _constrain_threshold_stability()：新增，稳定性约束裁剪阈值；
    - clear_gradient_history()：清空梯度历史（实验复用）。
    """
    def __init__(self, config=None):
        """
        初始化自适应裁剪DP优化器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.device = self.config.device
        self.epsilon = self.config.dp.epsilon  # DP隐私预算ε
        self.delta = self.config.dp.delta      # DP隐私预算δ（通常设为1e-5）
        self.base_clip_threshold = self.config.dp.base_clip_threshold  # 基础裁剪阈值

        # 2. 梯度历史缓存（支撑差值计算、时序校准、稳定性约束）
        # 结构：{param_name: {"prev_gradient": 上轮梯度, "diff_history": [历史差值列表], "threshold_history": [历史阈值列表]}}
        self.gradient_history = defaultdict(dict)
        # 滑动窗口配置（时序校准/稳定性约束用）
        self.sliding_window_size = self.config.dp.sliding_window_size  # 滑动窗口大小（如5轮）
        self.threshold_change_rate = self.config.dp.threshold_change_rate  # 阈值最大变化率（如0.2=±20%）

        # 3. 精细化差值处理配置
        self.diff_normalize_range = (0.0, 1.0)  # 差值归一化范围
        self.diff_levels = {  # 差值分级阈值（0~1，对应归一化后的值）
            "low": (0.0, 0.3),    # 低差值：小幅波动，宽松裁剪
            "medium": (0.3, 0.7), # 中差值：正常波动，标准裁剪
            "high": (0.7, 1.0)    # 高差值：大幅波动，严格裁剪
        }
        self.level_clip_coeff = {  # 不同级别裁剪系数（乘以基础阈值）
            "low": 1.2,    # 低差值：阈值×1.2（更宽松）
            "medium": 1.0, # 中差值：阈值×1.0（标准）
            "high": 0.8    # 高差值：阈值×0.8（更严格）
        }

        # 4. DP噪声配置
        self.noise_scale = self.base_clip_threshold * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon  # 噪声缩放因子
        self.epsilon_min = 1e-6  # 避免除零的极小值

        print(f"✅ 自适应裁剪DP优化器初始化完成")
        print(f"📌 DP隐私预算：ε={self.epsilon} | δ={self.delta} | 基础裁剪阈值={self.base_clip_threshold}")
        print(f"📌 时序/稳定性配置：滑动窗口={self.sliding_window_size} | 阈值变化率={self.threshold_change_rate}")
        print(f"📌 差值分级：低({self.diff_levels['low']}) | 中({self.diff_levels['medium']}) | 高({self.diff_levels['high']})")

    # ==============================================
    # 核心入口：自适应裁剪+添加DP噪声（客户端调用）
    # ==============================================
    def adaptive_clip_and_add_noise(self, model, current_gradient_dict: dict) -> dict:
        """
        核心入口：对模型梯度做自适应裁剪 + 添加DP噪声，返回带隐私保护的梯度
        Args:
            model: 客户端本地模型实例（用于获取参数名）
            current_gradient_dict: 当前轮次梯度字典 {param_name: gradient_tensor}
        Returns:
            protected_gradient_dict: 带DP保护的梯度字典 {param_name: protected_gradient}
        """
        protected_gradient_dict = {}

        for param_name, current_grad in current_gradient_dict.items():
            # 步骤1：计算本轮-上轮梯度差值（保留核心逻辑）
            grad_diff = self._calculate_gradient_diff(param_name, current_grad)

            # 步骤2：精细化处理梯度差值（归一化+分级）
            normalized_diff, diff_level = self._refine_gradient_diff(param_name, grad_diff)

            # 步骤3：基于时序特征校准裁剪阈值
            calibrated_threshold = self._calibrate_threshold_by_temporal(param_name, diff_level)

            # 步骤4：稳定性约束裁剪阈值（避免剧烈波动）
            stable_threshold = self._constrain_threshold_stability(param_name, calibrated_threshold)

            # 步骤5：自适应裁剪梯度
            clipped_gradient = self._clip_gradient(current_grad, stable_threshold)

            # 步骤6：添加DP噪声
            protected_gradient = self._add_dp_noise(clipped_gradient, stable_threshold)

            # 保存本轮梯度/阈值到历史（供下轮使用）
            self.gradient_history[param_name]["prev_gradient"] = deepcopy(current_grad.cpu())
            if "threshold_history" not in self.gradient_history[param_name]:
                self.gradient_history[param_name]["threshold_history"] = []
            self.gradient_history[param_name]["threshold_history"].append(stable_threshold)
            # 保存差值到历史（供时序校准/稳定性约束使用）
            if "diff_history" not in self.gradient_history[param_name]:
                self.gradient_history[param_name]["diff_history"] = []
            self.gradient_history[param_name]["diff_history"].append(normalized_diff)
            # 截断历史列表（仅保留滑动窗口内的数据）
            self._truncate_sliding_window(param_name)

            protected_gradient_dict[param_name] = protected_gradient.to(self.device)

            # 打印单参数处理结果（前3个参数，辅助调试）
            if len(protected_gradient_dict) <= 3:
                print(f"\n📌 参数 [{param_name}] DP处理结果：")
                print(f"   梯度差值（归一化）：{normalized_diff:.4f} | 差值级别：{diff_level}")
                print(f"   校准后阈值：{calibrated_threshold:.4f} | 稳定后阈值：{stable_threshold:.4f}")
                print(f"   噪声缩放因子：{self.noise_scale:.4f}")

        return protected_gradient_dict

    # ==============================================
    # 保留核心：计算本轮-上轮梯度差值
    # ==============================================
    def _calculate_gradient_diff(self, param_name: str, current_grad: torch.Tensor) -> float:
        """
        保留核心逻辑：计算本轮梯度与上轮梯度的L2差值（标量）
        Args:
            param_name: 参数名（如conv1.weight）
            current_grad: 当前轮次梯度张量
        Returns:
            grad_diff: 梯度差值（标量，L2距离）
        """
        # 首次计算：无上轮梯度，差值设为0
        if "prev_gradient" not in self.gradient_history[param_name] or self.gradient_history[param_name]["prev_gradient"] is None:
            self.gradient_history[param_name]["prev_gradient"] = deepcopy(current_grad.cpu())
            return 0.0

        # 计算L2差值（核心逻辑，完全保留）
        prev_grad = self.gradient_history[param_name]["prev_gradient"].to(self.device)
        grad_diff = torch.norm(current_grad - prev_grad, p=2).item()  # L2范数（差值大小）

        return grad_diff

    # ==============================================
    # 新增：精细化处理梯度差值（归一化+分级）
    # ==============================================
    def _refine_gradient_diff(self, param_name: str, grad_diff: float) -> tuple[float, str]:
        """
        新增：精细化处理梯度差值 → 归一化 + 分级
        Args:
            param_name: 参数名
            grad_diff: 原始梯度差值
        Returns:
            normalized_diff: 归一化后的差值（0~1）
            diff_level: 差值级别（low/medium/high）
        """
        # 1. 差值归一化（基于滑动窗口内的历史差值）
        diff_history = self.gradient_history[param_name].get("diff_history", [])
        if not diff_history:
            # 无历史差值：用基础阈值归一化
            normalized_diff = min(grad_diff / (self.base_clip_threshold + self.epsilon_min), 1.0)
        else:
            # 有历史差值：基于滑动窗口内的最大/最小值归一化
            window_diff = diff_history[-self.sliding_window_size:] if len(diff_history) >= self.sliding_window_size else diff_history
            diff_min = min(window_diff)
            diff_max = max(window_diff)
            if diff_max - diff_min < self.epsilon_min:
                normalized_diff = self.diff_normalize_range[0]
            else:
                norm_min, norm_max = self.diff_normalize_range
                normalized_diff = norm_min + (grad_diff - diff_min) * (norm_max - norm_min) / (diff_max - diff_min)
            # 限制在0~1区间
            normalized_diff = max(self.diff_normalize_range[0], min(self.diff_normalize_range[1], normalized_diff))

        # 2. 差值分级
        diff_level = "medium"  # 默认中级别
        if self.diff_levels["low"][0] <= normalized_diff < self.diff_levels["low"][1]:
            diff_level = "low"
        elif self.diff_levels["high"][0] <= normalized_diff <= self.diff_levels["high"][1]:
            diff_level = "high"

        return normalized_diff, diff_level

    # ==============================================
    # 新增：自身时序辅助校准裁剪阈值
    # ==============================================
    def _calibrate_threshold_by_temporal(self, param_name: str, diff_level: str) -> float:
        """
        新增：基于历史梯度时序特征（滑动窗口均值/方差）校准裁剪阈值
        逻辑：
        - 滑动窗口内差值均值高 → 梯度波动大 → 降低阈值；
        - 滑动窗口内差值方差高 → 梯度不稳定 → 降低阈值；
        - 结合差值级别，应用分级裁剪系数。
        Args:
            param_name: 参数名
            diff_level: 差值级别（low/medium/high）
        Returns:
            calibrated_threshold: 时序校准后的裁剪阈值
        """
        # 1. 基础分级阈值（乘以级别系数）
        level_coeff = self.level_clip_coeff[diff_level]
        level_threshold = self.base_clip_threshold * level_coeff

        # 2. 时序特征校准（基于滑动窗口内的差值统计）
        diff_history = self.gradient_history[param_name].get("diff_history", [])
        if len(diff_history) < 2:
            # 历史数据不足：直接返回分级阈值
            return level_threshold

        # 取滑动窗口内的差值
        window_diff = diff_history[-self.sliding_window_size:] if len(diff_history) >= self.sliding_window_size else diff_history
        diff_mean = np.mean(window_diff)  # 差值均值（波动大小）
        diff_var = np.var(window_diff)    # 差值方差（稳定性）

        # 校准系数：均值/方差越高，系数越小（阈值越低）
        mean_coeff = 1.0 - min(diff_mean * 0.5, 0.3)  # 均值校准系数（最多降30%）
        var_coeff = 1.0 - min(diff_var * 0.5, 0.2)    # 方差校准系数（最多降20%）
        temporal_coeff = mean_coeff * var_coeff

        # 最终校准阈值
        calibrated_threshold = level_threshold * temporal_coeff
        # 确保阈值为正
        calibrated_threshold = max(calibrated_threshold, self.epsilon_min)

        return calibrated_threshold

    # ==============================================
    # 新增：稳定性约束裁剪阈值
    # ==============================================
    def _constrain_threshold_stability(self, param_name: str, calibrated_threshold: float) -> float:
        """
        新增：稳定性约束 → 限制阈值变化率 + 滑动窗口平滑，避免阈值剧烈波动
        Args:
            param_name: 参数名
            calibrated_threshold: 时序校准后的阈值
        Returns:
            stable_threshold: 稳定性约束后的阈值
        """
        threshold_history = self.gradient_history[param_name].get("threshold_history", [])
        if not threshold_history:
            # 无历史阈值：直接返回校准阈值
            return calibrated_threshold

        # 1. 限制阈值变化率（不超过±threshold_change_rate）
        prev_threshold = threshold_history[-1]  # 上轮阈值
        max_increase = prev_threshold * (1 + self.threshold_change_rate)
        max_decrease = prev_threshold * (1 - self.threshold_change_rate)
        constrained_threshold = max(max_decrease, min(max_increase, calibrated_threshold))

        # 2. 滑动窗口平滑（进一步降低波动）
        window_threshold = threshold_history[-self.sliding_window_size:] if len(threshold_history) >= self.sliding_window_size else threshold_history
        window_mean = np.mean(window_threshold)
        stable_threshold = 0.7 * constrained_threshold + 0.3 * window_mean  # 70%当前值 + 30%窗口均值

        # 确保阈值为正
        stable_threshold = max(stable_threshold, self.epsilon_min)

        return stable_threshold

    # ==============================================
    # 辅助方法1：梯度裁剪（L2裁剪）
    # ==============================================
    def _clip_gradient(self, gradient: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        梯度裁剪：L2范数裁剪，确保梯度的L2范数不超过阈值
        """
        grad_norm = torch.norm(gradient, p=2)
        if grad_norm > threshold:
            clipped_gradient = gradient * (threshold / (grad_norm + self.epsilon_min))
        else:
            clipped_gradient = gradient
        return clipped_gradient

    # ==============================================
    # 辅助方法2：添加DP噪声（高斯噪声）
    # ==============================================
    def _add_dp_noise(self, gradient: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        添加DP高斯噪声：噪声尺度与裁剪阈值正相关
        """
        # 动态调整噪声尺度（与当前阈值匹配）
        dynamic_noise_scale = threshold * np.sqrt(2 * np.log(1.25 / self.delta)) / (self.epsilon + self.epsilon_min)
        # 生成高斯噪声（与梯度同形状、同设备）
        noise = torch.normal(0, dynamic_noise_scale, size=gradient.shape, device=self.device)
        # 添加噪声
        noisy_gradient = gradient + noise
        return noisy_gradient

    # ==============================================
    # 辅助方法3：截断滑动窗口（避免历史数据过多）
    # ==============================================
    def _truncate_sliding_window(self, param_name: str) -> None:
        """
        截断梯度差值/阈值的历史列表，仅保留滑动窗口内的数据
        """
        if "diff_history" in self.gradient_history[param_name]:
            self.gradient_history[param_name]["diff_history"] = self.gradient_history[param_name]["diff_history"][-self.sliding_window_size:]
        if "threshold_history" in self.gradient_history[param_name]:
            self.gradient_history[param_name]["threshold_history"] = self.gradient_history[param_name]["threshold_history"][-self.sliding_window_size:]

    # ==============================================
    # 辅助方法：清空梯度历史（实验复用）
    # ==============================================
    def clear_gradient_history(self) -> None:
        """
        清空所有梯度历史缓存（用于多次实验，避免历史数据干扰）
        """
        self.gradient_history = defaultdict(dict)
        print("✅ 梯度历史缓存已清空，DP优化器可重新使用")