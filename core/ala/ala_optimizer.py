# core/ala/ala_optimizer.py
"""
ALA优化器（ALAOptimizer）
核心职责：
1.  保留原有ALA核心逻辑：自适应本地聚合（缓解客户端异质性，优化本地模型初始化/更新）
2.  新增核心功能：extract_ala_features() 提取三大特征（偏差、稳定性、性能），支撑SA贡献度计算
3.  特征归一化处理，保证特征值在[0,1]区间，提升SA贡献度计算的稳定性
4.  独立模块设计，无业务侵入，仅对外暴露接口，兼容客户端/服务端调用
"""
import numpy as np
import torch
from copy import deepcopy

# 项目内模块导入
from configs.config_loader import load_config
from models import get_model

class ALAOptimizer:
    """
    ALA（Adaptive Local Aggregation）优化器
    核心方法：
    - ala_adaptive_update()：原有逻辑，客户端模型自适应更新
    - extract_ala_features()：新增逻辑，提取SA贡献度特征
    """
    def __init__(self, config=None):
        """
        初始化ALA优化器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.ala_alpha = self.config.fed.ala_alpha  # ALA自适应权重（0~1，全局配置）
        self.device = self.config.device

        # 2. 客户端历史参数缓存（持久化，用于ALA更新和特征计算）
        # 结构：{client_id: {"prev_params": 历史参数列表, "prev_metrics": 历史训练指标}}
        self.client_history = {}

        # 3. 特征计算超参数（从配置读取，避免硬编码）
        self.feature_norm_range = (0.0, 1.0)  # 特征归一化范围
        self.stability_window = 3  # 稳定性计算的滑动窗口（取最近N轮训练指标）
        self.bias_norm_type = "cosine"  # 偏差计算方式：cosine（余弦相似度）/l2（L2距离）

        print(f"✅ ALA优化器初始化完成")
        print(f"📌 ALA自适应权重：{self.ala_alpha} | 特征归一化范围：{self.feature_norm_range}")
        print(f"📌 偏差计算方式：{self.bias_norm_type} | 稳定性滑动窗口：{self.stability_window}")

    # ==============================================
    # 原有核心逻辑：ALA自适应本地聚合（客户端模型初始化/更新）
    # ==============================================
    def ala_adaptive_update(self, client_id: int, local_model, global_model_params=None, epoch=None) -> torch.nn.Module:
        """
        原有ALA核心逻辑：自适应更新客户端本地模型参数
        逻辑：w_init = α*w_global + (1-α)*w_prev（缓解客户端异质性，提升训练稳定性）
        Args:
            client_id: 客户端唯一标识
            local_model: 待更新的客户端本地模型实例
            global_model_params: 全局模型参数（None则使用本地模型当前参数作为参考）
            epoch: 当前本地训练轮次（用于更新历史缓存）
        Returns:
            更新后的本地模型实例
        """
        # 1. 初始化客户端历史缓存（首次调用时）
        if client_id not in self.client_history:
            self.client_history[client_id] = {
                "prev_params": None,  # 上一轮本地模型参数
                "prev_metrics": []    # 上几轮训练指标（loss/acc）
            }

        # 2. 获取全局参数（默认使用本地模型当前参数作为参考）
        if global_model_params is None:
            global_model_params = local_model.get_params()

        # 3. 执行ALA自适应更新
        local_params = local_model.get_params()
        prev_params = self.client_history[client_id]["prev_params"]

        if prev_params is None:
            # 首次更新：无历史参数，直接使用全局参数
            ala_updated_params = global_model_params
            print(f"📌 客户端 [{client_id}] 首次ALA更新，使用全局参数初始化")
        else:
            # 非首次更新：α*全局参数 + (1-α)*历史参数
            ala_updated_params = []
            for g_param, p_param in zip(global_model_params, prev_params):
                updated_param = self.ala_alpha * np.array(g_param) + (1 - self.ala_alpha) * np.array(p_param)
                ala_updated_params.append(updated_param)
            print(f"📌 客户端 [{client_id}] ALA更新完成（α={self.ala_alpha}）")

        # 4. 加载更新后的参数到本地模型
        local_model.set_params(ala_updated_params)

        # 5. 更新客户端历史缓存（保存当前参数，用于下一轮更新）
        if epoch is not None:
            self.client_history[client_id]["prev_params"] = deepcopy(local_model.get_params())

        return local_model

    # ==============================================
    # 新增核心逻辑：提取ALA特征（支撑SA贡献度计算）
    # ==============================================
    def extract_ala_features(self, client_id: int, local_model, train_metrics: dict, global_model=None) -> dict:
        """
        新增核心方法：提取客户端ALA三大特征（偏差、稳定性、性能），支撑SA贡献度计算
        Args:
            client_id: 客户端唯一标识
            local_model: 客户端本地模型实例
            train_metrics: 客户端本地训练指标（包含train_loss/train_acc/local_sample_num）
            global_model: 全局模型实例（None则自动初始化全局模型用于偏差计算）
        Returns:
            归一化后的ALA特征字典：
            {
                "bias_feature": 偏差特征（0~1，越小表示与全局偏差越小）,
                "stability_feature": 稳定性特征（0~1，越大表示训练越稳定）,
                "performance_feature": 性能特征（0~1，越大表示本地性能越好）
            }
        """
        # 前置检查：客户端历史缓存初始化
        if client_id not in self.client_history:
            self.client_history[client_id] = {
                "prev_params": deepcopy(local_model.get_params()),
                "prev_metrics": [train_metrics]
            }
        else:
            # 更新历史训练指标（用于稳定性计算）
            self.client_history[client_id]["prev_metrics"].append(train_metrics)

        # 步骤1：计算偏差特征（本地模型 vs 全局模型）
        bias_feature = self._calculate_bias_feature(local_model, global_model)

        # 步骤2：计算稳定性特征（本地训练指标的波动程度）
        stability_feature = self._calculate_stability_feature(client_id, train_metrics)

        # 步骤3：计算性能特征（本地训练的最终准确率/损失）
        performance_feature = self._calculate_performance_feature(train_metrics)

        # 步骤4：特征归一化（统一到[0,1]区间，方便SA贡献度融合）
        normalized_features = self._normalize_features({
            "bias_feature": bias_feature,
            "stability_feature": stability_feature,
            "performance_feature": performance_feature
        })

        # 打印特征结果（辅助调试）
        print(f"\n📊 客户端 [{client_id}] ALA特征提取完成：")
        print(f"   偏差特征（归一化）：{normalized_features['bias_feature']:.4f}")
        print(f"   稳定性特征（归一化）：{normalized_features['stability_feature']:.4f}")
        print(f"   性能特征（归一化）：{normalized_features['performance_feature']:.4f}")

        return normalized_features

    # ==============================================
    # 辅助方法：计算偏差特征（本地 vs 全局）
    # ==============================================
    def _calculate_bias_feature(self, local_model, global_model=None) -> float:
        """
        计算偏差特征：衡量本地模型与全局模型的参数差异
        - cosine方式：余弦相似度（1-相似度，值越小偏差越小）
        - l2方式：L2距离（归一化后，值越小偏差越小）
        """
        # 初始化全局模型（默认使用配置的基础模型）
        if global_model is None:
            global_model = get_model(config=self.config)
            global_model = global_model.to(self.device)

        # 提取本地/全局模型参数（展平为一维数组，便于计算）
        local_params_flat = self._flatten_params(local_model.get_params())
        global_params_flat = self._flatten_params(global_model.get_params())

        # 避免除零错误（参数全零）
        if np.linalg.norm(local_params_flat) == 0 or np.linalg.norm(global_params_flat) == 0:
            return 1.0  # 最大偏差

        # 按指定方式计算偏差
        if self.bias_norm_type == "cosine":
            # 余弦相似度：范围[-1,1] → 转换为偏差[0,1]（1 - (相似度+1)/2）
            cos_sim = np.dot(local_params_flat, global_params_flat) / (np.linalg.norm(local_params_flat) * np.linalg.norm(global_params_flat))
            bias = 1 - ((cos_sim + 1) / 2)  # 转换为0~1，越小偏差越小
        elif self.bias_norm_type == "l2":
            # L2距离：归一化到0~1
            l2_dist = np.linalg.norm(local_params_flat - global_params_flat)
            bias = l2_dist / (np.linalg.norm(local_params_flat) + np.linalg.norm(global_params_flat))  # 归一化
        else:
            raise ValueError(f"不支持的偏差计算方式：{self.bias_norm_type}，可选cosine/l2")

        return bias

    # ==============================================
    # 辅助方法：计算稳定性特征（训练指标波动）
    # ==============================================
    def _calculate_stability_feature(self, client_id: int, train_metrics: dict) -> float:
        """
        计算稳定性特征：衡量本地训练过程中指标的波动程度
        - 计算最近N轮loss/acc的方差，方差越小→稳定性越高→特征值越大
        """
        # 获取客户端历史训练指标
        history_metrics = self.client_history[client_id]["prev_metrics"]
        # 取最近stability_window轮指标（不足则取全部）
        recent_metrics = history_metrics[-self.stability_window:]

        if len(recent_metrics) < 2:
            # 只有1轮指标：无法计算方差，默认稳定性最高（1.0）
            return 1.0

        # 提取最近轮次的准确率（也可结合loss，这里优先用acc）
        recent_accs = [m["train_acc"][-1] for m in recent_metrics]  # 每轮最后一个acc
        # 计算方差（波动程度）
        acc_var = np.var(recent_accs)

        # 转换为稳定性特征：方差越小→稳定性越高→特征值越大（1 / (1 + 方差)）
        stability = 1.0 / (1.0 + acc_var)

        return stability

    # ==============================================
    # 辅助方法：计算性能特征（本地训练效果）
    # ==============================================
    def _calculate_performance_feature(self, train_metrics: dict) -> float:
        """
        计算性能特征：衡量客户端本地训练的最终效果
        - 优先用训练准确率（归一化到0~1），无acc则用loss（反向归一化）
        """
        if not train_metrics["train_acc"]:
            # 无准确率数据：用loss计算（loss越小→性能越好→特征值越大）
            final_loss = train_metrics["train_loss"][-1] if train_metrics["train_loss"] else 10.0
            # loss反向归一化（假设最大loss为10，可根据实际调整）
            performance = 1.0 - (final_loss / 10.0)
            performance = max(0.0, min(1.0, performance))  # 限制在0~1
        else:
            # 有准确率数据：直接取最终acc（已在0~1区间）
            final_acc = train_metrics["train_acc"][-1]
            performance = final_acc

        return performance

    # ==============================================
    # 辅助方法：特征归一化（统一到[0,1]区间）
    # ==============================================
    def _normalize_features(self, features: dict) -> dict:
        """
        特征归一化：将所有特征值映射到[self.feature_norm_range[0], self.feature_norm_range[1]]
        处理异常值（如NaN、Inf），保证特征有效性
        """
        normalized = {}
        min_val, max_val = self.feature_norm_range

        for feat_name, feat_val in features.items():
            # 处理异常值
            if np.isnan(feat_val) or np.isinf(feat_val):
                normalized[feat_name] = min_val  # 异常值默认最小
                continue

            # 归一化（已在0~1区间的特征直接保留，仅处理边界）
            normalized_val = max(min_val, min(max_val, feat_val))
            normalized[feat_name] = normalized_val

        return normalized

    # ==============================================
    # 工具方法：模型参数展平（便于计算相似度/距离）
    # ==============================================
    def _flatten_params(self, params_list: list) -> np.ndarray:
        """
        将模型参数列表（每层参数）展平为一维numpy数组
        Args:
            params_list: 模型参数列表（如model.get_params()返回的列表）
        Returns:
            展平后的一维数组
        """
        flat_params = []
        for param in params_list:
            flat = np.array(param).flatten()
            flat_params.extend(flat)
        return np.array(flat_params, dtype=np.float32)

    # ==============================================
    # 辅助方法：清空客户端历史缓存（便于多次实验）
    # ==============================================
    def clear_client_history(self, client_id=None) -> None:
        """
        清空客户端历史缓存
        Args:
            client_id: 可选，指定客户端ID；None则清空所有客户端
        """
        if client_id is None:
            self.client_history = {}
            print("✅ 所有客户端ALA历史缓存已清空")
        else:
            if client_id in self.client_history:
                del self.client_history[client_id]
                print(f"✅ 客户端 [{client_id}] ALA历史缓存已清空")
            else:
                print(f"⚠️  客户端 [{client_id}] 无ALA历史缓存，无需清空")