# core/shapley/shapley_calculator.py
"""
Shapley贡献度计算器（ShapleyCalculator）
核心职责：
1.  计算SA融合贡献度得分（Shapley+ALA）：
   - 先验SA得分：首轮无历史数据时，为公平选择提供基础得分；
   - 实际SA贡献度：有客户端上传数据时，融合ALA特征+样本数+Shapley原始值，支撑SA聚合。
2.  权重融合+归一化处理，保证得分稳定性；
3.  独立模块设计，兼容BaseServer/FairClientSelector调用，配置化调参。
"""
import numpy as np
from collections import defaultdict

# 项目内模块导入
from configs.config_loader import load_config

class ShapleyCalculator:
    """
    Shapley贡献度计算器
    核心方法：
    - calculate_prior_sa_scores()：计算先验SA得分（首轮/无历史数据）；
    - calculate_sa_contribution()：计算实际SA融合贡献度（支撑SA聚合）；
    - clear_contribution_history()：清空历史得分（实验复用）。
    """
    def __init__(self, config=None):
        """
        初始化Shapley计算器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.device = self.config.device

        # 2. SA融合权重（从配置读取，总和建议为1）
        self.ala_feature_weight = self.config.fed.ala_feature_weight  # ALA特征权重（如0.5）
        self.sample_num_weight = self.config.fed.sample_num_weight    # 样本数权重（如0.3）
        self.shapley_raw_weight = self.config.fed.shapley_raw_weight  # Shapley原始值权重（如0.2）

        # 3. Shapley原始值计算超参数
        self.shapley_epsilon = 1e-6  # 避免除零的极小值
        self.shapley_norm_range = (0.0, 1.0)  # Shapley原始值归一化范围

        # 4. 历史贡献度缓存（支撑多轮计算）
        # 结构：{client_id: {"sa_contribution_scores": [各轮得分], "total_contribution": 累计得分}}
        self.contribution_history = defaultdict(lambda: {"sa_contribution_scores": [], "total_contribution": 0.0})

        # 校验融合权重（总和建议为1，提示非强制）
        weight_sum = self.ala_feature_weight + self.sample_num_weight + self.shapley_raw_weight
        if not np.isclose(weight_sum, 1.0, atol=1e-2):
            print(f"⚠️  SA融合权重总和为 {weight_sum:.2f}（建议为1.0），已自动归一化处理")

        print(f"✅ Shapley贡献度计算器初始化完成")
        print(f"📌 SA融合权重：ALA特征={self.ala_feature_weight} | 样本数={self.sample_num_weight} | Shapley原始值={self.shapley_raw_weight}")
        print(f"📌 Shapley计算超参数：epsilon={self.shapley_epsilon} | 归一化范围={self.shapley_norm_range}")

    # ==============================================
    # 核心方法1：计算先验SA得分（首轮/无历史数据时）
    # ==============================================
    def calculate_prior_sa_scores(self, total_clients: int, round_idx: int, historical_client_data: dict) -> dict:
        """
        计算先验SA得分（首轮训练/无客户端上传数据时，为公平选择提供基础）
        逻辑：
        - 首轮（round_idx=1）：均匀得分（所有客户端得分相同）；
        - 非首轮但无数据：结合历史参与记录，给参与少的客户端略高得分（兼顾公平）。
        Args:
            total_clients: 客户端总数
            round_idx: 当前全局轮次
            historical_client_data: 服务端接收的历史客户端上传数据 {client_id: upload_data}
        Returns:
            prior_sa_scores: 先验SA得分字典 {client_id: prior_score}
        """
        prior_sa_scores = {}

        # 情况1：首轮训练（无任何历史数据）→ 均匀得分
        if round_idx == 1 or not historical_client_data:
            uniform_score = 1.0 / total_clients
            prior_sa_scores = {cid: uniform_score for cid in range(total_clients)}
            print(f"📌 首轮训练，先验SA得分均匀分配（每个客户端={uniform_score:.6f}）")
        # 情况2：非首轮，有历史数据但本轮无新上传 → 结合历史贡献度调整
        else:
            # 提取有历史贡献度的客户端
            has_history_cids = list(self.contribution_history.keys())
            # 初始化所有客户端得分为基础值
            base_score = 1.0 / total_clients
            prior_sa_scores = {cid: base_score for cid in range(total_clients)}
            # 对有历史贡献的客户端，按累计贡献度微调（贡献度高则略高）
            total_hist_contribution = sum([self.contribution_history[cid]["total_contribution"] for cid in has_history_cids]) + self.shapley_epsilon
            for cid in has_history_cids:
                hist_contribution = self.contribution_history[cid]["total_contribution"]
                # 微调得分：基础分 + 贡献度占比 * 基础分
                adjusted_score = base_score + (hist_contribution / total_hist_contribution) * base_score
                prior_sa_scores[cid] = adjusted_score
            # 归一化得分（确保总和为1）
            prior_sa_scores = self._normalize_scores(prior_sa_scores)
            print(f"📌 非首轮先验SA得分计算完成（基于历史贡献度微调）")

        return prior_sa_scores

    # ==============================================
    # 核心方法2：计算实际SA融合贡献度（支撑SA聚合）
    # ==============================================
    def calculate_sa_contribution(self, client_ids: list, client_features_list: list, local_sample_nums: list, global_model=None) -> list:
        """
        核心：计算客户端SA融合贡献度原始得分（支撑服务端SA加权聚合）
        公式：SA_raw(i) = α·ALA_feature + β·norm_sample_num + γ·shapley_raw
        Args:
            client_ids: 参与聚合的客户端ID列表
            client_features_list: 客户端ALA特征列表（每个元素是{"bias_feature":..., "stability_feature":..., "performance_feature":...}）
            local_sample_nums: 客户端本地样本数列表
            global_model: 全局模型实例（用于Shapley原始值计算，可选）
        Returns:
            sa_raw_scores: 客户端SA融合贡献度原始得分列表（与client_ids一一对应）
        """
        # 前置检查
        if len(client_ids) != len(client_features_list) or len(client_ids) != len(local_sample_nums):
            raise ValueError("客户端ID、ALA特征、样本数列表长度不一致")
        if not client_ids:
            return []

        # 步骤1：融合ALA三大特征为单一ALA得分（加权平均）
        ala_scores = []
        for features in client_features_list:
            # ALA特征权重：性能(0.5) > 稳定性(0.3) > 偏差(0.2)（可配置）
            ala_feature = 0.5 * features["performance_feature"] + 0.3 * features["stability_feature"] + 0.2 * (1 - features["bias_feature"])
            ala_scores.append(ala_feature)
        # 归一化ALA得分（0~1）
        ala_scores = self._normalize_list(ala_scores)

        # 步骤2：归一化本地样本数（0~1）
        norm_sample_nums = self._normalize_list(local_sample_nums)

        # 步骤3：计算Shapley原始值（简化版，基于模型性能差异）
        shapley_raw = self._calculate_shapley_raw(
            client_ids=client_ids,
            client_features_list=client_features_list,
            global_model=global_model
        )
        # 归一化Shapley原始值（0~1）
        shapley_raw = self._normalize_list(shapley_raw)

        # 步骤4：融合三大维度，计算SA原始得分
        sa_raw_scores = []
        fusion_weights = [self.ala_feature_weight, self.sample_num_weight, self.shapley_raw_weight]
        # 归一化融合权重（确保总和为1）
        fusion_weights = self._normalize_list(fusion_weights)
        α, β, γ = fusion_weights

        for i in range(len(client_ids)):
            sa_raw = α * ala_scores[i] + β * norm_sample_nums[i] + γ * shapley_raw[i]
            sa_raw_scores.append(sa_raw)
            # 打印单客户端融合过程（辅助调试）
            if i < 5:  # 仅打印前5个
                print(f"\n📌 客户端 [{client_ids[i]}] SA融合过程：")
                print(f"   ALA特征得分（归一化）：{ala_scores[i]:.4f} (权重α={α:.2f})")
                print(f"   样本数得分（归一化）：{norm_sample_nums[i]:.4f} (权重β={β:.2f})")
                print(f"   Shapley原始值（归一化）：{shapley_raw[i]:.4f} (权重γ={γ:.2f})")
                print(f"   SA原始得分：{sa_raw:.4f}")

        # 步骤5：更新客户端贡献度历史
        for cid, score in zip(client_ids, sa_raw_scores):
            self.contribution_history[cid]["sa_contribution_scores"].append(score)
            self.contribution_history[cid]["total_contribution"] += score

        return sa_raw_scores

    # ==============================================
    # 辅助方法1：计算Shapley原始值（简化版，适配联邦场景）
    # ==============================================
    def _calculate_shapley_raw(self, client_ids: list, client_features_list: list, global_model=None) -> list:
        """
        计算Shapley原始值（简化版，基于客户端本地性能与全局的差异）
        逻辑：Shapley_raw = 本地性能 / (全局基准性能 + ε) → 性能越好，原始值越高
        """
        # 若无全局模型，使用客户端性能特征作为近似
        if global_model is None:
            shapley_raw = [feat["performance_feature"] for feat in client_features_list]
            print(f"⚠️  无全局模型，使用客户端性能特征近似Shapley原始值")
        else:
            # （可选扩展）基于全局模型与本地模型的性能差异计算更精准的Shapley值
            # 此处为简化版，仍使用性能特征（实际可替换为精确Shapley计算逻辑）
            shapley_raw = [feat["performance_feature"] for feat in client_features_list]

        # 处理异常值（确保非负）
        shapley_raw = [max(score, self.shapley_epsilon) for score in shapley_raw]
        return shapley_raw

    # ==============================================
    # 辅助方法2：得分归一化（字典形式）
    # ==============================================
    def _normalize_scores(self, scores_dict: dict) -> dict:
        """
        归一化得分字典（确保所有得分总和为1）
        """
        total_score = sum(scores_dict.values()) + self.shapley_epsilon
        normalized = {cid: score / total_score for cid, score in scores_dict.items()}
        return normalized

    # ==============================================
    # 辅助方法3：列表归一化（0~1区间）
    # ==============================================
    def _normalize_list(self, values: list) -> list:
        """
        将列表值归一化到[self.shapley_norm_range[0], self.shapley_norm_range[1]]
        """
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        # 处理所有值相同的情况
        if max_val - min_val < self.shapley_epsilon:
            return [self.shapley_norm_range[1] for _ in values]
        # 线性归一化
        norm_min, norm_max = self.shapley_norm_range
        normalized = [norm_min + (val - min_val) * (norm_max - norm_min) / (max_val - min_val) for val in values]
        return normalized

    # ==============================================
    # 辅助方法：清空贡献度历史（实验复用）
    # ==============================================
    def clear_contribution_history(self) -> None:
        """
        清空所有客户端的贡献度历史记录（用于多次实验，避免历史干扰）
        """
        self.contribution_history = defaultdict(lambda: {"sa_contribution_scores": [], "total_contribution": 0.0})
        print("✅ 所有客户端Shapley贡献度历史已清空")

    # ==============================================
    # 辅助方法：打印贡献度统计（实验分析）
    # ==============================================
    def print_contribution_stats(self) -> None:
        """
        打印客户端SA贡献度统计（辅助分析贡献度分布）
        """
        print("\n" + "="*60 + " SA贡献度统计 " + "="*60)
        stats = sorted(
            [(cid, self.contribution_history[cid]["total_contribution"]) for cid in self.contribution_history],
            key=lambda x: x[1],
            reverse=True
        )
        for cid, total_score in stats[:10]:  # 仅打印前10个
            round_scores = self.contribution_history[cid]["sa_contribution_scores"]
            avg_score = np.mean(round_scores) if round_scores else 0.0
            print(f"客户端 [{cid}]：累计贡献度={total_score:.4f} | 平均贡献度={avg_score:.4f} | 参与轮次={len(round_scores)}")
        # 计算贡献度分布方差（越小越均衡）
        all_total_scores = [self.contribution_history[cid]["total_contribution"] for cid in self.contribution_history]
        if len(all_total_scores) > 0:
            score_var = np.var(all_total_scores)
            print(f"\n📊 贡献度分布方差（越小越均衡）：{score_var:.4f}")