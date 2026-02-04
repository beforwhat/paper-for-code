# core/fair_selection/fair_selector.py
"""
公平客户端选择器（FairClientSelector）
核心职责：
1.  平衡“贡献度”与“公平性”选择客户端：
   - 贡献度：基于SA得分，优先选对全局模型提升大的客户端；
   - 公平性：提升参与次数少的客户端权重，避免“马太效应”。
2.  记录客户端历史参与记录，动态调整选择权重；
3.  独立模块设计，兼容BaseServer调用，支持配置化调参。
"""
import numpy as np
import random
from collections import defaultdict

# 项目内模块导入
from configs.config_loader import load_config

class FairClientSelector:
    """
    公平客户端选择器
    核心方法：
    - select()：核心选择逻辑，输出选中的客户端ID列表；
    - update_selection_history()：更新客户端参与历史；
    - clear_selection_history()：清空历史记录（实验复用）。
    """
    def __init__(self, config=None):
        """
        初始化公平选择器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.fair_coeff = self.config.fed.fair_coeff  # 公平系数（0~1，0=纯贡献度，1=纯公平）
        self.total_clients = self.config.fed.num_clients  # 客户端总数

        # 2. 历史参与记录（支撑公平性计算）
        # 结构：{client_id: {"participate_rounds": [参与的轮次列表], "total_participate": 参与次数}}
        self.selection_history = defaultdict(lambda: {"participate_rounds": [], "total_participate": 0})

        # 3. 选择策略配置（支持轮盘赌/贪心/混合策略）
        self.selection_strategy = self.config.fed.selection_strategy  # "roulette"（轮盘赌）/"greedy"（贪心）/"hybrid"（混合）
        self.hybrid_ratio = self.config.fed.hybrid_ratio  # 混合策略中贪心占比（如0.7=70%贪心+30%轮盘赌）

        print(f"✅ 公平客户端选择器初始化完成")
        print(f"📌 公平系数：{self.fair_coeff} | 选择策略：{self.selection_strategy}")
        print(f"📌 客户端总数：{self.total_clients} | 混合策略贪心占比：{self.hybrid_ratio}")

    # ==============================================
    # 核心方法：公平选择客户端（BaseServer调用）
    # ==============================================
    def select(self, client_sa_scores: dict, select_num: int, round_idx: int) -> list:
        """
        核心：基于SA得分+公平性选择客户端
        Args:
            client_sa_scores: 客户端SA贡献度得分字典 {client_id: sa_score}
            select_num: 本轮需要选择的客户端数量
            round_idx: 当前全局轮次（用于更新历史记录）
        Returns:
            selected_clients: 选中的客户端ID列表（长度=select_num）
        """
        # 前置检查
        if not client_sa_scores:
            raise ValueError("客户端SA贡献度得分为空，无法选择客户端")
        if select_num <= 0 or select_num > len(client_sa_scores):
            raise ValueError(f"选择数量{select_num}无效（需满足 0 < 数量 ≤ 客户端总数{len(client_sa_scores)}）")

        # 步骤1：计算带公平性的选择权重
        fair_weights = self._calculate_fair_weights(client_sa_scores, round_idx)

        # 步骤2：按指定策略选择客户端
        if self.selection_strategy == "greedy":
            # 贪心策略：选权重最高的前N个（兼顾贡献度+公平性）
            selected_clients = self._greedy_selection(fair_weights, select_num)
        elif self.selection_strategy == "roulette":
            # 轮盘赌策略：按权重随机采样（更公平，避免绝对垄断）
            selected_clients = self._roulette_selection(fair_weights, select_num)
        elif self.selection_strategy == "hybrid":
            # 混合策略：部分贪心+部分轮盘赌
            selected_clients = self._hybrid_selection(fair_weights, select_num)
        else:
            raise ValueError(f"不支持的选择策略：{self.selection_strategy}，可选greedy/roulette/hybrid")

        # 步骤3：更新客户端参与历史（仅记录本轮选中的）
        self.update_selection_history(selected_clients, round_idx)

        return selected_clients

    # ==============================================
    # 辅助方法1：计算带公平性的选择权重
    # ==============================================
    def _calculate_fair_weights(self, client_sa_scores: dict, round_idx: int) -> dict:
        """
        计算带公平性的选择权重：
        weight = (1 - fair_coeff) * norm_sa_score + fair_coeff * norm_fair_score
        - norm_sa_score：归一化SA贡献度得分（0~1）；
        - norm_fair_score：归一化公平得分（参与次数越少，得分越高，0~1）。
        """
        # 1. 提取客户端ID列表
        client_ids = list(client_sa_scores.keys())

        # 2. 归一化SA贡献度得分（0~1）
        sa_scores = np.array([client_sa_scores[cid] for cid in client_ids])
        sa_scores = np.clip(sa_scores, 0, np.max(sa_scores))  # 处理负得分（异常值）
        if np.sum(sa_scores) == 0:
            norm_sa_scores = {cid: 1.0/len(client_ids) for cid in client_ids}  # 均分
        else:
            norm_sa_scores = {cid: score/np.sum(sa_scores) for cid, score in zip(client_ids, sa_scores)}

        # 3. 计算公平得分（参与次数越少，得分越高）
        fair_scores = {}
        max_participate = max([self.selection_history[cid]["total_participate"] for cid in client_ids]) + 1  # +1避免除零
        for cid in client_ids:
            participate_num = self.selection_history[cid]["total_participate"]
            # 公平得分：(最大参与次数 - 当前参与次数) / 最大参与次数 → 参与越少，得分越高
            fair_score = (max_participate - participate_num) / max_participate
            fair_scores[cid] = fair_score

        # 4. 归一化公平得分（0~1）
        fair_scores_arr = np.array(list(fair_scores.values()))
        if np.sum(fair_scores_arr) == 0:
            norm_fair_scores = {cid: 1.0/len(client_ids) for cid in client_ids}
        else:
            norm_fair_scores = {cid: score/np.sum(fair_scores_arr) for cid, score in fair_scores.items()}

        # 5. 融合贡献度+公平性，计算最终权重
        fair_weights = {}
        for cid in client_ids:
            weight = (1 - self.fair_coeff) * norm_sa_scores[cid] + self.fair_coeff * norm_fair_scores[cid]
            fair_weights[cid] = weight

        # 打印权重分布（前5个，辅助调试）
        print(f"\n📊 本轮客户端公平权重分布（前5个）：")
        sorted_weights = sorted(fair_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for cid, w in sorted_weights:
            print(f"   客户端 [{cid}]：SA得分={norm_sa_scores[cid]:.4f} | 公平得分={norm_fair_scores[cid]:.4f} | 最终权重={w:.4f}")

        return fair_weights

    # ==============================================
    # 辅助方法2：贪心选择（权重最高的前N个）
    # ==============================================
    def _greedy_selection(self, fair_weights: dict, select_num: int) -> list:
        """贪心选择：选权重最高的前N个客户端"""
        sorted_clients = sorted(fair_weights.items(), key=lambda x: x[1], reverse=True)
        selected_clients = [cid for cid, _ in sorted_clients[:select_num]]
        print(f"📌 贪心选择完成：选中权重最高的 {select_num} 个客户端")
        return selected_clients

    # ==============================================
    # 辅助方法3：轮盘赌选择（按权重随机采样，无放回）
    # ==============================================
    def _roulette_selection(self, fair_weights: dict, select_num: int) -> list:
        """轮盘赌选择：按权重随机采样，无放回（更公平，避免绝对垄断）"""
        client_ids = list(fair_weights.keys())
        weights = list(fair_weights.values())
        # 归一化权重（确保和为1）
        weights = np.array(weights) / np.sum(weights)
        # 无放回采样
        selected_clients = random.choices(client_ids, weights=weights, k=select_num)
        # 去重（极端情况权重集中可能重复，补充采样）
        while len(set(selected_clients)) < select_num:
            missing_num = select_num - len(set(selected_clients))
            additional = random.choices(client_ids, weights=weights, k=missing_num)
            selected_clients.extend(additional)
            selected_clients = list(set(selected_clients))[:select_num]
        print(f"📌 轮盘赌选择完成：按权重随机选中 {select_num} 个客户端")
        return selected_clients

    # ==============================================
    # 辅助方法4：混合选择（贪心+轮盘赌）
    # ==============================================
    def _hybrid_selection(self, fair_weights: dict, select_num: int) -> list:
        """混合选择：部分贪心+部分轮盘赌"""
        # 贪心选择数量 = 总数量 * 混合比例
        greedy_num = int(select_num * self.hybrid_ratio)
        roulette_num = select_num - greedy_num

        # 步骤1：贪心选前greedy_num个
        sorted_clients = sorted(fair_weights.items(), key=lambda x: x[1], reverse=True)
        greedy_selected = [cid for cid, _ in sorted_clients[:greedy_num]]

        # 步骤2：剩余客户端中轮盘赌选roulette_num个（排除已贪心选中的）
        remaining_clients = {cid: w for cid, w in fair_weights.items() if cid not in greedy_selected}
        if not remaining_clients:
            roulette_selected = []
        else:
            remaining_ids = list(remaining_clients.keys())
            remaining_weights = list(remaining_clients.values())
            remaining_weights = np.array(remaining_weights) / np.sum(remaining_weights)
            roulette_selected = random.choices(remaining_ids, weights=remaining_weights, k=roulette_num)

        # 合并结果
        selected_clients = greedy_selected + roulette_selected
        print(f"📌 混合选择完成：贪心选中 {greedy_num} 个 | 轮盘赌选中 {roulette_num} 个")
        return selected_clients

    # ==============================================
    # 辅助方法：更新客户端参与历史
    # ==============================================
    def update_selection_history(self, selected_clients: list, round_idx: int) -> None:
        """更新本轮选中客户端的参与历史"""
        for cid in selected_clients:
            self.selection_history[cid]["participate_rounds"].append(round_idx)
            self.selection_history[cid]["total_participate"] += 1
        print(f"✅ 本轮选中客户端参与历史已更新（轮次 {round_idx}）")

    # ==============================================
    # 辅助方法：清空选择历史（实验复用）
    # ==============================================
    def clear_selection_history(self) -> None:
        """清空所有客户端的参与历史（用于多次实验，避免历史干扰）"""
        self.selection_history = defaultdict(lambda: {"participate_rounds": [], "total_participate": 0})
        print("✅ 所有客户端选择历史已清空")

    # ==============================================
    # 辅助方法：打印客户端参与统计（实验分析）
    # ==============================================
    def print_participation_stats(self) -> None:
        """打印所有客户端的参与次数统计（辅助分析公平性）"""
        print("\n" + "="*60 + " 客户端参与统计 " + "="*60)
        stats = sorted(
            [(cid, self.selection_history[cid]["total_participate"]) for cid in self.selection_history],
            key=lambda x: x[1],
            reverse=True
        )
        for cid, cnt in stats:
            print(f"客户端 [{cid}]：参与次数 = {cnt}")
        # 计算公平性指标（参与次数的方差，越小越公平）
        all_participate = [self.selection_history[cid]["total_participate"] for cid in self.selection_history]
        if len(all_participate) > 0:
            participate_var = np.var(all_participate)
            print(f"\n📊 参与次数方差（越小越公平）：{participate_var:.4f}")