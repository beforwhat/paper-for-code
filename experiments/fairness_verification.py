 # experiments/fairness_verification.py
"""
公平性验证实验脚本
核心目标：
1. 量化验证联邦学习算法的客户端公平性，核心关注：
   - 性能分布公平性：客户端间准确率/损失的分布差异（基尼系数、方差/标准差）；
   - 性能保障公平性：最差客户端的性能下限（性能极差、最低准确率）；
   - 异构适应性公平性：不同数据异构程度下的公平性稳定性；
2. 对比5大基线算法（FedAvg/DP-FedAvg/FedProx/Ditto/FedShap）的公平性表现；
3. 模拟不同数据异构程度（低/中/高），验证算法的公平性鲁棒性；
4. 输出公平性量化报告、可视化对比图表，明确SA贡献度等组件的公平性提升效果。
设计原则：
- 基于“数据异构”场景设计实验（非IID是公平性问题的核心诱因）；
- 多维度公平性指标量化，避免单一指标的片面性；
- 结果可视化聚焦“算法对比”和“异构程度-公平性关系”；
- 复用基础实验框架，保证与其他实验的一致性。
"""
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import variation

# 项目内模块导入
from configs.config_loader import load_config
from baselines import (
    FedAvgServer, FedAvgClient,
    DPFedAvgServer, DPFedAvgClient,
    FedProxServer, FedProxClient,
    DITTOServer, DITTOClient,
    FedShapServer, FedShapClient
)
from datasets.non_iid_partitioner import NonIIDPartitioner as simulate_data_heterogeneity  # 数据异构模拟模块

# 可视化配置（与其他实验保持一致）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
PLOT_FORMAT = "png"
PLOT_DPI = 300
ALGORITHM_COLORS = {
    "FedAvg": "#1f77b4",
    "DP-FedAvg": "#ff7f0e",
    "FedProx": "#2ca02c",
    "Ditto": "#d62728",
    "FedShap": "#9467bd"
}
# 异构程度配置
HETEROGENEITY_LEVELS = {
    "low": 0.2,    # 低异构：客户端数据分布相似度80%
    "medium": 0.5, # 中异构：客户端数据分布相似度50%
    "high": 0.8    # 高异构：客户端数据分布相似度20%
}

# ======================== 公平性指标计算函数（核心） ========================
def calculate_gini_coefficient(values):
    """
    计算基尼系数（衡量分布公平性，取值0~1，0=完全公平，1=完全不公平）
    Args:
        values: 客户端性能指标列表（如准确率）
    Returns:
        gini: 基尼系数
    """
    if len(values) == 0 or np.all(values == values[0]):
        return 0.0
    values = np.array(values, dtype=np.float64)
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    # 基尼系数计算公式
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    return gini

def calculate_fairness_metrics(client_performances):
    """
    计算多维度公平性指标
    Args:
        client_performances: 客户端性能字典 {client_id: acc/loss}
    Returns:
        fairness_metrics: 公平性指标字典
    """
    performances = np.array(list(client_performances.values()), dtype=np.float64)
    mean_perf = np.mean(performances)
    std_perf = np.std(performances)
    var_perf = np.var(performances)
    cv_perf = variation(performances) if mean_perf != 0 else 0.0  # 变异系数
    min_perf = np.min(performances)
    max_perf = np.max(performances)
    range_perf = max_perf - min_perf  # 性能极差
    gini = calculate_gini_coefficient(performances)
    # 自定义公平性指数（综合指标，取值0~1，越高越公平）
    # 公式：(1 - 基尼系数) * (1 - 变异系数) * (min_perf / mean_perf)
    fairness_index = (1 - gini) * (1 - cv_perf) * (min_perf / mean_perf) if mean_perf != 0 else 0.0
    fairness_index = np.clip(fairness_index, 0, 1)  # 限制在0~1之间
    
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

# ======================== 核心实验类 ========================
class FairnessVerificationExperiment:
    def __init__(self, config=None, save_results=True, save_path="./experiment_results/fairness_verification"):
        """
        初始化公平性验证实验
        Args:
            config: 配置对象
            save_results: 是否保存结果
            save_path: 结果保存路径
        """
        self.config = config if config is not None else load_config()
        self.save_results = save_results
        self.save_path = save_path
        self.device = torch.device(self.config.device)
        
        # 创建保存目录
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "data"), exist_ok=True)
        
        # 算法列表（与基础性能实验一致）
        self.algorithms = [
            {"name": "FedAvg", "server_cls": FedAvgServer, "client_cls": FedAvgClient, "requires_dist": False},
            {"name": "DP-FedAvg", "server_cls": DPFedAvgServer, "client_cls": DPFedAvgClient, "requires_dist": False},
            {"name": "FedProx", "server_cls": FedProxServer, "client_cls": FedProxClient, "requires_dist": True},
            {"name": "Ditto", "server_cls": DITTOServer, "client_cls": DITTOClient, "requires_dist": True},
            {"name": "FedShap", "server_cls": FedShapServer, "client_cls": FedShapClient, "requires_dist": False}
        ]
        
        # 实验结果存储
        self.fairness_results = {
            "heterogeneity_levels": HETEROGENEITY_LEVELS,
            "per_heterogeneity": {},  # 不同异构程度下的结果
            "final_summary": {}       # 最终公平性汇总
        }
        
        print(f"✅ 公平性验证实验初始化完成 | 待验证算法：{[alg['name'] for alg in self.algorithms]}")
        print(f"📌 异构程度：{list(HETEROGENEITY_LEVELS.keys())} | 客户端数：{self.config.fed.num_clients} | 全局轮次：{self.config.fed.global_rounds}")

    def _simulate_heterogeneous_data(self, heterogeneity_level):
        """
        模拟指定异构程度的客户端数据分布
        Args:
            heterogeneity_level: 异构程度（low/medium/high）
        Returns:
            client_datasets: 各客户端的异构数据集
        """
        alpha = HETEROGENEITY_LEVELS[heterogeneity_level]
        print(f"\n📌 模拟{heterogeneity_level}异构数据（alpha={alpha}）")
        # 调用项目异构数据模拟模块，生成非IID数据集
        client_datasets = simulate_data_heterogeneity(
            dataset_name=self.config.data.dataset,
            num_clients=self.config.fed.num_clients,
            alpha=alpha,  # alpha越大，异构性越强
            seed=self.config.seed
        )
        return client_datasets

    def _run_algorithm_on_heterogeneity(self, algorithm, heterogeneity_level):
        """
        在指定异构程度下运行单个算法，记录公平性指标
        Args:
            algorithm: 算法配置
            heterogeneity_level: 异构程度（low/medium/high）
        Returns:
            alg_results: 该算法在该异构程度下的结果
        """
        alg_name = algorithm["name"]
        print(f"\n--- 运行 {alg_name} | 异构程度：{heterogeneity_level} ---")
        start_time = time.time()
        
        # 1. 模拟异构数据
        client_datasets = self._simulate_heterogeneous_data(heterogeneity_level)
        
        # 2. 初始化服务端和客户端
        server = algorithm["server_cls"](config=self.config)
        server.global_model.to(self.device)
        
        clients = []
        for client_id in range(self.config.fed.num_clients):
            client = algorithm["client_cls"](client_id=client_id, config=self.config)
            # 替换为异构数据集
            client.local_dataloader = client_datasets[client_id]
            client.local_model.to(self.device)
            clients.append(client)
        server.clients = clients
        
        # 3. 初始化指标记录
        round_fairness_metrics = []  # 每轮公平性指标
        round_client_performances = []  # 每轮客户端性能
        global_acc_list = []
        
        # 4. 多轮联邦训练
        for round_idx in range(self.config.fed.global_rounds):
            print(f"\n{alg_name} | {heterogeneity_level}异构 | 轮次 {round_idx+1}/{self.config.fed.global_rounds}")
            
            # 选择客户端
            selected_cids = server.select_clients(round_idx=round_idx)
            
            # 下发全局模型（如需）
            if algorithm["requires_dist"]:
                server.distribute_global_model(selected_client_ids=selected_cids)
            
            # 客户端本地训练
            client_outputs = []
            for cid in selected_cids:
                output = clients[cid].local_train()
                client_outputs.append(output)
            
            # 服务端聚合
            if alg_name == "FedShap":
                server.aggregate_local_results(client_results_list=client_outputs)
            else:
                client_params = [o for o in client_outputs]
                server.aggregate_local_results(client_params_list=client_params)
            
            # 评估全局准确率
            global_acc, _ = server.evaluate_global_model()
            global_acc_list.append(global_acc)
            
            # 评估客户端性能（计算公平性指标）
            client_performances = {}
            for cid in range(self.config.fed.num_clients):
                if alg_name == "Ditto":
                    # Ditto评估个性化模型（更能体现客户端适配性）
                    perf = clients[cid].evaluate_personal_model()
                else:
                    perf = clients[cid].evaluate_local_model()
                client_performances[cid] = perf
            
            # 计算本轮公平性指标
            fairness_metrics = calculate_fairness_metrics(client_performances)
            round_fairness_metrics.append(fairness_metrics)
            round_client_performances.append(client_performances)
            
            # 打印本轮核心公平性指标
            print(f"全局准确率：{global_acc:.2f}% | 基尼系数：{fairness_metrics['gini']:.4f} | 公平性指数：{fairness_metrics['fairness_index']:.4f}")
        
        # 5. 汇总结果
        total_time = time.time() - start_time
        # 提取最终轮次的公平性指标
        final_fairness = round_fairness_metrics[-1]
        # 计算各轮次公平性指标的均值（稳定性）
        avg_gini = np.mean([m["gini"] for m in round_fairness_metrics])
        avg_fairness_index = np.mean([m["fairness_index"] for m in round_fairness_metrics])
        # 最差客户端的平均性能
        avg_min_perf = np.mean([m["min"] for m in round_fairness_metrics])
        
        alg_results = {
            "round_fairness": round_fairness_metrics,
            "round_client_perfs": round_client_performances,
            "global_acc": global_acc_list,
            "final_fairness": final_fairness,
            "avg_gini": avg_gini,
            "avg_fairness_index": avg_fairness_index,
            "avg_min_perf": avg_min_perf,
            "total_time": total_time
        }
        
        print(f"\n✅ {alg_name} | {heterogeneity_level}异构 完成 | 最终基尼系数：{final_fairness['gini']:.4f} | 最终公平性指数：{final_fairness['fairness_index']:.4f}")
        return alg_results

    def run(self):
        """
        运行所有算法在不同异构程度下的公平性验证实验
        """
        # 遍历每个异构程度
        for hetero_level in HETEROGENEITY_LEVELS.keys():
            print(f"\n========== 开始验证 {hetero_level} 异构程度下的公平性 ==========")
            self.fairness_results["per_heterogeneity"][hetero_level] = {}
            
            # 遍历每个算法
            for algorithm in self.algorithms:
                alg_name = algorithm["name"]
                # 运行算法并记录结果
                alg_results = self._run_algorithm_on_heterogeneity(algorithm, hetero_level)
                self.fairness_results["per_heterogeneity"][hetero_level][alg_name] = alg_results
        
        # 生成最终汇总
        self._generate_final_summary()
        
        # 保存结果
        if self.save_results:
            self._save_results()
            self._generate_plots()
        
        # 打印公平性报告
        self._print_fairness_report()
        
        return self.fairness_results

    def _generate_final_summary(self):
        """
        生成各算法在不同异构程度下的公平性汇总
        """
        final_summary = {}
        for alg_name in [a["name"] for a in self.algorithms]:
            final_summary[alg_name] = {}
            for hetero_level in HETEROGENEITY_LEVELS.keys():
                alg_results = self.fairness_results["per_heterogeneity"][hetero_level][alg_name]
                final_summary[alg_name][hetero_level] = {
                    "final_gini": alg_results["final_fairness"]["gini"],
                    "final_fairness_index": alg_results["final_fairness"]["fairness_index"],
                    "final_mean_acc": alg_results["final_fairness"]["mean"],
                    "final_min_acc": alg_results["final_fairness"]["min"],
                    "avg_gini": alg_results["avg_gini"],
                    "avg_min_perf": alg_results["avg_min_perf"]
                }
        self.fairness_results["final_summary"] = final_summary

    def _save_results(self):
        """
        保存公平性实验结果
        """
        # 1. 完整结果（JSON）
        full_results_path = os.path.join(self.save_path, "data", "fairness_full_results.json")
        with open(full_results_path, "w", encoding="utf-8") as f:
            json.dump(self.fairness_results, f, ensure_ascii=False, indent=4)
        
        # 2. 最终汇总（CSV）
        summary_rows = []
        for alg_name, hetero_results in self.fairness_results["final_summary"].items():
            for hetero_level, metrics in hetero_results.items():
                row = {
                    "algorithm": alg_name,
                    "heterogeneity_level": hetero_level,
                    "final_gini": metrics["final_gini"],
                    "final_fairness_index": metrics["final_fairness_index"],
                    "final_mean_acc": metrics["final_mean_acc"],
                    "final_min_acc": metrics["final_min_acc"],
                    "avg_gini": metrics["avg_gini"],
                    "avg_min_perf": metrics["avg_min_perf"]
                }
                summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.save_path, "data", "fairness_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        
        print(f"\n📁 公平性实验结果已保存至：{self.save_path}/data")

    def _generate_plots(self):
        """
        生成公平性可视化图表
        """
        # 1. 各算法在不同异构程度下的基尼系数对比（核心公平性指标）
        plt.figure(figsize=(12, 6))
        hetero_levels = list(HETEROGENEITY_LEVELS.keys())
        x = np.arange(len(hetero_levels))
        width = 0.15  # 柱状图宽度
        alg_names = [a["name"] for a in self.algorithms]
        
        for i, alg_name in enumerate(alg_names):
            gini_values = [
                self.fairness_results["final_summary"][alg_name][level]["final_gini"]
                for level in hetero_levels
            ]
            plt.bar(x + i*width, gini_values, width, label=alg_name, color=ALGORITHM_COLORS[alg_name])
        
        plt.xlabel("数据异构程度", fontsize=12)
        plt.ylabel("最终基尼系数（越小越公平）", fontsize=12)
        plt.title("不同异构程度下各算法的公平性（基尼系数）对比", fontsize=14, fontweight="bold")
        plt.xticks(x + width*2, hetero_levels)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "gini_by_heterogeneity.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 2. 各算法在不同异构程度下的公平性指数对比
        plt.figure(figsize=(12, 6))
        for i, alg_name in enumerate(alg_names):
            fairness_index_values = [
                self.fairness_results["final_summary"][alg_name][level]["final_fairness_index"]
                for level in hetero_levels
            ]
            plt.bar(x + i*width, fairness_index_values, width, label=alg_name, color=ALGORITHM_COLORS[alg_name])
        
        plt.xlabel("数据异构程度", fontsize=12)
        plt.ylabel("最终公平性指数（越大越公平）", fontsize=12)
        plt.title("不同异构程度下各算法的综合公平性指数对比", fontsize=14, fontweight="bold")
        plt.xticks(x + width*2, hetero_levels)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "fairness_index_by_heterogeneity.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 3. 高异构程度下各算法的基尼系数收敛曲线
        plt.figure(figsize=(10, 6))
        hetero_level = "high"
        rounds = list(range(1, self.config.fed.global_rounds+1))
        for alg_name in alg_names:
            gini_values = [
                m["gini"] for m in self.fairness_results["per_heterogeneity"][hetero_level][alg_name]["round_fairness"]
            ]
            plt.plot(
                rounds, gini_values,
                label=alg_name,
                color=ALGORITHM_COLORS[alg_name],
                linewidth=2,
                marker="o",
                markersize=4
            )
        
        plt.xlabel("全局轮次", fontsize=12)
        plt.ylabel("基尼系数（越小越公平）", fontsize=12)
        plt.title(f"{hetero_level}异构程度下各算法基尼系数收敛曲线", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"gini_convergence_{hetero_level}.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 4. 各算法最差客户端准确率对比（性能保障公平性）
        plt.figure(figsize=(12, 6))
        for i, alg_name in enumerate(alg_names):
            min_acc_values = [
                self.fairness_results["final_summary"][alg_name][level]["final_min_acc"]
                for level in hetero_levels
            ]
            plt.bar(x + i*width, min_acc_values, width, label=alg_name, color=ALGORITHM_COLORS[alg_name])
        
        plt.xlabel("数据异构程度", fontsize=12)
        plt.ylabel("最差客户端最终准确率（%）", fontsize=12)
        plt.title("不同异构程度下各算法最差客户端准确率对比（性能保障）", fontsize=14, fontweight="bold")
        plt.xticks(x + width*2, hetero_levels)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "min_acc_by_heterogeneity.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        print(f"\n📊 公平性可视化图表已保存至：{self.save_path}/plots")

    def _print_fairness_report(self):
        """
        打印公平性验证最终报告
        """
        print("\n========== 公平性验证实验 - 最终报告 ==========")
        # 1. 高异构程度下的核心公平性指标对比
        print("\n【高异构程度下核心公平性指标对比】")
        print(f"{'算法':<10} {'基尼系数':<12} {'公平性指数':<12} {'平均准确率(%)':<15} {'最差客户端准确率(%)':<20}")
        print("-" * 70)
        hetero_level = "high"
        for alg_name in [a["name"] for a in self.algorithms]:
            metrics = self.fairness_results["final_summary"][alg_name][hetero_level]
            print(
                f"{alg_name:<10} "
                f"{metrics['final_gini']:<12.4f} "
                f"{metrics['final_fairness_index']:<12.4f} "
                f"{metrics['final_mean_acc']:<15.2f} "
                f"{metrics['final_min_acc']:<20.2f}"
            )
        
        # 2. 公平性提升率（以FedAvg为基准）
        print("\n【公平性提升率（以FedAvg为基准）- 高异构程度】")
        print(f"{'算法':<10} {'基尼系数降低率(%)':<18} {'公平性指数提升率(%)':<20} {'最差客户端准确率提升率(%)':<25}")
        print("-" * 75)
        fedavg_gini = self.fairness_results["final_summary"]["FedAvg"][hetero_level]["final_gini"]
        fedavg_fair_idx = self.fairness_results["final_summary"]["FedAvg"][hetero_level]["final_fairness_index"]
        fedavg_min_acc = self.fairness_results["final_summary"]["FedAvg"][hetero_level]["final_min_acc"]
        
        for alg_name in [a["name"] for a in self.algorithms if alg_name != "FedAvg"]:
            metrics = self.fairness_results["final_summary"][alg_name][hetero_level]
            # 基尼系数降低率（越大越好）
            gini_reduction = ((fedavg_gini - metrics["final_gini"]) / fedavg_gini) * 100 if fedavg_gini != 0 else 0.0
            # 公平性指数提升率（越大越好）
            fair_idx_improve = ((metrics["final_fairness_index"] - fedavg_fair_idx) / fedavg_fair_idx) * 100 if fedavg_fair_idx != 0 else 0.0
            # 最差客户端准确率提升率
            min_acc_improve = ((metrics["final_min_acc"] - fedavg_min_acc) / fedavg_min_acc) * 100 if fedavg_min_acc != 0 else 0.0
            
            print(
                f"{alg_name:<10} "
                f"{gini_reduction:<18.2f} "
                f"{fair_idx_improve:<20.2f} "
                f"{min_acc_improve:<25.2f}"
            )
        
        # 3. 关键结论
        print("\n【关键结论】")
        # 找出公平性最优的算法
        best_fair_alg = max(
            [(alg, self.fairness_results["final_summary"][alg]["high"]["final_fairness_index"]) for alg in alg_names],
            key=lambda x: x[1]
        )[0]
        print(f"1. 高异构场景下公平性最优的算法：{best_fair_alg}（公平性指数：{self.fairness_results['final_summary'][best_fair_alg]['high']['final_fairness_index']:.4f}）")
        # 最差客户端性能最优的算法
        best_min_acc_alg = max(
            [(alg, self.fairness_results["final_summary"][alg]["high"]["final_min_acc"]) for alg in alg_names],
            key=lambda x: x[1]
        )[0]
        print(f"2. 高异构场景下最差客户端性能最优的算法：{best_min_acc_alg}（最差准确率：{self.fairness_results['final_summary'][best_min_acc_alg]['high']['final_min_acc']:.2f}%）")
        # 异构适应性最好的算法（基尼系数随异构程度变化最小）
        alg_hetero_stability = {}
        for alg_name in alg_names:
            gini_values = [self.fairness_results["final_summary"][alg_name][level]["final_gini"] for level in hetero_levels]
            gini_var = np.var(gini_values)
            alg_hetero_stability[alg_name] = gini_var
        best_stable_alg = min(alg_hetero_stability.items(), key=lambda x: x[1])[0]
        print(f"3. 异构适应性最好的算法（基尼系数波动最小）：{best_stable_alg}（基尼系数方差：{alg_hetero_stability[best_stable_alg]:.6f}）")

# ======================== 外部调用函数 ========================
def run_fairness_verification_experiment(config=None, save_results=True, save_path="./experiment_results/fairness_verification"):
    """
    外部调用的核心函数：运行公平性验证实验
    Args:
        config: 配置对象
        save_results: 是否保存结果
        save_path: 结果保存路径
    Returns:
        fairness_results: 公平性实验结果
    """
    experiment = FairnessVerificationExperiment(config=config, save_results=save_results, save_path=save_path)
    results = experiment.run()
    return results

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 运行公平性验证实验
    results = run_fairness_verification_experiment(
        save_results=True,
        save_path="./experiment_results/fairness_verification_2026"
    )
    print("\n✅ 公平性验证实验全部完成！")