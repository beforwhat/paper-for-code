# experiments/basic_performance.py
"""
基础性能对比实验脚本
核心目标：
1. 统一运行FedAvg/DP-FedAvg/FedProx/Ditto/FedShap/FedAdaClip++/FedFairADP-ALA 7大算法的联邦训练；
2. 记录每轮全局准确率、全局损失、客户端本地损失/准确率、训练耗时；
3. 保存实验结果（CSV/JSON），生成收敛曲线、最终性能对比等可视化图表；
4. 输出量化对比报告，便于分析各算法的基础性能差异。
设计原则：
- 适配你的FederatedTrainer框架运行FedFairADP-ALA；
- 保留原有基线算法逻辑，保证对比公平性；
- 结果结构化保存，支持后续复现和分析；
- 可视化结果直观展示收敛速度、最终性能差异。
"""
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# 项目内模块导入
from configs.config_loader import load_config
from baselines import (
    FedAvgServer, FedAvgClient,
    DPFedAvgServer, DPFedAvgClient,
    FedProxServer, FedProxClient,
    DITTOServer, DITTOClient,
    FedShapServer, FedShapClient,
    FedAdaptiveDPServer, FedAdaptiveDPClient
)
# 导入你的核心联邦训练器
from core.federated.trainer import FederatedTrainer

# 设置matplotlib中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ======================== 实验配置常量 ========================
PLOT_FORMAT = "png"
PLOT_DPI = 300
# 颜色映射（区分不同算法）
ALGORITHM_COLORS = {
    "FedAvg": "#1f77b4",
    "DP-FedAvg": "#ff7f0e",
    "FedProx": "#2ca02c",
    "Ditto": "#d62728",
    "FedShap": "#9467bd",
    "FedAdaClip++": "#8c564b",
    "FedFairADP-ALA": "#e377c2"  # 你的方法（粉色突出）
}
# 标记映射
ALGORITHM_MARKERS = {
    "FedAvg": "o",
    "DP-FedAvg": "s",
    "FedProx": "^",
    "Ditto": "p",
    "FedShap": "*",
    "FedAdaClip++": "D",
    "FedFairADP-ALA": "X"  # 你的方法标记（叉形）
}

# ======================== 核心实验类 ========================
class BasicPerformanceExperiment:
    def __init__(self, config=None, save_results=True, save_path="./experiment_results/basic_performance"):
        """初始化基础性能对比实验"""
        # 加载配置
        self.config = config if config is not None else load_config()
        self.save_results = save_results
        self.save_path = save_path
        self.device = torch.device(self.config.device)
        
        # 创建保存目录
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "data"), exist_ok=True)
        
        # 初始化算法列表（适配你的FederatedTrainer）
        self.algorithms = [
            {
                "name": "FedAvg",
                "server_cls": FedAvgServer,
                "client_cls": FedAvgClient,
                "requires_global_distribution": False,
                "type": "baseline"
            },
            {
                "name": "DP-FedAvg",
                "server_cls": DPFedAvgServer,
                "client_cls": DPFedAvgClient,
                "requires_global_distribution": False,
                "type": "baseline"
            },
            {
                "name": "FedProx",
                "server_cls": FedProxServer,
                "client_cls": FedProxClient,
                "requires_global_distribution": True,
                "type": "baseline"
            },
            {
                "name": "Ditto",
                "server_cls": DITTOServer,
                "client_cls": DITTOClient,
                "requires_global_distribution": True,
                "type": "baseline"
            },
            {
                "name": "FedShap",
                "server_cls": FedShapServer,
                "client_cls": FedShapClient,
                "requires_global_distribution": False,
                "type": "baseline"
            },
            {
                "name": "FedAdaClip++",
                "server_cls": FedAdaptiveDPServer,
                "client_cls": FedAdaptiveDPClient,
                "requires_global_distribution": False,
                "type": "baseline"
            },
            {
                "name": "FedFairADP-ALA",  # 你的核心方法
                "type": "custom",  # 自定义类型
                "run_func": self._run_fedfairadp_ala  # 专用运行函数
            }
        ]
        
        # 实验结果存储
        self.experiment_results = {
            "global_metrics": {},  # 全局指标（每轮准确率/损失/耗时）
            "client_metrics": {},  # 客户端指标（最终本地准确率/损失）
            "final_summary": {}    # 最终性能汇总
        }
        
        print(f"✅ 基础性能对比实验初始化完成 | 待运行算法：{[alg['name'] for alg in self.algorithms]}")
        print(f"📌 实验配置：全局轮次={self.config.fed.num_global_rounds} | 客户端数={self.config.fed.num_clients} | 设备={self.device}")

    # ======================== 运行基线算法（原有逻辑） ========================
    def _run_baseline_algorithm(self, algorithm):
        """运行基线算法（FedAvg/DP-FedAvg等）"""
        alg_name = algorithm["name"]
        print(f"\n========== 开始运行 {alg_name} ==========")
        start_time = time.time()
        
        # 1. 初始化服务端
        server = algorithm["server_cls"](config=self.config)
        server.global_model.to(self.device)
        
        # 2. 初始化客户端
        clients = []
        for client_id in range(self.config.fed.num_clients):
            client = algorithm["client_cls"](client_id=client_id, config=self.config)
            client.local_model.to(self.device)
            clients.append(client)
        server.clients = clients
        
        # 3. 初始化指标记录
        global_acc_list = []
        global_loss_list = []
        round_time_list = []
        
        # 4. 多轮联邦训练
        for round_idx in range(self.config.fed.num_global_rounds):
            round_start = time.time()
            print(f"\n--- {alg_name} 全局轮次 {round_idx+1}/{self.config.fed.num_global_rounds} ---")
            
            # 选择客户端
            selected_cids = server.select_clients(round_idx=round_idx)
            print(f"📌 选中客户端ID：{selected_cids}")
            
            # 下发全局模型（如需要）
            if algorithm["requires_global_distribution"]:
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
                server.aggregate_local_results(client_params_list=client_outputs)
            
            # 评估全局模型
            global_acc, global_loss = server.evaluate_global_model()
            global_acc_list.append(global_acc)
            global_loss_list.append(global_loss)
            
            # 记录本轮耗时
            round_time = time.time() - round_start
            round_time_list.append(round_time)
            
            print(f"📌 {alg_name} 轮次 {round_idx+1} | 全局准确率：{global_acc:.2f}% | 全局损失：{global_loss:.4f} | 耗时：{round_time:.2f}s")
        
        # 5. 记录客户端最终本地指标
        client_final_metrics = {}
        for cid in range(self.config.fed.num_clients):
            if alg_name == "Ditto":
                client_acc = clients[cid].evaluate_personal_model()
                client_loss = clients[cid].personal_train_total_loss
            else:
                client_acc = clients[cid].evaluate_local_model()
                client_loss = clients[cid].local_train_loss
            client_final_metrics[cid] = {
                "acc": client_acc,
                "loss": client_loss
            }
        
        # 6. 计算总耗时和平均轮次耗时
        total_time = time.time() - start_time
        avg_round_time = np.mean(round_time_list)
        
        # 7. 整理结果
        return {
            "global_metrics": {
                "acc": global_acc_list,
                "loss": global_loss_list,
                "round_time": round_time_list,
                "total_time": total_time,
                "avg_round_time": avg_round_time
            },
            "client_metrics": client_final_metrics,
            "final_summary": {
                "final_global_acc": global_acc_list[-1],
                "final_global_loss": global_loss_list[-1],
                "avg_client_acc": np.mean([v["acc"] for v in client_final_metrics.values()]),
                "avg_client_loss": np.mean([v["loss"] for v in client_final_metrics.values()]),
                "total_time": total_time,
                "avg_round_time": avg_round_time
            }
        }

    # ======================== 运行你的FedFairADP-ALA（核心适配） ========================
    def _run_fedfairadp_ala(self, algorithm=None):
        """
        运行你的FedFairADP-ALA（基于FederatedTrainer）
        """
        alg_name = "FedFairADP-ALA"
        print(f"\n========== 开始运行 {alg_name} ==========")
        
        # 初始化你的联邦训练器
        trainer = FederatedTrainer(config=self.config)
        
        # 记录开始时间
        start_time = time.time()
        
        # 启动训练（你的核心逻辑）
        trainer.run_federated_training()
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 提取全局指标（从trainer的server中获取）
        global_acc_list = [m["acc"] * 100 for m in trainer.server.global_metrics["round_metrics"]]
        global_loss_list = [m["loss"] for m in trainer.server.global_metrics["round_metrics"]]
        round_time_list = trainer.training_metrics["round_duration"]
        avg_round_time = np.mean(round_time_list)
        
        # 提取客户端指标（模拟，可根据实际需求调整）
        client_final_metrics = {}
        for cid in range(self.config.fed.num_clients):
            if cid in trainer.clients:
                # 评估客户端本地模型
                client_acc = trainer.clients[cid].evaluate_local_model()
                client_loss = trainer.clients[cid].local_train_loss
                client_final_metrics[cid] = {
                    "acc": client_acc,
                    "loss": client_loss
                }
        
        # 整理结果（保持和基线算法一致的格式）
        avg_client_acc = np.mean([v["acc"] for v in client_final_metrics.values()])
        avg_client_loss = np.mean([v["loss"] for v in client_final_metrics.values()])
        
        return {
            "global_metrics": {
                "acc": global_acc_list,
                "loss": global_loss_list,
                "round_time": round_time_list,
                "total_time": total_time,
                "avg_round_time": avg_round_time
            },
            "client_metrics": client_final_metrics,
            "final_summary": {
                "final_global_acc": trainer.server.global_metrics["best_global_acc"] * 100,  # 最优准确率
                "final_global_loss": trainer.server.global_metrics["best_global_loss"],
                "avg_client_acc": avg_client_acc,
                "avg_client_loss": avg_client_loss,
                "total_time": total_time,
                "avg_round_time": avg_round_time
            }
        }

    # ======================== 主运行逻辑 ========================
    def run(self):
        """运行所有算法的性能对比实验"""
        for algorithm in self.algorithms:
            alg_name = algorithm["name"]
            
            # 运行算法
            if algorithm["type"] == "baseline":
                alg_results = self._run_baseline_algorithm(algorithm)
            else:  # custom（你的方法）
                alg_results = self._run_fedfairadp_ala(algorithm)
            
            # 保存结果
            self.experiment_results["global_metrics"][alg_name] = alg_results["global_metrics"]
            self.experiment_results["client_metrics"][alg_name] = alg_results["client_metrics"]
            self.experiment_results["final_summary"][alg_name] = alg_results["final_summary"]
        
        # 保存结果+生成可视化
        if self.save_results:
            self._save_results()
            self._generate_plots()
        
        # 输出最终报告
        self._print_final_report()
        
        return self.experiment_results

    # ======================== 保存结果 ========================
    def _save_results(self):
        """保存实验结果"""
        # 1. 全局指标（CSV）
        global_metrics_df = pd.DataFrame()
        for alg_name, metrics in self.experiment_results["global_metrics"].items():
            alg_df = pd.DataFrame({
                "round": list(range(1, len(metrics["acc"])+1)),
                "algorithm": alg_name,
                "global_acc": metrics["acc"],
                "global_loss": metrics["loss"],
                "round_time": metrics["round_time"]
            })
            global_metrics_df = pd.concat([global_metrics_df, alg_df], ignore_index=True)
        global_metrics_path = os.path.join(self.save_path, "data", "global_metrics.csv")
        global_metrics_df.to_csv(global_metrics_path, index=False, encoding="utf-8")
        
        # 2. 客户端指标（JSON）
        client_metrics_path = os.path.join(self.save_path, "data", "client_metrics.json")
        with open(client_metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.experiment_results["client_metrics"], f, ensure_ascii=False, indent=4)
        
        # 3. 最终汇总（CSV+JSON）
        final_summary_df = pd.DataFrame.from_dict(self.experiment_results["final_summary"], orient="index")
        final_summary_df.reset_index(inplace=True)
        final_summary_df.rename(columns={"index": "algorithm"}, inplace=True)
        
        final_summary_csv_path = os.path.join(self.save_path, "data", "final_summary.csv")
        final_summary_df.to_csv(final_summary_csv_path, index=False, encoding="utf-8")
        
        final_summary_json_path = os.path.join(self.save_path, "data", "final_summary.json")
        with open(final_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(self.experiment_results["final_summary"], f, ensure_ascii=False, indent=4)
        
        print(f"\n📁 实验结果已保存至：{self.save_path}/data")

    # ======================== 生成可视化 ========================
    def _generate_plots(self):
        """生成可视化图表"""
        # 1. 全局准确率收敛曲线（适配7个算法）
        plt.figure(figsize=(12, 6))
        for alg_name, metrics in self.experiment_results["global_metrics"].items():
            rounds = list(range(1, len(metrics["acc"])+1))
            plt.plot(
                rounds, metrics["acc"],
                label=alg_name,
                color=ALGORITHM_COLORS[alg_name],
                marker=ALGORITHM_MARKERS[alg_name],
                markersize=6,
                linewidth=2
            )
        plt.xlabel("全局轮次", fontsize=12)
        plt.ylabel("全局准确率（%）", fontsize=12)
        plt.title("各算法全局准确率收敛曲线", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"global_acc_convergence.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"📊 准确率收敛曲线已保存至：{plot_path}")
        
        # 2. 全局损失收敛曲线
        plt.figure(figsize=(12, 6))
        for alg_name, metrics in self.experiment_results["global_metrics"].items():
            rounds = list(range(1, len(metrics["loss"])+1))
            plt.plot(
                rounds, metrics["loss"],
                label=alg_name,
                color=ALGORITHM_COLORS[alg_name],
                marker=ALGORITHM_MARKERS[alg_name],
                markersize=6,
                linewidth=2
            )
        plt.xlabel("全局轮次", fontsize=12)
        plt.ylabel("全局损失", fontsize=12)
        plt.title("各算法全局损失收敛曲线", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"global_loss_convergence.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"📊 损失收敛曲线已保存至：{plot_path}")
        
        # 3. 最终全局准确率对比柱状图
        plt.figure(figsize=(12, 6))
        alg_names = list(self.experiment_results["final_summary"].keys())
        final_accs = [self.experiment_results["final_summary"][alg]["final_global_acc"] for alg in alg_names]
        colors = [ALGORITHM_COLORS[alg] for alg in alg_names]
        
        bars = plt.bar(alg_names, final_accs, color=colors, width=0.6)
        for bar, acc in zip(bars, final_accs):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{acc:.2f}%",
                ha="center", va="bottom", fontsize=10
            )
        plt.xlabel("算法", fontsize=12)
        plt.ylabel("最终全局准确率（%）", fontsize=12)
        plt.title("各算法最终全局准确率对比", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"final_global_acc_comparison.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"📊 最终准确率对比图已保存至：{plot_path}")
        
        # 4. 平均轮次耗时对比柱状图
        plt.figure(figsize=(12, 6))
        avg_round_times = [self.experiment_results["final_summary"][alg]["avg_round_time"] for alg in alg_names]
        bars = plt.bar(alg_names, avg_round_times, color=colors, width=0.6)
        for bar, t in zip(bars, avg_round_times):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{t:.2f}s",
                ha="center", va="bottom", fontsize=10
            )
        plt.xlabel("算法", fontsize=12)
        plt.ylabel("平均轮次耗时（s）", fontsize=12)
        plt.title("各算法平均轮次耗时对比", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", f"avg_round_time_comparison.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"📊 平均耗时对比图已保存至：{plot_path}")

    # ======================== 打印最终报告 ========================
    def _print_final_report(self):
        """打印最终性能对比报告"""
        print("\n========== 基础性能对比实验 - 最终报告 ==========")
        print(f"{'算法':<15} {'最终全局准确率(%)':<20} {'最终全局损失':<15} {'平均客户端准确率(%)':<20} {'总耗时(s)':<15} {'平均轮次耗时(s)':<15}")
        print("-" * 110)
        for alg_name, summary in self.experiment_results["final_summary"].items():
            print(
                f"{alg_name:<15} "
                f"{summary['final_global_acc']:<20.2f} "
                f"{summary['final_global_loss']:<15.4f} "
                f"{summary['avg_client_acc']:<20.2f} "
                f"{summary['total_time']:<15.2f} "
                f"{summary['avg_round_time']:<15.2f}"
            )
        print("-" * 110)

# ======================== 外部调用函数 ========================
def run_basic_performance_experiment(config=None, save_results=True, save_path="./experiment_results/basic_performance"):
    """外部调用的核心函数"""
    experiment = BasicPerformanceExperiment(config=config, save_results=save_results, save_path=save_path)
    experiment_results = experiment.run()
    return experiment_results

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 运行基础性能对比实验
    results = run_basic_performance_experiment(
        save_results=True,
        save_path="./experiment_results/basic_performance_2026"
    )
    print("\n✅ 基础性能对比实验全部完成！")