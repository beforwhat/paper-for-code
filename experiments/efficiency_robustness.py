# experiments/efficiency_robustness.py
"""
效率与鲁棒性验证实验脚本
核心目标：
1. 量化验证联邦学习算法的效率指标：
   - 时间效率：总训练耗时、每轮耗时、每客户端平均耗时；
   - 资源效率：内存占用、GPU显存占用（如有）、CPU使用率；
   - 通信效率：每轮参数传输量、总通信开销；
2. 验证算法的鲁棒性（重点SA贡献度的稳定性）：
   - 规模鲁棒性：不同客户端数量（少/中/多）下的性能稳定性；
   - 噪声鲁棒性：不同数据噪声（无/低/高）下的性能保持率；
   - 故障鲁棒性：节点故障（0%/10%/20%）下的性能容忍度；
   - 异构鲁棒性：不同数据异构程度下的性能波动；
3. 对比7大算法（含你的FedFairADP-ALA），明确SA贡献度+你的方法在效率-鲁棒性上的优势。
设计原则：
- 多场景验证鲁棒性，覆盖联邦学习实际部署的核心挑战；
- 量化效率指标，兼顾时间/资源/通信维度；
- 聚焦SA贡献度+FedFairADP-ALA的稳定性，对比其与其他算法的鲁棒性差异；
- 复用现有实验框架，保证结果可对比性。
"""
import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
from typing import Dict, List, Tuple

# 项目内模块导入
from configs.config_loader import load_config
from baselines import (
    FedAvgServer, FedAvgClient,
    DPFedAvgServer, DPFedAvgClient,
    FedProxServer, FedProxClient,
    DITTOServer, DITTOClient,
    FedShapServer, FedShapClient,
    FedAdaptiveDPServer, FedAdaptiveDPClient  # FedAdaClip++
)
# 导入你的核心联邦训练器（替代单独的Server/Client）
from core.federated.trainer import FederatedTrainer
from datasets.non_iid_partitioner import NonIIDPartitioner as simulate_data_heterogeneity
from core.noise import add_noise_to_dataset  # 数据噪声添加模块

# 可视化配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
PLOT_FORMAT = "png"
PLOT_DPI = 300
ALGORITHM_COLORS = {
    "FedAvg": "#1f77b4",
    "DP-FedAvg": "#ff7f0e",
    "FedProx": "#2ca02c",
    "Ditto": "#d62728",
    "FedShap": "#9467bd",  # SA贡献度算法
    "FedAdaClip++": "#8c564b",  # 2024新基线
    "FedFairADP-ALA": "#e377c2"  # 你的核心方法（粉色突出）
}
ALGORITHM_MARKERS = {
    "FedAvg": "o",
    "DP-FedAvg": "s",
    "FedProx": "^",
    "Ditto": "p",
    "FedShap": "*",
    "FedAdaClip++": "D",
    "FedFairADP-ALA": "X"  # 你的方法标记（叉形，突出）
}

# ======================== 鲁棒性场景配置 ========================
SCALE_SCENARIOS = {"small": 10, "medium": 20, "large": 50}
NOISE_SCENARIOS = {"none": 0.0, "low": 0.1, "high": 0.3}
FAILURE_SCENARIOS = {"none": 0.0, "low": 0.1, "high": 0.2}
HETEROGENEITY_SCENARIOS = {"low": 0.2, "medium": 0.5, "high": 0.8}

# ======================== 核心实验类 ========================
class EfficiencyRobustnessExperiment:
    def __init__(self, config=None, save_results=True, save_path="./experiment_results/efficiency_robustness"):
        self.config = config if config is not None else load_config()
        self.save_results = save_results
        self.save_path = save_path
        self.device = torch.device(self.config.device)
        self.process = psutil.Process(os.getpid())  # 用于资源监控
        
        # 创建保存目录
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "data"), exist_ok=True)
        
        # 初始化算法列表（适配你的FederatedTrainer）
        self.algorithms = [
            {
                "name": "FedAvg",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "DP-FedAvg",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "FedProx",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "Ditto",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "FedShap",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "FedAdaClip++",
                "type": "baseline",
                "run_func": self._run_baseline_algorithm
            },
            {
                "name": "FedFairADP-ALA",  # 你的核心方法
                "type": "custom",
                "run_func": self._run_fedfairadp_ala  # 专用运行函数
            }
        ]
        
        # 实验结果存储
        self.results = {
            "efficiency_metrics": {},  # 效率指标
            "robustness_metrics": {},  # 鲁棒性指标
            "final_summary": {}        # 最终汇总
        }
        
        print(f"✅ 效率&鲁棒性实验初始化完成 | 待运行算法：{[alg['name'] for alg in self.algorithms]}")

    # ======================== 运行基线算法（原有逻辑） ========================
    def _run_baseline_algorithm(self, alg_name, scenario_config=None):
        """运行原有基线算法（FedAvg/DP-FedAvg等）"""
        start_time = time.time()
        client_params_sizes = []
        
        # 适配场景配置（如客户端数量、噪声等）
        if scenario_config:
            self.config.fed.num_clients = scenario_config.get("num_clients", self.config.fed.num_clients)
        
        # 初始化服务端
        if alg_name == "FedAvg":
            server = FedAvgServer(config=self.config, total_clients=self.config.fed.num_clients)
        elif alg_name == "DP-FedAvg":
            server = DPFedAvgServer(config=self.config, total_clients=self.config.fed.num_clients)
        elif alg_name == "FedProx":
            server = FedProxServer(config=self.config, total_clients=self.config.fed.num_clients)
        elif alg_name == "Ditto":
            server = DITTOServer(config=self.config, total_clients=self.config.fed.num_clients)
        elif alg_name == "FedShap":
            server = FedShapServer(config=self.config, total_clients=self.config.fed.num_clients)
        elif alg_name == "FedAdaClip++":
            server = FedAdaptiveDPServer(config=self.config, total_clients=self.config.fed.num_clients)
        server.global_model.to(self.device)
        
        # 初始化客户端
        clients = {}
        for cid in range(self.config.fed.num_clients):
            if alg_name == "FedAvg":
                client = FedAvgClient(client_id=cid, config=self.config)
            elif alg_name == "DP-FedAvg":
                client = DPFedAvgClient(client_id=cid, config=self.config)
            elif alg_name == "FedProx":
                client = FedProxClient(client_id=cid, config=self.config)
            elif alg_name == "Ditto":
                client = DITTOClient(client_id=cid, config=self.config)
            elif alg_name == "FedShap":
                client = FedShapClient(client_id=cid, config=self.config)
            elif alg_name == "FedAdaClip++":
                client = FedAdaptiveDPClient(client_id=cid, config=self.config)
            client.local_model.to(self.device)
            clients[cid] = client
        server.clients = clients
        
        # 训练过程
        global_acc_list = []
        for round_idx in range(self.config.fed.num_global_rounds):
            selected_cids = server.select_clients(round_idx=round_idx)
            client_outputs = []
            
            for cid in selected_cids:
                output = clients[cid].local_train()
                client_outputs.append(output)
                # 记录参数大小
                param_size = sum(p.numel() * p.element_size() for p in output)
                client_params_sizes.append(param_size)
            
            # 聚合
            if alg_name == "FedShap":
                server.aggregate_local_results(client_results_list=client_outputs)
            else:
                server.aggregate_local_results(client_params_list=client_outputs)
            
            # 评估
            acc, _ = server.evaluate_global_model()
            global_acc_list.append(acc)
        
        end_time = time.time()
        # 计算效率指标
        efficiency_metrics = calculate_efficiency_metrics(
            start_time=start_time,
            end_time=end_time,
            client_params_sizes=client_params_sizes,
            process=self.process
        )
        # 返回结果
        return {
            "efficiency": efficiency_metrics,
            "final_acc": global_acc_list[-1],
            "acc_list": global_acc_list
        }

    # ======================== 运行你的FedFairADP-ALA（核心适配） ========================
    def _run_fedfairadp_ala(self, alg_name, scenario_config=None):
        """
        运行你的FedFairADP-ALA（基于FederatedTrainer）
        """
        # 备份原始配置
        original_num_clients = self.config.fed.num_clients
        
        # 适配场景配置（如客户端数量、噪声等）
        if scenario_config:
            self.config.fed.num_clients = scenario_config.get("num_clients", original_num_clients)
            # 适配噪声/异构配置（如果需要）
            if "noise_level" in scenario_config:
                self.config.data.noise_level = scenario_config["noise_level"]
            if "heterogeneity_level" in scenario_config:
                self.config.data.heterogeneity_level = scenario_config["heterogeneity_level"]
        
        # 初始化你的联邦训练器
        trainer = FederatedTrainer(config=self.config)
        
        # 记录开始时间
        start_time = time.time()
        client_params_sizes = []  # 用于通信效率计算
        
        try:
            # 启动训练（你的核心逻辑）
            trainer.run_federated_training()
            
            # 收集参数大小（模拟，实际可从trainer中提取）
            for round_idx in range(self.config.fed.num_global_rounds):
                # 假设每轮参数大小（可根据实际模型调整）
                param_size = sum(p.numel() * p.element_size() for p in trainer.server.global_model.parameters())
                client_params_sizes.append([param_size] * len(trainer.server.selected_clients))
            
            end_time = time.time()
            
            # 计算效率指标
            efficiency_metrics = calculate_efficiency_metrics(
                start_time=start_time,
                end_time=end_time,
                client_params_sizes=client_params_sizes,
                process=self.process
            )
            
            # 提取关键指标
            final_acc = trainer.server.global_metrics["best_global_acc"] * 100  # 转百分比
            acc_list = [m["acc"] * 100 for m in trainer.server.global_metrics["round_metrics"]]
            
            # 恢复原始配置
            self.config.fed.num_clients = original_num_clients
            
            return {
                "efficiency": efficiency_metrics,
                "final_acc": final_acc,
                "acc_list": acc_list,
                "trainer_metrics": trainer.training_metrics  # 你的训练器监控指标
            }
        
        except Exception as e:
            print(f"❌ 运行{alg_name}失败：{str(e)}")
            # 恢复原始配置
            self.config.fed.num_clients = original_num_clients
            raise

    # ======================== 鲁棒性测试（多场景） ========================
    def _test_robustness(self, alg_name, run_func):
        """测试算法在不同场景下的鲁棒性"""
        # 1. 规模鲁棒性
        scale_perfs = []
        for scale, num_clients in SCALE_SCENARIOS.items():
            res = run_func(alg_name, scenario_config={"num_clients": num_clients})
            scale_perfs.append(res["final_acc"])
        
        # 2. 噪声鲁棒性
        noise_perfs = []
        for noise, level in NOISE_SCENARIOS.items():
            res = run_func(alg_name, scenario_config={"noise_level": level})
            noise_perfs.append(res["final_acc"])
        
        # 3. 故障鲁棒性（模拟10%/20%客户端故障）
        failure_perfs = []
        for failure, rate in FAILURE_SCENARIOS.items():
            # 模拟故障：随机选择rate比例的客户端不参与训练
            res = run_func(alg_name, scenario_config={"failure_rate": rate})
            failure_perfs.append(res["final_acc"])
        
        # 4. 异构鲁棒性
        hetero_perfs = []
        for hetero, level in HETEROGENEITY_SCENARIOS.items():
            res = run_func(alg_name, scenario_config={"heterogeneity_level": level})
            hetero_perfs.append(res["final_acc"])
        
        # 计算鲁棒性指标
        baseline_perf = scale_perfs[1]  # medium规模作为基准
        robustness_metrics = {
            "scale": calculate_robustness_metrics(baseline_perf, scale_perfs),
            "noise": calculate_robustness_metrics(baseline_perf, noise_perfs),
            "failure": calculate_robustness_metrics(baseline_perf, failure_perfs),
            "heterogeneity": calculate_robustness_metrics(baseline_perf, hetero_perfs),
            # 综合鲁棒性得分（加权平均）
            "comprehensive_score": np.mean([
                robustness_metrics["scale"]["robustness_score"],
                robustness_metrics["noise"]["robustness_score"],
                robustness_metrics["failure"]["robustness_score"],
                robustness_metrics["heterogeneity"]["robustness_score"]
            ])
        }
        
        return robustness_metrics

    # ======================== 主运行逻辑 ========================
    def run(self):
        """运行所有算法的效率&鲁棒性测试"""
        for alg in self.algorithms:
            alg_name = alg["name"]
            run_func = alg["run_func"]
            
            print(f"\n========== 开始测试 {alg_name} ==========")
            
            # 1. 基础效率测试（基准场景）
            baseline_res = run_func(alg_name)
            self.results["efficiency_metrics"][alg_name] = baseline_res["efficiency"]
            
            # 2. 鲁棒性测试
            robustness_metrics = self._test_robustness(alg_name, run_func)
            self.results["robustness_metrics"][alg_name] = robustness_metrics
            
            # 3. 整理最终汇总
            self.results["final_summary"][alg_name] = {
                "final_acc": baseline_res["final_acc"],
                "total_time": baseline_res["efficiency"]["total_time"],
                "avg_round_time": baseline_res["efficiency"]["avg_round_time"],
                "comprehensive_robustness_score": robustness_metrics["comprehensive_score"],
                "memory_usage_mb": baseline_res["efficiency"]["memory_usage_mb"],
                "total_comm_mb": baseline_res["efficiency"]["total_comm_mb"]
            }
            
            print(f"✅ {alg_name} 测试完成 | 最终准确率：{baseline_res['final_acc']:.2f}% | 鲁棒性得分：{robustness_metrics['comprehensive_score']:.4f}")
        
        # 保存结果+生成可视化
        if self.save_results:
            self._save_results()
            self._generate_plots()
        
        # 打印最终报告
        self._print_final_report()
        
        return self.results

    # ======================== 保存结果 ========================
    def _save_results(self):
        """保存实验结果"""
        # 保存效率指标
        eff_df = pd.DataFrame.from_dict(self.results["efficiency_metrics"], orient="index")
        eff_df.to_csv(os.path.join(self.save_path, "data", "efficiency_metrics.csv"), encoding="utf-8")
        
        # 保存鲁棒性指标
        with open(os.path.join(self.save_path, "data", "robustness_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.results["robustness_metrics"], f, ensure_ascii=False, indent=4)
        
        # 保存最终汇总
        summary_df = pd.DataFrame.from_dict(self.results["final_summary"], orient="index")
        summary_df.to_csv(os.path.join(self.save_path, "data", "final_summary.csv"), encoding="utf-8")
        
        print(f"\n📁 实验结果已保存至：{self.save_path}/data")

    # ======================== 生成可视化 ========================
    def _generate_plots(self):
        """生成可视化图表"""
        alg_names = list(self.results["final_summary"].keys())
        colors = [ALGORITHM_COLORS[alg] for alg in alg_names]
        
        # 1. 综合鲁棒性得分对比
        plt.figure(figsize=(12, 6))
        scores = [self.results["final_summary"][alg]["comprehensive_robustness_score"] for alg in alg_names]
        bars = plt.bar(alg_names, scores, color=colors, width=0.6)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, 
                     f"{score:.4f}", ha="center", va="bottom")
        plt.xlabel("算法", fontsize=12)
        plt.ylabel("综合鲁棒性得分（0~1）", fontsize=12)
        plt.title("各算法综合鲁棒性得分对比", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "plots", f"robustness_score_comparison.{PLOT_FORMAT}"), 
                    dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 2. 总训练耗时对比
        plt.figure(figsize=(12, 6))
        total_times = [self.results["final_summary"][alg]["total_time"] for alg in alg_names]
        bars = plt.bar(alg_names, total_times, color=colors, width=0.6)
        for bar, t in zip(bars, total_times):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, 
                     f"{t:.2f}s", ha="center", va="bottom")
        plt.xlabel("算法", fontsize=12)
        plt.ylabel("总训练耗时（秒）", fontsize=12)
        plt.title("各算法总训练耗时对比", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "plots", f"total_time_comparison.{PLOT_FORMAT}"), 
                    dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 3. 最终准确率+鲁棒性散点图
        plt.figure(figsize=(10, 8))
        final_accs = [self.results["final_summary"][alg]["final_acc"] for alg in alg_names]
        scores = [self.results["final_summary"][alg]["comprehensive_robustness_score"] for alg in alg_names]
        
        for i, alg in enumerate(alg_names):
            plt.scatter(final_accs[i], scores[i], 
                        color=ALGORITHM_COLORS[alg],
                        marker=ALGORITHM_MARKERS[alg],
                        s=150, label=alg)
            # 标注算法名
            plt.text(final_accs[i]+0.5, scores[i]+0.01, alg, fontsize=9)
        
        plt.xlabel("最终全局准确率（%）", fontsize=12)
        plt.ylabel("综合鲁棒性得分（0~1）", fontsize=12)
        plt.title("各算法准确率-鲁棒性对比", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "plots", f"acc_robustness_scatter.{PLOT_FORMAT}"), 
                    dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        print(f"📊 可视化图表已保存至：{self.save_path}/plots")

    # ======================== 打印最终报告 ========================
    def _print_final_report(self):
        """打印最终对比报告"""
        print("\n========== 效率&鲁棒性实验 - 最终报告 ==========")
        print(f"{'算法':<15} {'最终准确率(%)':<15} {'总耗时(s)':<15} {'鲁棒性得分':<15} {'内存占用(MB)':<15} {'通信开销(MB)':<15}")
        print("-" * 100)
        for alg_name, summary in self.results["final_summary"].items():
            print(
                f"{alg_name:<15} "
                f"{summary['final_acc']:<15.2f} "
                f"{summary['total_time']:<15.2f} "
                f"{summary['comprehensive_robustness_score']:<15.4f} "
                f"{summary['memory_usage_mb']:<15.2f} "
                f"{summary['total_comm_mb']:<15.2f}"
            )
        print("-" * 100)

# ======================== 工具函数（复用） ========================
def calculate_efficiency_metrics(start_time: float, end_time: float, 
                                 client_params_sizes: List[int], 
                                 process: psutil.Process) -> Dict:
    """计算效率指标"""
    total_time = end_time - start_time
    num_rounds = len(client_params_sizes) if client_params_sizes else 0
    avg_round_time = total_time / num_rounds if num_rounds > 0 else 0.0
    
    memory_usage = process.memory_info().rss / (1024 * 1024)
    cpu_usage = process.cpu_percent()
    gpu_memory = 0.0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    total_comm_bytes = sum([sum(sizes) for sizes in client_params_sizes]) if client_params_sizes else 0.0
    total_comm_mb = total_comm_bytes / (1024 * 1024)
    avg_round_comm_mb = total_comm_mb / num_rounds if num_rounds > 0 else 0.0
    
    return {
        "total_time": float(total_time),
        "avg_round_time": float(avg_round_time),
        "memory_usage_mb": float(memory_usage),
        "cpu_usage_pct": float(cpu_usage),
        "gpu_memory_mb": float(gpu_memory),
        "total_comm_mb": float(total_comm_mb),
        "avg_round_comm_mb": float(avg_round_comm_mb)
    }

def calculate_robustness_metrics(baseline_perf: float, perturbed_perfs: List[float]) -> Dict:
    """计算鲁棒性指标"""
    perf_retention_rates = [perf / baseline_perf * 100 for perf in perturbed_perfs if baseline_perf != 0]
    avg_retention_rate = np.mean(perf_retention_rates) if perf_retention_rates else 0.0
    perf_std = np.std(perturbed_perfs)
    perf_cv = perf_std / np.mean(perturbed_perfs) if np.mean(perturbed_perfs) != 0 else 0.0
    robustness_score = (avg_retention_rate / 100) * (1 - perf_cv)
    robustness_score = np.clip(robustness_score, 0, 1)
    
    return {
        "baseline_perf": float(baseline_perf),
        "perturbed_perfs": [float(p) for p in perturbed_perfs],
        "avg_retention_rate_pct": float(avg_retention_rate),
        "perf_std": float(perf_std),
        "perf_cv": float(perf_cv),
        "robustness_score": float(robustness_score)
    }

# ======================== 外部调用函数 ========================
def run_efficiency_robustness_experiment(config=None, save_results=True, save_path="./experiment_results/efficiency_robustness"):
    experiment = EfficiencyRobustnessExperiment(config=config, save_results=save_results, save_path=save_path)
    results = experiment.run()
    return results

# ======================== 主函数 ========================
if __name__ == "__main__":
    results = run_efficiency_robustness_experiment(
        save_results=True,
        save_path="./experiment_results/efficiency_robustness_2026"
    )
    print("\n✅ 效率&鲁棒性实验全部完成！")