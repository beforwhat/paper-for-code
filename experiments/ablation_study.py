# experiments/ablation_study.py
"""
FedFairADP-ALA 核心模块消融实验脚本
核心目标：
1. 消融FedFairADP-ALA的5大核心模块，验证每个模块的必要性和收益：
   - 变体1：移除ALA（Adaptive Local Adjustment）→ 本地仅普通SGD更新
   - 变体2：移除伪标签 → 仅用真实标签训练，无高置信伪标签补充
   - 变体3：移除公平选择 → 随机选择客户端，无数据多样性-参与频率筛选
   - 变体4：Shapley聚合→全局平均聚合 → 无边际贡献量化
   - 变体5：自适应裁剪DP→固定裁剪DP → 无Shapley值驱动的裁剪调整
2. 严格遵循单一变量原则：仅关闭目标模块，其余参数/流程与基准版本完全一致；
3. 记录核心指标（性能：准确率/损失；公平性：基尼系数；隐私：ε有效值；效率：耗时）；
4. 输出消融对比报告、量化收益分析和可视化图表，明确各模块的独立贡献。
设计原则：
- 基于你的FederatedTrainer框架，仅修改目标模块逻辑，保证实验一致性；
- 每个变体仅差异目标模块，其余逻辑（如其他模块、训练参数）完全对齐；
- 结果结构化保存，支持量化分析各模块的贡献度。
"""
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# 项目内模块导入
from configs.config_loader import load_config
from core.federated.trainer import FederatedTrainer
from core.federated.server import BaseServer
from core.federated.client import BaseClient
from core.dp.adaptive_clipping_dp import AdaptiveClippingDP
from core.shapley.shapley_calculator import ShapleyCalculator

# 可视化配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
PLOT_FORMAT = "png"
PLOT_DPI = 300
# 颜色映射：基准版本（你的完整方法）vs 消融变体
COLOR_MAP = {
    "基准版本（FedFairADP-ALA）": "#1f77b4",
    "变体1（移除ALA）": "#ff7f0e",
    "变体2（移除伪标签）": "#2ca02c",
    "变体3（移除公平选择）": "#d62728",
    "变体4（平均聚合）": "#9467bd",
    "变体5（固定裁剪DP）": "#8c564b"
}
MARKER_MAP = {
    "基准版本（FedFairADP-ALA）": "o",
    "变体1（移除ALA）": "s",
    "变体2（移除伪标签）": "^",
    "变体3（移除公平选择）": "p",
    "变体4（平均聚合）": "*",
    "变体5（固定裁剪DP）": "D"
}

# ======================== 消融变体定义（核心：单一变量） ========================
# 每个变体仅修改目标模块，其余逻辑与基准版本完全一致
ABLATION_VARIANTS = [
    {
        "name": "基准版本（FedFairADP-ALA）",
        "description": "完整的FedFairADP-ALA（包含所有核心模块）",
        "modify_func": None,  # 无修改
        "focus_metrics": ["performance", "fairness", "privacy", "efficiency"]
    },
    {
        "name": "变体1（移除ALA）",
        "description": "移除自适应局部调整（ALA），本地仅用普通SGD更新",
        "modify_func": "disable_ala",
        "focus_metrics": ["performance", "stability"]
    },
    {
        "name": "变体2（移除伪标签）",
        "description": "移除高置信伪标签补充，仅用真实标签训练",
        "modify_func": "disable_pseudo_label",
        "focus_metrics": ["performance", "data_efficiency"]
    },
    {
        "name": "变体3（移除公平选择）",
        "description": "客户端随机选择，无数据多样性-参与频率筛选",
        "modify_func": "disable_fair_selection",
        "focus_metrics": ["fairness", "performance"]
    },
    {
        "name": "变体4（平均聚合）",
        "description": "Shapley边际贡献聚合 → 全局等权重平均聚合",
        "modify_func": "disable_shapley_aggregate",
        "focus_metrics": ["fairness", "performance"]
    },
    {
        "name": "变体5（固定裁剪DP）",
        "description": "Shapley驱动的自适应裁剪DP → 固定裁剪阈值DP",
        "modify_func": "disable_adaptive_clip_dp",
        "focus_metrics": ["privacy", "performance"]
    }
]

# ======================== 消融变体修改逻辑（核心：单一变量） ========================
class AblationClient(BaseClient):
    """带消融开关的客户端类（仅修改目标模块，其余逻辑与BaseClient一致）"""
    def __init__(self, client_id, config, dataset, ablation_config=None):
        super().__init__(client_id, config, dataset)
        self.ablation_config = ablation_config or {}
        
    # 变体1：移除ALA（自适应局部调整）
    def local_train(self):
        if self.ablation_config.get("disable_ala"):
            # 普通SGD更新（无ALA自适应调整）
            self._local_train_basic_sgd()
        else:
            # 原始ALA自适应局部调整
            super().local_train()
    
    def _local_train_basic_sgd(self):
        """普通SGD本地训练（无ALA）"""
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.config.train.lr)
        criterion = self.config.train.criterion
        
        for epoch in range(self.config.train.local_epochs):
            for data, target in self.dataset.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        self.local_train_loss = loss.item()
    
    # 变体2：移除伪标签
    def _generate_pseudo_labels(self):
        if self.ablation_config.get("disable_pseudo_label"):
            # 不生成伪标签，直接返回空
            return None
        else:
            # 原始伪标签生成逻辑
            return super()._generate_pseudo_labels()

class AblationServer(BaseServer):
    """带消融开关的服务端类（仅修改目标模块，其余逻辑与BaseServer一致）"""
    def __init__(self, config, total_clients, ablation_config=None):
        super().__init__(config, total_clients)
        self.ablation_config = ablation_config or {}
        self.shapley_calculator = ShapleyCalculator(config=config)
        
    # 变体3：移除公平选择（随机选择客户端）
    def select_clients(self, round_idx):
        if self.ablation_config.get("disable_fair_selection"):
            # 随机选择客户端（无公平性筛选）
            num_select = int(self.config.fed.client_selection_ratio * self.total_clients)
            selected_cids = random.sample(range(self.total_clients), num_select)
            return selected_cids
        else:
            # 原始公平选择逻辑（数据多样性+参与频率）
            return super().select_clients(round_idx)
    
    # 变体4：Shapley聚合→平均聚合
    def aggregate_local_results(self):
        if self.ablation_config.get("disable_shapley_aggregate"):
            # 等权重平均聚合
            client_params = [self.client_uploads[cid]["params"] for cid in self.client_uploads.keys()]
            avg_params = self._average_params(client_params)
            return avg_params
        else:
            # 原始Shapley边际贡献聚合
            return super().aggregate_local_results()
    
    def _average_params(self, params_list):
        """等权重平均聚合（无Shapley）"""
        avg_params = []
        for param_tensors in zip(*params_list):
            avg_tensor = torch.mean(torch.stack(param_tensors), dim=0)
            avg_params.append(avg_tensor)
        return avg_params
    
    # 变体5：自适应裁剪DP→固定裁剪DP
    def init_dp_optimizer(self):
        if self.ablation_config.get("disable_adaptive_clip_dp"):
            # 固定裁剪阈值DP
            self.dp_optimizer = AdaptiveClippingDP(config=self.config)
            self.dp_optimizer.adaptive = False  # 关闭自适应
            self.dp_optimizer.clip_threshold = self.config.dp.fixed_clip_threshold  # 固定阈值
        else:
            # 原始Shapley驱动的自适应裁剪DP
            super().init_dp_optimizer()

class AblationFederatedTrainer(FederatedTrainer):
    """带消融配置的联邦训练器（仅替换Server/Client为消融版本）"""
    def __init__(self, config, ablation_config=None):
        super().__init__(config)
        self.ablation_config = ablation_config or {}
    
    # 重写客户端初始化：使用AblationClient
    def init_clients(self):
        logger.info(f"📌 初始化消融实验客户端（配置：{self.ablation_config}）...")
        self.clients = {}
        
        for client_id in range(self.total_clients):
            try:
                client_dataset = get_client_dataset(config=self.config, client_id=client_id)
                # 使用消融客户端
                client = AblationClient(
                    client_id=client_id,
                    config=self.config,
                    dataset=client_dataset,
                    ablation_config=self.ablation_config
                )
                self.clients[client_id] = client
            except Exception as e:
                logger.error(f"❌ 客户端[{client_id}]初始化失败：{str(e)}")
                self.training_metrics["failed_client_ids"].append(client_id)
    
    # 重写服务端初始化：使用AblationServer
    def init_server(self):
        logger.info(f"📌 初始化消融实验服务端（配置：{self.ablation_config}）...")
        try:
            # 使用消融服务端
            self.server = AblationServer(
                config=self.config,
                total_clients=self.total_clients,
                ablation_config=self.ablation_config
            )
            self.global_test_dataloader = get_global_test_dataset(config=self.config).get_dataloader()
        except Exception as e:
            raise RuntimeError(f"服务端初始化失败：{str(e)}") from e

# ======================== 核心实验类 ========================
class FedFairADPAlaAblationExperiment:
    def __init__(self, config=None, save_results=True, save_path="./experiment_results/ablation_study_fedfairadp"):
        self.config = config if config is not None else load_config()
        self.save_results = save_results
        self.save_path = save_path
        self.device = torch.device(self.config.device)
        
        # 创建保存目录
        if self.save_results:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "data"), exist_ok=True)
        
        # 实验结果存储
        self.results = {
            "variant_metrics": {},  # 每个变体的详细指标
            "gain_analysis": {}     # 各模块的收益分析
        }
        
        print(f"✅ FedFairADP-ALA消融实验初始化完成 | 待运行变体数：{len(ABLATION_VARIANTS)}")
        print(f"📌 实验配置：全局轮次={self.config.fed.num_global_rounds} | 客户端数={self.config.fed.num_clients}")

    def _run_variant(self, variant):
        """运行单个消融变体（严格单一变量）"""
        variant_name = variant["name"]
        modify_func = variant["modify_func"]
        print(f"\n--- 运行变体：{variant_name} ---")
        print(f"变体描述：{variant['description']}")
        
        # 1. 构建消融配置（仅设置目标模块的关闭开关）
        ablation_config = {}
        if modify_func:
            ablation_config[modify_func] = True
        
        # 2. 初始化带消融配置的训练器
        trainer = AblationFederatedTrainer(
            config=self.config,
            ablation_config=ablation_config
        )
        
        # 3. 记录开始时间
        start_time = time.time()
        
        # 4. 启动训练（与基准版本流程完全一致）
        trainer.run_federated_training()
        
        # 5. 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        avg_round_time = np.mean(trainer.training_metrics["round_duration"])
        
        # 6. 提取核心指标
        # 性能指标
        global_acc_list = [m["acc"] * 100 for m in trainer.server.global_metrics["round_metrics"]]
        global_loss_list = [m["loss"] for m in trainer.server.global_metrics["round_metrics"]]
        final_global_acc = trainer.server.global_metrics["best_global_acc"] * 100
        final_global_loss = trainer.server.global_metrics["best_global_loss"]
        
        # 公平性指标（基尼系数）
        client_accs = [trainer.clients[cid].evaluate_local_model() for cid in trainer.clients.keys()]
        final_gini = self._calculate_gini(client_accs)
        
        # 隐私指标（DP ε值）
        avg_dp_epsilon = 0.0
        if hasattr(trainer.server, "dp_optimizer"):
            avg_dp_epsilon = trainer.server.dp_optimizer.calculate_epsilon()
        
        # 7. 整理结果
        variant_results = {
            "global_acc": global_acc_list,
            "global_loss": global_loss_list,
            "final_global_acc": final_global_acc,
            "final_global_loss": final_global_loss,
            "final_gini": final_gini,
            "avg_dp_epsilon": avg_dp_epsilon,
            "total_time": total_time,
            "avg_round_time": avg_round_time,
            "best_round": trainer.server.global_metrics["best_round"],
            "client_train_success": sum(trainer.training_metrics["client_train_success"]),
            "description": variant["description"]
        }
        
        print(f"✅ 变体 {variant_name} 运行完成 | 最终准确率：{final_global_acc:.2f}% | 基尼系数：{final_gini:.4f} | 总耗时：{total_time:.2f}s")
        return variant_results

    def _calculate_gini(self, values):
        """计算基尼系数（衡量客户端准确率公平性，越小越公平）"""
        if len(values) == 0:
            return 0.0
        values = np.array(values)
        values = np.sort(values)
        n = len(values)
        if values.sum() == 0:
            return 0.0
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _calculate_module_gain(self, baseline_results, variant_results, focus_metrics):
        """计算单个模块的收益（基准 - 变体，值越大模块贡献越高）"""
        gain = {}
        # 性能收益
        if "performance" in focus_metrics:
            gain["accuracy_gain"] = baseline_results["final_global_acc"] - variant_results["final_global_acc"]
            gain["loss_reduction"] = variant_results["final_global_loss"] - baseline_results["final_global_loss"]
        # 公平性收益
        if "fairness" in focus_metrics:
            gain["gini_reduction"] = variant_results["final_gini"] - baseline_results["final_gini"]
        # 隐私收益（ε越小越好）
        if "privacy" in focus_metrics:
            gain["epsilon_reduction"] = variant_results["avg_dp_epsilon"] - baseline_results["avg_dp_epsilon"]
        # 效率收益
        if "efficiency" in focus_metrics:
            gain["time_reduction"] = variant_results["total_time"] - baseline_results["total_time"]
        return gain

    def run(self):
        """运行所有消融变体，计算模块收益"""
        # 1. 先运行基准版本（必须第一个运行，作为收益计算的基准）
        baseline_variant = ABLATION_VARIANTS[0]
        baseline_results = self._run_variant(baseline_variant)
        self.results["variant_metrics"][baseline_variant["name"]] = baseline_results
        
        # 2. 运行所有消融变体
        for variant in ABLATION_VARIANTS[1:]:
            variant_results = self._run_variant(variant)
            self.results["variant_metrics"][variant["name"]] = variant_results
            
            # 3. 计算该模块的收益
            gain = self._calculate_module_gain(baseline_results, variant_results, variant["focus_metrics"])
            self.results["gain_analysis"][variant["name"]] = gain
        
        # 4. 保存结果+生成可视化
        if self.save_results:
            self._save_results()
            self._generate_plots()
        
        # 5. 输出消融报告
        self._print_ablation_report()
        
        return self.results

    def _save_results(self):
        """保存消融实验结果"""
        # 1. 变体详细指标（JSON）
        metrics_path = os.path.join(self.save_path, "data", "variant_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            # 转换numpy类型为Python原生类型
            serializable_results = {}
            for var_name, metrics in self.results["variant_metrics"].items():
                serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
                serializable_results[var_name] = serializable_metrics
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        # 2. 模块收益分析（CSV）
        gain_df = pd.DataFrame.from_dict(self.results["gain_analysis"], orient="index")
        gain_df.reset_index(inplace=True)
        gain_df.rename(columns={"index": "variant"}, inplace=True)
        gain_path = os.path.join(self.save_path, "data", "gain_analysis.csv")
        gain_df.to_csv(gain_path, index=False, encoding="utf-8")
        
        print(f"\n📁 消融实验结果已保存至：{self.save_path}/data")

    def _generate_plots(self):
        """生成消融实验可视化图表"""
        variants = list(self.results["variant_metrics"].keys())
        rounds = list(range(1, self.config.fed.num_global_rounds + 1))
        
        # 1. 全局准确率收敛曲线（核心对比）
        plt.figure(figsize=(14, 8))
        for var_name in variants:
            metrics = self.results["variant_metrics"][var_name]
            plt.plot(
                rounds, metrics["global_acc"],
                label=var_name,
                color=COLOR_MAP[var_name],
                marker=MARKER_MAP[var_name],
                markersize=6,
                linewidth=2
            )
        plt.xlabel("全局轮次", fontsize=12)
        plt.ylabel("全局准确率（%）", fontsize=12)
        plt.title("FedFairADP-ALA各消融变体准确率收敛对比", fontsize=14, fontweight="bold")
        plt.legend(fontsize=10, loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "accuracy_convergence.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 2. 最终全局准确率对比柱状图
        plt.figure(figsize=(14, 7))
        final_accs = [self.results["variant_metrics"][var]["final_global_acc"] for var in variants]
        colors = [COLOR_MAP[var] for var in variants]
        
        bars = plt.bar(variants, final_accs, color=colors, width=0.6)
        for bar, acc in zip(bars, final_accs):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%",
                ha="center", va="bottom", fontsize=9
            )
        plt.xlabel("消融变体", fontsize=12)
        plt.ylabel("最终全局准确率（%）", fontsize=12)
        plt.title("FedFairADP-ALA各消融变体最终准确率对比", fontsize=14, fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "final_accuracy_comparison.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        # 3. 公平性（基尼系数）对比
        plt.figure(figsize=(14, 7))
        gini_values = [self.results["variant_metrics"][var]["final_gini"] for var in variants]
        bars = plt.bar(variants, gini_values, color=colors, width=0.6)
        
        for bar, gini in zip(bars, gini_values):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{gini:.4f}",
                ha="center", va="bottom", fontsize=9
            )
        plt.xlabel("消融变体", fontsize=12)
        plt.ylabel("最终基尼系数（越小越公平）", fontsize=12)
        plt.title("FedFairADP-ALA各消融变体公平性对比", fontsize=14, fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(self.save_path, "plots", "gini_coefficient_comparison.png")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        
        print(f"📊 消融实验可视化图表已保存至：{self.save_path}/plots")

    def _print_ablation_report(self):
        """打印消融实验最终报告（量化各模块贡献）"""
        print("\n" + "="*100)
        print("FedFairADP-ALA 核心模块消融实验 - 最终报告")
        print("="*100)
        print(f"{'变体名称':<25} {'最终准确率(%)':<15} {'基尼系数':<15} {'DP ε值':<15} {'总耗时(s)':<15} {'模块收益(%)':<15}")
        print("-"*100)
        
        # 基准版本结果
        baseline_name = ABLATION_VARIANTS[0]["name"]
        baseline_acc = self.results["variant_metrics"][baseline_name]["final_global_acc"]
        
        for var_name in variants:
            if var_name == baseline_name:
                module_gain = "基准"
            else:
                module_gain = f"{baseline_acc - self.results['variant_metrics'][var_name]['final_global_acc']:.2f}"
            
            metrics = self.results["variant_metrics"][var_name]
            print(
                f"{var_name:<25} "
                f"{metrics['final_global_acc']:<15.2f} "
                f"{metrics['final_gini']:<15.4f} "
                f"{metrics['avg_dp_epsilon']:<15.2f} "
                f"{metrics['total_time']:<15.2f} "
                f"{module_gain:<15}"
            )
        
        print("-"*100)
        print("模块收益说明：")
        print("1. 模块收益 = 基准版本准确率 - 消融变体准确率 → 值越大，该模块对性能的贡献越高；")
        print("2. 基尼系数越小 → 客户端准确率分布越公平；")
        print("3. DP ε值越小 → 隐私保护效果越好；")
        print("4. 所有变体仅修改目标模块，其余逻辑与基准版本完全一致，保证单一变量原则。")
        print("="*100)

# ======================== 外部调用函数 ========================
def run_fedfairadp_ala_ablation(config=None, save_results=True, save_path="./experiment_results/ablation_study_fedfairadp"):
    """运行FedFairADP-ALA核心模块消融实验"""
    experiment = FedFairADPAlaAblationExperiment(config=config, save_results=save_results, save_path=save_path)
    results = experiment.run()
    return results

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 补充缺失的导入（与你的trainer保持一致）
    from datasets import get_client_dataset, get_global_test_dataset
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AblationExperiment")
    
    # 运行消融实验
    results = run_fedfairadp_ala_ablation(
        save_results=True,
        save_path="./experiment_results/ablation_study_fedfairadp_2026"
    )
    print("\n✅ FedFairADP-ALA核心模块消融实验全部完成！")