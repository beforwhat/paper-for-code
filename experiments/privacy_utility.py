# -*- coding: utf-8 -*-
"""
隐私-效用权衡实验（验证FedFairADP-ALA的隐私优势）
核心对比：FedFairADP-ALA vs DP-FedAvg vs FedAdaClip++（2024新基线）
实验变量：隐私预算ε（0.1/1/5/10/20）、DP方法
实验目标：强隐私下（ε=0.1）准确率领先DP-FedAvg≥8%，DLG攻击成功率≤10%
"""
import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
from copy import deepcopy

# 核心导入：严格对齐实际实现（删除不存在的导入）
from core.federated.server import BaseServer as FedFairADP_ALA_Server  # 本项目核心
from core.federated.trainer import FederatedTrainer as FedFairADP_ALA_Trainer
from baselines.dp_fedavg import DPFedAvgClient, DPFedAvgServer  
from baselines.fedadaclip import FedAdaClipPlusCore, FedAdaptiveDPClient, FedAdaptiveDPServer  # FedAdaClip++原始类
from datasets import get_global_test_dataset, get_client_dataset
from utils.logger import info, error, setup_global_logger
from utils.metrics import calculate_accuracy, dlg_attack
from configs.dp_config import get_dp_config
from configs.fed_config import get_fed_config
from configs.experiment_config import get_experiment_config

# ======================== 仅修改：补充DP-FedAvg训练器（极简适配原始代码） ========================
DP_CONFIG = get_dp_config()
FED_CONFIG = get_fed_config()
EXP_CONFIG = get_experiment_config()
class DPFedAvgTrainer:
    """
    适配DP-FedAvg原始代码的极简训练器（仅封装，不修改核心逻辑）
    完全保留原有接口，保证外部调用兼容
    """
    def __init__(self, config, dp_config, dataset_name, non_iid_alpha, model_name, device):
        self.config = config
        self.dp_config = dp_config
        self.dataset_name = dataset_name
        self.non_iid_alpha = non_iid_alpha
        self.model_name = model_name
        self.device = device
        
        # 初始化DP-FedAvg服务端（完全复用原始代码）
        self.server = DPFedAvgServer(config=config)
        # 初始化DP-FedAvg客户端列表（完全复用原始代码）
        self.clients = [
            DPFedAvgClient(client_id=i, config=config) 
            for i in range(config["num_clients"])
        ]
        
    def train(self, num_rounds):
        """DP-FedAvg联邦训练循环（完全复用原始代码逻辑）"""
        for round_idx in range(num_rounds):
            info(f"\n=== DP-FedAvg 第 {round_idx+1}/{num_rounds} 轮训练 ===")
            # 1. 客户端本地训练（带固定DP，复用原始local_train）
            client_params_list = []
            for client in self.clients:
                params = client.local_train()
                client_params_list.append(params)
            # 2. 服务端聚合（复用原始aggregate_local_results）
            self.server.aggregate_local_results(client_params_list)

# ======================== 仅修改：补充FedAdaClip++训练器（适配原始代码） ========================
class FedAdaptiveDPTrainer:
    """
    FedAdaClip++训练器（仅适配原始类，不修改核心逻辑）
    完全保留原有接口，保证外部调用兼容
    """
    def __init__(self, config, dp_config, dataset_name, non_iid_alpha, model_name, device):
        self.config = config
        self.dp_config = dp_config
        self.dataset_name = dataset_name
        self.non_iid_alpha = non_iid_alpha
        self.model_name = model_name
        self.device = device
        
        # 1. 初始化全局模型（复用项目逻辑，保证接口对齐）
        from core.models import build_model  # 项目原有模型构建函数
        self.global_model = build_model(model_name, dataset_name).to(device)
        
        # 2. 初始化FedAdaClip++ Server（复用原始类）
        self.server = FedAdaptiveDPServer(
            model=self.global_model,
            num_clients=config["num_clients"],
            device=device,
            dp_config=dp_config
        )
        
        # 3. 初始化客户端（复用DP-FedAvgClient，保证基线一致）
        self.clients = []
        for client_id in range(config["num_clients"]):
            # 获取客户端数据加载器（复用项目原有逻辑）
            train_loader, test_loader = get_client_dataset(
                client_id=client_id,
                dataset_name=dataset_name,
                non_iid_alpha=non_iid_alpha,
                num_clients=config["num_clients"]
            )
            # 复用DP-FedAvgClient（原始代码无修改）
            client = DPFedAvgClient(
                client_id=client_id,
                model=deepcopy(self.global_model),
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                dp_config=dp_config
            )
            self.clients.append(client)
        
        # 4. 传递客户端加载器（FedAdaClip++专属，不影响接口）
        self.server.set_client_loaders([c.train_loader for c in self.clients])
        # 绑定server.model，保证与其他训练器接口一致
        self.server.model = self.global_model
    
    def train(self, num_rounds):
        """FedAdaClip++训练循环（适配原始聚合逻辑，保留接口）"""
        for round_idx in range(num_rounds):
            info(f"\n=== FedAdaClip++ 第 {round_idx+1}/{num_rounds} 轮训练 ===")
            # 1. 客户端本地训练（复用DP-FedAvg原始local_train）
            client_models = []
            for client in self.clients:
                client.model.load_state_dict(self.global_model.state_dict())
                client.local_train(local_epochs=self.config["local_epochs"])
                client_models.append(deepcopy(client.model))
            
            # 2. 服务端聚合（复用FedAdaptiveDPServer原始aggregate）
            client_weights = [1/len(self.clients)] * len(self.clients)
            self.global_model = self.server.aggregate(client_models, client_weights)
            
            # 3. 同步全局模型（保证与其他训练器接口一致）
            self.server.model = self.global_model

# ======================== 完全保留：实验核心配置（无修改） ========================
PRIVACY_UTILITY_CONFIG = {
    "epsilon_list": [0.1, 1, 5, 10, 20],
    "dp_methods": ["FedFairADP-ALA", "DP-FedAvg", "FedAdaClip++"],
    "dataset_name": "cifar10",
    "non_iid_alpha": 0.1,
    "num_clients": 10,
    "num_rounds": 50,
    "local_epochs": 5,
    "model_name": "vgg11",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir": "./results/privacy_utility",
    "save_csv": True,
    "save_plot": True,
    "seed": 42,
}

# ======================== 完全保留：实验初始化工具（无修改） ========================
def setup_experiment():
    np.random.seed(PRIVACY_UTILITY_CONFIG["seed"])
    torch.manual_seed(PRIVACY_UTILITY_CONFIG["seed"])
    if PRIVACY_UTILITY_CONFIG["device"] == torch.device("cuda"):
        torch.cuda.manual_seed_all(PRIVACY_UTILITY_CONFIG["seed"])
    
    os.makedirs(PRIVACY_UTILITY_CONFIG["save_dir"], exist_ok=True)
    
    log_path = os.path.join(PRIVACY_UTILITY_CONFIG["save_dir"], "privacy_utility_exp.log")
    setup_global_logger(log_path)
    info(f"实验初始化完成 | 配置：{json.dumps(PRIVACY_UTILITY_CONFIG, indent=2)}")

# ======================== 仅修改：单方法实验函数（仅调整Trainer初始化，保留其他逻辑） ========================
def run_single_method(
    dp_method: str,
    epsilon: float,
    config: Dict = PRIVACY_UTILITY_CONFIG
) -> Dict:
    info(f"\n===== 运行 {dp_method} | ε={epsilon} =====")
    
    # 1. 适配DP配置（无修改）
    dp_config = DP_CONFIG.copy()
    dp_config["epsilon"] = epsilon
    dp_config["noise_multiplier"] = {0.1: 8.0, 1: 4.0, 5: 1.0, 10: 0.5, 20: 0.2}[epsilon]
    dp_config["fixed_clip_norm"] = 1.0
    
    # 2. 初始化训练器（仅修改此处，适配原始类，保留参数和接口）
    if dp_method == "FedFairADP-ALA":
        trainer = FedFairADP_ALA_Trainer(
            fed_config=FED_CONFIG,
            dp_config=dp_config,
            dataset_name=config["dataset_name"],
            non_iid_alpha=config["non_iid_alpha"],
            model_name=config["model_name"],
            device=config["device"]
        )
    elif dp_method == "DP-FedAvg":
        # 仅适配DP-FedAvg原始Trainer，参数完全对齐
        trainer = DPFedAvgTrainer(
            config=FED_CONFIG,
            dp_config=dp_config,
            dataset_name=config["dataset_name"],
            non_iid_alpha=config["non_iid_alpha"],
            model_name=config["model_name"],
            device=config["device"]
        )
    elif dp_method == "FedAdaClip++":
        # 仅适配FedAdaClip++原始类，参数完全对齐
        trainer = FedAdaptiveDPTrainer(
            config=FED_CONFIG,
            dp_config=dp_config,
            dataset_name=config["dataset_name"],
            non_iid_alpha=config["non_iid_alpha"],
            model_name=config["model_name"],
            device=config["device"]
        )
    else:
        raise ValueError(f"不支持的DP方法：{dp_method}")
    
    # 3. 训练模型（无修改）
    trainer.train(num_rounds=config["num_rounds"])
    
    # 4. 评估效用（无修改）
    test_loader = get_global_test_dataset(
        dataset_name=config["dataset_name"],
        num_clients=config["num_clients"]
    )
    global_acc = calculate_accuracy(trainer.server.model, test_loader, config["device"])
    info(f"{dp_method} | ε={epsilon} | 全局准确率：{global_acc:.2f}%")
    
    # 5. 评估隐私（无修改）
    client_loader = get_client_dataset(client_id=0, **config)
    dlg_success_rate = dlg_attack(
        target_model=trainer.server.model,
        client_loader=client_loader,
        device=config["device"],
        num_attack_trials=100
    )
    info(f"{dp_method} | ε={epsilon} | DLG攻击成功率：{dlg_success_rate:.2f}%")
    
    return {
        "dp_method": dp_method,
        "epsilon": epsilon,
        "global_acc": global_acc,
        "dlg_success_rate": dlg_success_rate
    }

# ======================== 完全保留：实验主函数及后续所有逻辑（无修改） ========================
def run_privacy_utility_experiment() -> Dict:
    setup_experiment()
    all_results = []
    
    for dp_method in PRIVACY_UTILITY_CONFIG["dp_methods"]:
        method_results = []
        for epsilon in tqdm(PRIVACY_UTILITY_CONFIG["epsilon_list"], desc=f"Running {dp_method}"):
            res = run_single_method(dp_method, epsilon)
            method_results.append(res)
        
        # 计算准确率保留率
        base_acc = [r["global_acc"] for r in method_results if r["epsilon"] == 20][0]
        for res in method_results:
            res["acc_retention_rate"] = (res["global_acc"] / base_acc) * 100
            info(f"{dp_method} | ε={res['epsilon']} | 准确率保留率：{res['acc_retention_rate']:.2f}%")
        
        all_results.extend(method_results)
    
    save_results(all_results)
    plot_results(all_results)
    verify_experiment_goals(all_results)
    
    return all_results

def save_results(all_results: List[Dict]):
    save_dir = PRIVACY_UTILITY_CONFIG["save_dir"]
    
    csv_path = os.path.join(save_dir, "privacy_utility_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dp_method", "epsilon", "global_acc", "acc_retention_rate", "dlg_success_rate"])
        writer.writeheader()
        writer.writerows(all_results)
    info(f"量化指标已保存至 {csv_path}")
    
    json_path = os.path.join(save_dir, "privacy_utility_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    info(f"完整结果已保存至 {json_path}")

def plot_results(all_results: List[Dict]):
    save_dir = PRIVACY_UTILITY_CONFIG["save_dir"]
    dp_methods = PRIVACY_UTILITY_CONFIG["dp_methods"]
    epsilon_list = PRIVACY_UTILITY_CONFIG["epsilon_list"]
    
    # 隐私-效用曲线
    plt.figure(figsize=(10, 6))
    for dp_method in dp_methods:
        method_acc = [r["global_acc"] for r in all_results if r["dp_method"] == dp_method]
        plt.plot(epsilon_list, method_acc, marker="o", label=dp_method)
    plt.xlabel("Privacy Budget ε (Smaller = Stronger Privacy)")
    plt.ylabel("Global Accuracy (%)")
    plt.title("Privacy-Utility Trade-off Curve (CIFAR-10, α=0.1)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "privacy_utility_curve.png"))
    plt.close()
    info("隐私-效用曲线已保存")
    
    # DLG攻击成功率柱状图
    plt.figure(figsize=(8, 5))
    dlg_rates = [
        [r["dlg_success_rate"] for r in all_results if r["dp_method"] == m and r["epsilon"] == 0.1][0]
        for m in dp_methods
    ]
    sns.barplot(x=dp_methods, y=dlg_rates)
    plt.xlabel("DP Method")
    plt.ylabel("DLG Attack Success Rate (%)")
    plt.title("DLG Attack Success Rate (ε=0.1, Strong Privacy)")
    plt.axhline(y=10, color="r", linestyle="--", label="Target (≤10%)")
    plt.axhline(y=30, color="orange", linestyle="--", label="DP-FedAvg Target (≥30%)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "dlg_attack_rate.png"))
    plt.close()
    info("DLG攻击成功率柱状图已保存")

def verify_experiment_goals(all_results: List[Dict]):
    info("\n===== 验证实验目标 =====")
    eps_01_results = [r for r in all_results if r["epsilon"] == 0.1]
    
    # 提取所有方法的核心指标
    fedfair_acc = [r["global_acc"] for r in eps_01_results if r["dp_method"] == "FedFairADP-ALA"][0]
    dpfedavg_acc = [r["global_acc"] for r in eps_01_results if r["dp_method"] == "DP-FedAvg"][0]
    fedadaclip_acc = [r["global_acc"] for r in eps_01_results if r["dp_method"] == "FedAdaClip++"][0]
    
    fedfair_dlg = [r["dlg_success_rate"] for r in eps_01_results if r["dp_method"] == "FedFairADP-ALA"][0]
    dpfedavg_dlg = [r["dlg_success_rate"] for r in eps_01_results if r["dp_method"] == "DP-FedAvg"][0]
    fedadaclip_dlg = [r["dlg_success_rate"] for r in eps_01_results if r["dp_method"] == "FedAdaClip++"][0]
    
    # 验证本项目vs DP-FedAvg
    acc_gap = fedfair_acc - dpfedavg_acc
    info(f"FedFairADP-ALA vs DP-FedAvg 准确率差距：{acc_gap:.2f}%")
    if acc_gap >= 8:
        info("✅ 准确率领先目标达成（≥8%）")
    else:
        info("❌ 准确率领先目标未达成（需≥8%）")
    
    # 验证本项目vs FedAdaClip++
    acc_gap_2 = fedfair_acc - fedadaclip_acc
    info(f"FedFairADP-ALA vs FedAdaClip++ 准确率差距：{acc_gap_2:.2f}%")
    
    # 验证DLG攻击成功率
    info(f"FedFairADP-ALA DLG攻击成功率：{fedfair_dlg:.2f}%")
    if fedfair_dlg <= 10:
        info("✅ DLG攻击成功率目标达成（≤10%）")
    else:
        info("❌ DLG攻击成功率目标未达成（需≤10%）")
    
    info(f"DP-FedAvg DLG攻击成功率：{dpfedavg_dlg:.2f}%")
    if dpfedavg_dlg >= 30:
        info("✅ DP-FedAvg DLG攻击成功率目标达成（≥30%）")
    else:
        info("❌ DP-FedAvg DLG攻击成功率目标未达成（需≥30%）")
    
    info(f"FedAdaClip++ DLG攻击成功率：{fedadaclip_dlg:.2f}%")

if __name__ == "__main__":
    results = run_privacy_utility_experiment()
    info("\n===== 隐私-效用权衡实验完成 =====")