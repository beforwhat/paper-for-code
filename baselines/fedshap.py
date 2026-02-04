# baselines/fedshap.py
"""
FedShap算法实现（结合Shapley值的联邦学习基线）
核心定位：基于Shapley值计算客户端贡献度，加权聚合模型参数（替代FedAvg等权重）
核心逻辑：
1. 客户端（FedShapClient）：完全复用FedAvgClient的本地训练逻辑，仅上传模型参数+本地性能指标（供Shapley计算）；
2. 服务端（FedShapServer）：
   - 新增Shapley贡献度计算：基于客户端本地训练损失/准确率，计算每个客户端对全局模型的贡献度；
   - 加权聚合：用归一化后的Shapley值作为权重，替代FedAvg的等权重平均；
设计原则：
- 客户端无修改（保证与FedAvg的唯一差异是服务端聚合权重）；
- 复用项目Shapley计算模块，避免重复造轮子；
- 接口与FedAvg对齐，便于对比“等权重”与“贡献度加权”的聚合效果。
"""
import torch
import numpy as np
from tqdm import tqdm

# 项目内模块导入
from baselines.fedavg import FedAvgClient, FedAvgServer  # 复用基础FedAvg
from core.shapley.shapley_calculator import ShapleyCalculator  # 复用Shapley计算模块
from configs.config_loader import load_config

class FedShapClient(FedAvgClient):
    """
    FedShap客户端：完全复用FedAvgClient的本地训练逻辑
    核心：仅新增返回本地性能指标（损失/样本数），供服务端计算Shapley贡献度
    """
    def __init__(self, client_id: int, config=None):
        """
        初始化FedShap客户端（完全复用FedAvgClient）
        Args:
            client_id: 客户端唯一标识
            config: 配置对象（需包含shapley相关配置）
        """
        super().__init__(client_id=client_id, config=config)
        # 记录本地性能指标（供Shapley计算）
        self.local_samples_num = len(self.local_dataloader.dataset)  # 本地样本数
        self.local_acc = 0.0  # 本地训练后模型准确率
        
        print(f"✅ FedShap客户端 [{self.client_id}] 初始化完成（复用FedAvg训练，新增Shapley指标记录）")

    def local_train(self):
        """
        复用FedAvgClient的local_train，仅新增本地准确率计算（供Shapley贡献度评估）
        """
        # 1. 执行FedAvg的本地训练逻辑
        client_params = super().local_train()
        
        # 2. 计算本地训练后模型的准确率（供Shapley计算贡献度）
        self.local_acc = self.evaluate_local_model()
        print(f"📌 FedShap客户端 [{self.client_id}] 本地准确率：{self.local_acc:.2f}% | 样本数：{self.local_samples_num}")
        
        # 3. 返回参数+本地指标（扩展返回值，供服务端计算Shapley）
        return {
            "params": client_params,
            "loss": self.local_train_loss,
            "acc": self.local_acc,
            "samples_num": self.local_samples_num,
            "client_id": self.client_id
        }

class FedShapServer(FedAvgServer):
    """
    FedShap服务端：核心修改聚合逻辑，基于Shapley贡献度加权聚合
    替代FedAvg的等权重平均，解决“贡献不均导致的聚合低效”问题
    """
    def __init__(self, config=None):
        """
        初始化FedShap服务端（复用FedAvgServer，新增Shapley计算器）
        Args:
            config: 配置对象（需包含shapley计算参数：metric、normalization等）
        """
        super().__init__(config=config)
        
        # ========== 初始化Shapley计算器（核心新增） ==========
        self.shapley_calculator = ShapleyCalculator(
            metric=self.config.shapley.metric,  # 贡献度评估指标：loss/acc/samples
            normalization=self.config.shapley.normalization  # 权重归一化方式：min-max/softmax
        )
        # 记录历史Shapley权重（便于跟踪贡献度变化）
        self.history_shapley_weights = []
        
        print(f"✅ FedShap服务端初始化完成（Shapley评估指标：{self.config.shapley.metric}）")

    def calculate_shapley_weights(self, client_results_list: list):
        """
        核心：计算每个客户端的Shapley贡献度，并归一化为聚合权重
        Args:
            client_results_list: 客户端返回的参数+指标列表
        Returns:
            shapley_weights: 归一化后的Shapley权重字典 {client_id: weight}
        """
        # 1. 提取Shapley计算所需的客户端指标
        client_metrics = {}
        for client_result in client_results_list:
            cid = client_result["client_id"]
            # 根据配置的metric选择评估指标（loss/acc/samples）
            if self.config.shapley.metric == "loss":
                client_metrics[cid] = client_result["loss"]  # 损失越小，贡献度越高
                higher_better = False  # loss是越小越好
            elif self.config.shapley.metric == "acc":
                client_metrics[cid] = client_result["acc"]  # 准确率越高，贡献度越高
                higher_better = True
            elif self.config.shapley.metric == "samples":
                client_metrics[cid] = client_result["samples_num"]  # 样本数越多，贡献度越高
                higher_better = True
            else:
                raise ValueError(f"不支持的Shapley指标：{self.config.shapley.metric}，可选：loss/acc/samples")
        
        # 2. 调用Shapley计算器计算贡献度
        shapley_values = self.shapley_calculator.calculate(
            client_metrics=client_metrics,
            higher_better=higher_better
        )
        
        # 3. 归一化Shapley值为聚合权重（确保权重和为1）
        shapley_weights = self.shapley_calculator.normalize_weights(
            shapley_values=shapley_values,
            method=self.config.shapley.normalization
        )
        
        # 4. 记录历史权重（便于分析）
        self.history_shapley_weights.append(shapley_weights)
        
        # 打印权重分布（便于调试）
        print(f"\n📌 FedShap服务端 Shapley权重分布：")
        for cid, weight in shapley_weights.items():
            print(f"   客户端 [{cid}]：权重={weight:.4f}")
        
        return shapley_weights

    def aggregate_local_results(self, client_results_list: list, client_ids: list = None):
        """
        重写聚合逻辑：基于Shapley权重加权聚合，替代FedAvg的等权重平均
        Args:
            client_results_list: 客户端返回的参数+指标列表
            client_ids: 客户端ID列表（兼容接口，无实际作用）
        Returns:
            global_params: 加权聚合后的全局模型参数
        """
        # 前置检查：无客户端结果则返回当前全局参数
        if not client_results_list:
            print("⚠️  无客户端参数可聚合，返回当前全局参数")
            return self.get_model_parameters()

        # ========== 步骤1：计算Shapley贡献度权重 ==========
        shapley_weights = self.calculate_shapley_weights(client_results_list)

        # ========== 步骤2：Shapley加权聚合参数 ==========
        print(f"\n📌 FedShap服务端开始聚合 | 参与客户端数：{len(client_results_list)} | 聚合策略：Shapley加权")
        # 初始化聚合参数（以第一个客户端参数为模板）
        global_params = {}
        first_client_params = client_results_list[0]["params"]
        client_num = len(client_results_list)

        # 遍历每个参数名，加权累加所有客户端的该参数
        for param_name, param_tensor in first_client_params.items():
            # 初始化参数累加器
            param_sum = torch.zeros_like(param_tensor, device=self.device)
            # 加权累加：Σ (shapley_weight_i * client_params_i)
            for client_result in client_results_list:
                cid = client_result["client_id"]
                weight = shapley_weights[cid]
                client_param = client_result["params"][param_name].to(self.device)
                param_sum += weight * client_param
            # 加权聚合结果作为全局参数
            global_params[param_name] = param_sum

        # ========== 步骤3：更新全局模型参数 ==========
        self.set_model_parameters(global_params)
        print(f"✅ FedShap服务端聚合完成 | 全局模型参数已更新（Shapley加权替代等权重）")

        return global_params

# ======================== 独立测试示例（验证FedShap功能） ========================
if __name__ == "__main__":
    """
    测试FedShap核心逻辑：客户端返回训练指标 → 服务端计算Shapley权重 → 加权聚合
    对比FedAvg：仅聚合权重从“等权重”变为“Shapley贡献度加权”
    """
    # 1. 加载配置
    config = load_config()
    # 测试用配置
    config.fed.num_clients = 3
    config.fed.local_epochs = 1
    config.fed.local_lr = 0.01
    config.shapley.metric = "acc"  # 基于准确率计算Shapley贡献度
    config.shapley.normalization = "softmax"  # softmax归一化权重

    # 2. 初始化FedShap服务端
    fedshap_server = FedShapServer(config=config)

    # 3. 初始化FedShap客户端
    client_list = []
    for client_id in range(config.fed.num_clients):
        client = FedShapClient(client_id=client_id, config=config)
        client_list.append(client)

    # 4. 模拟一轮FedShap联邦训练
    print("\n=== 模拟FedShap一轮联邦训练 ===")
    # 4.1 客户端本地训练（返回参数+指标）
    client_results_list = []
    for client in client_list:
        client_result = client.local_train()
        client_results_list.append(client_result)

    # 4.2 服务端Shapley加权聚合
    fedshap_server.aggregate_local_results(client_results_list=client_results_list)

    # 4.3 打印结果
    print("\n=== FedShap一轮训练完成 ===")
    print(f"服务端全局模型参数示例（conv1.weight.shape）：{fedshap_server.global_model.conv1.weight.shape}")
    print(f"历史Shapley权重：{fedshap_server.history_shapley_weights[-1]}")