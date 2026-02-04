# baselines/fedavg.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 项目内基础组件导入（复用核心基类）
from core.federated.server import BaseServer
from core.federated.client import BaseClient
from configs.config_loader import load_config

class FedAvgClient(BaseClient):
    """
    FedAvg客户端：纯本地训练，无任何特殊优化
    仅实现最基础的本地训练逻辑，作为基准对比
    """
    def __init__(self, client_id: int, config=None):
        """
        初始化FedAvg客户端（复用BaseClient的初始化逻辑）
        Args:
            client_id: 客户端唯一标识
            config: 配置对象（默认加载全局配置）
        """
        super().__init__(client_id=client_id, config=config)
        # FedAvg无额外初始化，仅打印标识（便于日志区分）
        print(f"✅ FedAvg客户端 [{self.client_id}] 初始化完成（纯本地训练，无特殊优化）")

    def local_train(self):
        """
        重写本地训练逻辑：纯基础训练，无DP、无ALA、无伪标签、无任何优化
        核心：前向传播→计算损失→反向传播→梯度下降
        """
        # 1. 初始化训练环境（复用BaseClient的模型/数据/优化器）
        self.local_model.train()
        # FedAvg使用基础SGD优化器（保证基准纯粹性）
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.config.fed.local_lr,
            momentum=self.config.fed.momentum
        )
        loss_fn = F.cross_entropy if self.config.model.num_classes > 2 else F.binary_cross_entropy_with_logits

        # 2. 本地训练循环（纯基础逻辑）
        for epoch in range(self.config.fed.local_epochs):
            epoch_loss = 0.0
            total_samples = 0
            pbar = tqdm(self.local_dataloader, desc=f"FedAvg客户端 [{self.client_id}] 训练Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                # 数据迁移到指定设备
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = loss_fn(outputs, labels)
                
                # 反向传播 + 梯度下降（纯基础逻辑，无任何修改）
                loss.backward()
                optimizer.step()
                
                # 统计损失
                epoch_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                pbar.set_postfix({"batch_loss": loss.item(), "avg_loss": epoch_loss/total_samples})

        # 3. 训练完成，记录本地训练损失（用于日志）
        self.local_train_loss = epoch_loss / total_samples
        print(f"\n📌 FedAvg客户端 [{self.client_id}] 本地训练完成 | 平均损失：{self.local_train_loss:.4f}")

        # 4. 返回本地模型参数（供服务端聚合）
        return self.get_model_parameters()

class FedAvgServer(BaseServer):
    """
    FedAvg服务端：等权重平均所有客户端参数，无加权聚合
    仅实现最基础的参数平均逻辑，作为基准对比
    """
    def __init__(self, config=None):
        """
        初始化FedAvg服务端（复用BaseServer的初始化逻辑）
        Args:
            config: 配置对象（默认加载全局配置）
        """
        super().__init__(config=config)
        # FedAvg无额外初始化，仅打印标识
        print(f"✅ FedAvg服务端初始化完成（等权重参数平均，无加权聚合）")

    def aggregate_local_results(self, client_params_list: list, client_ids: list = None):
        """
        重写聚合逻辑：FedAvg核心——等权重平均所有客户端上传的参数
        Args:
            client_params_list: 客户端参数列表（每个元素是{param_name: param_tensor}）
            client_ids: 客户端ID列表（FedAvg中无作用，仅兼容接口）
        Returns:
            global_params: 聚合后的全局模型参数
        """
        # 前置检查：无客户端参数则返回当前全局参数
        if not client_params_list:
            print("⚠️  无客户端参数可聚合，返回当前全局参数")
            return self.get_model_parameters()

        # 1. 初始化聚合参数（以第一个客户端参数为模板）
        global_params = {}
        first_client_params = client_params_list[0]
        client_num = len(client_params_list)  # 客户端数量（等权重分母）

        # 2. 等权重平均所有客户端的每个参数
        print(f"\n📌 FedAvg服务端开始聚合 | 参与客户端数：{client_num} | 聚合策略：等权重平均")
        for param_name, param_tensor in first_client_params.items():
            # 初始化参数累加器
            param_sum = torch.zeros_like(param_tensor, device=self.device)
            # 累加所有客户端的该参数
            for client_params in client_params_list:
                param_sum += client_params[param_name].to(self.device)
            # 等权重平均
            global_params[param_name] = param_sum / client_num

        # 3. 更新全局模型参数
        self.set_model_parameters(global_params)
        print(f"✅ FedAvg服务端聚合完成 | 全局模型参数已更新")

        return global_params

# ======================== 独立测试示例（便于验证功能） ========================
if __name__ == "__main__":
    """
    测试FedAvg的核心逻辑：服务端初始化→客户端本地训练→服务端聚合
    """
    # 1. 加载配置
    config = load_config()
    config.fed.num_clients = 3  # 测试用客户端数
    config.fed.local_epochs = 2  # 测试用本地训练轮次
    config.fed.local_lr = 0.01   # 测试用学习率

    # 2. 初始化FedAvg服务端
    fedavg_server = FedAvgServer(config=config)

    # 3. 初始化多个FedAvg客户端
    client_list = []
    for client_id in range(config.fed.num_clients):
        client = FedAvgClient(client_id=client_id, config=config)
        client_list.append(client)

    # 4. 模拟联邦训练一轮
    print("\n=== 模拟FedAvg一轮联邦训练 ===")
    # 4.1 客户端本地训练
    client_params_list = []
    for client in client_list:
        client_params = client.local_train()
        client_params_list.append(client_params)

    # 4.2 服务端聚合
    fedavg_server.aggregate_local_results(client_params_list=client_params_list)

    # 4.3 打印结果
    print("\n=== FedAvg一轮训练完成 ===")
    print(f"服务端全局模型参数示例（conv1.weight.shape）：{fedavg_server.global_model.conv1.weight.shape}")
    for idx, client in enumerate(client_list):
        print(f"客户端 [{idx}] 本地训练损失：{client.local_train_loss:.4f}")