# baselines/fedprox.py
"""
FedProx算法实现（解决数据/模型异构性的经典联邦基线）
核心定位：FedAvg基础上新增近端正则项，缓解异构场景下的训练不稳定问题
核心逻辑：
1. 客户端（FedProxClient）：继承FedAvgClient，仅在损失函数中加入近端项（Proximal Term）；
2. 服务端（FedProxServer）：完全复用FedAvgServer的等权重聚合逻辑，无任何修改；
设计原则：
- 仅新增近端项相关逻辑，其余完全复用FedAvg（保证与基础FedAvg的唯一差异是近端正则）；
- 近端系数μ可配置，适配不同异构程度的场景；
- 接口与FedAvg完全对齐，便于公平对比（仅多μ配置项）。
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

# 项目内模块导入
from baselines.fedavg import FedAvgClient, FedAvgServer  # 复用基础FedAvg
from configs.config_loader import load_config

class FedProxClient(FedAvgClient):
    """
    FedProx客户端：继承FedAvgClient，核心修改是损失函数加入近端正则项
    近端项作用：约束本地模型参数不偏离全局模型参数过远，缓解异构场景下的训练震荡
    """
    def __init__(self, client_id: int, config=None):
        """
        初始化FedProx客户端（复用FedAvgClient初始化，新增近端系数μ）
        Args:
            client_id: 客户端唯一标识
            config: 配置对象（需包含fedprox相关配置：mu（近端系数））
        """
        super().__init__(client_id=client_id, config=config)
        
        # FedProx核心超参数：近端系数μ（越大，约束越强，适配高异构场景）
        self.mu = self.config.fedprox.mu
        # 保存服务端下发的全局模型参数（用于计算近端项）
        self.global_model_params = None

        print(f"✅ FedProx客户端 [{self.client_id}] 初始化完成（基于FedAvg + 近端正则项）")
        print(f"📌 FedProx配置：近端系数μ={self.mu}（μ越大，本地参数约束越强）")

    def set_global_model_params(self, global_params: dict):
        """
        新增方法：接收服务端下发的全局模型参数（用于计算近端项）
        适配BaseClient接口，服务端分发全局模型时调用
        """
        self.global_model_params = global_params
        # 将全局参数加载到临时模型（便于计算参数差）
        self.global_model = self._build_model()  # 复用BaseClient的模型构建方法
        self.set_model_parameters(self.global_model, self.global_model_params)
        self.global_model.to(self.device)
        # 冻结全局模型参数（仅用于计算近端项，不参与训练）
        for param in self.global_model.parameters():
            param.requires_grad = False

    def _calculate_proximal_term(self):
        """
        计算近端正则项：(μ/2) * ||θ_local - θ_global||²
        θ_local：当前本地模型参数；θ_global：服务端下发的全局模型参数
        """
        proximal_term = 0.0
        # 遍历本地模型和全局模型的参数，计算L2范数的平方和
        for (local_param, global_param) in zip(
            self.local_model.parameters(), 
            self.global_model.parameters()
        ):
            proximal_term += torch.norm(local_param - global_param, p=2) ** 2
        # 乘以近端系数μ/2
        proximal_term = (self.mu / 2) * proximal_term
        return proximal_term

    def local_train(self):
        """
        重写FedAvgClient的local_train：核心修改是损失函数加入近端项
        核心流程：前向传播→计算基础损失→计算近端项→总损失=基础损失+近端项→反向传播→梯度下降
        """
        # 前置检查：必须先接收全局模型参数（否则无法计算近端项）
        if self.global_model_params is None:
            raise ValueError(f"FedProx客户端 [{self.client_id}] 未接收全局模型参数，无法计算近端项！")
        
        # 1. 初始化训练环境（完全复用FedAvg的逻辑）
        self.local_model.train()
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.config.fed.local_lr,
            momentum=self.config.fed.momentum
        )
        loss_fn = F.cross_entropy if self.config.model.num_classes > 2 else F.binary_cross_entropy_with_logits

        # 2. 本地训练循环（核心新增近端项计算）
        for epoch in range(self.config.fed.local_epochs):
            epoch_loss = 0.0
            epoch_proximal_loss = 0.0  # 统计近端项损失
            total_samples = 0
            pbar = tqdm(self.local_dataloader, desc=f"FedProx客户端 [{self.client_id}] 训练Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播（复用FedAvg）
                optimizer.zero_grad()
                outputs = self.local_model(images)
                # 基础损失（任务损失，如交叉熵）
                base_loss = loss_fn(outputs, labels)

                # ==============================================
                # 核心新增：计算近端正则项，合并为总损失
                # ==============================================
                proximal_term = self._calculate_proximal_term()
                # FedProx总损失 = 基础任务损失 + 近端正则项
                total_loss = base_loss + proximal_term
                # ==============================================

                # 反向传播（基于总损失）
                total_loss.backward()
                # 梯度下降（复用FedAvg）
                optimizer.step()
                
                # 统计损失（区分基础损失和近端项损失，便于分析）
                epoch_loss += base_loss.item() * images.size(0)
                epoch_proximal_loss += proximal_term.item() * images.size(0)
                total_samples += images.size(0)
                pbar.set_postfix({
                    "base_loss": base_loss.item(),
                    "proximal_loss": proximal_term.item(),
                    "total_loss": total_loss.item(),
                    "avg_total_loss": (epoch_loss + epoch_proximal_loss)/total_samples
                })

        # 3. 训练完成，记录损失（区分基础损失和总损失）
        self.local_train_base_loss = epoch_loss / total_samples
        self.local_train_total_loss = (epoch_loss + epoch_proximal_loss) / total_samples
        print(f"\n📌 FedProx客户端 [{self.client_id}] 本地训练完成：")
        print(f"   基础任务损失：{self.local_train_base_loss:.4f} | 近端项损失：{epoch_proximal_loss/total_samples:.4f} | 总损失：{self.local_train_total_loss:.4f}")

        # 4. 返回本地模型参数（复用FedAvg）
        return self.get_model_parameters()

class FedProxServer(FedAvgServer):
    """
    FedProx服务端：完全复用FedAvgServer的等权重聚合逻辑
    核心：FedProx的核心修改仅在客户端（近端项），服务端无需任何调整
    """
    def __init__(self, config=None):
        """
        初始化FedProx服务端（完全复用FedAvgServer）
        Args:
            config: 配置对象（仅需FedAvg相关配置，无需额外FedProx配置）
        """
        super().__init__(config=config)
        print(f"✅ FedProx服务端初始化完成（完全复用FedAvg等权重聚合，无额外修改）")

    def distribute_global_model(self, selected_client_ids: list):
        """
        重写分发全局模型方法：向选中的客户端下发全局参数（供客户端计算近端项）
        适配FedProxClient的set_global_model_params方法
        """
        global_params = self.get_model_parameters()
        for cid in selected_client_ids:
            # 假设self.clients是客户端列表，索引为client_id
            self.clients[cid].set_global_model_params(global_params)
        print(f"📌 FedProx服务端已向 {len(selected_client_ids)} 个客户端下发全局模型参数（用于计算近端项）")

# ======================== 独立测试示例（验证FedProx功能） ========================
if __name__ == "__main__":
    """
    测试FedProx核心逻辑：服务端下发全局参数 → 客户端带近端项训练 → 服务端等权重聚合
    对比FedAvg：仅客户端损失多了近端项，服务端完全一致
    """
    # 1. 加载配置（需包含fedprox配置）
    config = load_config()
    # 测试用配置
    config.fed.num_clients = 2
    config.fed.local_epochs = 1
    config.fed.local_lr = 0.01
    config.fedprox.mu = 0.1  # 近端系数（小值适配低异构场景）

    # 2. 初始化FedProx服务端
    fedprox_server = FedProxServer(config=config)

    # 3. 初始化FedProx客户端
    client_list = []
    for client_id in range(config.fed.num_clients):
        client = FedProxClient(client_id=client_id, config=config)
        client_list.append(client)
    # 绑定客户端到服务端（供分发全局参数）
    fedprox_server.clients = client_list

    # 4. 模拟一轮联邦训练
    print("\n=== 模拟FedProx一轮联邦训练 ===")
    # 4.1 服务端选择客户端并下发全局参数
    selected_cids = [0, 1]
    fedprox_server.distribute_global_model(selected_client_ids=selected_cids)

    # 4.2 选中客户端本地训练（带近端项）
    client_params_list = []
    for cid in selected_cids:
        client_params = client_list[cid].local_train()
        client_params_list.append(client_params)

    # 4.3 服务端聚合（复用FedAvg等权重）
    fedprox_server.aggregate_local_results(client_params_list=client_params_list)

    # 4.4 打印结果
    print("\n=== FedProx一轮训练完成 ===")
    print(f"服务端全局模型参数示例（conv1.weight.shape）：{fedprox_server.global_model.conv1.weight.shape}")
    for idx, client in enumerate(client_list):
        print(f"FedProx客户端 [{idx}] 总训练损失：{client.local_train_total_loss:.4f}")