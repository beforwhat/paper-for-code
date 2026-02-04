# baselines/ditto.py
"""
DITTO: Fair and Robust Federated Learning Through Personalization
补充专属DittoServer类，适配baselines/__init__.py导入需求
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

# 导入现有项目模块
from datasets import get_client_dataset, get_global_test_dataset
from models import get_model, FedModel
from core.federated.server import BaseServer  # 继承通用Server类
from utils.logger import info, error

# ======================== 核心配置 ========================
DITTO_CONFIG = {
    "dataset_name": "simulation",
    "model_name": "simple_mlp",
    "num_clients": 10,
    "num_rounds": 50,
    "local_epochs": 5,
    "lr": 0.01,
    "lambda_p": 0.1,  # 个性化正则项权重
    "batch_size": 32,
    "num_classes": 10,
    "feature_dim": 784,
    "non_iid_alpha": 0.5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ======================== 专属DittoServer类（核心补充） ========================
class DITTOServer(BaseServer):
    """
    DITTO专属服务器类（继承自通用Server）
    扩展DITTO特有的功能：个性化模型评估、双模型聚合适配、个性化模型下发等
    """
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        device: torch.device,
        config: Dict = DITTO_CONFIG
    ):
        super().__init__(model=model, num_clients=num_clients, device=device)  # 调用父类构造
        self.config = config
        self.personal_models: Dict[int, nn.Module] = {}  # 存储各客户端的个性化模型（可选）
        self.test_loader = get_global_test_dataset(  # 全局测试集（评估用）
            dataset_name=config["dataset_name"],
            num_clients=num_clients,
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"]
        )

    def aggregate(self, client_models: List[nn.Module], client_weights: List[float]) -> nn.Module:
        """
        重写聚合方法（DITTO仅聚合global_model，逻辑与FedAvg一致，可扩展）
        Args:
            client_models: 各客户端训练后的global_model列表
            client_weights: 各客户端的聚合权重（如数据量占比）
        Returns:
            聚合后的全局模型
        """
        info("DittoServer: 执行FedAvg聚合客户端global_model")
        # 父类Server的aggregate方法（FedAvg）
        aggregated_model = super().aggregate(client_models=client_models, client_weights=client_weights)
        return aggregated_model

    def save_personal_models(self, client_id: int, personal_model: nn.Module):
        """保存客户端的个性化模型（DITTO特有）"""
        self.personal_models[client_id] = personal_model
        info(f"DittoServer: 保存客户端{client_id}的个性化模型")

    def evaluate_personal_models(self) -> Dict[int, float]:
        """评估所有客户端的个性化模型准确率（DITTO核心评估指标）"""
        personal_accs = {}
        for client_id, model in self.personal_models.items():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in self.test_loader:
                    X, y = X.to(self.config["device"]), y.to(self.device)
                    pred = model(X)
                    _, predicted = torch.max(pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            acc = 100 * correct / total
            personal_accs[client_id] = acc
            info(f"DittoServer: 客户端{client_id}个性化模型准确率：{acc:.2f}%")
        return personal_accs

# ======================== DITTO客户端类（适配DittoServer） ========================
class DITTOClient:
    """DITTO客户端：维护全局模型+本地个性化模型"""
    def __init__(
        self,
        client_id: int,
        config: Dict = DITTO_CONFIG,
        global_model: Optional[nn.Module] = None
    ):
        self.client_id = client_id
        self.config = config
        self.device = config["device"]
        
        # 1. 初始化双模型
        if global_model is None:
            self.global_model = get_model(
                model_name=config["model_name"],
                input_dim=config["feature_dim"],
                num_classes=config["num_classes"]
            ).to(self.device)
        else:
            self.global_model = global_model.to(self.device)
        
        self.personal_model = get_model(
            model_name=config["model_name"],
            input_dim=config["feature_dim"],
            num_classes=config["num_classes"]
        ).to(self.device)
        
        # 2. 加载客户端数据
        self.train_loader = get_client_dataset(
            dataset_name=config["dataset_name"],
            client_id=client_id,
            num_clients=config["num_clients"],
            sample_per_client=1000,
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"],
            non_iid_alpha=config["non_iid_alpha"]
        )
        
        # 3. 优化器
        self.optimizer = optim.SGD(
            list(self.personal_model.parameters()) + list(self.global_model.parameters()),
            lr=config["lr"],
            momentum=0.9
        )
        self.criterion = nn.CrossEntropyLoss()

    def _compute_l2_distance(self, model1: nn.Module, model2: nn.Module) -> torch.Tensor:
        """计算两个模型参数的L2距离（个性化正则项）"""
        distance = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.norm(p1 - p2, p=2)
        return distance

    def local_train(self) -> Tuple[nn.Module, nn.Module, float]:
        """
        客户端本地训练（返回global_model+personal_model+损失）
        Returns:
            (训练后的global_model, 训练后的personal_model, 平均损失)
        """
        self.global_model.train()
        self.personal_model.train()
        
        total_loss = 0.0
        total_batches = len(self.train_loader)
        
        for epoch in range(self.config["local_epochs"]):
            epoch_loss = 0.0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # 前向传播
                pred = self.personal_model(X)
                task_loss = self.criterion(pred, y)
                
                # 个性化正则项
                reg_loss = self._compute_l2_distance(self.personal_model, self.global_model)
                
                # 总损失
                loss = task_loss + self.config["lambda_p"] * reg_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / total_batches
            total_loss += avg_epoch_loss
            info(f"客户端{self.client_id} 轮次{epoch+1} 损失：{avg_epoch_loss:.4f}")
        
        return self.global_model, self.personal_model, total_loss / self.config["local_epochs"]

# ======================== DITTO训练器（协调DittoServer+客户端） ========================
class DITTOTrainer:
    """DITTO联邦训练器：协调DittoServer+多客户端训练"""
    def __init__(self, config: Dict = DITTO_CONFIG):
        self.config = config
        self.device = config["device"]
        
        # 1. 初始化全局模型
        self.global_model = get_model(
            model_name=config["model_name"],
            input_dim=config["feature_dim"],
            num_classes=config["num_classes"]
        ).to(self.device)
        
        # 2. 初始化DittoServer（替换通用Server）
        self.ditto_server = DittoServer(
            model=self.global_model,
            num_clients=config["num_clients"],
            device=self.device,
            config=config
        )
        
        # 3. 加载全局测试集
        self.test_loader = get_global_test_dataset(
            dataset_name=config["dataset_name"],
            num_clients=config["num_clients"],
            feature_dim=config["feature_dim"],
            num_classes=config["num_classes"]
        )
        
        # 4. 初始化客户端
        self.clients = [
            DITTOClient(client_id=i, config=config, global_model=self.global_model)
            for i in range(config["num_clients"])
        ]

    def _evaluate_global_model(self) -> float:
        """评估全局模型准确率"""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.global_model(X)
                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = 100 * correct / total
        info(f"全局模型测试准确率：{acc:.2f}%")
        return acc

    def train(self) -> Dict:
        """执行DITTO联邦训练"""
        history = {"rounds": [], "global_acc": [], "avg_client_loss": []}
        
        for round_idx in tqdm(range(self.config["num_rounds"]), desc="DITTO训练轮次"):
            info(f"\n===== 联邦轮次 {round_idx+1}/{self.config['num_rounds']} =====")
            
            # 1. 客户端本地训练
            client_global_models = []
            client_personal_models = []
            client_losses = []
            for client in self.clients:
                # 下发全局模型
                client.global_model.load_state_dict(self.global_model.state_dict())
                # 本地训练
                global_model, personal_model, loss = client.local_train()
                client_global_models.append(global_model)
                client_personal_models.append(personal_model)
                client_losses.append(loss)
                # 客户端上传个性化模型至服务器
                self.ditto_server.save_personal_models(client.client_id, personal_model)
            
            # 2. DittoServer聚合global_model
            self.global_model = self.ditto_server.aggregate(
                client_models=client_global_models,
                client_weights=[1/self.config["num_clients"]]*self.config["num_clients"]
            )
            
            # 3. 评估并记录
            avg_loss = np.mean(client_losses)
            global_acc = self._evaluate_global_model()
            history["rounds"].append(round_idx+1)
            history["global_acc"].append(global_acc)
            history["avg_client_loss"].append(avg_loss)
            
            info(f"轮次{round_idx+1} 平均客户端损失：{avg_loss:.4f} | 全局准确率：{global_acc:.2f}%")
        
        # 4. 评估所有客户端的个性化模型（DITTO特有）
        info("\n===== 评估客户端个性化模型 =====")
        personal_accs = self.ditto_server.evaluate_personal_models()
        history["personal_accs"] = personal_accs
        
        # 5. 保存模型
        self._save_model()
        return history

    def _save_model(self):
        """保存全局模型和个性化模型"""
        save_dir = "./ditto_models"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存全局模型
        torch.save(
            self.global_model.state_dict(),
            os.path.join(save_dir, "ditto_global_model.pth")
        )
        
        # 保存个性化模型（从DittoServer中读取）
        for client_id, model in self.ditto_server.personal_models.items():
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"ditto_personal_model_client_{client_id}.pth")
            )
        info(f"模型已保存至 {save_dir}")

# ======================== 导出至baselines/__init__.py ========================
# 确保baselines/__init__.py能导入DittoServer、DITTOClient、DITTOTrainer
__all__ = ["DITTOServer", "DITTOClient", "DITTOTrainer", "main"]

# ======================== 主函数 ========================
def main(config: Dict = DITTO_CONFIG):
    """运行DITTO训练"""
    info("===== 启动DITTO联邦训练 =====")
    trainer = DITTOTrainer(config=config)
    training_history = trainer.train()
    info("===== DITTO训练完成 =====")
    return training_history

if __name__ == "__main__":
    main(DITTO_CONFIG)