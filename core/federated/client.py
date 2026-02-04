# core/federated/client.py
"""
联邦学习客户端基类（BaseClient）
核心职责：
1.  封装客户端通用流程：下载全局模型 → 本地训练 → 提取贡献度特征 → 上传本地结果
2.  兼容现有核心模块：ALA（自适应更新）、伪标签（数据增强）、DP（差分隐私）、Shapley（SA贡献度）
3.  作为基类预留扩展接口，方便后续子类定制（如分类/回归任务客户端）
4.  无修改核心通信逻辑，仅嵌入辅助模块支撑，保持通用性
"""
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# 项目内模块导入（兼容core下的其他核心模块）
from configs.config_loader import load_config
from models import get_model, BaseModel
from core.ala.ala_optimizer import ALAOptimizer
from core.pseudo_label.pseudo_label import PseudoLabelGenerator
from core.dp.adaptive_clipping_dp import AdaptiveClippingDP

class BaseClient:
    """
    联邦学习客户端基类
    核心流程：download_global_model() → local_train() → extract_client_contribution_features() → upload_local_results()
    """
    def __init__(self, client_id: int, config=None, model=None, dataset=None):
        """
        初始化客户端
        Args:
            client_id: 客户端唯一标识（不可重复）
            config: 配置对象（默认加载全局配置）
            model: 本地模型实例（默认从models获取，与全局模型结构一致）
            dataset: 客户端本地数据集（已划分好的客户端私有数据）
        """
        # 1. 基础属性初始化
        self.client_id = client_id
        self.config = config if config is not None else load_config()
        self.device = self.config.device
        self.local_epochs = self.config.fed.num_local_epochs
        self.local_lr = self.config.fed.local_lr
        self.local_momentum = self.config.fed.local_momentum

        # 2. 核心对象初始化（模型、数据集）
        self.local_model = model if model is not None else self._init_local_model()
        self.local_dataset = dataset
        self.local_dataloader = self._init_local_dataloader() if self.local_dataset else None

        # 3. 辅助模块初始化（兼容现有核心模块，按需启用）
        self.ala_optimizer = ALAOptimizer(config=self.config)  # ALA自适应更新 + 特征提取
        self.pseudo_label_generator = PseudoLabelGenerator(config=self.config)  # 高置信伪标签生成
        self.adaptive_dp = AdaptiveClippingDP(config=self.config)  # DP自适应裁剪（带精细化优化）

        # 4. 训练/评估指标记录（用于后续贡献度计算、模型分析）
        self.train_metrics = {
            "train_loss": [],
            "train_acc": [],
            "local_sample_num": len(self.local_dataset.target_dataset) if self.local_dataset else 0
        }
        self.client_features = {}  # 客户端贡献度特征（ALA提取：偏差、稳定性、性能）
        self.trained_local_params = None  # 本地训练后的模型参数（用于上传服务端）

    def _init_local_model(self) -> BaseModel:
        """
        初始化本地模型（与全局模型结构一致，复用models下的基础模型）
        Returns:
            初始化完成的本地模型实例（已移至指定设备）
        """
        local_model = get_model(config=self.config)
        local_model = local_model.to(self.device)
        print(f"✅ 客户端 [{self.client_id}] 本地模型初始化完成（设备：{self.device}）")
        return local_model

    def _init_local_dataloader(self):
        """初始化本地数据集DataLoader（复用数据集模块的加载逻辑）"""
        if not self.local_dataset:
            raise ValueError(f"客户端 [{self.client_id}] 未传入有效数据集，无法初始化DataLoader")
        return self.local_dataset.get_dataloader()

    # ==============================================
    # 核心方法1：下载全局模型（从服务端获取全局参数，更新本地模型）
    # ==============================================
    def download_global_model(self, global_model_params: list) -> None:
        """
        下载服务端全局模型参数，更新本地模型
        Args:
            global_model_params: 服务端下发的全局模型参数（与本地模型结构一致的numpy列表）
        """
        if not global_model_params:
            raise ValueError("全局模型参数为空，无法更新本地模型")
        
        # 加载全局参数到本地模型
        self.local_model.set_params(global_model_params)
        print(f"✅ 客户端 [{self.client_id}] 已成功下载并加载全局模型参数")

    # ==============================================
    # 核心方法2：本地训练（整合ALA、伪标签、DP，无修改核心训练流程）
    # ==============================================
    def local_train(self) -> None:
        """
        客户端本地训练（核心流程，兼容所有辅助模块，保持基类通用性）
        流程：ALA自适应初始化 → 生成伪标签 → 带DP裁剪的本地训练 → ALA特征提取
        """
        if not self.local_dataloader:
            raise RuntimeError(f"客户端 [{self.client_id}] 无有效数据加载器，无法进行本地训练")
        
        # 1. 初始化优化器（带DP自适应裁剪的梯度优化）
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.local_lr,
            momentum=self.local_momentum
        )

        # 2. 前置准备：ALA自适应本地模型初始化（缓解客户端异质性）
        self.local_model = self.ala_optimizer.ala_adaptive_update(
            client_id=self.client_id,
            local_model=self.local_model,
            global_model_params=self.local_model.get_params()  # 初始为全局参数
        )

        # 3. 生成高置信度伪标签（提升本地数据利用率，半监督训练）
        pseudo_images, pseudo_labels = self.pseudo_label_generator.generate_high_conf_pseudo_labels(
            model=self.local_model,
            dataloader=self.local_dataloader
        )
        has_pseudo_data = pseudo_images is not None and pseudo_labels is not None

        # 4. 本地训练循环（核心：带DP裁剪、ALA自适应调整）
        self.local_model.train()
        print(f"🚀 客户端 [{self.client_id}] 开始本地训练（{self.local_epochs} 轮）")
        for epoch in tqdm(range(self.local_epochs), desc=f"客户端 [{self.client_id}] 本地训练"):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (images, labels) in enumerate(self.local_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # （1）前向传播：真实标签数据训练
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = self.local_model.loss_fn(outputs, labels)

                # （2）伪标签联合训练（按需启用，加权融合损失）
                if has_pseudo_data and batch_idx % int(1/self.config.fed.pseudo_batch_ratio) == 0:
                    pseudo_batch = self.pseudo_label_generator.get_pseudo_batch(
                        pseudo_images=pseudo_images,
                        pseudo_labels=pseudo_labels,
                        batch_size=32
                    )
                    if pseudo_batch:
                        pseudo_imgs, pseudo_labs = pseudo_batch
                        pseudo_imgs, pseudo_labs = pseudo_imgs.to(self.device), pseudo_labs.to(self.device)
                        pseudo_outputs = self.local_model(pseudo_imgs)
                        pseudo_loss = self.local_model.loss_fn(pseudo_outputs, pseudo_labs)
                        loss = 0.7 * loss + 0.3 * pseudo_loss  # 真实标签权重优先

                # （3）反向传播：带DP自适应裁剪（精细化梯度处理，保证隐私）
                loss.backward()
                # DP自适应裁剪（优化后：归一化+分级+时序校准+稳定性约束）
                self.adaptive_dp.clip_gradient(self.local_model.parameters())
                optimizer.step()

                # （4）记录批次指标
                epoch_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_total += images.size(0)

            # 5. 记录轮次指标
            avg_epoch_loss = epoch_loss / epoch_total
            avg_epoch_acc = epoch_correct / epoch_total
            self.train_metrics["train_loss"].append(avg_epoch_loss)
            self.train_metrics["train_acc"].append(avg_epoch_acc)

            # 6. ALA自适应调整（每轮训练后更新，提升模型稳定性）
            self.local_model = self.ala_optimizer.ala_adaptive_update(
                client_id=self.client_id,
                local_model=self.local_model,
                epoch=epoch
            )

        # 7. 训练完成后：提取客户端贡献度特征（支撑SA融合贡献度计算）
        self.extract_client_contribution_features()

        # 8. 保存训练后的本地模型参数（用于上传服务端）
        self.trained_local_params = self.local_model.get_params()
        print(f"🎉 客户端 [{self.client_id}] 本地训练完成，最优训练准确率：{max(self.train_metrics['train_acc']):.4f}")

    # ==============================================
    # 核心方法3：提取客户端贡献度特征（支撑Shapley SA贡献度计算）
    # ==============================================
    def extract_client_contribution_features(self) -> None:
        """
        提取客户端贡献度核心特征（调用ALA模块，提取3类特征：偏差、稳定性、性能）
        特征结果存入self.client_features，用于后续服务端SA融合贡献度计算
        """
        self.client_features = self.ala_optimizer.extract_ala_features(
            client_id=self.client_id,
            local_model=self.local_model,
            train_metrics=self.train_metrics
        )
        print(f"✅ 客户端 [{self.client_id}] 已提取SA贡献度特征，特征维度：{len(self.client_features)}")

    # ==============================================
    # 核心方法4：上传本地结果（给服务端，用于聚合与贡献度评估）
    # ==============================================
    def upload_local_results(self) -> dict:
        """
        整理客户端本地训练结果，上传至服务端
        Returns:
            客户端本地结果字典（包含模型参数、样本数、贡献度特征、训练指标）
        """
        if not self.trained_local_params:
            raise RuntimeError(f"客户端 [{self.client_id}] 未完成本地训练，无有效结果可上传")
        
        upload_data = {
            "client_id": self.client_id,
            "local_params": self.trained_local_params,
            "local_sample_num": self.train_metrics["local_sample_num"],
            "client_features": self.client_features,  # SA贡献度特征
            "train_metrics": self.train_metrics  # 训练损失/准确率（辅助评估）
        }
        print(f"✅ 客户端 [{self.client_id}] 已整理上传数据，准备发送至服务端")
        return upload_data

    # ==============================================
    # 辅助方法：本地模型评估（可选，用于客户端自验证）
    # ==============================================
    def evaluate_local_model(self, test_dataloader=None) -> tuple[float, float]:
        """
        评估本地模型性能（自验证，不影响联邦聚合流程）
        Args:
            test_dataloader: 测试集DataLoader（默认使用客户端本地验证集）
        Returns:
            avg_loss: 平均测试损失
            avg_acc: 平均测试准确率
        """
        eval_dataloader = test_dataloader if test_dataloader else self.local_dataloader
        if not eval_dataloader:
            raise ValueError(f"客户端 [{self.client_id}] 无有效评估数据加载器")
        
        self.local_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in eval_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.local_model(images)
                loss = self.local_model.loss_fn(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"📊 客户端 [{self.client_id}] 本地模型评估：损失={avg_loss:.4f} | 准确率={avg_acc:.4f}")
        return avg_loss, avg_acc