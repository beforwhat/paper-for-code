# models/fed_model.py
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# 直接相对导入：models/下的核心组件（无需嵌套子目录，更简洁）
from . import get_model
from .base_model import BaseModel

# 全局导入项目公共组件（配置、数据集）
from configs.config_loader import load_config
from datasets import get_dataset

class FedModel:
    """
    核心联邦模型（直接放在models/下）：整合 ALA（自适应本地聚合）+ 伪标签（Pseudo Label）
    1.  无嵌套子目录，核心逻辑直达，便于快速修改和调试
    2.  复用models/下的基础模型（CustomCNN/VGG11），兼容现有配置和数据集
    3.  作为实验组核心模型，与baselines/下的FedAvg形成对比
    """
    def __init__(self, config=None):
        """初始化核心联邦模型（ALA+伪标签）"""
        # 1. 加载配置与超参数
        self.config = config if config is not None else load_config()
        self.fed_cfg = self.config.fed
        self.model_cfg = self.config.model
        self.dataset_cfg = self.config.dataset
        self.device = self.config.device

        # 2. 原有联邦超参数（与基线FedAvg保持一致，保证对比公平）
        self.total_clients = self.fed_cfg.num_clients
        self.select_ratio = self.fed_cfg.client_selection_ratio
        self.select_clients = int(self.total_clients * self.select_ratio)
        self.global_rounds = self.fed_cfg.num_global_rounds
        self.local_epochs = self.fed_cfg.num_local_epochs
        self.local_lr = self.fed_cfg.local_lr
        self.local_momentum = self.fed_cfg.local_momentum

        # 3. 新增：ALA + 伪标签专属超参数（从配置读取，无硬编码）
        self.ala_alpha = self.fed_cfg.ala_alpha  # ALA自适应权重
        self.pseudo_conf_thresh = self.fed_cfg.pseudo_conf_thresh  # 伪标签置信度阈值
        self.pseudo_batch_ratio = self.fed_cfg.pseudo_batch_ratio  # 伪标签批次占比
        self.client_prev_params = {}  # 客户端历史参数持久化（用于ALA）

        # 4. 核心对象初始化
        self.global_model = None
        self.client_datasets = {}
        self.test_dataset = None
        self.best_acc = 0.0

        # 5. 初始化流程（复用models/下的模型，加载数据）
        self._init_global_model()
        self._init_client_datasets()
        self._init_test_dataset()
        # 初始化客户端历史参数（第一轮为None，直接用全局参数）
        for client_id in range(self.total_clients):
            self.client_prev_params[client_id] = None

    def _init_global_model(self):
        """初始化全局模型（直接复用models/下的get_model()，无需嵌套导入）"""
        self.global_model = get_model(config=self.config)
        print(f"✅ [核心FedModel] 全局模型 [{self.model_cfg.backbone}] 初始化完成（设备：{self.device}）")
        print(f"✅ 已启用 ALA + 伪标签训练逻辑，直接放在models/下便于调试\n")

    def _init_client_datasets(self):
        """加载客户端本地数据集（与基线FedAvg复用同一套数据）"""
        print(f"📥 [核心FedModel] 开始加载 {self.total_clients} 个客户端的本地数据集...")
        for client_id in tqdm(range(self.total_clients), desc="客户端数据加载（核心模型）"):
            self.client_datasets[client_id] = get_dataset(
                config=self.config,
                is_train=True,
                client_id=client_id
            )
        print("✅ 所有客户端数据集加载完成\n")

    def _init_test_dataset(self):
        """加载全局测试集（与基线FedAvg保持一致，保证对比公平）"""
        self.test_dataset = get_dataset(
            config=self.config,
            is_train=False,
            client_id=None
        )
        print("✅ 全局测试集加载完成\n")

    def _ala_init_local_model(self, client_id, local_model):
        """
        ALA 自适应本地聚合：初始化本地模型参数（核心改进点1）
        解决客户端异质性导致的模型震荡问题
        """
        global_params = self.global_model.get_params()
        client_prev_params = self.client_prev_params[client_id]

        # 第一轮无历史参数，直接使用全局参数
        if client_prev_params is None:
            local_model.set_params(global_params)
            return local_model

        # 非第一轮，执行ALA加权聚合（w_init = α*w_global + (1-α)*w_prev）
        ala_init_params = []
        for g_param, p_param in zip(global_params, client_prev_params):
            init_param = self.ala_alpha * g_param + (1 - self.ala_alpha) * p_param
            ala_init_params.append(init_param)

        local_model.set_params(ala_init_params)
        return local_model

    def _generate_pseudo_labels(self, local_model, client_dataloader):
        """
        生成高置信度伪标签（核心改进点2）：提升本地数据利用率
        """
        local_model.eval()
        pseudo_images = []
        pseudo_labels = []

        with torch.no_grad():
            for images, _ in client_dataloader:
                images = images.to(self.device)
                outputs = local_model(images)

                # 计算置信度与硬伪标签
                confs, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
                # 筛选置信度高于阈值的样本
                high_conf_mask = confs >= self.pseudo_conf_thresh

                if high_conf_mask.sum() > 0:
                    pseudo_images.append(images[high_conf_mask].cpu())
                    pseudo_labels.append(preds[high_conf_mask].cpu())

        # 拼接伪标签数据（无合格样本则返回空）
        if len(pseudo_images) == 0:
            return None, None
        return torch.cat(pseudo_images, dim=0), torch.cat(pseudo_labels, dim=0)

    def _client_local_train_with_ala_pseudo(self, client_id):
        """
        客户端本地训练：整合 ALA 初始化 + 伪标签联合训练（核心流程）
        """
        # 1. 准备客户端数据
        client_data = self.client_datasets[client_id]
        client_dl = client_data.get_dataloader()
        local_sample_num = len(client_data.target_dataset)

        # 2. 初始化本地模型（复用models/下的基础模型）
        local_model = get_model(config=self.config)

        # 3. ALA 自适应初始化（核心改进1）
        local_model = self._ala_init_local_model(client_id, local_model)

        # 4. 生成高置信度伪标签（核心改进2）
        pseudo_images, pseudo_labels = self._generate_pseudo_labels(local_model, client_dl)
        has_pseudo_data = pseudo_images is not None and pseudo_labels is not None

        # 5. 初始化优化器（与基线FedAvg超参数一致，保证对比公平）
        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.local_lr,
            momentum=self.local_momentum
        )

        # 6. 联合训练（真实标签 + 伪标签）
        local_model.train()
        for _ in range(self.local_epochs):
            for batch_idx, (images, labels) in enumerate(client_dl):
                # 真实标签数据训练
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = local_model(images)
                loss = local_model.loss_fn(outputs, labels)

                # 伪标签数据联合训练（按需执行，避免内存溢出）
                if has_pseudo_data and batch_idx % int(1/self.pseudo_batch_ratio) == 0:
                    pseudo_batch_size = min(32, len(pseudo_images))
                    pseudo_idx = np.random.choice(len(pseudo_images), pseudo_batch_size, replace=False)
                    batch_pseudo_imgs = pseudo_images[pseudo_idx].to(self.device)
                    batch_pseudo_labels = pseudo_labels[pseudo_idx].to(self.device)

                    # 伪标签损失计算（加权融合，真实标签权重更高）
                    pseudo_outputs = local_model(batch_pseudo_imgs)
                    pseudo_loss = local_model.loss_fn(pseudo_outputs, batch_pseudo_labels)
                    loss = 0.7 * loss + 0.3 * pseudo_loss

                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 7. 持久化当前客户端参数（用于下一轮ALA初始化）
        self.client_prev_params[client_id] = local_model.get_params()

        # 8. 返回训练结果（用于服务端聚合）
        return local_model.get_params(), local_sample_num

    def _server_aggregate(self, client_params_list):
        """
        服务端加权聚合（与基线FedAvg逻辑一致，保证对比公平性）
        """
        new_global_params = []
        for param in self.global_model.get_params():
            new_global_params.append(np.zeros_like(param, dtype=np.float32))

        total_samples = sum([sample_num for (_, sample_num) in client_params_list])

        for local_params, local_sample_num in client_params_list:
            weight = local_sample_num / total_samples
            for i in range(len(new_global_params)):
                new_global_params[i] += local_params[i] * weight

        return new_global_params

    def _evaluate_global_model(self):
        """
        全局模型评估（与基线FedAvg逻辑一致，保证对比结果有效）
        """
        self.global_model.eval()
        test_dl = self.test_dataset.get_dataloader()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_dl:
                outputs = self.global_model(images)
                labels = labels.to(self.device)

                loss = self.global_model.loss_fn(outputs, labels)
                total_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_acc, avg_loss

    def train(self):
        """
        端到端核心联邦训练（ALA+伪标签）：与基线FedAvg形成对比
        """
        print(f"🚀 开始 [核心FedModel] 联邦训练（全局轮次：{self.global_rounds}）")
        print(f"📌 ALA权重：{self.ala_alpha} | 伪标签置信度阈值：{self.pseudo_conf_thresh}")
        print(f"📌 每轮选择 {self.select_clients}/{self.total_clients} 个客户端\n")

        for global_round in range(1, self.global_rounds + 1):
            print(f"=== 全局轮次 {global_round}/{self.global_rounds} ===")

            # 步骤1：随机选择客户端
            selected_client_ids = np.random.choice(
                self.total_clients,
                size=self.select_clients,
                replace=False
            )
            print(f"🔍 选中的客户端 ID：{sorted(selected_client_ids)}")

            # 步骤2：客户端本地训练（ALA+伪标签，核心改进流程）
            client_params = []
            for client_id in tqdm(selected_client_ids, desc="客户端本地训练（ALA+伪标签）"):
                local_params, local_samples = self._client_local_train_with_ala_pseudo(client_id)
                client_params.append((local_params, local_samples))

            # 步骤3：服务端聚合，更新全局模型
            new_global_params = self._server_aggregate(client_params)
            self.global_model.set_params(new_global_params)
            print("🔄 服务端参数聚合完成，全局模型已更新")

            # 步骤4：评估全局模型，保存最优模型
            test_acc, test_loss = self._evaluate_global_model()
            print(f"📊 全局模型评估：损失={test_loss:.4f} | 准确率={test_acc:.4f}")

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.global_model.save_model(
                    epoch=global_round,
                    model_name=f"{self.model_cfg.backbone}_fedmodel_best"
                )
            print(f"🏆 当前最优准确率：{self.best_acc:.4f}\n")

        # 训练完成，保存最终核心模型
        self.global_model.save_model(model_name=f"{self.model_cfg.backbone}_fedmodel_final")
        print(f"🎉 [核心FedModel] 训练完成！最终最优准确率：{self.best_acc:.4f}")