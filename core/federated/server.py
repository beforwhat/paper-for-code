# core/federated/server.py
"""
联邦学习服务端基类（BaseServer）
核心职责：
1.  封装服务端通用流程：初始化全局模型 → 选择客户端 → 下发模型 → 接收结果 → 聚合更新 → 评估保存
2.  核心修改：替换传统FedAvg样本数加权，改为「SA融合贡献度加权聚合」（结合ALA特征+Shapley值）
3.  兼容现有核心模块：Shapley（SA贡献度计算）、ALA（特征支撑）、公平选择（客户端筛选）
4.  作为基类预留扩展接口，方便后续子类定制（如多任务服务端、异步联邦服务端）
"""
import os
import numpy as np
import torch
from tqdm import tqdm

# 项目内模块导入
from configs.config_loader import load_config
from models import get_model, BaseModel
from core.shapley.shapley_calculator import ShapleyCalculator
from core.fair_selection.fair_selector import FairClientSelector
from core.ala.ala_optimizer import ALAOptimizer

class BaseServer:
    """
    联邦学习服务端基类（核心修改：SA融合贡献度加权聚合）
    核心流程：select_clients() → distribute_global_model() → receive_client_uploads() → aggregate_local_results() → update_global_model()
    """
    def __init__(self, config=None, global_model=None, total_clients=None):
        """
        初始化服务端
        Args:
            config: 配置对象（默认加载全局配置）
            global_model: 全局模型实例（默认从models获取，与客户端模型结构一致）
            total_clients: 联邦系统中客户端总数（默认从配置读取）
        """
        # 1. 基础属性初始化
        self.config = config if config is not None else load_config()
        self.device = self.config.device
        self.global_rounds = self.config.fed.num_global_rounds
        self.model_save_path = self.config.model.model_save_path
        self.total_clients = total_clients if total_clients else self.config.fed.num_clients
        self.select_ratio = self.config.fed.client_selection_ratio

        # 2. 核心对象初始化（全局模型、结果存储）
        self.global_model = global_model if global_model is not None else self._init_global_model()
        self.global_model_params = self.global_model.get_params()  # 全局模型参数（用于下发客户端）
        self.received_client_data = {}  # 接收的客户端上传数据：{client_id: upload_data}
        self.selected_clients = []  # 本轮选中的客户端列表

        # 3. 辅助模块初始化（兼容现有核心模块，支撑SA聚合与公平选择）
        self.shapley_calculator = ShapleyCalculator(config=self.config)  # SA融合贡献度计算
        self.fair_selector = FairClientSelector(config=self.config)  # 公平客户端选择（复用SA贡献度）
        self.ala_optimizer = ALAOptimizer(config=self.config)  # ALA特征解析（支撑SA贡献度）

        # 4. 全局训练/评估指标记录（用于后续分析、模型对比）
        self.global_metrics = {
            "global_round": [],
            "global_loss": [],
            "global_acc": [],
            "best_global_acc": 0.0,
            "best_round": 0
        }

        # 确保模型保存目录存在
        os.makedirs(self.model_save_path, exist_ok=True)
        print(f"✅ 联邦服务端初始化完成（全局轮次：{self.global_rounds} | 客户端总数：{self.total_clients}）")
        print(f"✅ 核心修改：启用SA融合贡献度加权聚合（替代传统FedAvg样本数加权）")

    def _init_global_model(self) -> BaseModel:
        """
        初始化全局模型（与客户端模型结构一致，复用models下的基础模型）
        Returns:
            初始化完成的全局模型实例（已移至指定设备）
        """
        global_model = get_model(config=self.config)
        global_model = global_model.to(self.device)
        print(f"✅ 全局模型初始化完成（设备：{self.device} | 模型结构：{self.config.model.backbone}）")
        return global_model

    # ==============================================
    # 核心方法1：公平选择参与本轮训练的客户端（复用SA贡献度提升精准度）
    # ==============================================
    def select_clients(self, round_idx: int) -> list:
        """
        选择参与本轮训练的客户端（公平选择，结合SA贡献度筛选优质客户端）
        Args:
            round_idx: 当前全局训练轮次（用于动态调整选择策略）
        Returns:
            选中的客户端ID列表
        """
        # 1. 计算待选客户端的SA贡献度（首轮无历史数据，采用均匀分布；后续结合客户端上传特征）
        client_sa_scores = self.shapley_calculator.calculate_prior_sa_scores(
            total_clients=self.total_clients,
            round_idx=round_idx,
            historical_client_data=self.received_client_data
        )

        # 2. 调用公平选择器，筛选符合条件的客户端（兼顾公平性与贡献度）
        select_num = int(self.total_clients * self.select_ratio)
        self.selected_clients = self.fair_selector.select(
            client_sa_scores=client_sa_scores,
            select_num=select_num,
            round_idx=round_idx
        )

        print(f"\n=== 全局轮次 [{round_idx}] 客户端选择完成 ===")
        print(f"🔍 选中客户端数量：{len(self.selected_clients)} | 选中列表：{sorted(self.selected_clients)}")
        return self.selected_clients

    # ==============================================
    # 核心方法2：向选中的客户端下发全局模型参数
    # ==============================================
    def distribute_global_model(self) -> list:
        """
        向本轮选中的所有客户端下发最新全局模型参数
        Returns:
            全局模型参数列表（与客户端模型结构一致，供所有选中客户端下载）
        """
        if not self.selected_clients:
            raise RuntimeError("未选中任何客户端，无法下发全局模型参数")
        
        # 刷新全局模型参数（确保下发最新版本）
        self.global_model_params = self.global_model.get_params()
        print(f"✅ 全局模型参数已刷新，准备下发至 {len(self.selected_clients)} 个客户端")
        return self.global_model_params

    # ==============================================
    # 核心方法3：接收选中客户端的本地训练结果上传
    # ==============================================
    def receive_client_uploads(self, client_upload_data: dict) -> None:
        """
        接收单个客户端的上传数据，整理并存储（由训练器协调，批量接收）
        Args:
            client_upload_data: 客户端上传的结果字典（来自BaseClient.upload_local_results()）
        """
        client_id = client_upload_data["client_id"]
        if client_id not in self.selected_clients:
            print(f"⚠️  客户端 [{client_id}] 未被选中本轮训练，拒绝接收其上传数据")
            return
        
        # 存储客户端上传数据（去重，避免重复接收）
        self.received_client_data[client_id] = client_upload_data
        print(f"✅ 已接收客户端 [{client_id}] 上传数据（包含参数、SA特征、训练指标）")

    # ==============================================
    # 核心方法4：SA融合贡献度加权聚合（核心修改，替代传统FedAvg）
    # ==============================================
    def aggregate_local_results(self) -> list:
        """
        核心：SA融合贡献度加权聚合客户端本地参数
        流程：1. 提取客户端SA特征 2. 计算每个客户端的SA贡献度权重 3. 加权聚合生成新全局参数
        Returns:
            聚合后的新全局模型参数列表
        """
        if not self.received_client_data:
            raise RuntimeError("未接收任何客户端上传数据，无法进行聚合操作")
        
        # 1. 提取聚合所需基础数据（客户端ID、本地参数、SA特征、样本数）
        client_ids = list(self.received_client_data.keys())
        local_params_list = [self.received_client_data[cid]["local_params"] for cid in client_ids]
        client_features_list = [self.received_client_data[cid]["client_features"] for cid in client_ids]
        local_sample_nums = [self.received_client_data[cid]["local_sample_num"] for cid in client_ids]

        # 2. 核心：计算每个客户端的SA融合贡献度权重（结合ALA特征+Shapley值+样本数）
        sa_weights = self._calculate_sa_contribution_weights(
            client_ids=client_ids,
            client_features_list=client_features_list,
            local_sample_nums=local_sample_nums
        )

        # 3. SA加权聚合：生成新全局模型参数
        print(f"🚀 开始SA融合贡献度加权聚合（共 {len(client_ids)} 个客户端参与）")
        new_global_params = []
        # 遍历模型每一层参数，进行加权求和
        for param_layer in zip(*local_params_list):
            layer_np_arrays = [np.array(p) for p in param_layer]
            # 按SA权重加权聚合当前层参数
            aggregated_layer = np.sum([w * arr for w, arr in zip(sa_weights, layer_np_arrays)], axis=0)
            new_global_params.append(aggregated_layer)

        print(f"✅ SA融合贡献度聚合完成，新全局模型参数已生成")
        return new_global_params

    def _calculate_sa_contribution_weights(self, client_ids: list, client_features_list: list, local_sample_nums: list) -> list:
        """
        辅助：计算每个客户端的SA融合贡献度权重（核心逻辑封装，归一化处理）
        Args:
            client_ids: 参与聚合的客户端ID列表
            client_features_list: 客户端ALA特征列表（偏差、稳定性、性能）
            local_sample_nums: 客户端本地样本数列表
        Returns:
            归一化后的SA贡献度权重列表（和为1）
        """
        # 1. 调用Shapley模块，计算SA融合贡献度原始得分
        sa_raw_scores = self.shapley_calculator.calculate_sa_contribution(
            client_ids=client_ids,
            client_features_list=client_features_list,
            local_sample_nums=local_sample_nums,
            global_model=self.global_model
        )

        # 2. 权重归一化（确保所有客户端权重和为1，避免数值溢出）
        sa_scores_sum = sum(sa_raw_scores)
        if sa_scores_sum <= 0:
            # 异常处理：得分和为非正数时，采用均匀权重
            print(f"⚠️  SA原始得分异常，切换为均匀权重")
            sa_weights = [1.0 / len(client_ids) for _ in client_ids]
        else:
            sa_weights = [score / sa_scores_sum for score in sa_raw_scores]

        # 3. 打印权重分布（辅助分析）
        print(f"📊 本轮客户端SA贡献度权重分布（前5个）：")
        for i, (cid, w) in enumerate(zip(client_ids[:5], sa_weights[:5])):
            print(f"   客户端 [{cid}]：权重={w:.6f}")
        if len(client_ids) > 5:
            print(f"   ... 剩余 {len(client_ids)-5} 个客户端权重已省略")

        return sa_weights

    # ==============================================
    # 核心方法5：用聚合后的参数更新全局模型
    # ==============================================
    def update_global_model(self, new_global_params: list) -> None:
        """
        用SA聚合后的新参数更新服务端全局模型
        Args:
            new_global_params: SA聚合生成的新全局模型参数列表
        """
        if not new_global_params:
            raise ValueError("聚合后的全局参数为空，无法更新全局模型")
        
        # 加载新参数到全局模型
        self.global_model.set_params(new_global_params)
        # 刷新全局模型参数缓存（用于下一轮下发）
        self.global_model_params = self.global_model.get_params()
        print(f"✅ 全局模型已更新为SA聚合后的新版本")

    # ==============================================
    # 核心方法6：评估全局模型性能（在全局测试集上验证）
    # ==============================================
    def evaluate_global_model(self, test_dataloader, round_idx: int) -> tuple[float, float]:
        """
        在全局测试集上评估当前全局模型性能，记录全局指标
        Args:
            test_dataloader: 全局测试集DataLoader
            round_idx: 当前全局训练轮次
        Returns:
            avg_loss: 全局平均测试损失
            avg_acc: 全局平均测试准确率
        """
        if not test_dataloader:
            raise ValueError("全局测试集DataLoader为空，无法评估模型")
        
        self.global_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc=f"全局模型评估（轮次 {round_idx}）"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = self.global_model.loss_fn(outputs, labels)

                # 累计指标
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += images.size(0)

        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # 记录全局指标
        self.global_metrics["global_round"].append(round_idx)
        self.global_metrics["global_loss"].append(avg_loss)
        self.global_metrics["global_acc"].append(avg_acc)

        # 更新最优模型记录
        if avg_acc > self.global_metrics["best_global_acc"]:
            self.global_metrics["best_global_acc"] = avg_acc
            self.global_metrics["best_round"] = round_idx
            # 保存最优全局模型
            self._save_global_model(
                model_name=f"{self.config.model.backbone}_sa_global_best",
                epoch=round_idx
            )

        # 打印评估结果
        print(f"\n=== 全局轮次 [{round_idx}] 模型评估结果 ===")
        print(f"📊 全局测试损失：{avg_loss:.4f} | 全局测试准确率：{avg_acc:.4f}")
        print(f"🏆 目前最优准确率：{self.global_metrics['best_global_acc']:.4f}（轮次 {self.global_metrics['best_round']}）")

        return avg_loss, avg_acc

    # ==============================================
    # 辅助方法：保存全局模型（最优/最终版本）
    # ==============================================
    def _save_global_model(self, model_name: str, epoch: int) -> None:
        """
        保存全局模型到指定路径
        Args:
            model_name: 模型保存名称
            epoch: 训练轮次（用于标注模型版本）
        """
        save_path = os.path.join(
            self.model_save_path,
            f"{model_name}_round_{epoch}.pth"
        )
        self.global_model.save_model(save_path=save_path)
        print(f"✅ 全局模型已保存至：{save_path}")

    # ==============================================
    # 核心方法7：端到端联邦训练主流程（协调所有步骤）
    # ==============================================
    def run_federated_training(self, global_test_dataloader, client_manager):
        """
        启动端到端联邦训练（由FederatedTrainer协调客户端通信，此处封装核心流程）
        Args:
            global_test_dataloader: 全局测试集DataLoader（用于评估全局模型）
            client_manager: 客户端管理器（用于协调客户端本地训练，封装通信细节）
        """
        print("\n" + "="*80)
        print("🚀 开始端到端联邦训练（SA融合贡献度聚合）")
        print("="*80)

        for round_idx in range(1, self.global_rounds + 1):
            print("\n" + "-"*60 + f" 全局轮次 [{round_idx}/{self.global_rounds}] " + "-"*60)

            # 步骤1：选择参与本轮训练的客户端
            self.select_clients(round_idx=round_idx)

            # 步骤2：下发全局模型参数到选中客户端
            self.distribute_global_model()

            # 步骤3：协调客户端执行本地训练，并接收上传结果（由client_manager封装通信）
            self.received_client_data = {}  # 清空上一轮接收的数据
            client_manager.run_client_local_training(
                server=self,
                round_idx=round_idx,
                selected_clients=self.selected_clients
            )

            # 步骤4：SA融合贡献度加权聚合本地结果
            new_global_params = self.aggregate_local_results()

            # 步骤5：更新全局模型
            self.update_global_model(new_global_params=new_global_params)

            # 步骤6：评估当前全局模型性能
            self.evaluate_global_model(
                test_dataloader=global_test_dataloader,
                round_idx=round_idx
            )

        # 训练完成：保存最终全局模型
        print("\n" + "="*80)
        print("🎉 端到端联邦训练完成（SA融合贡献度聚合）")
        print("="*80)
        self._save_global_model(
            model_name=f"{self.config.model.backbone}_sa_global_final",
            epoch=self.global_rounds
        )

        # 打印训练总结
        print("\n" + "="*60 + " 联邦训练总结 " + "="*60)
        print(f"📌 总全局轮次：{self.global_rounds}")
        print(f"📌 最优全局准确率：{self.global_metrics['best_global_acc']:.4f}（轮次 {self.global_metrics['best_round']}）")
        print(f"📌 最终全局准确率：{self.global_metrics['global_acc'][-1]:.4f}（轮次 {self.global_rounds}）")