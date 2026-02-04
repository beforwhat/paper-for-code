# core/federated/trainer.py
"""
联邦训练器（FederatedTrainer）
核心定位：服务端与客户端的“协调者”，无修改核心业务逻辑，仅封装通信与流程协调
核心职责：
1.  管理客户端实例（初始化、状态监控、异常处理）
2.  协调服务端与客户端的通信流程：下发全局模型 → 触发客户端本地训练 → 收集上传结果
3.  封装端到端联邦训练逻辑，对外暴露简洁的启动接口
4.  监控训练进度，记录全局日志，提升联邦训练的可维护性
"""
import time
import logging
import numpy as np
from tqdm import tqdm

# 项目内模块导入
from configs.config_loader import load_config
from core.federated.server import BaseServer
from core.federated.client import BaseClient
from datasets import get_client_dataset, get_global_test_dataset

# 配置日志（监控训练流程，便于问题排查）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FederatedTrainer")

class FederatedTrainer:
    """
    联邦训练器（协调者）
    核心流程：init_clients() → init_server() → run_federated_training()
    """
    def __init__(self, config=None):
        """
        初始化联邦训练器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.total_clients = self.config.fed.num_clients
        self.global_rounds = self.config.fed.num_global_rounds
        self.device = self.config.device

        # 2. 核心组件初始化（延迟初始化，支持动态调整）
        self.server = None  # 服务端实例（BaseServer）
        self.clients = {}   # 客户端实例字典：{client_id: BaseClient}
        self.global_test_dataloader = None  # 全局测试集DataLoader

        # 3. 训练监控指标（记录耗时、成功率、异常客户端）
        self.training_metrics = {
            "round_start_time": [],
            "round_end_time": [],
            "round_duration": [],  # 每轮训练耗时（秒）
            "client_train_success": [],  # 每轮成功训练的客户端数
            "client_train_failed": [],   # 每轮训练失败的客户端数
            "failed_client_ids": []      # 失败客户端ID（用于后续分析）
        }

        logger.info("✅ 联邦训练器初始化完成，等待启动训练流程")

    # ==============================================
    # 核心方法1：初始化所有客户端实例（统一管理，避免重复创建）
    # ==============================================
    def init_clients(self) -> None:
        """
        初始化所有客户端实例（按配置的客户端总数创建）
        逻辑：为每个客户端加载专属数据集，初始化BaseClient实例
        """
        logger.info(f"📌 开始初始化 {self.total_clients} 个客户端实例...")
        self.clients = {}  # 清空原有客户端实例

        for client_id in tqdm(range(self.total_clients), desc="客户端实例初始化"):
            try:
                # 加载客户端专属数据集
                client_dataset = get_client_dataset(
                    config=self.config,
                    client_id=client_id
                )
                # 初始化客户端实例
                client = BaseClient(
                    client_id=client_id,
                    config=self.config,
                    dataset=client_dataset
                )
                self.clients[client_id] = client
                logger.debug(f"✅ 客户端 [{client_id}] 初始化成功")
            except Exception as e:
                logger.error(f"❌ 客户端 [{client_id}] 初始化失败：{str(e)}")
                self.training_metrics["failed_client_ids"].append(client_id)

        # 校验客户端初始化结果
        success_num = len(self.clients)
        failed_num = len(self.training_metrics["failed_client_ids"])
        logger.info(f"📊 客户端初始化完成：成功 {success_num} 个 | 失败 {failed_num} 个")
        if failed_num > 0:
            logger.warning(f"⚠️  失败客户端ID列表：{self.training_metrics['failed_client_ids']}")

    # ==============================================
    # 核心方法2：初始化服务端实例 + 全局测试集
    # ==============================================
    def init_server(self) -> None:
        """
        初始化服务端实例 + 全局测试集（用于服务端评估全局模型）
        """
        logger.info("📌 开始初始化服务端实例...")
        try:
            # 1. 初始化服务端
            self.server = BaseServer(
                config=self.config,
                total_clients=self.total_clients
            )
            # 2. 加载全局测试集
            self.global_test_dataloader = get_global_test_dataset(
                config=self.config
            ).get_dataloader()
            logger.info("✅ 服务端 + 全局测试集初始化完成")
        except Exception as e:
            logger.error(f"❌ 服务端初始化失败：{str(e)}")
            raise RuntimeError("服务端初始化失败，无法启动联邦训练") from e

    # ==============================================
    # 核心方法3：协调单轮联邦训练（服务端+客户端通信闭环）
    # ==============================================
    def run_single_round_training(self, round_idx: int) -> None:
        """
        协调单轮联邦训练流程（核心协调逻辑，无业务修改）
        Args:
            round_idx: 当前全局训练轮次
        """
        logger.info(f"\n=== 开始协调全局轮次 [{round_idx}/{self.global_rounds}] 训练 ===")
        round_start = time.time()

        # 步骤1：服务端选择本轮参与训练的客户端
        selected_clients = self.server.select_clients(round_idx=round_idx)
        if not selected_clients:
            logger.error(f"❌ 轮次 [{round_idx}] 未选中任何客户端，跳过本轮训练")
            self.training_metrics["client_train_success"].append(0)
            self.training_metrics["client_train_failed"].append(0)
            return

        # 步骤2：服务端下发最新全局模型参数
        global_params = self.server.distribute_global_model()

        # 步骤3：协调选中的客户端执行本地训练，并收集上传结果
        success_count = 0
        failed_count = 0
        round_failed_clients = []

        for client_id in tqdm(selected_clients, desc=f"协调客户端训练（轮次 {round_idx}）"):
            try:
                # 跳过初始化失败的客户端
                if client_id not in self.clients:
                    raise ValueError("客户端未初始化")
                
                client = self.clients[client_id]
                # 子步骤1：客户端下载全局模型参数
                client.download_global_model(global_model_params=global_params)
                # 子步骤2：客户端执行本地训练（整合ALA/伪标签/DP）
                client.local_train()
                # 子步骤3：客户端上传训练结果，服务端接收
                upload_data = client.upload_local_results()
                self.server.receive_client_uploads(client_upload_data=upload_data)
                
                success_count += 1
                logger.debug(f"✅ 客户端 [{client_id}] 轮次 [{round_idx}] 训练+上传成功")
            except Exception as e:
                failed_count += 1
                round_failed_clients.append(client_id)
                logger.error(f"❌ 客户端 [{client_id}] 轮次 [{round_idx}] 训练失败：{str(e)}")

        # 步骤4：服务端执行SA融合贡献度聚合 + 更新全局模型
        try:
            new_global_params = self.server.aggregate_local_results()
            self.server.update_global_model(new_global_params=new_global_params)
            logger.info(f"✅ 轮次 [{round_idx}] 服务端SA聚合 + 全局模型更新成功")
        except Exception as e:
            logger.error(f"❌ 轮次 [{round_idx}] 服务端聚合失败：{str(e)}")
            raise RuntimeError(f"轮次 [{round_idx}] 聚合失败，训练中断") from e

        # 步骤5：服务端评估全局模型性能
        try:
            self.server.evaluate_global_model(
                test_dataloader=self.global_test_dataloader,
                round_idx=round_idx
            )
            logger.info(f"✅ 轮次 [{round_idx}] 全局模型评估成功")
        except Exception as e:
            logger.warning(f"⚠️  轮次 [{round_idx}] 全局模型评估失败：{str(e)}")

        # 记录本轮训练指标
        round_end = time.time()
        self.training_metrics["round_start_time"].append(round_start)
        self.training_metrics["round_end_time"].append(round_end)
        self.training_metrics["round_duration"].append(round_end - round_start)
        self.training_metrics["client_train_success"].append(success_count)
        self.training_metrics["client_train_failed"].append(failed_count)
        self.training_metrics["failed_client_ids"].extend(round_failed_clients)

        # 打印本轮训练小结
        logger.info(f"\n=== 轮次 [{round_idx}] 训练小结 ===")
        logger.info(f"⏱️  本轮耗时：{round_end - round_start:.2f} 秒")
        logger.info(f"📊 客户端训练：成功 {success_count} 个 | 失败 {failed_count} 个")
        if round_failed_clients:
            logger.warning(f"⚠️  本轮失败客户端ID：{round_failed_clients}")

    # ==============================================
    # 核心方法4：端到端联邦训练主流程（对外暴露的核心接口）
    # ==============================================
    def run_federated_training(self) -> None:
        """
        启动端到端联邦训练（完整闭环，无需外部额外协调）
        流程：初始化客户端 → 初始化服务端 → 逐轮协调训练 → 输出训练总结
        """
        # 前置检查：初始化客户端和服务端
        if not self.clients:
            logger.info("📌 未检测到已初始化的客户端，自动执行客户端初始化...")
            self.init_clients()
        if not self.server:
            logger.info("📌 未检测到已初始化的服务端，自动执行服务端初始化...")
            self.init_server()
        if not self.global_test_dataloader:
            raise RuntimeError("全局测试集未加载，无法评估全局模型")

        # 启动端到端训练
        logger.info("\n" + "="*80)
        logger.info("🚀 启动端到端联邦训练（FederatedTrainer协调）")
        logger.info(f"📌 总全局轮次：{self.global_rounds} | 客户端总数：{self.total_clients}")
        logger.info("="*80)

        total_start = time.time()
        for round_idx in range(1, self.global_rounds + 1):
            self.run_single_round_training(round_idx=round_idx)

        # 训练完成：输出全局总结
        total_end = time.time()
        total_duration = total_end - total_start
        avg_round_duration = np.mean(self.training_metrics["round_duration"])
        total_success = sum(self.training_metrics["client_train_success"])
        total_failed = sum(self.training_metrics["client_train_failed"])

        logger.info("\n" + "="*80)
        logger.info("🎉 端到端联邦训练全部完成！")
        logger.info("="*80)
        logger.info(f"📊 全局训练总结：")
        logger.info(f"⏱️  总耗时：{total_duration:.2f} 秒（平均每轮 {avg_round_duration:.2f} 秒）")
        logger.info(f"📈 客户端训练：累计成功 {total_success} 次 | 累计失败 {total_failed} 次")
        logger.info(f"🏆 全局模型最优准确率：{self.server.global_metrics['best_global_acc']:.4f}（轮次 {self.server.global_metrics['best_round']}）")
        logger.info(f"📁 最优模型保存路径：{self.server.model_save_path}")

        # （可选）保存训练监控指标（用于后续分析）
        self._save_training_metrics()

    # ==============================================
    # 辅助方法：保存训练监控指标（便于后续分析训练效率）
    # ==============================================
    def _save_training_metrics(self) -> None:
        """
        将训练监控指标保存为JSON文件（可选，便于后续可视化/分析）
        """
        import json
        import os

        save_path = os.path.join(self.config.log.log_save_path, "federated_trainer_metrics.json")
        os.makedirs(self.config.log.log_save_path, exist_ok=True)

        # 转换numpy类型为Python原生类型（避免JSON序列化报错）
        metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.training_metrics.items()
        }
        # 补充全局指标
        metrics["global_best_acc"] = self.server.global_metrics["best_global_acc"]
        metrics["global_best_round"] = self.server.global_metrics["best_round"]
        metrics["total_training_time"] = sum(self.training_metrics["round_duration"])

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✅ 训练监控指标已保存至：{save_path}")

    # ==============================================
    # 辅助方法：重置训练器（重新初始化，便于多次实验）
    # ==============================================
    def reset(self) -> None:
        """
        重置联邦训练器（清空客户端、服务端、监控指标）
        """
        self.server = None
        self.clients = {}
        self.global_test_dataloader = None
        self.training_metrics = {
            "round_start_time": [],
            "round_end_time": [],
            "round_duration": [],
            "client_train_success": [],
            "client_train_failed": [],
            "failed_client_ids": []
        }
        logger.info("✅ 联邦训练器已重置，可重新启动训练")