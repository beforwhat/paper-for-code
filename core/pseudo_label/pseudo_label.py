# core/pseudo_label/pseudo_label.py
"""
伪标签生成器（PseudoLabelGenerator）
核心职责：
1.  无修改核心逻辑：仅实现高置信度伪标签生成+批次采样，为客户端半监督训练提供数据增强
2.  核心流程：模型推理生成伪标签 → 置信度筛选 → 批次采样（避免内存溢出）
3.  独立模块设计：兼容BaseClient调用，支持配置化调参，鲁棒性强
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 项目内模块导入
from configs.config_loader import load_config

class PseudoLabelGenerator:
    """
    高置信度伪标签生成器
    核心方法：
    - generate_high_conf_pseudo_labels()：生成高置信伪标签数据
    - get_pseudo_batch()：随机采样伪标签批次（用于客户端联合训练）
    """
    def __init__(self, config=None):
        """
        初始化伪标签生成器
        Args:
            config: 配置对象（默认加载全局配置）
        """
        # 1. 基础配置初始化
        self.config = config if config is not None else load_config()
        self.confidence_threshold = self.config.fed.pseudo_conf_thresh  # 伪标签置信度阈值（如0.9）
        self.pseudo_batch_size = self.config.fed.pseudo_batch_size  # 伪标签批次大小（如32）
        self.device = self.config.device

        # 2. 伪标签数据缓存（避免重复生成，提升效率）
        # 结构：{client_id: {"pseudo_images": tensor, "pseudo_labels": tensor}}
        self.pseudo_data_cache = {}

        print(f"✅ 伪标签生成器初始化完成")
        print(f"📌 置信度阈值：{self.confidence_threshold} | 伪标签批次大小：{self.pseudo_batch_size}")
        print(f"📌 设备：{self.device} | 模式：高置信度筛选 + 批次采样")

    # ==============================================
    # 核心方法1：生成高置信度伪标签（核心逻辑，无修改）
    # ==============================================
    def generate_high_conf_pseudo_labels(self, model, dataloader, client_id: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        生成高置信度伪标签数据
        逻辑：模型推理 → 计算置信度 → 筛选阈值以上样本 → 缓存结果
        Args:
            model: 客户端本地模型实例（已完成ALA初始化，训练前/训练中均可调用）
            dataloader: 客户端本地数据集DataLoader（含真实标签，推理时忽略标签）
            client_id: 客户端ID（用于缓存伪标签数据，避免重复生成）
        Returns:
            pseudo_images: 高置信度伪标签图像（Tensor，shape [N, C, H, W]）
            pseudo_labels: 高置信度伪标签（硬标签，Tensor，shape [N]）
            若无高置信样本，返回 (None, None)
        """
        # 前置检查
        if model is None or dataloader is None:
            raise ValueError("模型或DataLoader为空，无法生成伪标签")
        model.eval()  # 推理模式，关闭Dropout/BatchNorm

        # 初始化伪标签数据容器
        pseudo_images = []
        pseudo_labels = []

        print(f"\n📌 开始生成高置信度伪标签（置信度阈值：{self.confidence_threshold}）")
        with torch.no_grad():  # 推理阶段禁用梯度计算，节省内存
            for images, _ in tqdm(dataloader, desc="伪标签生成推理"):
                # 数据迁移到指定设备
                images = images.to(self.device)

                # 模型推理，获取预测概率
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)  # 转换为概率分布
                confs, preds = torch.max(probs, dim=1)  # 置信度 + 伪标签（硬标签）

                # 筛选高置信度样本（置信度 ≥ 阈值）
                high_conf_mask = confs >= self.confidence_threshold
                high_conf_images = images[high_conf_mask]
                high_conf_preds = preds[high_conf_mask]

                # 收集高置信样本
                if high_conf_images.size(0) > 0:
                    pseudo_images.append(high_conf_images.cpu())  # 转回CPU，避免GPU内存占用
                    pseudo_labels.append(high_conf_preds.cpu())

        # 拼接伪标签数据
        if len(pseudo_images) == 0:
            print(f"⚠️  无高置信度伪标签样本（置信度阈值 {self.confidence_threshold} 过高）")
            if client_id is not None:
                self.pseudo_data_cache[client_id] = (None, None)
            return None, None
        else:
            pseudo_images = torch.cat(pseudo_images, dim=0)
            pseudo_labels = torch.cat(pseudo_labels, dim=0)
            print(f"✅ 伪标签生成完成：共筛选出 {pseudo_images.size(0)} 个高置信样本")

            # 缓存伪标签数据（客户端ID指定时）
            if client_id is not None:
                self.pseudo_data_cache[client_id] = (pseudo_images, pseudo_labels)
                print(f"✅ 客户端 [{client_id}] 伪标签数据已缓存，可通过get_pseudo_batch()采样")

            return pseudo_images, pseudo_labels

    # ==============================================
    # 核心方法2：伪标签批次采样（避免内存溢出，适配客户端训练）
    # ==============================================
    def get_pseudo_batch(self, client_id: int, batch_size: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从缓存的伪标签数据中随机采样一个批次（用于客户端联合训练）
        Args:
            client_id: 客户端ID（用于获取缓存的伪标签数据）
            batch_size: 批次大小（默认使用配置中的pseudo_batch_size）
        Returns:
            batch_pseudo_images: 伪标签图像批次
            batch_pseudo_labels: 伪标签批次
            若无缓存数据，返回 (None, None)
        """
        batch_size = batch_size if batch_size is not None else self.pseudo_batch_size

        # 检查客户端缓存
        if client_id not in self.pseudo_data_cache:
            print(f"⚠️  客户端 [{client_id}] 无伪标签缓存数据，请先调用generate_high_conf_pseudo_labels()")
            return None, None
        
        pseudo_images, pseudo_labels = self.pseudo_data_cache[client_id]
        if pseudo_images is None or pseudo_labels is None:
            return None, None
        
        # 随机采样批次（避免顺序采样导致的过拟合）
        total_pseudo_samples = pseudo_images.size(0)
        if total_pseudo_samples <= batch_size:
            # 伪标签样本不足一个批次，返回全部
            return pseudo_images.to(self.device), pseudo_labels.to(self.device)
        else:
            # 随机索引采样
            indices = torch.randperm(total_pseudo_samples)[:batch_size]
            batch_pseudo_images = pseudo_images[indices].to(self.device)
            batch_pseudo_labels = pseudo_labels[indices].to(self.device)
            return batch_pseudo_images, batch_pseudo_labels

    # ==============================================
    # 辅助方法：清空伪标签缓存（便于多次实验/轮次训练）
    # ==============================================
    def clear_pseudo_cache(self, client_id=None) -> None:
        """
        清空伪标签缓存数据
        Args:
            client_id: 可选，指定客户端ID；None则清空所有客户端缓存
        """
        if client_id is None:
            self.pseudo_data_cache = {}
            print("✅ 所有客户端伪标签缓存已清空")
        else:
            if client_id in self.pseudo_data_cache:
                del self.pseudo_data_cache[client_id]
                print(f"✅ 客户端 [{client_id}] 伪标签缓存已清空")
            else:
                print(f"⚠️  客户端 [{client_id}] 无伪标签缓存数据，无需清空")

    # ==============================================
    # 辅助方法：动态调整置信度阈值（实验调参用）
    # ==============================================
    def adjust_confidence_threshold(self, new_threshold: float) -> None:
        """
        动态调整伪标签置信度阈值（无需重新初始化，支持实验调参）
        Args:
            new_threshold: 新的置信度阈值（0 < new_threshold < 1）
        """
        if not (0 < new_threshold < 1):
            raise ValueError("置信度阈值必须在(0, 1)区间内")
        self.confidence_threshold = new_threshold
        print(f"✅ 伪标签置信度阈值已调整为：{self.confidence_threshold}")
        # 阈值调整后，建议清空对应客户端缓存，重新生成伪标签
        self.clear_pseudo_cache()