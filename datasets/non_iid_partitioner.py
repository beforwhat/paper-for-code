# datasets/non_iid_partitioner.py
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any

class NonIIDPartitioner:
    """
    Non-IID划分工具类：封装多种Non-IID划分算法，供所有数据集复用
    核心改进：补充seed参数保证复现，完善边界处理
    """
    def __init__(self, num_clients: int, alpha: float, seed: int = 42):
        """
        初始化划分器
        Args:
            num_clients: 客户端数量
            alpha: Dirichlet分布的α值（越小Non-IID程度越高）
            seed: 随机种子（核心！保证实验可复现）
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        # 固定随机种子（全局生效）
        np.random.seed(self.seed)
    
    def dirichlet_partition(self, labels: np.ndarray) -> List[List[int]]:
        """
        核心方法：Dirichlet分布划分（标签异质性，最常用）
        Args:
            labels: 原始数据集的标签数组（np.array）
        Returns:
            client_indices: 客户端样本索引列表（shape: [num_clients, num_samples_per_client]）
        """
        # 边界检查
        if len(labels) == 0:
            raise ValueError("标签数组不能为空！")
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(self.num_clients)]
        
        # 1. 按类别分组，保存每个类别的样本索引
        class_to_samples = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_samples[label].append(idx)
        
        # 2. 对每个类别，用Dirichlet分布分配给各客户端
        for class_id, sample_indices in class_to_samples.items():
            # 打乱当前类别的样本索引（保证分配随机性）
            np.random.shuffle(sample_indices)
            
            # 生成每个客户端对该类别的分配比例（Dirichlet分布）
            class_proportions = np.random.dirichlet(
                alpha=[self.alpha] * self.num_clients,
                size=1
            ).flatten()
            
            # 归一化比例（避免因浮点误差导致总和≠1）
            class_proportions = class_proportions / class_proportions.sum()
            
            # 按比例分配该类别的样本给各客户端
            num_samples_in_class = len(sample_indices)
            sample_indices_per_client = self._split_samples_by_proportion(
                sample_indices, class_proportions, num_samples_in_class
            )
            
            # 合并到客户端索引列表
            for client_id, indices in enumerate(sample_indices_per_client):
                client_indices[client_id].extend(indices)
        
        # 3. 打乱每个客户端的样本顺序（提升训练稳定性）
        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])
            
            # 容错：确保每个客户端至少有1个样本
            if len(client_indices[client_id]) == 0:
                client_indices[client_id] = [np.random.choice(len(labels))]
        
        return client_indices
    
    def _split_samples_by_proportion(self, sample_indices: List[int], proportions: np.ndarray, num_samples: int) -> List[List[int]]:
        """
        辅助方法：按比例拆分样本索引（内部调用，不对外暴露）
        """
        sample_indices_per_client = []
        start_idx = 0
        for idx, prop in enumerate(proportions):
            # 计算该客户端分配到的样本数量
            num_samples_for_client = int(prop * num_samples)
            
            # 处理最后一个客户端，避免遗漏样本（浮点误差导致的总数不匹配）
            if idx == self.num_clients - 1:
                num_samples_for_client = num_samples - start_idx
            
            # 边界保护：避免负数或超出范围
            num_samples_for_client = max(0, min(num_samples_for_client, num_samples - start_idx))
            
            # 拆分样本索引
            end_idx = start_idx + num_samples_for_client
            sample_indices_per_client.append(sample_indices[start_idx:end_idx])
            start_idx = end_idx
        
        return sample_indices_per_client
    
    def quantity_unbalanced_partition(self, labels: np.ndarray, min_sample_ratio: float = 0.1) -> List[List[int]]:
        """
        扩展方法：数量异质性划分（客户端样本数量不一致）
        Args:
            labels: 原始标签数组
            min_sample_ratio: 最小样本比例（相对于均匀分配的数量）
        Returns:
            client_indices: 数量异构的客户端索引列表
        """
        # 1. 先按Dirichlet得到标签异构的基础索引
        base_indices = self.dirichlet_partition(labels)
        
        # 2. 计算均匀分配时的最小样本数（作为下限）
        uniform_sample_num = len(labels) // self.num_clients
        min_samples_per_client = max(1, int(uniform_sample_num * min_sample_ratio))
        
        # 3. 随机调整各客户端样本数量（实现数量异质性）
        client_indices = []
        for indices in base_indices:
            # 随机保留 [min_samples, len(indices)] 范围内的样本
            keep_num = np.random.randint(min_samples_per_client, len(indices) + 1)
            np.random.shuffle(indices)
            client_indices.append(indices[:keep_num])
        
        return client_indices

class DirichletPartitioner(NonIIDPartitioner):
    """
    狄利克雷分布Non-IID划分（正式版，复用父类逻辑）
    核心改进：修正继承逻辑，补充seed，复用父类方法
    """
    def __init__(
        self,
        labels: np.ndarray,
        num_clients: int,
        alpha: float = 0.5,  # 狄利克雷分布参数，越小越非IID
        seed: int = 42
    ):
        # 修正继承：调用父类正确的初始化参数
        super().__init__(num_clients=num_clients, alpha=alpha, seed=seed)
        self.labels = labels  # 保存标签数组
        self.num_classes = len(np.unique(labels))  # 类别数
    
    def partition(self) -> Dict[int, List[int]]:
        """
        正式版划分逻辑（复用父类dirichlet_partition，返回字典格式）
        Returns:
            键：客户端ID，值：该客户端的样本索引列表
        """
        # 复用父类的核心划分方法
        client_indices_list = self.dirichlet_partition(self.labels)
        
        # 转换为字典格式（客户端ID -> 索引列表）
        client_indices_dict = {
            client_id: indices for client_id, indices in enumerate(client_indices_list)
        }
        
        return client_indices_dict