# datasets/dataset_factory.py
"""数据集工厂函数（正式版）
包含联邦训练必需的get_client_dataset/get_global_test_dataset，满足正常训练逻辑
"""
import numpy as np
from typing import Optional, Dict, List, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

# 导入核心依赖
from .base_dataset import BaseDataset
from .simulation_dataset import get_simulation_dataset
from .non_iid_partitioner import DirichletPartitioner  # 正式使用狄利克雷划分
# 可选：导入真实数据集（如CIFAR10）
# from .cifar10_dataset import CIFAR10Dataset

def get_dataset(
    dataset_name: str = "simulation",
    num_clients: int = 10,
    sample_per_client: int = 1000,
    feature_dim: int = 784,
    num_classes: int = 10,
    val_ratio: float = 0.1,  # 验证集比例
    **kwargs
) -> Tuple[BaseDataset, BaseDataset]:
    """
    正式版：获取完整数据集（训练+测试）
    Returns:
        (训练集, 测试集)
    """
    # 1. 生成/加载基础数据集
    if dataset_name.lower() in ["simulation", "fake"]:
        full_dataset = get_simulation_dataset(
            num_clients=num_clients,
            sample_per_client=sample_per_client,
            feature_dim=feature_dim,
            num_classes=num_classes
        )
    # 2. 真实数据集逻辑（可选，按需补充）
    # elif dataset_name.lower() == "cifar10":
    #     full_dataset = CIFAR10Dataset(**kwargs)
    else:
        raise ValueError(f"不支持的数据集：{dataset_name}")
    
    # 3. 划分训练/测试集（正式训练必需）
    total_samples = len(full_dataset.X)
    test_size = int(total_samples * val_ratio)
    train_size = total_samples - test_size
    
    # 随机划分索引
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 构建训练/测试集
    train_dataset = BaseDataset(
        X=full_dataset.X[train_indices],
        y=full_dataset.y[train_indices]
    )
    test_dataset = BaseDataset(
        X=full_dataset.X[test_indices],
        y=full_dataset.y[test_indices]
    )
    
    return train_dataset, test_dataset

def get_client_dataset(
    dataset_name: str = "simulation",
    client_id: int = 0,
    num_clients: int = 10,
    sample_per_client: int = 1000,
    feature_dim: int = 784,
    num_classes: int = 10,
    non_iid_alpha: float = 0.5,  # 狄利克雷划分参数
    **kwargs
) -> DataLoader:
    """
    正式版：获取指定客户端的Non-IID训练数据集（返回DataLoader，满足训练直接使用）
    Args:
        client_id: 客户端ID（从0开始）
        num_clients: 总客户端数
        non_iid_alpha: 狄利克雷分布参数（越小越非IID）
    Returns:
        客户端专属的DataLoader（可直接用于训练）
    """
    # 1. 获取全局训练集
    train_dataset, _ = get_dataset(
        dataset_name=dataset_name,
        num_clients=num_clients,
        sample_per_client=sample_per_client,
        feature_dim=feature_dim,
        num_classes=num_classes,
        **kwargs
    )
    
    # 2. 用狄利克雷划分生成客户端专属索引（正式Non-IID逻辑）
    partitioner = DirichletPartitioner(
        labels=train_dataset.y,
        num_clients=num_clients,
        alpha=non_iid_alpha
    )
    client_indices = partitioner.partition()
    
    # 3. 校验客户端ID合法性
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"客户端ID {client_id} 超出范围（0~{num_clients-1}）")
    
    # 4. 构建客户端专属数据集
    client_X = train_dataset.X[client_indices[client_id]]
    client_y = train_dataset.y[client_indices[client_id]]
    
    # 5. 转换为PyTorch DataLoader（满足训练批量加载）
    tensor_dataset = TensorDataset(
        torch.from_numpy(client_X).float(),
        torch.from_numpy(client_y).long()
    )
    client_loader = DataLoader(
        dataset=tensor_dataset,
        batch_size=32,  # 训练批量（可配置）
        shuffle=True,
        num_workers=2
    )
    
    return client_loader

def get_global_test_dataset(
    dataset_name: str = "simulation",
    num_clients: int = 10,
    sample_per_client: int = 1000,
    feature_dim: int = 784,
    num_classes: int = 10,
    batch_size: int = 64,
    **kwargs
) -> DataLoader:
    """
    正式版：获取全局统一的测试数据集（返回DataLoader，用于服务器评估）
    Returns:
        全局测试集的DataLoader
    """
    # 1. 获取全局测试集
    _, test_dataset = get_dataset(
        dataset_name=dataset_name,
        num_clients=num_clients,
        sample_per_client=sample_per_client,
        feature_dim=feature_dim,
        num_classes=num_classes,
        **kwargs
    )
    
    # 2. 转换为PyTorch DataLoader
    tensor_dataset = TensorDataset(
        torch.from_numpy(test_dataset.X).float(),
        torch.from_numpy(test_dataset.y).long()
    )
    test_loader = DataLoader(
        dataset=tensor_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不洗牌
        num_workers=2
    )
    
    return test_loader