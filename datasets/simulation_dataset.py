# datasets/simulation_dataset.py
"""仿真数据集生成函数"""
import numpy as np
from .base_dataset import BaseDataset

def get_simulation_dataset(
    num_clients: int = 10,
    sample_per_client: int = 1000,
    feature_dim: int = 784,
    num_classes: int = 10,
    non_iid_alpha: float = 0.5
) -> BaseDataset:
    """
    生成仿真数据集（示例模板，需根据你的业务逻辑修改）
    Args:
        num_clients: 客户端数量
        sample_per_client: 每个客户端的样本数
        feature_dim: 特征维度（如MNIST为784）
        num_classes: 类别数
        non_iid_alpha: Non-IID划分的alpha值（越小越非IID）
    Returns:
        封装后的BaseDataset实例
    """
    # 示例：生成随机仿真数据
    X = np.random.randn(num_clients * sample_per_client, feature_dim)
    y = np.random.randint(0, num_classes, size=num_clients * sample_per_client)
    
    # 封装为BaseDataset（需适配你的BaseDataset构造函数）
    dataset = BaseDataset(X, y)
    return dataset