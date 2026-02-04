# models/model_factory.py
"""模型工厂函数（测试版，极简实现）"""
from typing import Optional, Dict

# 导入基础模型（测试阶段仅需SimpleMLP）
from .simple_mlp import SimpleMLP

def get_model(
    model_name: str = "simple_mlp",
    input_dim: int = 784,
    num_classes: int = 10,
    **kwargs  # 兼容测试时的额外参数
) -> object:
    """
    测试版模型工厂函数：仅返回基础模型实例，满足导入和基础调用
    Args:
        model_name: 模型名（测试仅支持simple_mlp）
        input_dim: 输入维度（如MNIST为784）
        num_classes: 类别数
        **kwargs: 兼容其他参数（测试阶段忽略）
    Returns:
        基础模型实例（SimpleMLP）
    """
    # 测试版：仅实现simple_mlp，其他模型暂返回SimpleMLP（避免报错）
    if model_name.lower() in ["simple_mlp", "mlp"]:
        return SimpleMLP(input_dim=input_dim, num_classes=num_classes, **kwargs)
    else:
        # 测试容错：未知模型名默认返回SimpleMLP
        print(f"测试阶段暂不支持模型{model_name}，返回SimpleMLP")
        return SimpleMLP(input_dim=input_dim, num_classes=num_classes, **kwargs)