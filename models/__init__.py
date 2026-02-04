# models/__init__.py
"""
模型包：导出基础模型、自定义模型、核心联邦模型（FedModel）
"""
# 原有基础模型导出
from .base_model import BaseModel
from .custom_cnn import CustomCNNModel
from .vgg11 import VGG11Model
from .simple_mlp import SimpleMLP
from .model_factory import get_model
# 【关键修正】明确从fed_model.py导入FedModel，确保导出生效
# 语法：from .[文件名] import [类名]
from .fed_model import FedModel

# 原有支持的基础模型列表
SUPPORTED_MODELS = [
    "custom_cnn",
    "vgg11"
]

# 原有模型工厂函数（用于初始化基础模型，无需修改）
def get_model(config=None):
    if config is None:
        from configs.config_loader import load_config
        config = load_config()
    backbone = config.model.backbone.lower()

    if backbone not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型：{backbone}，支持列表：{SUPPORTED_MODELS}")

    if backbone == "custom_cnn":
        return CustomCNNModel(config=config)
    elif backbone == "vgg11":
        return VGG11Model(config=config)

# 【关键修正】更新__all__列表，明确包含FedModel（规范批量导出）
# __all__的作用：定义from models import * 时能导入的对象，同时让外部工具识别导出的类
__all__ = [
    "BaseModel",
    "CustomCNNModel",
    "VGG11Model",
    "SUPPORTED_MODELS",
    "get_model",
    "FedModel"  # 确保FedModel被明确导出，外部可正常调用
]