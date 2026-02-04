# models/base_model.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
from abc import ABC, abstractmethod

# 兜底：将项目根目录添加到sys.path，确保能导入configs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import load_config

class BaseModel(nn.Module, ABC):
    """
    模型基类：继承nn.Module和ABC，抽离所有模型的通用逻辑，定义统一接口
    所有模型子类（如VGG11、CustomCNN）必须继承此类，并实现抽象方法
    """
    def __init__(self, config=None):
        super(BaseModel, self).__init__()
        
        # 1. 初始化配置（默认加载全局配置，也可传入自定义配置）
        self.config = config if config is not None else load_config()
        self.model_config = self.config.model  # 提取模型专属配置
        self.device = self.config.device  # 提取硬件设备（CPU/GPU）
        
        # 2. 提取模型核心参数（从config中读取，无需硬编码）
        self.input_size = self.model_config.input_size  # 输入尺寸：(C, H, W)
        self.num_classes = self.model_config.num_classes  # 输出类别数
        self.dropout_prob = self.model_config.dropout_prob  # Dropout概率
        self.backbone = self.model_config.backbone  # 骨干网络名称
        
        # 3. 核心变量初始化（子类构建网络后赋值）
        self.features = None  # 特征提取层（如卷积层、池化层）
        self.classifier = None  # 分类层（如全连接层）
        self.loss_fn = None  # 损失函数
        
        # 4. 通用流程：构建网络→初始化参数→迁移设备→定义损失函数
        self._build_network()  # 抽象方法，子类实现专属网络结构
        self._init_weights()  # 通用方法，参数初始化
        self.to(self.device)  # 迁移模型到指定设备（GPU/CPU）
        self._init_loss_fn()  # 通用方法，初始化损失函数
    
    @abstractmethod
    def _build_network(self):
        """
        抽象方法：构建模型专属网络结构（特征提取层+分类层）
        子类必须实现，需赋值self.features和self.classifier（或直接定义网络层）
        """
        pass
    
    def _init_weights(self):
        """
        通用方法：模型参数默认初始化（正态分布）
        子类可重写此方法，实现专属参数初始化（如Xavier、Kaiming）
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 卷积层和全连接层：正态分布初始化
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：权重1，偏置0
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
    
    def _init_loss_fn(self):
        """
        通用方法：初始化默认损失函数（交叉熵损失，适配分类任务）
        子类可重写此方法，实现专属损失函数（如MSE、Focal Loss）
        """
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
    
    def forward(self, x):
        """
        通用方法：模型前向传播（兼容PyTorch生态）
        子类可重写此方法，实现复杂前向逻辑（如残差连接、多分支输出）
        """
        # 步骤1：将输入数据迁移到指定设备（避免设备不匹配报错）
        x = x.to(self.device)
        
        # 步骤2：特征提取（子类在_build_network中定义self.features）
        if self.features is not None:
            x = self.features(x)
        
        # 步骤3：展平特征图（适配全连接层输入）
        x = torch.flatten(x, 1)  # 保留batch维度，展平其余维度：(B, C, H, W) → (B, C*H*W)
        
        # 步骤4：分类输出（子类在_build_network中定义self.classifier）
        if self.classifier is not None:
            x = self.classifier(x)
        
        return x
    
    def get_params(self):
        """
        联邦学习核心方法：获取模型所有可训练参数（返回numpy数组）
        用于客户端将本地模型参数上传到服务端进行聚合
        """
        params = []
        for param in self.parameters():
            # 迁移到CPU→转为numpy数组→加入列表（兼容不同设备，方便聚合）
            params.append(param.detach().cpu().numpy())
        return params
    
    def set_params(self, new_params):
        """
        联邦学习核心方法：设置模型参数（接收服务端聚合后的参数）
        用于客户端加载服务端下发的全局聚合参数
        Args:
            new_params: 聚合后的模型参数列表（与get_params返回格式一致）
        """
        for param, new_param in zip(self.parameters(), new_params):
            # 转换为torch张量→迁移到指定设备→更新模型参数
            new_param_tensor = torch.tensor(new_param, dtype=torch.float32).to(self.device)
            param.data = new_param_tensor.data
    
    def save_model(self, epoch=None, model_name=None):
        """
        通用方法：保存模型权重到config.model_save_path
        Args:
            epoch: 训练轮次（用于命名，区分不同阶段的模型）
            model_name: 模型名称（默认使用backbone+数据集名称）
        """
        # 1. 确定模型保存名称
        if model_name is None:
            model_name = f"{self.backbone}_{self.model_config.dataset_name}"
        if epoch is not None:
            model_name = f"{model_name}_epoch_{epoch}"
        model_path = os.path.join(self.config.model_save_path, f"{model_name}.pth")
        
        # 2. 保存模型权重（仅保存状态字典，节省空间，便于加载）
        torch.save(self.state_dict(), model_path)
        print(f"模型已保存到：{model_path}")
    
    def load_model(self, model_path=None, epoch=None, model_name=None):
        """
        通用方法：从指定路径加载模型权重
        Args:
            model_path: 直接指定模型路径（优先级最高）
            epoch: 训练轮次（用于拼接路径）
            model_name: 模型名称（用于拼接路径）
        """
        # 1. 确定模型加载路径
        if model_path is None:
            if model_name is None:
                model_name = f"{self.backbone}_{self.model_config.dataset_name}"
            if epoch is not None:
                model_name = f"{model_name}_epoch_{epoch}"
            model_path = os.path.join(self.config.model_save_path, f"{model_name}.pth")
        
        # 2. 加载模型权重并迁移到指定设备
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"模型已从：{model_path} 加载完成")



