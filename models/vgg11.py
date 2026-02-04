# models/vgg11.py
import torch
import torch.nn as nn
from .base_model import BaseModel

class VGG11Model(BaseModel):
    """
    VGG11模型：继承BaseModel，实现_build_network()抽象方法
    遵循VGG11经典结构，适配小尺寸数据集（32×32/28×28），支持config动态配置
    """
    def _build_network(self):
        """
        实现抽象方法：构建VGG11网络结构（features + classifier）
        从config中读取专属参数，自动适配不同数据集
        """
        # 1. 从config中提取VGG11专属配置
        vgg11_config = self.model_config.vgg11_config
        num_filters = vgg11_config["num_filters"]  # 卷积层滤波器数量列表
        fc_dim = vgg11_config["fc_dim"]  # 全连接层维度
        
        # 2. 构建特征提取层（self.features）
        # 结构：Conv→ReLU→Conv→ReLU→MaxPool → 重复该模式，最后一组Conv后接MaxPool
        features_layers = []
        in_channels = self.input_size[0]  # 输入通道数（3 for RGB，1 for 灰度图）
        
        for out_channels in num_filters:
            # 添加卷积层（3×3卷积，padding=1保持特征图尺寸不变）
            features_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            # 添加ReLU激活函数
            features_layers.append(nn.ReLU(inplace=True))
            # 更新输入通道数（下一层卷积的输入=当前层输出）
            in_channels = out_channels
        
        # 插入最大池化层（每2个卷积层后插入1个，VGG11共4个池化层）
        # 处理后的features结构：[Conv, ReLU, Conv, ReLU, MaxPool, ...]
        pool_positions = [3, 7, 11, 15]  # 对应每2组Conv+ReLU后的插入位置
        for pos in reversed(pool_positions):  # 倒序插入，避免打乱索引
            if pos < len(features_layers):
                features_layers.insert(
                    pos + 1,
                    nn.MaxPool2d(kernel_size=2, stride=2)  # 池化后尺寸缩小为1/2
                )
        
        # 封装为nn.Sequential，赋值给self.features
        self.features = nn.Sequential(*features_layers)
        
        # 3. 计算特征提取层输出尺寸（用于全连接层输入）
        # 模拟输入，自动计算输出尺寸（适配不同input_size，无需硬编码）
        with torch.no_grad():
            fake_input = torch.randn(1, *self.input_size)  # (batch=1, C, H, W)
            fake_feature = self.features(fake_input)
            feature_out_size = fake_feature.numel()  # 展平后的总元素个数
        
        # 4. 构建分类层（self.classifier）
        # 结构：Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear
        self.classifier = nn.Sequential(
            nn.Linear(feature_out_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(fc_dim, self.num_classes)  # 最后一层输出类别数
        )
    
    # 可选：重写参数初始化方法，适配VGG的最佳实践（Xavier初始化）
    def _init_weights(self):
        """
        重写基类的_init_weights，使用Xavier均匀分布初始化，提升模型收敛速度
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Xavier均匀分布初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # 全连接层：Xavier均匀分布初始化
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：保持基类的初始化逻辑
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)