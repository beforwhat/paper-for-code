# models/simple_mlp.py
"""测试版SimpleMLP模型（极简实现）"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """测试用简单MLP模型"""
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [256, 128],
        num_classes: int = 10,
        dropout: float = 0.1  # 测试用默认值
    ):
        super().__init__()
        # 极简MLP结构（满足测试导入+前向传播）
        layers = []
        prev_dim = input_dim
        for hid_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hid_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hid_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（测试核心）"""
        # 适配输入形状（如MNIST的[batch, 1, 28, 28] → [batch, 784]）
        if len(x.shape) > 2:
            x = x.flatten(1)
        return self.model(x)

# 测试用例（可选，验证模型可运行）
if __name__ == "__main__":
    model = SimpleMLP()
    test_input = torch.randn(32, 784)  # batch_size=32，输入维度784
    output = model(test_input)
    print(f"测试输出形状：{output.shape}")  # 应输出torch.Size([32, 10])