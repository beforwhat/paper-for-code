import torch
import torch.nn as nn
from .base_model import BaseModel

class CustomCNNModel(BaseModel):
    def _build_network(self):
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels=self.input_size[0],out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=self.dropout_prob),
            # 第二层
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=self.dropout_prob),
            # 第三层
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=9),
            nn.Dropout(p=self.dropout_prob),
        )
        feature_out_size = 128 * (self.input_size[1] // 8) * (self.input_size[2] // 8)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_out_size,out_features=self.num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(in_features=self.model_config.hidden_dim,out_features=self.num_classes),
        )