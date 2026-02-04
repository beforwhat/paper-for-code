# datasets/femnist_dataset.py
import torchvision
from torchvision import transforms
from .base_dataset import BaseDataset

class FEMNISTDataset(BaseDataset):
    """
    FEMNIST专属数据集类：用EMNIST替代（快速落地，兼容框架）
    核心：EMNIST byclass与FEMNIST数据分布接近，均为62类手写字符
    """
    def _get_transform(self):
        """
        实现抽象方法：FEMNIST专属数据增强+归一化（单通道灰度图，28×28）
        """
        # 1. FEMNIST专属归一化参数（单通道，固定统计值）
        femnist_mean = [0.1751]
        femnist_std = [0.3332]
        
        if self.is_train:
            # 2. 训练集：随机旋转+随机平移+归一化
            transform = transforms.Compose([
                transforms.RandomRotation(15),  # 随机旋转±15度，适配手写字符
                transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移10%
                transforms.ToTensor(),
                transforms.Normalize(mean=femnist_mean, std=femnist_std)
            ])
        else:
            # 3. 测试集：仅归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=femnist_mean, std=femnist_std)
            ])
        
        return transform
    
    def _load_raw_data(self):
        """
        实现抽象方法：用EMNIST byclass替代FEMNIST，快速落地
        """
        self.raw_dataset = torchvision.datasets.EMNIST(
            root=self.raw_data_path,  # 对应data/raw/femnist/
            split="byclass",  # 62类（0-9, a-z, A-Z），与FEMNIST一致
            train=self.is_train,
            download=True,
            transform=self._get_transform()
        )