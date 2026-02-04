# datasets/emnist_dataset.py
import torchvision
from torchvision import transforms
from .base_dataset import BaseDataset

class EMNISTDataset(BaseDataset):
    """
    EMNIST专属数据集类：继承BaseDataset，实现专属抽象方法
    """
    def _get_transform(self):
        """EMNIST专属Transform（单通道灰度图）"""
        mean = [0.1307]
        std = [0.3081]
        
        if self.is_train:
            transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return transform
    
    def _load_raw_data(self):
        """加载EMNIST原始数据"""
        self.raw_dataset = torchvision.datasets.EMNIST(
            root=self.raw_data_path,
            split="byclass",  # EMNIST专属参数
            train=self.is_train,
            download=True,
            transform=self._get_transform()
        )