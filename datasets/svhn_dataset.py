# datasets/svhn_dataset.py
import torchvision
from torchvision import transforms
from .base_dataset import BaseDataset

class SVHNDataset(BaseDataset):
    """
    SVHN专属数据集类：继承BaseDataset，实现两个抽象方法
    核心差异：SVHN用split参数区分训练/测试集，而非train参数
    """
    def _get_transform(self):
        """
        实现抽象方法：SVHN专属数据增强+归一化（3通道RGB，32×32）
        数据增强策略与CIFAR10一致，仅替换归一化参数
        """
        # 1. SVHN专属归一化参数（3通道，固定统计值）
        svhn_mean = [0.4377, 0.4438, 0.4728]
        svhn_std = [0.1980, 0.2010, 0.1970]
        
        if self.is_train:
            # 2. 训练集：随机裁剪+随机水平翻转+归一化（和CIFAR10一致）
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=svhn_mean, std=svhn_std)
            ])
        else:
            # 3. 测试集：仅归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=svhn_mean, std=svhn_std)
            ])
        
        return transform
    
    def _load_raw_data(self):
        """
        实现抽象方法：加载/下载SVHN原始数据
        核心差异：使用split参数，而非train参数
        """
        # 1. 根据self.is_train映射split参数（基类用is_train，适配SVHN的split）
        split = "train" if self.is_train else "test"
        
        # 2. 调用torchvision.datasets.SVHN加载数据
        self.raw_dataset = torchvision.datasets.SVHN(
            root=self.raw_data_path,  # 对应data/raw/svhn/，基类已创建
            split=split,  # SVHN专属参数：区分训练/测试集
            download=True,  # 自动下载（仅第一次运行）
            transform=self._get_transform()  # 传入专属transform
        )