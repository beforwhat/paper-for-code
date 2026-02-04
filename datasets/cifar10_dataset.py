# datasets/cifar10_dataset.py
import torchvision
from torchvision import transforms
from .base_dataset import BaseDataset

class CIFAR10Dataset(BaseDataset):
    """
    CIFAR-10专属数据集类：继承BaseDataset，实现专属抽象方法
    """
    def _get_transform(self):
        """
        实现抽象方法：CIFAR-10专属数据增强/归一化
        区分训练集和测试集（训练集数据增强，测试集仅归一化）
        """
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        if self.is_train:
            # 训练集：随机裁剪、随机翻转、归一化
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 测试集：仅归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return transform
    
    def _load_raw_data(self):
        """
        实现抽象方法：下载/加载CIFAR-10原始数据
        赋值给self.raw_dataset，供基类后续处理
        """
        self.raw_dataset = torchvision.datasets.CIFAR10(
            root=self.raw_data_path,  # 对应data/raw/cifar10/
            train=self.is_train,
            download=True,  # 自动下载（若不存在）
            transform=self._get_transform()  # 调用专属Transform
        )