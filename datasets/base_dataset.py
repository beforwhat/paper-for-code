# datasets/base_dataset.py
import os
import numpy as np
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, Subset
from configs.base_config import get_base_config


class BaseDataset(ABC):


    def __init__(self, config, is_train=True, client_id=None):
        # 1. 保存配置参数
        self.config = config
        self.dataset_name = config.dataset.name  # 如 "cifar10"
        self.num_clients = config.fed.num_clients  # 客户端数量
        self.non_iid_alpha = config.dataset.non_iid_alpha  # Non-IID α值
        self.batch_size = config.train.batch_size
        self.is_train = is_train  # 训练集/测试集标识
        self.client_id = client_id  # 客户端ID（None表示全局数据集）

        base_config_dict = get_base_config()
        self.RAW_DATA_PATH_FROM_CONFIG = base_config_dict["raw_data_path"]
        self.PROCESSED_DATA_PATH_FROM_CONFIG = base_config_dict["processed_data_path"]
        self.raw_data_path = os.path.join(self.RAW_DATA_PATH_FROM_CONFIG, self.dataset_name)
        self.processed_data_path = os.path.join(
            self.PROCESSED_DATA_PATH_FROM_CONFIG,  # 使用提取到的路径，不再直接用PROCESSED_DATA_PATH
            f"{self.dataset_name}_non_iid_alpha_{self.non_iid_alpha}"
        )
        # 创建目录（若不存在）
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)

        # 3. 初始化核心变量
        self.raw_dataset = None  # 原始数据集
        self.client_indices = None  # 客户端样本索引（shape: [num_clients, num_samples_per_client]）
        self.target_dataset = None  # 最终目标数据集（全局/客户端子集）

        # 4. 加载数据（通用流程：加载原始数据 → 划分/加载客户端索引 → 生成目标数据集）
        self._load_raw_data()  # 抽象方法，子类实现
        self._handle_client_partition()  # 通用逻辑：处理Non-IID划分/加载
        self._generate_target_dataset()  # 通用逻辑：生成目标数据集

    @abstractmethod
    def _get_transform(self):
        """
        抽象方法：返回数据集专属的数据增强/归一化Transform
        子类必须实现（不同数据集的Transform不同）
        """
        pass

    @abstractmethod
    def _load_raw_data(self):
        """
        抽象方法：加载/下载原始数据集，赋值给self.raw_dataset
        子类必须实现（不同数据集的下载/加载逻辑不同）
        """
        pass

    def _partition_data(self):
        """
        通用方法：调用NonIIDPartitioner进行Non-IID划分（仅训练集需要）
        子类可重写（若有专属划分逻辑，如FEMNIST按用户分组）
        """
        if not self.is_train:
            # 测试集无需划分，全局共享
            return None

        # 提取原始数据集的标签（适配PyTorch数据集格式）
        try:
            labels = np.array(self.raw_dataset.targets)
        except AttributeError:
            labels = np.array([y for _, y in self.raw_dataset])

        # 调用Non-IID划分工具类
        partitioner = NonIIDPartitioner(
            num_clients=self.num_clients,
            alpha=self.non_iid_alpha
        )
        # 执行Dirichlet划分（默认），返回客户端索引
        client_indices = partitioner.dirichlet_partition(labels)

        # 保存划分结果到data/processed/，方便后续复用
        self._save_processed_data(client_indices, "client_indices.npy")

        return client_indices

    def _save_processed_data(self, data, filename):
        """
        通用方法：保存处理后的数据到data/processed/（缓存复用）
        """
        save_path = os.path.join(self.processed_data_path, filename)
        np.save(save_path, data)

    def _load_processed_data(self, filename):
        """
        通用方法：从data/processed/加载处理后的数据（缓存复用）
        """
        load_path = os.path.join(self.processed_data_path, filename)
        if not os.path.exists(load_path):
            return None
        return np.load(load_path, allow_pickle=True).item()

    def _handle_client_partition(self):
        """
        通用逻辑：处理客户端划分（优先加载缓存，无缓存则重新划分）
        """
        if self.client_id is None and not self.is_train:
            # 全局测试集，无需划分
            self.client_indices = None
            return

        # 尝试加载缓存的客户端索引
        self.client_indices = self._load_processed_data("client_indices.npy")

        # 无缓存则重新划分
        if self.client_indices is None:
            self.client_indices = self._partition_data()

    def _generate_target_dataset(self):
        """
        通用逻辑：生成最终目标数据集（全局数据集/指定客户端子集）
        """
        if self.client_id is None:
            # 无client_id，返回全局数据集
            self.target_dataset = self.raw_dataset
        else:
            # 有client_id，返回该客户端的子集（仅训练集有效）
            if not self.is_train:
                raise ValueError("测试集不支持按客户端划分，全局共享")
            if self.client_id < 0 or self.client_id >= self.num_clients:
                raise ValueError(f"client_id超出范围[0, {self.num_clients - 1}]")

            # 提取该客户端的样本索引
            client_sample_indices = self.client_indices[self.client_id]
            self.target_dataset = Subset(self.raw_dataset, client_sample_indices)

    def get_dataloader(self, shuffle=True):
        """
        核心对外接口：统一返回PyTorch DataLoader
        外部调用仅需该方法，无需关心内部实现
        """
        dataloader = DataLoader(
            dataset=self.target_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and self.is_train,  # 训练集shuffle，测试集不shuffle
            num_workers=self.config.train.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        return dataloader