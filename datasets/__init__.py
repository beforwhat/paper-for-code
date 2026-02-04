# datasets/__init__.py
# 导出基类
from .base_dataset import BaseDataset

# 导出专属数据集子类
from .cifar10_dataset import CIFAR10Dataset
from .emnist_dataset import EMNISTDataset
from .svhn_dataset import SVHNDataset
from .femnist_dataset import FEMNISTDataset
from .simulation_dataset import get_simulation_dataset
# 导出Non-IID划分工具类
from .non_iid_partitioner import NonIIDPartitioner
from .dataset_factory import (
    get_dataset,
    get_client_dataset,  # 新增：正式训练必需
    get_global_test_dataset  # 新增：正式训练必需
)

# 定义支持的数据集列表，方便外部判断
SUPPORTED_DATASETS = [
    "cifar10",
    "emnist",
    "svhn",
    "femnist",
    "simulation"
]