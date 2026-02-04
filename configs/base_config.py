import os
import torch


def get_base_config():
    """
    基础公共配置：存放所有模块依赖的公共参数，提供合理默认值
    """
    # 1. 项目根路径（自动获取，无需修改，适配不同环境）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. 数据存储路径（与data/目录对应，无需硬编码）
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH = os.path.join(DATA_ROOT, "raw")
    PROCESSED_DATA_PATH = os.path.join(DATA_ROOT, "processed")

    # 3. 结果/模型保存路径（与results/目录对应）
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
    MODEL_SAVE_PATH = os.path.join(RESULTS_ROOT, "saved_models")
    LOG_SAVE_PATH = os.path.join(RESULTS_ROOT, "logs")

    # 4. 硬件配置（自动适配GPU/CPU，无需手动修改）
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    PIN_MEMORY = True if torch.cuda.is_available() else False  # 加速GPU数据加载

    # 5. 数据集基础配置（与datasets/模块协同）
    DATASET_CONFIG = {
        "name": "femnist",  # 默认数据集：cifar10，可选：emnist、svhn、femnist
        "non_iid_alpha": 0.1,  # Non-IID α值，越小Non-IID程度越高，默认0.1（极端Non-IID）
        "num_classes": 62,  # 默认类别数：cifar10为10，femnist emnist为62，可后续被模型配置覆盖
    }

    # 6. 训练基础配置（所有训练流程的公共参数）
    TRAIN_CONFIG = {
        "batch_size": 64,  # 批次大小，默认64（可根据GPU显存调整）
        "num_workers": 4,  # 数据加载工作线程数，默认4（根据CPU核心数调整）
        "lr": 0.01,  # 基础学习率，默认0.01
        "weight_decay": 5e-4,  # 权重衰减，防止过拟合，默认5e-4
        "momentum": 0.9,  # 优化器动量，默认0.9（适用于SGD）
        "max_epochs": 200,  # 全局通信轮次（联邦学习）/ 训练轮次（单机）
    }

    # 7. 日志/模型保存配置
    SAVE_CONFIG = {
        "save_model": True,  # 是否保存训练好的模型
        "save_freq": 50,  # 每50轮保存一次模型
        "enable_log": True,  # 是否开启日志记录
        "log_freq": 10,  # 每10轮打印一次训练日志
    }

    # 8. 整合所有基础配置为字典（方便后续统一加载）
    base_config = {
        "project_root": PROJECT_ROOT,
        "data_root": DATA_ROOT,
        "raw_data_path": RAW_DATA_PATH,
        "processed_data_path": PROCESSED_DATA_PATH,
        "results_root": RESULTS_ROOT,
        "model_save_path": MODEL_SAVE_PATH,
        "log_save_path": LOG_SAVE_PATH,
        "device": DEVICE,
        "num_gpus": NUM_GPUS,
        "pin_memory": PIN_MEMORY,
        "dataset": DATASET_CONFIG,
        "train": TRAIN_CONFIG,
        "save": SAVE_CONFIG,
    }

    # 创建目录（若不存在），确保路径有效
    for path in [DATA_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH, RESULTS_ROOT, MODEL_SAVE_PATH, LOG_SAVE_PATH]:
        os.makedirs(path, exist_ok=True)

    return base_config