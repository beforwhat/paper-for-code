# -*- coding: utf-8 -*-
"""
Utils 工具包 - 联邦学习实验通用工具集
核心功能：
1. 统一日志管理（logger）：实验过程日志记录、等级控制、文件输出
2. 指标计算（metrics）：包含SA贡献度精准度验证（皮尔逊相关系数）、公平性/鲁棒性/效率指标
3. 模型 checkpoint 管理：训练过程中模型保存/加载、断点续训
4. 可视化工具：SA贡献度波动图、阈值稳定性曲线、实验结果对比图等
5. 并行计算：加速Shapley值计算、多客户端并行训练
"""

# 包基本信息（可选，增强可读性）
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "通用工具包，支撑联邦学习SA贡献度验证实验的全流程"

# ======================== 核心组件导出（对外统一接口） ========================
# 遵循「最小暴露」原则：只导出外部常用的核心类/函数，不暴露内部实现细节

# 1. 日志工具 - 核心类
from .logger import Logger, setup_global_logger

# 2. 指标计算工具 - 核心类+关键函数（重点包含SA贡献度验证相关）
from .metrics import (
    MetricsCalculator,          # 通用指标计算主类
    calculate_pearson_corr,     # 皮尔逊相关系数（验证SA贡献度精准度）
    calculate_gini_coefficient, # 基尼系数（复用公平性实验）
    calculate_fairness_metrics, # 多维度公平性指标
    calculate_robustness_metrics, # 鲁棒性指标
    calculate_efficiency_metrics  # 效率指标
)

# 3. Checkpoint 工具 - 核心类
from .checkpoint import CheckpointManager

# 4. 可视化工具 - 核心类+常用函数（SA贡献度相关可视化）
from .visualization import (
    Visualizer,                 # 可视化主类
    plot_sa_contribution_trend, # SA贡献度波动趋势图
    plot_threshold_stability,   # 自适应阈值稳定性曲线
    plot_experiment_comparison, # 多算法实验结果对比
    plot_robustness_heatmap     # 鲁棒性场景热力图
)

# 5. 并行计算工具 - 核心类+函数
from .parallel import (
    ParallelRunner,             # 多客户端并行训练器
    parallel_shapley_calculate  # 并行计算Shapley值（加速SA贡献度计算）
)

# ======================== 批量导入控制（支持 from utils import *） ========================
# 定义__all__，明确导出的成员列表，避免导入无关内容
__all__ = [
    # 日志工具
    "Logger",
    "setup_global_logger",
    # 指标计算
    "MetricsCalculator",
    "calculate_pearson_corr",
    "calculate_gini_coefficient",
    "calculate_fairness_metrics",
    "calculate_robustness_metrics",
    "calculate_efficiency_metrics",
    # Checkpoint
    "CheckpointManager",
    # 可视化
    "Visualizer",
    "plot_sa_contribution_trend",
    "plot_threshold_stability",
    "plot_experiment_comparison",
    "plot_robustness_heatmap",
    # 并行计算
    "ParallelRunner",
    "parallel_shapley_calculate"
]