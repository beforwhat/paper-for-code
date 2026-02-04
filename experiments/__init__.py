# experiments/__init__.py
"""
实验脚本模块（experiments）
核心定位：统一管理5大核心实验，无缝衔接项目新核心模块（SA贡献度、自适应裁剪DP等）
实验分类：
1. 基础性能对比：验证5大基线算法（FedAvg/DP-FedAvg/FedProx/Ditto/FedShap）的基础性能；
2. 隐私-效用权衡：验证自适应裁剪DP优化后的隐私（ε/δ）与模型效用（准确率/损失）Trade-off；
3. 组件消融：单独消融SA贡献度、优化后自适应裁剪等核心组件，验证各组件的作用；
4. 公平性验证：验证SA贡献度提升后的客户端选择/聚合公平性（如基尼系数、准确率方差）；
5. 效率与鲁棒性：验证SA贡献度在不同客户端数/异构程度下的稳定性、训练效率；
设计原则：
- 统一导出各实验的核心执行函数，简化外部调用；
- 所有实验函数均兼容项目核心配置，无缝衔接新核心模块；
- 新增实验仅需补充导入和__all__，无需修改调用逻辑。
"""

# ======================== 导入各实验脚本的核心执行函数 ========================
# 1. 基础性能对比实验（核心：5大基线算法性能对比）
from .basic_performance import run_basic_performance_experiment

# 2. 隐私-效用权衡实验（核心：验证自适应裁剪DP的优化效果）
from .privacy_utility import run_privacy_utility_experiment

# 3. 组件消融实验（核心：单独消融SA贡献度/自适应裁剪等组件）
from .ablation_study import run_fedfairadp_ala_ablation

# 4. 公平性验证实验（核心：验证SA贡献度的公平性提升）
from .fairness_verification import run_fairness_verification_experiment

# 5. 效率与鲁棒性实验（核心：验证SA贡献度的稳定性/效率）
from .efficiency_robustness import run_efficiency_robustness_experiment

# ======================== 声明模块公开接口（核心） ========================
# __all__ 定义「from experiments import *」时导入的实验函数，规范外部调用
__all__ = [
    # 基础性能对比
    "run_basic_performance_experiment",
    # 隐私-效用权衡
    "run_privacy_utility_experiment",
    # 组件消融
    "run_fedfairadp_ala_ablation",
    # 公平性验证
    "run_fairness_verification_experiment",
    # 效率与鲁棒性
    "run_efficiency_robustness_experiment"
]

# ======================== 可选：批量运行所有实验的便捷函数 ========================
def run_all_experiments(config=None, save_results=True, results_dir="./experiment_results"):
    """
    批量运行所有5大实验（便捷入口，适合全量验证）
    Args:
        config: 配置对象（默认加载全局配置）
        save_results: 是否保存实验结果（日志/图表/数据文件）
        results_dir: 结果保存目录
    """
    import os
    # 创建结果保存目录
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n=== 开始批量运行所有实验 | 结果保存至：{results_dir} ===")
    
    # 1. 基础性能对比
    print("\n--- 1/5 运行基础性能对比实验 ---")
    run_basic_performance_experiment(config=config, save_results=save_results, save_path=os.path.join(results_dir, "basic_performance"))
    
    # 2. 隐私-效用权衡
    print("\n--- 2/5 运行隐私-效用权衡实验 ---")
    run_privacy_utility_tradeoff_experiment(config=config, save_results=save_results, save_path=os.path.join(results_dir, "privacy_utility"))
    
    # 3. 组件消融
    print("\n--- 3/5 运行组件消融实验 ---")
    run_ablation_study_experiment(config=config, save_results=save_results, save_path=os.path.join(results_dir, "ablation_study"))
    
    # 4. 公平性验证
    print("\n--- 4/5 运行公平性验证实验 ---")
    run_fairness_verification_experiment(config=config, save_results=save_results, save_path=os.path.join(results_dir, "fairness_verification"))
    
    # 5. 效率与鲁棒性
    print("\n--- 5/5 运行效率与鲁棒性实验 ---")
    run_efficiency_robustness_experiment(config=config, save_results=save_results, save_path=os.path.join(results_dir, "efficiency_robustness"))
    
    print(f"\n=== 所有实验运行完成 | 结果已保存至：{results_dir} ===")

# ======================== 模块元信息（可选） ========================
__version__ = "1.0.0"  # 实验模块版本
__author__ = "Your Team/Name"  # 模块维护者
__description__ = "联邦学习核心实验集合，支撑SA贡献度+自适应裁剪DP的全维度验证"