#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目入口文件（适配FedFairADP-ALA结构）
核心功能：
1. 命令行参数解析（实验类型、配置路径、测试模式等）；
2. 批量运行指定实验（基础性能/隐私效用/消融/公平性/效率鲁棒性）；
3. 模块功能自测（验证utils/模型/数据集等核心组件）；
4. 统一日志/结果管理。
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ========== 项目路径配置（确保所有模块可导入） ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 核心模块导入（适配新目录结构） ==========
# 工具类（路径正确，无修改）
from utils.logger import setup_global_logger, info, error
from utils.metrics import MetricsCalculator
from utils.checkpoint import create_checkpoint_manager
from utils.visualization import create_visualizer, plot_experiment_summary
from utils.parallel import accelerate_shapley_calculation, accelerate_client_training

# 数据集（路径正确，无修改）
from datasets import get_simulation_dataset  # 注：需在datasets/__init__.py导出该函数
from datasets.non_iid_partitioner import DirichletPartitioner  # 测试Non-IID划分

# 模型（路径正确，无修改）
from models import SimpleMLP  # 注：需在models/__init__.py导出该类（或替换为vgg11/custom_cnn）
from models.fed_model import FedModel  # 联邦模型封装

# 核心模块（适配新层级：core/federated/core/shapley/core/ala等）
from core.federated.server import BaseServer  # 联邦服务器（修改后聚合逻辑）
from core.federated.client import BaseClient  # 联邦客户端
from core.shapley.shapley_calculator import ShapleyCalculator  # SA贡献度计算
from core.ala.ala_optimizer import ALAOptimizer  # ALA特征提取
from core.dp.adaptive_clipping_dp import AdaptiveClippingDP  # 优化后自适应裁剪

# 基线算法（路径正确，无修改）
from baselines import fedavg, dp_fedavg, fedprox, ditto, fedshap

# 实验脚本（适配新名称：privacy_utility.py新增）
from experiments import (
    basic_performance,
    privacy_utility,
    ablation_study,
    fairness_verification,
    efficiency_robustness
)

# 配置文件（适配新configs结构）
from configs import (
    base_config,
    model_config,
    fed_config,
    dp_config,
    shapley_config,
    experiment_config
)

# ========== 命令行参数解析（适配新实验类型：新增privacy_utility） ==========
def parse_args():
    parser = argparse.ArgumentParser(description="FedFairADP-ALA 联邦学习SA贡献度验证实验入口")
    # 核心参数
    parser.add_argument("--experiment", type=str, default="test",
                        choices=["baseline", "privacy", "ablation", "fairness", "efficiency", "test"],
                        help="实验类型：baseline(基础性能)/privacy(隐私效用)/ablation(消融)/fairness(公平性)/efficiency(效率鲁棒性)/test(模块自测)")
    parser.add_argument("--config", type=str, default="configs/base_config.py",
                        help="实验配置文件路径（base/model/fed/dp/shapley/experiment）")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="日志等级")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="并行进程数（默认CPU核心数-1）")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="GPU ID列表（逗号分隔，如0,1,2）")
    return parser.parse_args()

# ========== 模块自测功能（适配新结构） ==========
def run_module_test():
    """
    运行所有核心模块的简单自测，验证功能可用性（适配FedFairADP-ALA结构）
    """
    info("========== 开始模块自测（适配FedFairADP-ALA） ==========")
    test_results = {"success": [], "failed": []}
    
    # ---------------- 1. 日志模块测试（路径正确） ----------------
    try:
        setup_global_logger(
            experiment_name="module_test",
            log_level="info",
            log_dir="./logs/test"
        )
        info("✅ 日志模块初始化成功")
        test_results["success"].append("logger")
    except Exception as e:
        error(f"❌ 日志模块测试失败：{str(e)}")
        test_results["failed"].append(f"logger: {str(e)}")
    
    # ---------------- 2. 指标计算模块测试（路径正确） ----------------
    try:
        calculator = MetricsCalculator()
        # 测试SA贡献度精准度
        true_contrib = [0.8, 0.5, 0.3, 0.9, 0.2]
        pred_contrib = [0.78, 0.52, 0.29, 0.89, 0.21]
        sa_metrics = calculator.calculate_sa_contribution_metrics(true_contrib, pred_contrib)
        # 测试公平性指标
        client_perfs = {0: 85.2, 1: 84.8, 2: 86.1, 3: 85.5}
        fairness_metrics = calculator.calculate_fairness(client_perfs)
        info(f"✅ 指标模块测试成功（皮尔逊系数：{sa_metrics['pearson_corr']:.4f}，基尼系数：{fairness_metrics['gini']:.4f}）")
        test_results["success"].append("metrics")
    except Exception as e:
        error(f"❌ 指标模块测试失败：{str(e)}")
        test_results["failed"].append(f"metrics: {str(e)}")
    
    # ---------------- 3. Checkpoint模块测试（路径正确） ----------------
    try:
        # 初始化联邦模型（适配models/fed_model.py）
        model = FedModel(backbone="custom_cnn", num_classes=62)  # 适配FEMNIST
        # 创建管理器
        ckpt_manager = create_checkpoint_manager(
            experiment_name="module_test",
            checkpoint_dir="./checkpoints/test",
            device="cpu"
        )
        # 保存模型
        ckpt_manager.save_checkpoint(
            round_idx=1,
            models={"fed_model": model},
            metrics={"accuracy": 85.0}
        )
        # 加载模型
        ckpt_manager.load_checkpoint(target="latest", models={"fed_model": model})
        info("✅ Checkpoint模块测试成功（保存/加载联邦模型）")
        test_results["success"].append("checkpoint")
    except Exception as e:
        error(f"❌ Checkpoint模块测试失败：{str(e)}")
        test_results["failed"].append(f"checkpoint: {str(e)}")
    
    # ---------------- 4. 可视化模块测试（路径正确） ----------------
    try:
        viz = create_visualizer(
            experiment_name="module_test",
            save_dir="./visualizations/test"
        )
        # 测试SA贡献度趋势图
        sa_contributions = {0: [0.12, 0.34], 1: [0.15, 0.32], 2: [0.14, 0.35]}
        viz.plot_sa_contribution_trend(sa_contributions)
        # 测试公平性柱状图
        fairness_data = {"FedAvg": 0.2345, "FedShap": 0.1234}
        viz.plot_fairness_metrics(fairness_data)
        info("✅ 可视化模块测试成功（生成测试图表）")
        test_results["success"].append("visualization")
    except Exception as e:
        error(f"❌ 可视化模块测试失败：{str(e)}")
        test_results["failed"].append(f"visualization: {str(e)}")
    
    # ---------------- 5. 并行计算模块测试（路径正确） ----------------
    try:
        # 测试1：并行计算Shapley值（适配core/shapley/shapley_calculator.py）
        def mock_shapley_func(client_id, data, model, **kwargs):
            calculator = ShapleyCalculator(sampling_method="group_monte_carlo")
            return calculator.calculate_raw_shapley(client_id, data, model)  # 模拟原始Shapley计算
        client_data = {0: None, 1: None, 2: None}
        model = SimpleMLP(input_dim=10, output_dim=2)
        shapley_values = accelerate_shapley_calculation(
            client_data=client_data,
            model=model,
            calculate_func=mock_shapley_func,
            auto_gpu=False,  # 测试CPU模式
            n_workers=2
        )
        # 测试2：并行客户端训练（适配core/federated/client.py）
        client_datasets = {0: [(None, None)] * 2, 1: [(None, None)] * 2}
        train_config = {
            "model_cls": SimpleMLP,
            "model_kwargs": {"input_dim": 10, "output_dim": 2},
            "optimizer_cls": lambda params, **kwargs: None,  # 模拟优化器
            "loss_fn": lambda x, y: 0.1,
            "epochs": 1
        }
        client_results = accelerate_client_training(
            client_datasets=client_datasets,
            global_model=model,
            train_config=train_config,
            auto_gpu=False,
            n_workers=2
        )
        info(f"✅ 并行模块测试成功（Shapley值：{shapley_values}，训练客户端数：{len(client_results)}）")
        test_results["success"].append("parallel")
    except Exception as e:
        error(f"❌ 并行模块测试失败：{str(e)}")
        test_results["failed"].append(f"parallel: {str(e)}")
    
    # ---------------- 6. 数据集模块测试（路径正确） ----------------
    try:
        # 加载模拟数据集 + 测试Non-IID划分
        train_data, test_data = get_simulation_dataset(num_samples=100, input_dim=10)
        partitioner = DirichletPartitioner(train_data, num_clients=3, alpha=0.5)  # Non-IID划分
        client_indices = partitioner.partition()
        info(f"✅ 数据集模块测试成功（训练集大小：{len(train_data)}，Non-IID划分客户端数：{len(client_indices)}）")
        test_results["success"].append("datasets")
    except Exception as e:
        error(f"❌ 数据集模块测试失败：{str(e)}")
        test_results["failed"].append(f"datasets: {str(e)}")
    
    # ---------------- 7. 模型模块测试（路径正确） ----------------
    try:
        model = SimpleMLP(input_dim=10, output_dim=2)  # 基础模型
        fed_model = FedModel(backbone="vgg11", num_classes=10)  # 联邦模型封装
        dummy_input = model.dummy_input()
        output = fed_model(dummy_input)
        info(f"✅ 模型模块测试成功（VGG11输入维度：{dummy_input.shape}，输出维度：{output.shape}）")
        test_results["success"].append("models")
    except Exception as e:
        error(f"❌ 模型模块测试失败：{str(e)}")
        test_results["failed"].append(f"models: {str(e)}")
    
    # ---------------- 8. 核心算法模块测试（适配新层级） ----------------
    try:
        # 测试1：SA贡献度计算（core/shapley/shapley_calculator.py）
        shap_calc = ShapleyCalculator(sampling_method="group_monte_carlo", smooth_coeff=0.1)
        # 测试2：ALA特征提取（core/ala/ala_optimizer.py）
        ala_optimizer = ALAOptimizer(learning_rate=0.001)
        ala_features = ala_optimizer.extract_ala_features(model, dummy_input)  # 新增的特征提取
        # 测试3：自适应裁剪DP（core/dp/adaptive_clipping_dp.py）
        dp_module = AdaptiveClippingDP(lamda=0.5, theta=0.1, adjust_upper=0.05)
        info(f"✅ 核心算法模块测试成功（ALA特征维度：{len(ala_features)}，DP参数λ={dp_module.lamda}）")
        test_results["success"].append("core")
    except Exception as e:
        error(f"❌ 核心算法模块测试失败：{str(e)}")
        test_results["failed"].append(f"core: {str(e)}")
    
    # ---------------- 9. 基线算法模块测试（路径正确） ----------------
    try:
        fedavg_alg = fedavg.FedAvg(model=model, num_clients=3)
        fedavg_alg.train_round(round_idx=1, client_datasets={0: None, 1: None})
        info("✅ 基线算法模块测试成功（FedAvg训练轮次）")
        test_results["success"].append("baselines")
    except Exception as e:
        error(f"❌ 基线算法模块测试失败：{str(e)}")
        test_results["failed"].append(f"baselines: {str(e)}")
    
    # ---------------- 10. 配置模块测试（适配新configs结构） ----------------
    try:
        # 加载各配置
        base_cfg = base_config.get_base_config()
        dp_cfg = dp_config.get_dp_config()
        shap_cfg = shapley_config.get_shapley_config()
        info(f"✅ 配置模块测试成功（基础配置客户端数：{base_cfg['num_clients']}，DPλ：{dp_cfg['lamda']}，Shapley平滑系数：{shap_cfg['smooth_coeff']}）")
        test_results["success"].append("configs")
    except Exception as e:
        error(f"❌ 配置模块测试失败：{str(e)}")
        test_results["failed"].append(f"configs: {str(e)}")
    
    # ========== 自测结果汇总 ==========
    info("========== 自测结果汇总 ==========")
    info(f"✅ 成功模块：{test_results['success']}（共{len(test_results['success'])}个）")
    if test_results["failed"]:
        error(f"❌ 失败模块：{test_results['failed']}（共{len(test_results['failed'])}个）")
    else:
        info("🎉 所有模块自测通过！")
    return test_results

# ========== 实验运行调度（适配新实验脚本） ==========
def run_experiment(args):
    """
    根据命令行参数调度对应实验（适配FedFairADP-ALA实验脚本）
    """
    info(f"========== 开始运行实验：{args.experiment} ==========")
    start_time = time.time()
    
    # 解析GPU ID
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if args.gpu_ids else None
    
    # 加载配置（适配新configs结构）
    if "base" in args.config:
        config = base_config.get_base_config()
    elif "dp" in args.config:
        config = dp_config.get_dp_config()
    elif "shapley" in args.config:
        config = shapley_config.get_shapley_config()
    elif "experiment" in args.config:
        config = experiment_config.get_experiment_config()
    else:
        config = base_config.get_base_config()
    
    config.update({
        "n_workers": args.n_workers,
        "gpu_ids": gpu_ids,
        "log_level": args.log_level
    })
    
    # 运行对应实验
    if args.experiment == "baseline":
        # 基础性能实验（FedShap vs 基线算法）
        basic_performance.run_basic_performance(config)
    elif args.experiment == "privacy":
        # 隐私-效用权衡实验（验证自适应裁剪优化效果）
        privacy_utility.run_privacy_utility(config)
    elif args.experiment == "ablation":
        # 消融实验（验证SA组件、自适应裁剪有效性）
        ablation_study.run_ablation_study(config)
    elif args.experiment == "fairness":
        # 公平性验证实验
        fairness_verification.run_fairness_verification(config)
    elif args.experiment == "efficiency":
        # 效率鲁棒性验证实验
        efficiency_robustness.run_efficiency_robustness(config)
    elif args.experiment == "test":
        # 模块自测
        run_module_test()
    
    # 实验耗时
    elapsed_time = time.time() - start_time
    info(f"========== 实验完成！总耗时：{elapsed_time:.2f}秒 ==========")

# ========== 主函数 ==========
def main():
    # 解析参数
    args = parse_args()
    
    # 初始化全局日志（先于所有操作）
    setup_global_logger(
        experiment_name=f"exp_{args.experiment}",
        log_level=args.log_level,
        log_dir="./logs"
    )
    
    # 运行实验/测试
    run_experiment(args)

if __name__ == "__main__":
    main()