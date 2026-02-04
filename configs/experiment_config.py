
def get_experiment_config():

    experiment_config = {
        # 1. 实验基本信息（方便结果区分与复现）
        "experiment_name": "FedFairADP-ALA_SA_DP_Opt",  # 实验名称
        "experiment_desc": "验证SA融合贡献度与优化后自适应裁剪的效果",  # 实验描述
        "experiment_id": "{experiment_name}_v1",  # 实验唯一标识（用于保存结果）

        # 2. 实验类型配置（选择需要运行的实验）
        "experiment_type": "all",  # 默认all，可选：basic_performance、privacy_utility、ablation、fairness、efficiency
        "run_basic_performance": True,  # 是否运行基础性能对比实验
        "run_privacy_utility": True,  # 是否运行隐私-效用权衡实验
        "run_ablation": True,  # 是否运行消融实验
        "run_fairness": True,  # 是否运行公平性验证实验
        "run_efficiency": True,  # 是否运行效率与鲁棒性实验

        # 3. 消融实验变量配置（指定需要消融的组件）
        "ablation_components": [
            "sa_fusion",  # 消融SA融合贡献度
            "optimized_dp_clipping",  # 消融优化后自适应裁剪
            "ala_module",  # 消融ALA模块
        ],

        # 4. 结果可视化与保存配置
        "enable_visualization": True,  # 是否开启结果可视化
        "save_figure": True,  # 是否保存可视化图表
        "figure_format": "png",  # 图表保存格式，默认png，可选：pdf、svg
        "figure_dpi": 300,  # 图表分辨率，默认300
        "save_result_csv": True,  # 是否保存实验结果到CSV文件（方便后续分析）
    }

    return experiment_config