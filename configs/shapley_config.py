# configs/shapley_config.py
def get_shapley_config():
    shapley_config = {
        # 1. 基础Shapley值计算参数
        "enable_shapley": True,  # 是否开启Shapley值计算，默认True
        "monte_carlo_groups": 10,  # 分组蒙特卡洛采样组数，默认10（平衡计算效率与精度）
        "monte_carlo_samples": 100,  # 每组采样次数，默认100
        "shapley_calc_freq": 1,  # 每轮都计算Shapley值（可设为5，每5轮计算一次，提升效率）

        # 2. SA融合贡献度参数（核心）
        "enable_sa_fusion": True,  # 是否开启SA融合贡献度，默认True
        "sa_weights": {
            "shapley": 0.6,  # 归一化Shapley值权重
            "ala_bias": 0.2,  # ALA偏差因子权重
            "ala_stab": 0.1,  # ALA稳定因子权重
            "ala_perf": 0.1,  # ALA性能因子权重
        },
        "sa_smooth_coeff": 0.8,  # SA融合贡献度滑动平均平滑系数，默认0.8
        "sa_contribution_norm": True,  # 是否对SA贡献度做归一化，默认True

        # 3. 计算优化参数
        "parallel_calc": True,  # 是否开启并行计算Shapley值，默认True
        "num_parallel_workers": 4,  # 并行计算工作线程数，默认4
    }

    return shapley_config