def get_fed_config():
    """
    联邦学习专属配置：与core/federated/模块协同，定义联邦通信与局部训练参数
    """
    fed_config = {
        # 1. 联邦客户端配置
        "num_clients": 100,  # 总客户端数量，默认100（经典联邦场景）
        "client_sample_ratio": 0.1,  # 每轮参与训练的客户端比例，默认0.1（即每轮选10个客户端）
        "min_clients": 5,  # 每轮最少参与客户端数量，兜底保障，默认5

        # 2. 局部训练配置（每个客户端的本地训练参数）
        "local_epochs": 5,  # 每个客户端的局部训练轮次，默认5
        "local_lr": 0.01,  # 局部学习率（可与全局学习率不同）
        "local_lr_decay": 0.99,  # 局部学习率衰减系数，每轮局部训练后衰减
        "local_optimizer": "sgd",  # 局部优化器，默认sgd，可选：adam

        # 3. 联邦聚合配置
        "aggregation_strategy": "sa_contribution",  # 聚合策略，默认sa_contribution（SA融合贡献度），可选：shapley、fedavg
        "aggregation_weight_decay": 0.0,  # 聚合权重衰减，默认0.0

        # 4. 联邦公平性配置（可选，与core/fair_selection/模块协同）
        "fairness_constraint": True,  # 是否开启公平性约束，默认True
        "min_contribution_threshold": 0.1,  # 客户端最小贡献度阈值，默认0.1
    }

    return fed_config