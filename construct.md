# 项目结构
FedFairADP-ALA/  # 项目根目录
├── configs/      # 统一配置中心（新增SA贡献度配置，补充自适应裁剪优化参数）
│   ├── __init__.py  导出配置 便于引用
    ├── base_config.pybase_config.py       # 基础配置（路径、硬件、统一超参数，无修改）
│   ├── model_config.py      # 模型配置（Backbone参数，无修改）
│   ├── fed_config.py        # 联邦学习配置（客户端数量、通信轮次，无修改）
│   ├── dp_config.py         # 差分隐私配置（新增：自适应裁剪的λ、θ、单轮调整上限等优化参数）
│   ├── shapley_config.py    # Shapley配置（新增：SA融合贡献度的权重分配、平滑系数）
│   └── experiment_config.py # 实验配置（实验变量，无修改）
├── datasets/     # 数据集处理目录（无修改，支撑多数据集、多Non-IID场景）
│   ├── __init__.py
│   ├── base_dataset.py      # 数据集基类（统一接口）
│   ├── cifar10_dataset.py   # CIFAR-10专属处理
│   ├── emnist_dataset.py    # EMNIST专属处理
│   ├── svhn_dataset.py      # SVHN专属处理
│   ├── femnist_dataset.py   # FEMNIST专属处理（对接LEAF）
│   └── non_iid_partitioner.py # Non-IID划分工具类（Dirichlet等）
├── models/       # 模型定义目录（无修改，统一Backbone，封装ALA+伪标签）
│   ├── __init__.py
│   ├── base_model.py        # 模型基类（统一训练/评估接口）
│   ├── vgg11.py             # VGG11实现（适配CIFAR-10/SVHN）
│   ├── custom_cnn.py        # 定制CNN（适配EMNIST/FEMNIST，62类输出）
│   └── fed_model.py         # 联邦模型封装（整合ALA+伪标签训练逻辑）
├── core/         # 核心模块（两处核心修改落地，其他模块无变动）
│   ├── __init__.py
│   ├── federated/           # 联邦学习基础组件（仅server.py修改聚合逻辑）
│   │   ├── client.py        # 客户端基类（本地训练、上传下载，无修改）
│   │   ├── server.py        # 服务器基类（核心修改：替换为SA融合贡献度加权聚合）
│   │   └── trainer.py       # 联邦训练器（协调通信流程，无修改）
│   ├── ala/                 # ALA模块（新增：ALA特征提取，支撑SA贡献度）
│   │   └── ala_optimizer.py # 核心修改：① 保留原有ALA自适应更新 ② 新增extract_ala_features()提取偏差/稳定性/性能特征
│   ├── pseudo_label/        # 伪标签模块（无修改，复用高置信伪标签生成）
│   │   └── pseudo_label.py
│   ├── fair_selection/      # 公平客户端选择模块（无修改，可复用SA贡献度提升选择精准度）
│   │   └── fair_selector.py
│   ├── shapley/             # Shapley模块（核心修改：新增SA融合贡献度计算）
│   │   └── shapley_calculator.py # ① 保留原有分组蒙特卡洛采样计算原始Shapley值 ② 新增calculate_sa_contribution()融合ALA特征
│   └── dp/                  # 差分隐私模块（核心修改：优化自适应裁剪逻辑）
│       └── adaptive_clipping_dp.py # ① 保留「本轮-上轮梯度差值」核心 ② 新增精细化差值处理（归一化+分级） ③ 新增自身时序辅助校准 ④ 新增稳定性约束
├── baselines/    # 基线方法目录（无修改，复用基础组件，对比实验公平）
│   ├── __init__.py
│   ├── fedavg.py
│   ├── dp_fedavg.py
│   ├── fedprox.py
│   ├── ditto.py
│   ├── fedshap.py
│   ├── fedadp.py（选做）
│   └── fairfedshap.py（选做）
├── experiments/  # 实验脚本目录（无修改，直接支撑5大实验模块，无缝衔接新核心）
│   ├── __init__.py
│   ├── basic_performance.py # 基础性能对比
│   ├── privacy_utility.py   # 隐私-效用权衡（验证自适应裁剪优化效果）
│   ├── ablation_study.py    # 组件消融（可单独消融SA贡献度、优化后自适应裁剪）
│   ├── fairness_verification.py # 公平性验证
│   └── efficiency_robustness.py # 效率与鲁棒性验证（验证SA贡献度的稳定性）
├── utils/        # 工具类目录（无修改，统一日志、指标、可视化）
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   ├── metrics.py           # 指标计算（含皮尔逊相关系数，验证SA贡献度精准度）
│   ├── checkpoint.py        # 模型保存/加载
│   ├── visualization.py     # 可视化工具（绘制贡献度波动、阈值稳定性等图表）
│   └── parallel.py          # 并行计算（加速Shapley值、多客户端训练）
├── data/         # 数据存储目录（自动生成，无修改）
│   ├── raw/                 # 原始数据集
│   └── processed/           # 处理后数据（Non-IID划分索引）
├── results/      # 实验结果存储目录（自动生成，无修改，按实验类型分类）
│   ├── basic_performance/
│   ├── privacy_utility/
│   ├── ablation_study/
│   ├── fairness_verification/
│   └── efficiency_robustness/
└── main.py       # 项目入口文件（可选，批量运行实验、解析命令行参数）
└── test_env.py   #测试各个模块功能是否正常
└── requirements.txt #项目依赖