"""
联邦学习基线算法模块（baselines）
核心定位：复用项目基础组件（BaseServer/BaseClient/DP/Shapley等），作为对比实验的公平基准
包含基线算法：
1. FedAvg: 基础联邦平均（联邦学习最核心的基准，无任何优化）
2. DP-FedAvg: 带差分隐私的FedAvg（集成AdaptiveClippingDP，隐私保护基线）
3. FedProx: 带近端项的联邦算法（解决数据/模型异构性的经典基线）
4. Ditto: 个性化联邦学习基线（客户端保留个性化模型，兼顾全局与本地）
5. FedShap: 结合Shapley值的联邦基线（基于贡献度的聚合策略）
设计原则：
- 所有基线算法的Server/Client类均兼容项目BaseServer/BaseClient接口；
- 统一导出核心类，简化外部调用；
- 新增基线仅需在此文件补充导入，无需修改调用逻辑。
"""

# ======================== 导入所有基线算法的核心类 ========================
# 1. 基础FedAvg（核心基准，必选）
from .fedavg import FedAvgServer, FedAvgClient

# 2. 带差分隐私的FedAvg（DP-FedAvg）
from .dp_fedavg import DPFedAvgServer, DPFedAvgClient

# 3. FedProx（异构场景基线）
from .fedprox import FedProxServer, FedProxClient

# 4. Ditto（个性化联邦基线）
from .ditto import DITTOServer,DITTOTrainer,DITTOClient

# 5. FedShap（Shapley贡献度聚合基线）
from .fedshap import FedShapServer, FedShapClient
# 6. FedAdaClip
from .fedadaclip import FedAdaptiveDPServer, FedAdaptiveDPClient

# ======================== 声明模块公开接口（核心） ========================
# __all__ 定义「from baselines import *」时导入的对象，避免导出内部辅助类/函数
__all__ = [
    # 基础FedAvg
    "FedAvgServer", "FedAvgClient",
    # DP-FedAvg
    "DPFedAvgServer", "DPFedAvgClient",
    # FedProx
    "FedProxServer", "FedProxClient",
    # Ditto（个性化）
    "DITTOServer", "DITTOTrainer","DITTOClient",
    # FedShap（Shapley贡献度）
    "FedShapServer", "FedShapClient"
    # FedAdaClip
    "FedAdaptiveDPServer", "FedAdaptiveDPTrainer"
]

# ======================== 模块元信息（可选，便于调试/协作） ========================
__version__ = "1.0.0"          # 基线模块版本
__author__ = "chenhaolong"  # 模块维护者（可选）
__description__ = "联邦学习基线算法集合，适配项目基础组件，用于对比实验"