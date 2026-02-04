# core/__init__.py
"""
核心模块（联邦学习核心功能封装）
说明：
1.  包含6个子模块，其中「federated、ala、shapley、dp」为核心修改模块，「pseudo_label、fair_selection」为无修改复用模块
2.  统一导出各子模块的核心类/函数，外部无需深层导入，直接从core调用即可
3.  保持接口简洁，仅暴露核心业务对象，隐藏内部实现细节
"""

# ==============================================
# 1. 联邦学习基础组件（federated）- 核心修改：server.py 替换SA融合贡献度加权聚合
# ==============================================
from .federated.client import BaseClient
from .federated.server import BaseServer  # 核心修改：已替换为SA融合贡献度加权聚合
from .federated.trainer import FederatedTrainer

# ==============================================
# 2. ALA模块（ala）- 核心修改：新增extract_ala_features() 提取三大特征
# ==============================================
from .ala.ala_optimizer import ALAOptimizer  # 核心修改：保留自适应更新 + 新增特征提取

# ==============================================
# 3. 伪标签模块（pseudo_label）- 无修改：复用高置信伪标签生成
# ==============================================
from .pseudo_label.pseudo_label import PseudoLabelGenerator

# ==============================================
# 4. 公平客户端选择模块（fair_selection）- 无修改：复用SA贡献度提升选择精准度
# ==============================================
from .fair_selection.fair_selector import FairClientSelector

# ==============================================
# 5. Shapley模块（shapley）- 核心修改：新增calculate_sa_contribution() 融合ALA特征
# ==============================================
from .shapley.shapley_calculator import ShapleyCalculator  # 核心修改：保留蒙特卡洛采样 + 新增SA贡献度计算

# ==============================================
# 6. 差分隐私模块（dp）- 核心修改：优化自适应裁剪逻辑（4点核心优化）
# ==============================================
from .dp.adaptive_clipping_dp import AdaptiveClippingDP  # 核心修改：精细化差值处理 + 时序校准 + 稳定性约束

# ==============================================
# 规范批量导出（__all__）：定义外部可导入的核心对象列表
# ==============================================
__all__ = [
    # 联邦学习基础组件
    "BaseClient",
    "BaseServer",
    "FederatedTrainer",
    # ALA模块
    "ALAOptimizer",
    # 伪标签模块
    "PseudoLabelGenerator",
    # 公平客户端选择模块
    "FairClientSelector",
    # Shapley模块
    "ShapleyCalculator",
    # 差分隐私模块
    "AdaptiveClippingDP"
]