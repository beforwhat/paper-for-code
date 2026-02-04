
from types import SimpleNamespace
from .base_config import get_base_config
from .model_config import get_model_config
from .fed_config import get_fed_config
from .dp_config import get_dp_config
from .shapley_config import get_shapley_config
from .experiment_config import get_experiment_config


def _dict_to_namespace(d):
    """
    辅助函数：将嵌套字典转换为SimpleNamespace对象，支持「.」访问
    """
    if not isinstance(d, dict):
        return d
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v))
    return ns


def load_config():
    """
    全局统一配置加载函数：整合所有配置，返回可通过「.」访问的配置对象
    外部调用：from configs.config_loader import load_config; config = load_config()
    """
    # 1. 加载基础配置
    base_config = get_base_config()

    # 2. 加载模型配置（传入基础配置的数据集名称）
    model_config = get_model_config(base_config["dataset"]["name"])

    # 3. 加载其他配置
    fed_config = get_fed_config()
    dp_config = get_dp_config()
    shapley_config = get_shapley_config()
    experiment_config = get_experiment_config()

    # 4. 整合所有配置为一个嵌套字典
    combined_config = {
        **base_config,
        "model": model_config,
        "fed": fed_config,
        "dp": dp_config,
        "shapley": shapley_config,
        "experiment": experiment_config,
    }

    # 5. 转换为SimpleNamespace对象，支持「.」访问
    config_namespace = _dict_to_namespace(combined_config)

    return config_namespace