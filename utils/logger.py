# -*- coding: utf-8 -*-
"""
日志工具模块
核心功能：
1. 封装Python标准logging模块，提供简洁的日志接口；
2. 支持控制台+文件双输出，日志文件按实验名称+日期自动命名；
3. 统一日志格式（时间、等级、模块、进程ID、消息）；
4. 支持全局日志配置，一键初始化；
5. 适配多实验场景，避免日志混乱。
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

# 定义日志等级映射（便于外部配置）
LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 默认日志格式
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(module)s - %(process)d - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class Logger:
    """日志管理类"""
    def __init__(
        self,
        name: str = "federated_learning_sa",  # 日志器名称
        log_dir: str = "./logs",              # 日志文件保存目录
        log_level: str = "info",              # 日志等级
        enable_console: bool = True,          # 是否启用控制台输出
        enable_file: bool = True,             # 是否启用文件输出
        experiment_name: Optional[str] = None # 实验名称（用于日志文件命名）
    ):
        # 初始化日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL_MAP.get(log_level.lower(), logging.INFO))
        self.logger.propagate = False  # 避免重复输出
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 清除已存在的处理器（避免重复输出）
        self.logger.handlers.clear()
        
        # 1. 添加控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(LOG_LEVEL_MAP.get(log_level.lower(), logging.INFO))
            console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # 2. 添加文件处理器
        if enable_file:
            # 生成日志文件名：实验名称（可选）+ 日期时间
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                log_filename = f"{experiment_name}_{timestamp}.log"
            else:
                log_filename = f"federated_learning_{timestamp}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_handler.setLevel(LOG_LEVEL_MAP.get(log_level.lower(), logging.INFO))
            file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"日志文件已创建：{log_filepath}")
    
    def get_logger(self) -> logging.Logger:
        """获取底层logging.Logger对象（兼容原生接口）"""
        return self.logger
    
    # 封装常用日志方法
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

# 全局日志器实例（供整个项目复用）
_global_logger: Optional[Logger] = None

def setup_global_logger(
    name: str = "federated_learning_sa",
    log_dir: str = "./logs",
    log_level: str = "info",
    enable_console: bool = True,
    enable_file: bool = True,
    experiment_name: Optional[str] = None
) -> Logger:
    """
    初始化全局日志器（项目中只需调用一次）
    Args:
        name: 日志器名称
        log_dir: 日志文件目录
        log_level: 日志等级（debug/info/warning/error/critical）
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        experiment_name: 实验名称（如ablation_study/fairness_verification）
    Returns:
        初始化后的全局Logger实例
    """
    global _global_logger
    _global_logger = Logger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file,
        experiment_name=experiment_name
    )
    return _global_logger

def get_global_logger() -> Logger:
    """
    获取全局日志器（需先调用setup_global_logger初始化）
    Raises:
        RuntimeError: 全局日志器未初始化
    Returns:
        全局Logger实例
    """
    if _global_logger is None:
        # 未初始化时，自动创建默认日志器
        return setup_global_logger()
    return _global_logger

# 快捷日志函数（无需实例化，直接调用）
def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    get_global_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args: Any, **kwargs: Any) -> None:
    get_global_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    get_global_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args: Any, **kwargs: Any) -> None:
    get_global_logger().error(msg, *args, **kwargs)

def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    get_global_logger().critical(msg, *args, **kwargs)