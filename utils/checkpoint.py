# -*- coding: utf-8 -*-
"""
模型Checkpoint管理模块
核心功能：
1. 联邦学习场景适配：支持服务端全局模型+多客户端本地模型的批量保存/加载；
2. 断点续训：保存训练轮次、优化器状态、训练指标，支持从指定轮次恢复训练；
3. 灵活保存策略：支持最新模型、最佳模型、间隔轮次保存，自动清理冗余checkpoint；
4. 设备兼容：自动处理GPU/CPU模型加载，适配不同实验环境；
5. 目录管理：按实验名称+时间戳自动创建checkpoint目录，避免文件覆盖。
"""

import os
import json
import shutil
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path

# 支持的模型文件格式
SUPPORTED_FORMATS = {"pt", "pth"}
# 默认保存策略
DEFAULT_SAVE_STRATEGY = {
    "save_latest": True,    # 始终保存最新模型
    "save_best": True,      # 保存性能最佳模型
    "save_interval": 5,     # 每5轮保存一次快照
    "keep_last_n": 3        # 保留最后3个非最佳/最新checkpoint
}

class CheckpointManager:
    """联邦学习模型Checkpoint管理器"""
    def __init__(
        self,
        experiment_name: str,               # 实验名称（用于目录命名）
        checkpoint_dir: str = "./checkpoints",  # 根目录
        save_strategy: Optional[Dict] = None,   # 保存策略
        device: Union[str, torch.device] = "cpu"  # 模型加载设备
    ):
        self.experiment_name = experiment_name
        self.root_dir = Path(checkpoint_dir)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.save_strategy = DEFAULT_SAVE_STRATEGY.copy()
        if save_strategy is not None:
            self.save_strategy.update(save_strategy)
        
        # 生成实验专属目录（实验名+时间戳，避免覆盖）
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.root_dir / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态记录
        self.best_metric = -float("inf")  # 最佳性能指标（越大越好）
        self.best_round = 0               # 最佳模型对应的轮次
        self.checkpoint_history = []      # 保存的checkpoint列表
        
        # 日志（复用全局日志器，避免重复依赖）
        try:
            from utils import get_global_logger
            self.logger = get_global_logger()
        except ImportError:
            self.logger = None
    
    def _log(self, msg: str, level: str = "info") -> None:
        """内部日志函数（兼容无日志器场景）"""
        if self.logger:
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
        else:
            print(f"[{level.upper()}] {msg}")
    
    def _get_checkpoint_path(self, round_idx: int, suffix: str = "pt") -> Path:
        """生成checkpoint文件路径"""
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的文件格式：{suffix}，仅支持{SUPPORTED_FORMATS}")
        return self.exp_dir / f"checkpoint_round_{round_idx}.{suffix}"
    
    def _get_best_path(self, suffix: str = "pt") -> Path:
        """生成最佳模型路径"""
        return self.exp_dir / f"best_checkpoint.{suffix}"
    
    def _get_latest_path(self, suffix: str = "pt") -> Path:
        """生成最新模型路径"""
        return self.exp_dir / f"latest_checkpoint.{suffix}"
    
    def _clean_old_checkpoints(self) -> None:
        """按策略清理旧checkpoint（保留最后N个）"""
        keep_last_n = self.save_strategy["keep_last_n"]
        if len(self.checkpoint_history) <= keep_last_n:
            return
        
        # 排序（按轮次升序），删除最早的
        self.checkpoint_history.sort(key=lambda x: x["round"])
        to_delete = self.checkpoint_history[:-keep_last_n]
        
        for item in to_delete:
            path = Path(item["path"])
            if path.exists():
                try:
                    path.unlink()
                    self._log(f"清理旧checkpoint：{path}")
                except Exception as e:
                    self._log(f"清理checkpoint失败：{e}", level="warning")
        
        # 更新历史记录
        self.checkpoint_history = self.checkpoint_history[-keep_last_n:]
    
    def save_checkpoint(
        self,
        round_idx: int,
        models: Dict[str, Any],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metric_name: str = "accuracy"  # 用于判断最佳模型的指标名
    ) -> None:
        """
        保存Checkpoint（核心方法）
        Args:
            round_idx: 当前训练轮次
            models: 模型字典，支持联邦学习多模型场景：
                    示例1（服务端）：{"global_model": global_model}
                    示例2（客户端）：{"client_0": client0_model, "client_1": client1_model}
                    示例3（混合）：{"global": global_model, "clients": {0: c0_model, 1: c1_model}}
            optimizers: 优化器字典（可选），格式与models对应
            metrics: 当前轮次性能指标（可选）
            metric_name: 用于判断最佳模型的指标名称
        """
        try:
            # 构建checkpoint数据
            checkpoint_data = {
                "round": round_idx,
                "timestamp": datetime.now().isoformat(),
                "experiment_name": self.experiment_name,
                "models": {},
                "optimizers": {},
                "metrics": metrics or {}
            }
            
            # 保存模型（剥离设备，兼容CPU/GPU加载）
            for model_name, model in models.items():
                if isinstance(model, torch.nn.Module):
                    checkpoint_data["models"][model_name] = model.state_dict()
                elif isinstance(model, dict):  # 客户端模型字典
                    checkpoint_data["models"][model_name] = {
                        k: v.state_dict() if isinstance(v, torch.nn.Module) else v
                        for k, v in model.items()
                    }
                else:
                    checkpoint_data["models"][model_name] = model  # 非模型数据
            
            # 保存优化器状态
            if optimizers is not None:
                for opt_name, optimizer in optimizers.items():
                    checkpoint_data["optimizers"][opt_name] = optimizer.state_dict()
            
            # 保存到文件
            suffix = "pt"
            checkpoint_path = self._get_checkpoint_path(round_idx, suffix)
            torch.save(checkpoint_data, checkpoint_path)
            
            # 记录历史
            self.checkpoint_history.append({
                "round": round_idx,
                "path": str(checkpoint_path),
                "metrics": metrics or {}
            })
            
            self._log(f"保存Checkpoint：{checkpoint_path}（轮次：{round_idx}）")
            
            # 保存最新模型（软链接/复制）
            if self.save_strategy["save_latest"]:
                latest_path = self._get_latest_path(suffix)
                if latest_path.exists():
                    latest_path.unlink()
                shutil.copy2(checkpoint_path, latest_path)
                self._log(f"更新最新模型：{latest_path}")
            
            # 保存最佳模型（如果当前指标更优）
            if self.save_strategy["save_best"] and metrics and metric_name in metrics:
                current_metric = metrics[metric_name]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_round = round_idx
                    best_path = self._get_best_path(suffix)
                    if best_path.exists():
                        best_path.unlink()
                    shutil.copy2(checkpoint_path, best_path)
                    self._log(f"更新最佳模型：{best_path}（{metric_name}={current_metric:.4f}）")
            
            # 按间隔保存+清理旧文件
            if self.save_strategy["save_interval"] > 0:
                if round_idx % self.save_strategy["save_interval"] == 0:
                    self._clean_old_checkpoints()
        
        except Exception as e:
            self._log(f"保存Checkpoint失败：{str(e)}", level="error")
            raise
    
    def load_checkpoint(
        self,
        target: Union[int, str] = "latest",  # 加载目标：轮次/ "latest"/ "best"
        models: Optional[Dict[str, Any]] = None,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None
    ) -> Dict[str, Any]:
        """
        加载Checkpoint（核心方法）
        Args:
            target: 加载目标：
                    - 整数：指定轮次
                    - "latest"：最新模型
                    - "best"：最佳模型
            models: 待加载的模型字典（需与保存时的key一致）
            optimizers: 待加载的优化器字典（可选）
        Returns:
            checkpoint_data: 完整的checkpoint数据（含轮次、指标等）
        """
        try:
            # 确定加载路径
            if target == "latest":
                load_path = self._get_latest_path()
                if not load_path.exists():
                    raise FileNotFoundError(f"最新模型文件不存在：{load_path}")
            elif target == "best":
                load_path = self._get_best_path()
                if not load_path.exists():
                    raise FileNotFoundError(f"最佳模型文件不存在：{load_path}")
            elif isinstance(target, int):
                load_path = self._get_checkpoint_path(target)
                if not load_path.exists():
                    raise FileNotFoundError(f"轮次{target}的模型文件不存在：{load_path}")
            else:
                raise ValueError(f"不支持的加载目标：{target}，仅支持轮次/ latest/ best")
            
            # 加载checkpoint数据
            checkpoint_data = torch.load(
                load_path,
                map_location=self.device,  # 自动适配设备
                weights_only=True  # 安全加载（PyTorch 2.0+）
            )
            
            self._log(f"加载Checkpoint成功：{load_path}（轮次：{checkpoint_data.get('round', '未知')}）")
            
            # 加载模型权重
            if models is not None and "models" in checkpoint_data:
                for model_name, model in models.items():
                    if model_name not in checkpoint_data["models"]:
                        self._log(f"模型{model_name}不在checkpoint中，跳过加载", level="warning")
                        continue
                    
                    model_state = checkpoint_data["models"][model_name]
                    if isinstance(model, torch.nn.Module):
                        # 加载模型权重
                        model.load_state_dict(model_state)
                        model.to(self.device)
                    elif isinstance(model, dict):  # 客户端模型字典
                        for client_id, client_model in model.items():
                            if client_id not in model_state:
                                self._log(f"客户端{client_id}模型不在checkpoint中，跳过", level="warning")
                                continue
                            client_model.load_state_dict(model_state[client_id])
                            client_model.to(self.device)
            
            # 加载优化器状态
            if optimizers is not None and "optimizers" in checkpoint_data:
                for opt_name, optimizer in optimizers.items():
                    if opt_name in checkpoint_data["optimizers"]:
                        optimizer.load_state_dict(checkpoint_data["optimizers"][opt_name])
                    else:
                        self._log(f"优化器{opt_name}不在checkpoint中，跳过加载", level="warning")
            
            return checkpoint_data
        
        except Exception as e:
            self._log(f"加载Checkpoint失败：{str(e)}", level="error")
            raise
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """获取Checkpoint管理信息（用于实验记录）"""
        return {
            "experiment_dir": str(self.exp_dir),
            "timestamp": self.timestamp,
            "best_metric": self.best_metric,
            "best_round": self.best_round,
            "checkpoint_count": len(self.checkpoint_history),
            "save_strategy": self.save_strategy,
            "checkpoint_history": self.checkpoint_history
        }
    
    def export_checkpoint_info(self, save_path: Optional[str] = None) -> None:
        """导出Checkpoint信息到JSON文件"""
        info = self.get_checkpoint_info()
        save_path = save_path or str(self.exp_dir / "checkpoint_info.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)
        self._log(f"Checkpoint信息已导出：{save_path}")
    
    def clean_all_checkpoints(self, confirm: bool = False) -> None:
        """清理当前实验的所有Checkpoint（谨慎使用）"""
        if not confirm:
            self._log("清理操作需要confirm=True确认，跳过", level="warning")
            return
        
        try:
            shutil.rmtree(self.exp_dir)
            self._log(f"已清理实验{self.experiment_name}的所有Checkpoint：{self.exp_dir}")
        except Exception as e:
            self._log(f"清理Checkpoint失败：{e}", level="error")

# ======================== 快捷函数（简化实验调用） ========================
def create_checkpoint_manager(
    experiment_name: str,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu"
) -> CheckpointManager:
    """
    创建Checkpoint管理器（快捷函数）
    """
    return CheckpointManager(
        experiment_name=experiment_name,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

def load_best_model(
    experiment_dir: str,
    models: Dict[str, Any],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    快速加载指定实验目录的最佳模型（无需初始化管理器）
    """
    # 从目录名解析实验信息（兼容自动生成的目录格式）
    exp_name = Path(experiment_dir).name.split("_")[0]
    manager = CheckpointManager(
        experiment_name=exp_name,
        checkpoint_dir=str(Path(experiment_dir).parent),
        device=device
    )
    # 覆盖exp_dir为指定目录
    manager.exp_dir = Path(experiment_dir)
    return manager.load_checkpoint(target="best", models=models)