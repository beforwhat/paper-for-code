# -*- coding: utf-8 -*-
"""
可视化工具模块
核心功能：
1. SA贡献度专属可视化：贡献度波动趋势、阈值稳定性曲线、贡献度分布对比；
2. 公平性可视化：基尼系数对比、客户端性能分布箱线图/直方图；
3. 鲁棒性可视化：多场景鲁棒性得分对比、热力图、性能保持率曲线；
4. 效率可视化：时间/资源/通信效率对比柱状图、效率-鲁棒性权衡散点图；
5. 通用可视化：多算法实验结果对比、训练曲线、指标热力图；
6. 统一样式配置：支持中文显示、高清导出、自定义颜色/主题。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from utils.logger import info, error 

# ======================== 全局可视化配置（核心） ========================
# 中文显示支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 兼容Linux/Mac
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 300  # 默认高清
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.facecolor"] = "white"  # 白底（便于论文/报告使用）

# 颜色配置（贴合实验场景，突出SA贡献度算法）
COLOR_PALETTE = {
    "sa_primary": "#9467bd",    # SA贡献度主色（紫色）
    "sa_secondary": "#d3b7e3",  # SA贡献度辅助色
    "baseline": "#1f77b4",      # 基线算法主色（蓝色）
    "contrast": "#d62728",      # 对比色（红色）
    "neutral": "#7f7f7f",       # 中性色（灰色）
    # 多算法配色（与效率鲁棒性实验一致）
    "FedAvg": "#1f77b4",
    "DP-FedAvg": "#ff7f0e",
    "FedProx": "#2ca02c",
    "Ditto": "#d62728",
    "FedShap": "#9467bd"
}

# 默认样式
DEFAULT_STYLE = {
    "figure_size": (10, 6),
    "font_size": 12,
    "title_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
    "line_width": 2,
    "marker_size": 6,
    "bar_width": 0.35
}

class Visualizer:
    """
    统一可视化类，整合所有实验场景的可视化方法
    """
    def __init__(
        self,
        experiment_name: str,               # 实验名称（用于文件命名）
        save_dir: str = "./visualizations", # 保存目录
        style_config: Optional[Dict] = None # 样式配置
    ):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 样式配置
        self.style = DEFAULT_STYLE.copy()
        if style_config is not None:
            self.style.update(style_config)
        
        # 日志（复用全局日志器）
        try:
            from utils import get_global_logger
            self.logger = get_global_logger()
        except ImportError:
            self.logger = None
        
        # 初始化seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette([COLOR_PALETTE[k] for k in ["sa_primary", "baseline", "contrast", "neutral"]])
    
    def _log(self, msg: str, level: str = "info") -> None:
        """内部日志函数"""
        if self.logger:
            if level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
        else:
            print(f"[{level.upper()}] {msg}")
    
    def _get_save_path(self, plot_name: str, fmt: str = "png") -> Path:
        """生成图表保存路径"""
        valid_fmts = {"png", "pdf", "svg", "jpg"}
        if fmt not in valid_fmts:
            self._log(f"不支持的格式{fmt}，使用默认png", level="warning")
            fmt = "png"
        return self.save_dir / f"{plot_name}.{fmt}"
    
    def _setup_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """初始化图表画布"""
        figsize = figsize or self.style["figure_size"]
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("", fontsize=self.style["label_size"])
        ax.set_ylabel("", fontsize=self.style["label_size"])
        ax.tick_params(axis="both", labelsize=self.style["tick_size"])
        return fig, ax
    
    def _save_figure(self, fig: plt.Figure, plot_name: str, fmt: str = "png", tight_layout: bool = True) -> None:
        """保存图表"""
        try:
            save_path = self._get_save_path(plot_name, fmt)
            if tight_layout:
                fig.tight_layout()
            fig.savefig(save_path, format=fmt, dpi=300, bbox_inches="tight")
            plt.close(fig)
            self._log(f"图表已保存：{save_path}")
        except Exception as e:
            self._log(f"保存图表失败：{str(e)}", level="error")
            plt.close(fig)
    
    # ======================== SA贡献度专属可视化 ========================
    def plot_sa_contribution_trend(
        self,
        contributions: Dict[int, List[float]],  # {轮次: [客户端1贡献度, 客户端2...]}
        client_ids: Optional[List[int]] = None,
        highlight_client: Optional[int] = None,
        plot_name: str = "sa_contribution_trend",
        fmt: str = "png"
    ) -> None:
        """
        绘制SA贡献度波动趋势图（核心）
        """
        fig, ax = self._setup_figure()
        
        # 整理数据
        rounds = sorted(contributions.keys())
        client_ids = client_ids or list(range(len(contributions[rounds[0]])))
        
        # 绘制所有客户端贡献度趋势
        for idx, cid in enumerate(client_ids):
            c_contrib = [contributions[r][idx] for r in rounds]
            color = COLOR_PALETTE["sa_primary"] if cid == highlight_client else COLOR_PALETTE["neutral"]
            linewidth = self.style["line_width"] + 1 if cid == highlight_client else self.style["line_width"]
            ax.plot(rounds, c_contrib, label=f"客户端{cid}", color=color, linewidth=linewidth, marker="o", markersize=self.style["marker_size"])
        
        # 样式配置
        ax.set_title("SA贡献度随训练轮次的波动趋势", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("训练轮次", fontsize=self.style["label_size"])
        ax.set_ylabel("SA贡献度值", fontsize=self.style["label_size"])
        ax.legend(fontsize=self.style["legend_size"], loc="best")
        ax.grid(True, alpha=0.3)
        
        # 保存
        self._save_figure(fig, plot_name, fmt)
    
    def plot_threshold_stability(
        self,
        thresholds: Dict[int, float],  # {轮次: 自适应阈值}
        contributions_mean: Dict[int, float],  # {轮次: 贡献度均值}
        plot_name: str = "threshold_stability",
        fmt: str = "png"
    ) -> None:
        """
        绘制SA贡献度自适应阈值稳定性曲线
        """
        fig, ax1 = self._setup_figure()
        
        # 整理数据
        rounds = sorted(thresholds.keys())
        threshold_vals = [thresholds[r] for r in rounds]
        mean_vals = [contributions_mean[r] for r in rounds]
        
        # 阈值曲线（主坐标轴）
        ax1.plot(rounds, threshold_vals, color=COLOR_PALETTE["contrast"], 
                 linewidth=self.style["line_width"], marker="s", markersize=self.style["marker_size"],
                 label="自适应阈值")
        ax1.set_xlabel("训练轮次", fontsize=self.style["label_size"])
        ax1.set_ylabel("自适应阈值", fontsize=self.style["label_size"], color=COLOR_PALETTE["contrast"])
        ax1.tick_params(axis="y", labelcolor=COLOR_PALETTE["contrast"])
        
        # 贡献度均值曲线（次坐标轴）
        ax2 = ax1.twinx()
        ax2.plot(rounds, mean_vals, color=COLOR_PALETTE["sa_primary"],
                 linewidth=self.style["line_width"], marker="o", markersize=self.style["marker_size"],
                 label="贡献度均值")
        ax2.set_ylabel("贡献度均值", fontsize=self.style["label_size"], color=COLOR_PALETTE["sa_primary"])
        ax2.tick_params(axis="y", labelcolor=COLOR_PALETTE["sa_primary"])
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=self.style["legend_size"], loc="best")
        
        ax1.set_title("SA贡献度自适应阈值稳定性", fontsize=self.style["title_size"], fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        self._save_figure(fig, plot_name, fmt)
    
    def plot_contribution_distribution(
        self,
        true_contrib: Union[np.ndarray, List[float]],
        pred_contrib: Union[np.ndarray, List[float]],
        plot_name: str = "contribution_distribution",
        fmt: str = "png"
    ) -> None:
        """
        绘制SA贡献度预测值vs真实值分布对比（验证精准度）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 转换为数组
        true_arr = np.array(true_contrib).flatten()
        pred_arr = np.array(pred_contrib).flatten()
        
        # 直方图
        sns.histplot(true_arr, ax=ax1, color=COLOR_PALETTE["baseline"], label="真实贡献度", kde=True)
        ax1.set_title("真实SA贡献度分布", fontsize=self.style["title_size"]-1, fontweight="bold")
        ax1.set_xlabel("贡献度值", fontsize=self.style["label_size"])
        ax1.set_ylabel("频次", fontsize=self.style["label_size"])
        ax1.legend(fontsize=self.style["legend_size"])
        
        sns.histplot(pred_arr, ax=ax2, color=COLOR_PALETTE["sa_primary"], label="预测贡献度", kde=True)
        ax2.set_title("预测SA贡献度分布", fontsize=self.style["title_size"]-1, fontweight="bold")
        ax2.set_xlabel("贡献度值", fontsize=self.style["label_size"])
        ax2.set_ylabel("频次", fontsize=self.style["label_size"])
        ax2.legend(fontsize=self.style["legend_size"])
        
        # 整体标题
        fig.suptitle("SA贡献度真实值vs预测值分布对比", fontsize=self.style["title_size"], fontweight="bold")
        
        self._save_figure(fig, plot_name, fmt)
    
    # ======================== 公平性可视化 ========================
    def plot_fairness_metrics(
        self,
        fairness_data: Dict[str, float],  # {算法名: 基尼系数/公平性指数}
        metric_name: str = "gini_coefficient",
        plot_name: str = "fairness_metrics",
        fmt: str = "png"
    ) -> None:
        """
        绘制多算法公平性指标对比（柱状图）
        """
        fig, ax = self._setup_figure()
        
        # 整理数据
        algorithms = list(fairness_data.keys())
        values = list(fairness_data.values())
        colors = [COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"]) for alg in algorithms]
        
        # 高亮SA算法
        for i, alg in enumerate(algorithms):
            if alg == "FedShap":
                colors[i] = COLOR_PALETTE["sa_primary"]
        
        # 绘制柱状图
        bars = ax.bar(algorithms, values, width=self.style["bar_width"], color=colors)
        
        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=self.style["tick_size"])
        
        # 样式配置
        metric_label = "基尼系数" if metric_name == "gini_coefficient" else "综合公平性指数"
        ax.set_title(f"各算法{metric_label}对比", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("算法", fontsize=self.style["label_size"])
        ax.set_ylabel(metric_label, fontsize=self.style["label_size"])
        ax.grid(True, alpha=0.3, axis="y")
        
        self._save_figure(fig, plot_name, fmt)
    
    def plot_client_performance_dist(
        self,
        client_perfs: Dict[str, List[float]],  # {算法名: [客户端1性能, 客户端2...]}
        plot_name: str = "client_performance_dist",
        fmt: str = "png"
    ) -> None:
        """
        绘制客户端性能分布箱线图（展示公平性）
        """
        fig, ax = self._setup_figure(figsize=(12, 6))
        
        # 整理数据为DataFrame
        data = []
        for alg, perfs in client_perfs.items():
            for perf in perfs:
                data.append({"algorithm": alg, "performance": perf})
        df = pd.DataFrame(data)
        
        # 绘制箱线图
        box_plot = sns.boxplot(x="algorithm", y="performance", data=df, ax=ax,
                               palette=[COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"]) for alg in client_perfs.keys()])
        
        # 样式配置
        ax.set_title("各算法客户端性能分布（箱线图）", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("算法", fontsize=self.style["label_size"])
        ax.set_ylabel("客户端性能（准确率%）", fontsize=self.style["label_size"])
        ax.grid(True, alpha=0.3, axis="y")
        
        self._save_figure(fig, plot_name, fmt)
    
    # ======================== 鲁棒性可视化 ========================
    def plot_robustness_scores(
        self,
        robustness_data: Dict[str, Dict[str, float]],  # {算法名: {场景: 鲁棒性得分}}
        plot_name: str = "robustness_scores",
        fmt: str = "png"
    ) -> None:
        """
        绘制多算法多场景鲁棒性得分对比（分组柱状图）
        """
        fig, ax = self._setup_figure(figsize=(12, 6))
        
        # 整理数据
        algorithms = list(robustness_data.keys())
        scenarios = list(robustness_data[algorithms[0]].keys())
        x = np.arange(len(scenarios))
        width = self.style["bar_width"]
        
        # 绘制分组柱状图
        for i, alg in enumerate(algorithms):
            values = [robustness_data[alg][scen] for scen in scenarios]
            color = COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"])
            # 高亮SA算法
            if alg == "FedShap":
                color = COLOR_PALETTE["sa_primary"]
                edgecolor = "black"
                linewidth = 2
            else:
                edgecolor = None
                linewidth = 1
            
            ax.bar(x + (i - len(algorithms)/2 + 0.5) * width, values, width,
                   label=alg, color=color, edgecolor=edgecolor, linewidth=linewidth)
        
        # 样式配置
        ax.set_title("各算法多场景鲁棒性得分对比", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("鲁棒性场景", fontsize=self.style["label_size"])
        ax.set_ylabel("鲁棒性得分（0~1）", fontsize=self.style["label_size"])
        ax.set_xticks(x)
        scen_labels = {"scale": "规模", "noise": "噪声", "failure": "故障", "heterogeneity": "异构"}
        ax.set_xticklabels([scen_labels.get(s, s) for s in scenarios])
        ax.legend(fontsize=self.style["legend_size"], loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        
        self._save_figure(fig, plot_name, fmt)
    
    def plot_robustness_heatmap(
        self,
        robustness_matrix: pd.DataFrame,  # 行：算法，列：场景，值：鲁棒性得分
        plot_name: str = "robustness_heatmap",
        fmt: str = "png"
    ) -> None:
        """
        绘制鲁棒性得分热力图
        """
        fig, ax = self._setup_figure(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(robustness_matrix, ax=ax, annot=True, fmt=".4f", cmap="RdYlBu_r",
                    cbar_kws={"label": "鲁棒性得分（0~1）"}, linewidths=0.5)
        
        # 样式配置
        ax.set_title("算法-场景鲁棒性得分热力图", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("鲁棒性场景", fontsize=self.style["label_size"])
        ax.set_ylabel("算法", fontsize=self.style["label_size"])
        
        self._save_figure(fig, plot_name, fmt)
    
    # ======================== 效率可视化 ========================
    def plot_efficiency_metrics(
        self,
        efficiency_data: Dict[str, Dict[str, float]],  # {算法名: {指标: 值}}
        metrics: List[str] = ["total_time", "memory_usage_mb", "total_comm_mb"],
        plot_name: str = "efficiency_metrics",
        fmt: str = "png"
    ) -> None:
        """
        绘制多算法效率指标对比（子图）
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # 指标标签映射
        metric_labels = {
            "total_time": "总训练时间（s）",
            "avg_round_time": "每轮平均时间（s）",
            "memory_usage_mb": "内存占用（MB）",
            "cpu_usage_pct": "CPU使用率（%）",
            "gpu_memory_mb": "GPU显存（MB）",
            "total_comm_mb": "总通信量（MB）",
            "avg_round_comm_mb": "每轮平均通信量（MB）"
        }
        
        algorithms = list(efficiency_data.keys())
        colors = [COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"]) for alg in algorithms]
        # 高亮SA算法
        for i, alg in enumerate(algorithms):
            if alg == "FedShap":
                colors[i] = COLOR_PALETTE["sa_primary"]
        
        # 绘制每个指标的柱状图
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [efficiency_data[alg].get(metric, 0.0) for alg in algorithms]
            
            bars = ax.bar(algorithms, values, width=self.style["bar_width"], color=colors)
            
            # 标注数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=self.style["tick_size"])
            
            ax.set_title(metric_labels.get(metric, metric), fontsize=self.style["title_size"]-1, fontweight="bold")
            ax.set_xlabel("算法", fontsize=self.style["label_size"])
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=self.style["label_size"])
            ax.grid(True, alpha=0.3, axis="y")
        
        # 整体标题
        fig.suptitle("各算法效率指标对比", fontsize=self.style["title_size"]+2, fontweight="bold")
        
        self._save_figure(fig, plot_name, fmt)
    
    def plot_efficiency_robustness_tradeoff(
        self,
        tradeoff_data: Dict[str, Tuple[float, float]],  # {算法名: (效率值, 鲁棒性得分)}
        efficiency_metric: str = "total_time",
        plot_name: str = "efficiency_robustness_tradeoff",
        fmt: str = "png"
    ) -> None:
        """
        绘制效率-鲁棒性权衡散点图
        """
        fig, ax = self._setup_figure()
        
        # 整理数据
        algorithms = list(tradeoff_data.keys())
        eff_values = [tradeoff_data[alg][0] for alg in algorithms]
        rob_values = [tradeoff_data[alg][1] for alg in algorithms]
        colors = [COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"]) for alg in algorithms]
        
        # 绘制散点图
        for i, alg in enumerate(algorithms):
            # 高亮SA算法
            marker_size = 100 if alg == "FedShap" else 60
            ax.scatter(eff_values[i], rob_values[i], label=alg, color=colors[i],
                       s=marker_size, alpha=0.8, edgecolors="black" if alg == "FedShap" else None)
            # 标注算法名
            ax.annotate(alg, (eff_values[i], rob_values[i]), xytext=(5, 5), textcoords="offset points")
        
        # 样式配置
        eff_label = "总训练时间（s）" if efficiency_metric == "total_time" else efficiency_metric
        ax.set_title("算法效率-鲁棒性权衡分析", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel(eff_label, fontsize=self.style["label_size"])
        ax.set_ylabel("鲁棒性得分（0~1）", fontsize=self.style["label_size"])
        ax.legend(fontsize=self.style["legend_size"], loc="best")
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, plot_name, fmt)
    
    # ======================== 通用可视化 ========================
    def plot_training_curve(
        self,
        training_data: Dict[str, List[float]],  # {算法名: [轮次1性能, 轮次2...]}
        plot_name: str = "training_curve",
        fmt: str = "png"
    ) -> None:
        """
        绘制训练曲线（准确率/损失随轮次变化）
        """
        fig, ax = self._setup_figure()
        
        # 整理数据
        algorithms = list(training_data.keys())
        rounds = list(range(1, len(training_data[algorithms[0]]) + 1))
        
        # 绘制曲线
        for alg in algorithms:
            values = training_data[alg]
            color = COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"])
            # 高亮SA算法
            linewidth = self.style["line_width"] + 1 if alg == "FedShap" else self.style["line_width"]
            ax.plot(rounds, values, label=alg, color=color, linewidth=linewidth,
                    marker="o", markersize=self.style["marker_size"])
        
        # 样式配置
        ax.set_title("算法训练曲线（准确率）", fontsize=self.style["title_size"], fontweight="bold")
        ax.set_xlabel("训练轮次", fontsize=self.style["label_size"])
        ax.set_ylabel("准确率（%）", fontsize=self.style["label_size"])
        ax.legend(fontsize=self.style["legend_size"], loc="best")
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, plot_name, fmt)
    
    def plot_experiment_comparison(
        self,
        comparison_data: pd.DataFrame,  # 行：算法，列：指标
        plot_name: str = "experiment_comparison",
        fmt: str = "png"
    ) -> None:
        """
        绘制多算法多指标综合对比图（雷达图）
        """
        # 仅支持数值型指标
        numeric_cols = comparison_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            self._log("雷达图需要至少3个数值指标，改用柱状图", level="warning")
            # 降级为柱状图
            fig, ax = self._setup_figure()
            comparison_data.plot(kind="bar", ax=ax, color=[COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"]) for alg in comparison_data.index])
            ax.set_title("多算法多指标综合对比", fontsize=self.style["title_size"], fontweight="bold")
            ax.set_xlabel("算法", fontsize=self.style["label_size"])
            ax.legend(fontsize=self.style["legend_size"], loc="best")
            self._save_figure(fig, plot_name, fmt)
            return
        
        # 归一化数据（雷达图需要0~1范围）
        df_normalized = (comparison_data[numeric_cols] - comparison_data[numeric_cols].min()) / (comparison_data[numeric_cols].max() - comparison_data[numeric_cols].min())
        df_normalized = df_normalized.fillna(0)
        
        # 绘制雷达图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # 角度设置
        angles = np.linspace(0, 2*np.pi, len(numeric_cols), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 绘制每个算法的雷达图
        for idx, (alg, row) in enumerate(df_normalized.iterrows()):
            values = row.values.tolist()
            values += values[:1]  # 闭合
            
            color = COLOR_PALETTE.get(alg, COLOR_PALETTE["neutral"])
            # 高亮SA算法
            linewidth = self.style["line_width"] + 1 if alg == "FedShap" else self.style["line_width"]
            ax.plot(angles, values, label=alg, color=color, linewidth=linewidth)
            ax.fill(angles, values, color=color, alpha=0.2)
        
        # 样式配置
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(numeric_cols, fontsize=self.style["label_size"])
        ax.set_ylim(0, 1)
        ax.set_title("多算法多指标综合对比（归一化）", fontsize=self.style["title_size"], fontweight="bold", pad=20)
        ax.legend(fontsize=self.style["legend_size"], loc="upper right", bbox_to_anchor=(1.2, 1.0))
        
        self._save_figure(fig, plot_name, fmt)

# ======================== 快捷函数（简化实验调用） ========================
def create_visualizer(
    experiment_name: str,
    save_dir: str = "./visualizations",
    style_config: Optional[Dict] = None
) -> Visualizer:
    """
    创建可视化器（快捷函数）
    """
    return Visualizer(
        experiment_name=experiment_name,
        save_dir=save_dir,
        style_config=style_config
    )

def plot_sa_contribution_trend(
    contributions: Dict[int, List[float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "sa_contribution_trend"
) -> None:
    """
    快捷绘制SA贡献度波动趋势图
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_sa_contribution_trend(contributions, plot_name=plot_name)

def plot_threshold_stability(
    thresholds: Dict[int, float],
    contributions_mean: Dict[int, float],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "threshold_stability"
) -> None:
    """
    快捷绘制SA贡献度自适应阈值稳定性曲线
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_threshold_stability(thresholds, contributions_mean, plot_name=plot_name)

def plot_contribution_distribution(
    true_contrib: Union[np.ndarray, List[float]],
    pred_contrib: Union[np.ndarray, List[float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "contribution_distribution"
) -> None:
    """
    快捷绘制SA贡献度预测值vs真实值分布对比
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_contribution_distribution(true_contrib, pred_contrib, plot_name=plot_name)

def plot_fairness_metrics(
    fairness_data: Dict[str, float],
    experiment_name: str,
    save_dir: str = "./visualizations",
    metric_name: str = "gini_coefficient",
    plot_name: str = "fairness_metrics"
) -> None:
    """
    快捷绘制多算法公平性指标对比
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_fairness_metrics(fairness_data, metric_name=metric_name, plot_name=plot_name)

def plot_client_performance_dist(
    client_perfs: Dict[str, List[float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "client_performance_dist"
) -> None:
    """
    快捷绘制客户端性能分布箱线图
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_client_performance_dist(client_perfs, plot_name=plot_name)

def plot_robustness_scores(
    robustness_data: Dict[str, Dict[str, float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "robustness_scores"
) -> None:
    """
    快捷绘制多算法多场景鲁棒性得分对比
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_robustness_scores(robustness_data, plot_name=plot_name)

def plot_robustness_heatmap(
    robustness_matrix: pd.DataFrame,
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "robustness_heatmap"
) -> None:
    """
    快捷绘制鲁棒性得分热力图
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_robustness_heatmap(robustness_matrix, plot_name=plot_name)

def plot_efficiency_metrics(
    efficiency_data: Dict[str, Dict[str, float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    metrics: List[str] = ["total_time", "memory_usage_mb", "total_comm_mb"],
    plot_name: str = "efficiency_metrics"
) -> None:
    """
    快捷绘制多算法效率指标对比
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_efficiency_metrics(efficiency_data, metrics=metrics, plot_name=plot_name)

def plot_efficiency_robustness_tradeoff(
    tradeoff_data: Dict[str, Tuple[float, float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    efficiency_metric: str = "total_time",
    plot_name: str = "efficiency_robustness_tradeoff"
) -> None:
    """
    快捷绘制效率-鲁棒性权衡散点图
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_efficiency_robustness_tradeoff(tradeoff_data, efficiency_metric=efficiency_metric, plot_name=plot_name)

def plot_training_curve(
    training_data: Dict[str, List[float]],
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "training_curve"
) -> None:
    """
    快捷绘制训练曲线
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_training_curve(training_data, plot_name=plot_name)

def plot_experiment_comparison(
    comparison_data: pd.DataFrame,
    experiment_name: str,
    save_dir: str = "./visualizations",
    plot_name: str = "experiment_comparison"
) -> None:
    """
    快捷绘制多算法多指标综合对比图
    """
    viz = create_visualizer(experiment_name, save_dir)
    viz.plot_experiment_comparison(comparison_data, plot_name=plot_name)

def plot_experiment_summary(
    experiment_data: Dict[str, Any],
    experiment_name: str,
    save_dir: str = "./visualizations"
) -> None:
    """
    一键生成实验所有核心图表
    """
    viz = create_visualizer(experiment_name, save_dir)
    
    # 1. SA贡献度趋势（如有）
    if "sa_contributions" in experiment_data:
        viz.plot_sa_contribution_trend(experiment_data["sa_contributions"])
    
    # 2. 阈值稳定性（如有）
    if "thresholds" in experiment_data and "contributions_mean" in experiment_data:
        viz.plot_threshold_stability(experiment_data["thresholds"], experiment_data["contributions_mean"])
    
    # 3. 贡献度分布（如有）
    if "true_contrib" in experiment_data and "pred_contrib" in experiment_data:
        viz.plot_contribution_distribution(experiment_data["true_contrib"], experiment_data["pred_contrib"])
    
    # 4. 公平性指标（如有）
    if "fairness_metrics" in experiment_data:
        viz.plot_fairness_metrics(experiment_data["fairness_metrics"])
    
    # 5. 客户端性能分布（如有）
    if "client_perfs" in experiment_data:
        viz.plot_client_performance_dist(experiment_data["client_perfs"])
    
    # 6. 鲁棒性得分（如有）
    if "robustness_scores" in experiment_data:
        viz.plot_robustness_scores(experiment_data["robustness_scores"])
    
    # 7. 效率指标（如有）
    if "efficiency_metrics" in experiment_data:
        viz.plot_efficiency_metrics(experiment_data["efficiency_metrics"])
    
    # 8. 训练曲线（如有）
    if "training_curves" in experiment_data:
        viz.plot_training_curve(experiment_data["training_curves"])
    
    # 9. 综合对比（如有）
    if "comparison_data" in experiment_data:
        viz.plot_experiment_comparison(experiment_data["comparison_data"])
    
    viz._log(f"实验{experiment_name}所有可视化图表已生成")