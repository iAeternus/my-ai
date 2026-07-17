"""通用实验目录与产物管理器"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# CJK 字体配置
_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
]


def configure_cjk_font(matplotlib) -> None:
    """检测并设置可用 CJK 字体，避免中文绘图警告。"""
    from matplotlib.font_manager import FontProperties, fontManager

    for name in _CJK_FONT_CANDIDATES:
        try:
            font = FontProperties(family=name)
            path = fontManager.findfont(font, fallback_to_default=False)
            if path:
                matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams[
                    "font.sans-serif"
                ]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue


@dataclass
class PlotSpec:
    """单张图的规格描述

    Attributes:
        title: 图表标题（同时用作输出文件名推导）
        train_key: history 中的训练指标键（如 ``"loss"``）
        val_key: history 中的验证指标键（如 ``"val_loss"``）
        ylabel: Y 轴标签
        output_filename: 输出文件名（为空则从 ``title`` 推导，如 ``"Loss"`` -> ``"loss.png"``）
        multi_keys: 多线图的 key 列表（如 ``["val_hits@1", "val_hits@3", "val_hits@10"]``）
    """

    title: str
    train_key: str = ""
    val_key: str = ""
    ylabel: str = ""
    output_filename: str = ""
    multi_keys: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.output_filename:
            self.output_filename = self.title.lower().replace(" ", "_") + ".png"


class ExperimentManager:
    """通用实验目录和产物管理器

    支持单 seed / 多 seed 两种模式，通过 ``dir_segments`` 泛化路径结构

    单 seed::

        {save_dir}/{seg0}/{seg1}/.../{prefix}_{timestamp}/
        ├── config.json
        ├── history.json
        ├── metrics.json
        ├── logs/
        │   └── train.log
        ├── checkpoints/
        │   ├── best.pt
        │   └── latest.pt
        └── plots/

    多 seed::

        {save_dir}/{seg0}/{seg1}/.../{prefix}_{timestamp}_multi/
        ├── config.json
        ├── summary.json
        ├── logs/                  <- 共享日志
        │   └── train.log
        ├── seed_42/
        │   ├── history.json
        │   ├── checkpoints/
        │   │   └── best.pt
        │   └── plots/
        ├── seed_123/
        │   └── ...
        └── seed_456/
            └── ...

    Usage::

        # GNN
        exp = ExperimentManager("outputs", dir_segments=["cora", "gcn"])
        # KGE
        exp = ExperimentManager("outputs", dir_segments=["fb15k-237", "trans-e"])
    """

    def __init__(
        self,
        save_dir: str | Path,
        name_prefix: str = "exp",
        *,
        dir_segments: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.name_prefix = name_prefix
        self.dir_segments = dir_segments or []
        self.seeds = seeds or [42]
        self._root_dir: Path | None = None

    @property
    def root_dir(self) -> Path:
        """实验根目录（需先调用 ``setup()`` 或 ``setup_multi()``）"""
        if self._root_dir is None:
            raise RuntimeError("请先调用 setup() 或 setup_multi()")
        return self._root_dir

    @property
    def is_multi(self) -> bool:
        """是否为多 seed 模式"""
        return len(self.seeds) > 1

    def setup(self) -> Path:
        """创建单 seed 实验目录并返回其路径"""
        self._root_dir = self._build_root(suffix="")
        self._ensure_subdirs("checkpoints", "plots", "logs")
        return self.root_dir

    def setup_multi(self) -> Path:
        """创建多 seed 实验根目录（仅 logs 共享）并返回路径"""
        self._root_dir = self._build_root(suffix="_multi")
        self._ensure_subdirs("logs")
        return self.root_dir

    def seed_run_dir(self, seed: int) -> Path:
        """创建并返回单个 seed 的产物目录（checkpoints + plots）"""
        path = self.root_dir / f"seed_{seed}"
        _mkdir(path)
        _mkdir(path / "checkpoints")
        _mkdir(path / "plots")
        return path

    def _build_root(self, suffix: str = "") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.name_prefix}_{timestamp}{suffix}"
        return Path(self.save_dir, *self.dir_segments, name)

    def _ensure_subdirs(self, *names: str) -> None:
        for name in names:
            _mkdir(self.root_dir / name)

    def save_config(self, config_dict: dict[str, Any]) -> None:
        """保存实验配置为 JSON

        Args:
            config_dict: 任意可 JSON 序列化的配置字典
        """
        _write_json(self.root_dir / "config.json", config_dict)

    def save_history(
        self,
        history: dict[str, list[float]],
        *,
        run_dir: Path | None = None,
    ) -> None:
        """保存训练历史

        Args:
            history: 训练历史（key -> epoch 值列表）
            run_dir: 指定保存目录，默认 root_dir
        """
        _write_json((run_dir or self.root_dir) / "history.json", history)

    def save_metrics(
        self,
        metrics: dict[str, Any],
        *,
        run_dir: Path | None = None,
    ) -> None:
        """保存实验指标

        Args:
            metrics: 指标字典
            run_dir: 指定保存目录，默认 root_dir
        """
        _write_json((run_dir or self.root_dir) / "metrics.json", metrics)

    def summarize_seeds(self, results: dict[int, float]) -> dict[str, Any]:
        """统计多 seed 实验结果并写入 ``summary.json``

        Args:
            results: ``{seed: metric_value}`` 字典

        Returns:
            summary 字典（含 mean/std/n_seeds）
        """
        values = list(results.values())
        summary: dict[str, Any] = {
            "seeds": results,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
        _write_json(self.root_dir / "summary.json", summary)
        return summary

    def plot_metrics(
        self,
        history: dict[str, list[float]],
        *,
        specs: list[PlotSpec] | None = None,
        run_dir: Path | None = None,
    ) -> None:
        """绘制训练曲线

        Args:
            history: 训练历史（key -> epoch 值列表）
            specs: 绘图规格列表。为 ``None`` 时自动从 history 推断单线图
            run_dir: 指定保存目录，默认 root_dir
        """
        import matplotlib

        matplotlib.use("Agg")
        configure_cjk_font(matplotlib)

        base = run_dir or self.root_dir
        plots_dir = base / "plots"
        _mkdir(plots_dir)

        epochs = list(range(1, len(history.get("loss", [])) + 1))
        if not epochs:
            return

        if specs is None:
            # 自动推断：每条 history key 生成一张单线图
            specs = [
                PlotSpec(
                    title=k.replace("val_", "").replace("_", " ").title(),
                    val_key=k,
                    ylabel=k,
                )
                for k in history
                if k.startswith("val_")
            ]

        for spec in specs:
            save_path = plots_dir / spec.output_filename
            if spec.multi_keys:
                _draw_multi_line_plot(
                    history, epochs, save_path, keys=spec.multi_keys, ylabel=spec.ylabel
                )
            else:
                _draw_line_plot(
                    history,
                    epochs,
                    save_path,
                    train_key=spec.train_key,
                    val_key=spec.val_key,
                    ylabel=spec.ylabel or spec.title,
                )


def _mkdir(path: Path) -> None:
    """创建目录（幂等）"""
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any) -> None:
    """写入 JSON 文件"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _draw_line_plot(
    history: dict[str, list[float]],
    epochs: list[int],
    save_path: Path,
    *,
    train_key: str = "",
    val_key: str = "",
    ylabel: str = "",
    title: str = "",
) -> None:
    """绘制单张 train+val 双子图"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4))
    has_data = False

    if train_key and train_key in history:
        plt.plot(epochs, history[train_key], label=f"train {train_key}")
        has_data = True
    if val_key and val_key in history:
        plt.plot(epochs, history[val_key], label=f"val {val_key}")
        has_data = True

    if has_data:
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _draw_multi_line_plot(
    history: dict[str, list[float]],
    epochs: list[int],
    save_path: Path,
    *,
    keys: list[str],
    ylabel: str = "",
    title: str = "",
) -> None:
    """绘制多条曲线（如多个 K 值的 Hits@K）"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4))
    has_data = False

    for key in keys:
        if key in history:
            vals = history[key]
            n_vals = len(vals)
            if n_vals > 0:
                x = (
                    epochs[:n_vals]
                    if len(epochs) >= n_vals
                    else list(range(1, n_vals + 1))
                )
                plt.plot(x, vals, label=key)
                has_data = True

    if has_data:
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
