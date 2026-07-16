"""ExperimentManager — 实验目录、产物和绘图管理器"""

from __future__ import annotations
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from kge.config.schema import Config

# 常见 CJK 字体优先级列表（跨平台）
_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "AR PL UMing CN",
]


def _configure_cjk_font(matplotlib) -> None:
    """检测并设置可用 CJK 字体，避免中文绘图警告"""
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


def _mkdir(path: Path) -> None:
    """创建目录（幂等）"""
    path.mkdir(parents=True, exist_ok=True)


class ExperimentManager:
    """实验目录和产物管理器，统一支持单 seed / 多 seed 两种模式

    单 seed::

        {save_dir}/{dataset}/{encoder}/{prefix}_{timestamp}/
        ├── config.json
        ├── history.json
        ├── metrics.json
        ├── logs/
        │   └── train.log
        ├── checkpoints/
        │   ├── best.pt
        │   └── latest.pt
        └── plots/
            ├── loss.png
            ├── mrr.png
            └── hits_at_k.png

    多 seed::

        {save_dir}/{dataset}/{encoder}/{prefix}_{timestamp}_multi/
        ├── config.json
        ├── summary.json
        ├── logs/                  <- 共享日志
        │   └── train.log
        ├── seed_42/
        │   ├── history.json
        │   ├── checkpoints/
        │   │   └── best.pt
        │   └── plots/
        │       ├── loss.png
        │       └── mrr.png
        ├── seed_123/
        │   └── ...
        └── seed_456/
            └── ...
    """

    def __init__(self, cfg: Config) -> None:
        self._config = cfg
        self._root_dir: Path | None = None

    @property
    def root_dir(self) -> Path:
        if self._root_dir is None:
            raise RuntimeError("请先调用 setup() 或 setup_multi()")
        return self._root_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def plot_dir(self) -> Path:
        return self.root_dir / "plots"

    @property
    def is_multi(self) -> bool:
        return len(self._config.experiment.seeds) > 1

    def setup(self) -> Path:
        """创建单 seed 实验目录"""
        self._root_dir = (
            Path(self._config.experiment.save_dir)
            / self._config.dataset.name
            / self._config.model.encoder_name
            / self._generate_name()
        )
        _mkdir(self._root_dir)
        for sub in ["checkpoints", "plots", "logs"]:
            _mkdir(self._root_dir / sub)
        self.save_config()
        return self.root_dir

    def setup_multi(self) -> Path:
        """创建多 seed 实验目录（仅 logs 在 root 级别共享）"""
        self._root_dir = (
            Path(self._config.experiment.save_dir)
            / self._config.dataset.name
            / self._config.model.encoder_name
            / f"{self._generate_name()}_multi"
        )
        _mkdir(self._root_dir)
        _mkdir(self._root_dir / "logs")  # 共享日志
        self.save_config()
        return self.root_dir

    def seed_run_dir(self, seed: int) -> Path:
        """创建单个 seed 的产物目录（checkpoints + plots）"""
        path = self.root_dir / f"seed_{seed}"
        _mkdir(path)
        _mkdir(path / "checkpoints")
        _mkdir(path / "plots")
        return path

    def _generate_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self._config.experiment.name_prefix
        return f"{prefix}_{timestamp}"

    def save_config(self) -> None:
        """保存实验配置"""
        self._config.to_json(self.root_dir / "config.json")

    def save_history(
        self,
        history: dict[str, list[float]],
        *,
        run_dir: Path | None = None,
    ) -> None:
        """保存训练历史

        Args:
            history: 训练历史记录
            run_dir: 指定保存目录，默认 root_dir
        """
        base = run_dir if run_dir is not None else self.root_dir
        path = base / "history.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def save_metrics(
        self,
        metrics: dict[str, Any],
        *,
        run_dir: Path | None = None,
    ) -> None:
        """保存实验指标"""
        base = run_dir if run_dir is not None else self.root_dir
        path = base / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def summarize_seeds(self, results: dict[int, float]) -> dict[str, Any]:
        """统计多 seed 实验结果，写入 summary.json"""
        values = list(results.values())
        summary: dict[str, Any] = {
            "seeds": results,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        }
        path = self.root_dir / "summary.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def plot_metrics(
        self,
        history: dict[str, list[float]],
        *,
        run_dir: Path | None = None,
    ) -> None:
        """绘制训练曲线 (loss、MRR、Hits@K)

        Args:
            history: 训练历史记录
            run_dir: 指定保存目录，默认 root_dir
        """
        import matplotlib

        matplotlib.use("Agg")
        _configure_cjk_font(matplotlib)

        base = run_dir if run_dir is not None else self.root_dir
        plots_dir = base / "plots"
        _mkdir(plots_dir)

        epochs = list(range(1, len(history.get("loss", [])) + 1))
        if not epochs:
            return

        # Loss
        _draw_line_plot(
            history,
            epochs,
            plots_dir / "loss.png",
            train_key="loss",
            val_key="val_loss",
            ylabel="Loss",
        )

        # MRR
        mrr_keys = [k for k in history if "mrr" in k.lower()]
        if mrr_keys:
            _draw_multi_line_plot(
                history,
                epochs,
                plots_dir / "mrr.png",
                keys=mrr_keys,
                ylabel="MRR",
            )

        # Hits@K
        hits_keys = [k for k in history if "hits" in k.lower()]
        if hits_keys:
            _draw_multi_line_plot(
                history,
                epochs,
                plots_dir / "hits_at_k.png",
                keys=hits_keys,
                ylabel="Hits@K",
            )


def _draw_line_plot(
    history: dict[str, list[float]],
    epochs: list[int],
    save_path: Path,
    *,
    train_key: str,
    val_key: str,
    ylabel: str,
) -> None:
    """绘制单张双子图（train + val）"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 4))
    has_data = False

    if train_key in history:
        plt.plot(epochs, history[train_key], label=f"train {train_key}")
        has_data = True
    if val_key in history:
        plt.plot(epochs, history[val_key], label=f"val {val_key}")
        has_data = True

    if has_data:
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
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
    ylabel: str,
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
                plt.plot(
                    epochs[:n_vals]
                    if len(epochs) >= n_vals
                    else list(range(1, n_vals + 1)),
                    vals,
                    label=key,
                )
                has_data = True

    if has_data:
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
