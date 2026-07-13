import json
from pathlib import Path
from datetime import datetime
import statistics
from typing import Any

from gnn.config import Config

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
    """实验目录和产物管理器，统一支持单 seed / 多 seed 两种模式。

    单 seed::

        {save_dir}/{dataset}/{model}/{name_prefix}_{timestamp}/
        ├── config.json
        ├── history.json
        ├── metrics.json
        ├── logs/
        │   └── train.log
        ├── checkpoints/
        │   └── best.pt
        └── plots/
            ├── loss.png
            └── accuracy.png

    多 seed::

        {save_dir}/{dataset}/{model}/{name_prefix}_{timestamp}_multi/
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
        │       └── accuracy.png
        ├── seed_123/
        │   └── ...
        └── seed_456/
            └── ...
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._root_dir: Path | None = None

    @property
    def root_dir(self) -> Path:
        if self._root_dir is None:
            raise RuntimeError("Please call setup() or setup_multi() first")
        return self._root_dir

    @property
    def log_dir(self) -> Path:
        return self.root_dir / "logs"

    def setup(self) -> Path:
        """创建单 seed 实验目录"""
        self._root_dir = (
            Path(self._config.experiment.save_dir)
            / self._config.dataset.name
            / self._config.model.name
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
            / self._config.model.name
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
        """保存实验指标（单 seed 模式）"""
        base = run_dir if run_dir is not None else self.root_dir
        path = base / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def summarize_seeds(
        self,
        results: dict[int, float],
    ) -> dict[str, Any]:
        """统计多 seed 实验结果，写入 summary.json"""
        values = list(results.values())
        summary = {
            "seeds": results,
            "mean": statistics.mean(values),
            "std": (statistics.stdev(values) if len(values) > 1 else 0.0),
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
        """绘制训练曲线，自适应节点分类 / 链接预测。

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

        # Performance metric
        if "val_auc" in history:
            _draw_line_plot(
                history,
                epochs,
                plots_dir / "accuracy.png",
                train_key="auc",
                val_key="val_auc",
                ylabel="AUC",
            )
        elif "val_acc" in history:
            _draw_line_plot(
                history,
                epochs,
                plots_dir / "accuracy.png",
                train_key="acc",
                val_key="val_acc",
                ylabel="Accuracy",
            )
        # else: 无可用性能指标，跳过


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
