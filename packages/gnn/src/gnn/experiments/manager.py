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


class ExperimentManager:
    """实验目录和实验产物管理器

    普通实验:

    ::

        {save_dir}/{dataset}/{model}/{name_prefix}_{timestamp}/
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
            └── accuracy.png

    多seed实验:

    ::

        {name_prefix}_{timestamp}_multi/
        ├── config.json
        ├── summary.json
        ├── seed_42/
        ├── seed_123/

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
    def checkpoint_dir(self) -> Path:
        return self.root_dir / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.root_dir / "plots"

    @property
    def log_dir(self) -> Path:
        return self.root_dir / "logs"

    def setup(self) -> Path:
        """创建普通实验目录"""
        self._root_dir = (
            Path(self._config.experiment.save_dir)
            / self._config.dataset.name
            / self._config.model.name
            / self._generate_name()
        )

        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._create_dirs()
        self.save_config()
        return self.root_dir

    def setup_multi(self) -> Path:
        """创建多seed实验目录"""
        self._root_dir = (
            Path(self._config.experiment.save_dir)
            / self._config.dataset.name
            / self._config.model.name
            / f"{self._generate_name()}_multi"
        )

        self._root_dir.mkdir(parents=True, exist_ok=True)
        self.save_config()
        return self.root_dir

    def _create_dirs(self) -> None:
        for dir in [self.checkpoint_dir, self.plots_dir, self.log_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def _generate_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self._config.experiment.name_prefix
        return f"{prefix}_{timestamp}"

    def seed_run_dir(self, seed: int) -> Path:
        """创建某个seed运行目录"""
        path = self.root_dir / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_config(self) -> None:
        """保存实验配置"""
        self._config.to_json(self.root_dir / "config.json")

    def save_history(self, history: dict[str, list[float]]) -> None:
        """保存训练历史"""
        path = self.root_dir / "history.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """保存实验指标"""
        path = self.root_dir / "metrics.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def summarize_seeds(
        self,
        results: dict[int, float],
    ) -> dict[str, Any]:
        """统计多seed实验结果"""
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

    def plot_metrics(self, history: dict[str, list[float]]) -> None:
        """绘制训练曲线"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _configure_cjk_font(matplotlib)

        epochs = range(1, len(history.get("loss", [])) + 1)

        # Loss
        fig = plt.figure(figsize=(6, 4))

        if "loss" in history:
            plt.plot(epochs, history["loss"], label="train loss")

        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="val loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        fig.savefig(self.plots_dir / "loss.png", dpi=150)
        plt.close(fig)

        # Accuracy
        fig = plt.figure(figsize=(6, 4))

        if "acc" in history:
            plt.plot(epochs, history["acc"], label="train acc")

        if "val_acc" in history:
            plt.plot(epochs, history["val_acc"], label="val acc")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        fig.savefig(self.plots_dir / "accuracy.png", dpi=150)
        plt.close(fig)
