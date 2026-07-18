from dataclasses import dataclass, field

from core.utils.early_stopping import MONITOR_MODES


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """运行时配置基类"""

    device: str = "auto"
    compile: str | bool = "auto"


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    """实验配置基类"""

    name_prefix: str = "experiment"
    save_dir: str = ""
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass(slots=True, frozen=True)
class EarlyStoppingConfig:
    """早停配置基类"""

    enabled: bool = True
    patience: int = 30
    monitor: str = "val_loss"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience 必须 >= 0")
        if self.monitor not in MONITOR_MODES:
            raise ValueError(
                f"不支持的 early_stopping.monitor: {self.monitor!r}，"
                f"可选: {list(MONITOR_MODES.keys())}"
            )


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """优化器配置基类"""

    name: str = "adam"
    params: dict[str, object] = field(default_factory=dict)
