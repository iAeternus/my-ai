from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    LINK_PREDICTION = "link_prediction"
    TRIPLE_CLASSIFICATION = "triple_classification"
    RELATION_PREDICTION = "relation_prediction"


class LossType(str, Enum):
    MARGIN_RANKING = "margin_ranking"
    BCE = "bce"
    CROSS_ENTROPY = "cross_entropy"


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """数据集配置

    Attributes:
        name: 数据集名称，对应 KGDatasetRegistry 中的注册名
        root: 数据集根目录
        batch_size: 训练 batch 大小
        num_negative_samples: 每个正例的负采样数
        num_workers: DataLoader 工作进程数
    """

    name: str = "fb15k-237"
    root: str = "packages/kge/data"
    batch_size: int = 1024
    num_negative_samples: int = 128
    num_workers: int = 0


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """模型配置

    Attributes:
        encoder_name: KGE encoder 名 (trans-e, rotate, compl-ex, ...)
        head_name: 预测头名 (link_prediction, relation_prediction, ...)
        params: encoder/head 专属参数字典
            常用键:
            - embedding_dim (int): 嵌入维度，默认 100
            - gamma (float): margin 值，平移系列默认 12.0
            - p_norm (int): L1/L2 范数，默认 1
            - epsilon (float): 数值稳定常数，RotatE 默认 2.0
            - kernel_size (int): ConvE 卷积核尺寸，默认 3
            - conv_out_channels (int): ConvE 输出通道，默认 32
            - hidden_dropout (float): 隐层 dropout，默认 0.3
            - input_dropout (float): 输入 dropout，ConvE 默认 0.2
            - feature_dropout (float): 特征图 dropout，ConvE 默认 0.2
            - hidden_dim (int): Head 隐层维度，默认 256
            - pretrained_encoder (str): 预训练 encoder 路径
    """

    encoder_name: str = "trans-e"
    head_name: str = "link_prediction"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """优化器配置"""

    name: str = "adam"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EarlyStoppingConfig:
    """早停配置"""

    enabled: bool = True
    patience: int = 30
    monitor: str = "val_mrr"
    min_delta: float = 0.0


@dataclass(slots=True, frozen=True)
class TrainConfig:
    """训练配置"""

    epochs: int = 500
    lr: float = 0.001
    weight_decay: float = 0.0
    loss_type: LossType = LossType.MARGIN_RANKING
    margin: float = 1.0
    adversarial_temperature: float = 0.0
    label_smoothing: float = 0.0
    regularization_weight: float = 0.0
    eval_interval: int = 10
    eval_ks: list[int] = field(default_factory=lambda: [1, 3, 10])
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """运行时配置"""

    device: str = "auto"
    compile: str | bool = "auto"


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    """实验配置"""

    name_prefix: str = "kge"
    save_dir: str = "packages/kge/outputs"
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass(slots=True, frozen=True)
class Config:
    """根配置"""

    task: TaskType = TaskType.LINK_PREDICTION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """校验配置合法性"""
        from kge.utils.early_stopping import MONITOR_MODES

        monitor = self.train.early_stopping.monitor
        if monitor not in MONITOR_MODES:
            raise ValueError(
                f"不支持的 early_stopping.monitor: {monitor!r}，"
                f"可选: {list(MONITOR_MODES.keys())}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
