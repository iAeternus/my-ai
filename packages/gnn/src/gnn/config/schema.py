from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any
from gnn.utils.early_stopping import MONITOR_MODES


class TaskType(str, Enum):
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str = "cora"
    root: str = "packages/gnn/data"


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """模型配置

    Attributes:
        name:
            模型名称 (gcn、gat、gin、sage)

        params:
            - hidden_dim (int): 隐藏层维度
            - num_layers (int): GNN 层数
            - dropout (float): Dropout 概率
            - heads (int): GAT 多头注意力头数
            - aggr (str): GraphSAGE 聚合方式 (mean、sum、max)
            - link_predictor (str): 链接预测器 (dot_product、mlp)
            - norm (str): 归一化层类型 (batch、layer)
            - dropedge (float): DropEdge 边丢弃概率 (0.0~0.2)
    """

    name: str = "gcn"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    """优化器配置

    Attributes:
        name:
            优化器名称 (adam、adamw、sgd)

        params:
            - lr (float): 学习率
            - weight_decay (float): 权重衰减
            - momentum (float): SGD 动量
    """

    name: str = "adam"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SchedulerConfig:
    """学习率调度器配置

    Attributes:
        enabled:
            是否启用学习率调度器

        name:
            调度器名称 (step、cosine、plateau)

        params:
            - step_size (int): StepLR 更新步长
            - gamma (float): 学习率衰减系数
            - T_max (int): CosineAnnealing 最大周期
            - eta_min (float): CosineAnnealing 最小学习率
            - factor (float): ReduceLROnPlateau 衰减系数
            - patience (int): ReduceLROnPlateau 等待轮数
    """

    enabled: bool = False
    name: str | None = None
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EarlyStoppingConfig:
    """早停配置

    Attributes:
        enabled:
            是否启用早停

        patience:
            连续无改善的容忍轮数

        monitor:
            监控指标 (val_loss、val_acc、val_auc、val_ap)
    """

    enabled: bool = True
    patience: int = 30
    monitor: str = "val_loss"


@dataclass(slots=True, frozen=True)
class TrainConfig:
    epochs: int = 100
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    device: str = "auto"
    compile: str | bool = "auto"


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    name_prefix: str = "default"
    save_dir: str = "packages/gnn/outputs"
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass(slots=True, frozen=True)
class Config:
    task: TaskType = TaskType.NODE_CLASSIFICATION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        self._validate()

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典"""
        return asdict(self)

    def to_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> None:
        """保存配置为 JSON 文件

        Args:
            path:
                输出文件路径

            indent:
                JSON 缩进
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def _validate(self) -> None:
        PLANETOID_NAMES = {"cora", "citeseer", "pubmed"}
        if self.dataset.name not in PLANETOID_NAMES:
            return  # 非 Planetoid 不做校验

        if self.task not in (TaskType.NODE_CLASSIFICATION, TaskType.LINK_PREDICTION):
            raise ValueError(
                f"Planetoid 数据集 '{cfg.dataset.name}' 不支持任务 '{cfg.task.value}'。"
                f"仅支持: node_classification, link_prediction"
            )

        if self.train.early_stopping.monitor not in MONITOR_MODES.keys():
            raise ValueError(
                "Unsupported early_stopping.monitor: "
                f"{cfg.train.early_stopping.monitor!r}. "
                f"Available: {', '.join(MONITOR_MODES.keys())}"
            )
