from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.config import (
    BaseEarlyStoppingConfig,
    BaseExperimentConfig,
    BaseOptimizerConfig,
    BaseRuntimeConfig,
    SerializableConfig,
    validate_monitor,
)
from core.utils import MONITOR_MODES


class TaskType(str, Enum):
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str = "cora"
    root: str = "data"  # 相对于 PACKAGE_ROOT，由 from_dict() 做绝对路径解析


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
class OptimizerConfig(BaseOptimizerConfig):
    """GNN 优化器配置。继承 core 的 ``name`` + ``params`` 默认值。

    params 常用键: lr, weight_decay, momentum
    """


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
class EarlyStoppingConfig(BaseEarlyStoppingConfig):
    """GNN 早停配置。继承 core 的 ``enabled`` / ``patience`` / ``monitor`` / ``min_delta``。

    monitor 可选: val_loss、val_acc、val_auc、val_ap
    """


@dataclass(slots=True, frozen=True)
class TrainConfig:
    epochs: int = 100
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(slots=True, frozen=True)
class RuntimeConfig(BaseRuntimeConfig):
    """GNN 运行时配置。继承 core 的 ``device`` + ``compile`` 默认值。"""


@dataclass(slots=True, frozen=True)
class ExperimentConfig(BaseExperimentConfig):
    """GNN 实验配置。继承 core 的 ``name_prefix`` / ``save_dir`` / ``seeds``。

    包特定默认值（如 ``name_prefix="default"``）在 ``from_dict()`` 中处理。
    """


@dataclass(slots=True, frozen=True)
class Config(SerializableConfig):
    """GNN 根配置。继承 ``SerializableConfig`` 获得 ``to_dict()`` / ``to_json()``。"""

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

    def _validate(self) -> None:
        PLANETOID_NAMES = {"cora", "citeseer", "pubmed"}
        if self.dataset.name not in PLANETOID_NAMES:
            return  # 非 Planetoid 不做校验

        if self.task not in (TaskType.NODE_CLASSIFICATION, TaskType.LINK_PREDICTION):
            raise ValueError(
                f"Planetoid 数据集 '{self.dataset.name}' 不支持任务 '{self.task.value}'。"
                f"仅支持: node_classification, link_prediction"
            )

        # 委托给 core 统一校验
        validate_monitor(
            self.train.early_stopping.monitor, monitor_modes=MONITOR_MODES
        )
