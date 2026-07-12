from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str = "cora"
    root: str = "packages/gnn/data"


@dataclass(slots=True, frozen=True)
class ModelConfig:
    name: str = "gcn"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OptimizerConfig:
    name: str = "adam"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SchedulerConfig:
    enabled: bool = False
    name: str | None = None
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EarlyStoppingConfig:
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
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
