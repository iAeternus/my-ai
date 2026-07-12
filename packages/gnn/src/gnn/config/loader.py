from dataclasses import asdict
from pathlib import Path
from typing import Any
import yaml

from gnn.config.schema import (
    Config,
    DatasetConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainConfig,
)


def from_yaml(path: str | Path) -> Config:
    """从yaml加载配置
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return from_dict(data)


def from_dict(data: dict[str, Any]) -> Config:
    """从dict加载配置
    """
    dataset_data = data.get("dataset", {})
    model_data = data.get("model", {})
    optimizer_data = data.get("optimizer", {})
    scheduler_data = data.get("scheduler", {})
    train_data = data.get("train", {})
    runtime_data = data.get("runtime", {})
    experiment_data = data.get("experiment", {})

    return Config(
        dataset=DatasetConfig(**dataset_data),
        model=ModelConfig(**model_data),
        optimizer=OptimizerConfig(**optimizer_data),
        scheduler=SchedulerConfig(**scheduler_data),
        train=TrainConfig(**_parse_train(train_data)),
        runtime=RuntimeConfig(**runtime_data),
        experiment=ExperimentConfig(**experiment_data),
    )


def _parse_train(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    early_stopping = data.pop("early_stopping", {})
    return {**data, "early_stopping": EarlyStoppingConfig(**early_stopping)}


def from_cli(
    yaml_path: str | Path | None,
    *,
    overrides: dict[str, Any],
) -> Config:
    """从CLI加载配置
    优先级: Default > Yaml > CLI
    """
    config = Config()
    if yaml_path:
        config = from_yaml(yaml_path)

    raw = asdict(config)
    apply_cli_overrides(raw, overrides)
    return from_dict(raw)


def apply_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None: ...
