from dataclasses import asdict
from pathlib import Path
from typing import Any
from gnn.utils.early_stopping import MONITOR_MODES
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
    TaskType,
    TrainConfig,
)


def from_yaml(path: str | Path) -> Config:
    """从yaml加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return from_dict(data)


def from_dict(data: dict[str, Any]) -> Config:
    """从dict加载配置"""
    task_raw = data.get("task", "node_classification")
    task = TaskType(task_raw)

    dataset_data = data.get("dataset", {})
    model_data = data.get("model", {})
    optimizer_data = data.get("optimizer", {})
    scheduler_data = data.get("scheduler", {})
    train_data = data.get("train", {})
    runtime_data = data.get("runtime", {})
    experiment_data = data.get("experiment", {})

    return Config(
        task=task,
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


def apply_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    """CLI 参数覆盖 YAML 配置（原地修改 raw dict）

    仅覆盖非 None 的项
    """

    # 将 CLI 参数名映射为嵌套字典路径（逐层 key 列表）
    _OVERRIDE_MAP: dict[str, list[str]] = {
        "task": ["task"],
        "dataset": ["dataset", "name"],
        "root": ["dataset", "root"],
        "model": ["model", "name"],
        "hidden_dim": ["model", "params", "hidden_dim"],
        "dropout": ["model", "params", "dropout"],
        "num_layers": ["model", "params", "num_layers"],
        "lr": ["optimizer", "params", "lr"],
        "weight_decay": ["optimizer", "params", "weight_decay"],
        "epochs": ["train", "epochs"],
        "patience": ["train", "early_stopping", "patience"],
        "device": ["runtime", "device"],
        "compile": ["runtime", "compile"],
        "seeds": ["experiment", "seeds"],
    }

    for cli_key, value in overrides.items():
        if value is None:
            continue
        path = _OVERRIDE_MAP.get(cli_key)
        if path is None:
            continue
        _set_nested(config, path, value)


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """按 key 路径设置嵌套字典值，中间层不存在则自动创建。

    Example:
        _set_nested(d, ["model", "params", "hidden_dim"], 128)
        # => d["model"]["params"]["hidden_dim"] = 128
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
