"""GNN 配置加载器。

委托给 ``core.config`` 的通用加载函数（``load_config_from_yaml``、
``load_config_from_cli``、``apply_overrides``、``set_nested``），
仅保留 GNN 特有的 ``from_dict`` 工厂函数和 ``_OVERRIDE_MAP`` 映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import load_config_from_cli, load_config_from_yaml

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
from gnn.utils.paths import DATA_DIR, OUTPUT_DIR, resolve_config, resolve_path

# CLI arg name to nested dict path mapping (GNN-specific)

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


# Config loading (delegates to core)


def from_yaml(path: str | Path) -> Config:
    """从 YAML 加载配置（路径解析后委托给 core）。"""
    path = resolve_config(path)
    return load_config_from_yaml(path, factory=from_dict)


def from_dict(data: dict[str, Any]) -> Config:
    """从字典构建 Config（GNN 特有工厂函数）。

    对 ``dataset.root`` 和 ``experiment.save_dir`` 做绝对路径解析。
    """
    task_raw = data.get("task", "node_classification")
    task = TaskType(task_raw)

    dataset_data: dict[str, Any] = dict(data.get("dataset", {}))
    model_data: dict[str, Any] = dict(data.get("model", {}))
    optimizer_data: dict[str, Any] = dict(data.get("optimizer", {}))
    scheduler_data: dict[str, Any] = dict(data.get("scheduler", {}))
    train_data: dict[str, Any] = dict(data.get("train", {}))
    runtime_data: dict[str, Any] = dict(data.get("runtime", {}))
    experiment_data: dict[str, Any] = dict(data.get("experiment", {}))

    # 路径解析：确保 root / save_dir 为绝对路径
    if "root" not in dataset_data:
        dataset_data["root"] = str(DATA_DIR)
    dataset_data["root"] = str(resolve_path(dataset_data["root"]))

    if "save_dir" not in experiment_data:
        experiment_data["save_dir"] = str(OUTPUT_DIR)
    experiment_data["save_dir"] = str(resolve_path(experiment_data["save_dir"]))

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


def from_cli(
    yaml_path: str | Path | None,
    *,
    overrides: dict[str, Any],
) -> Config:
    """分层加载配置：默认值 → YAML → CLI 覆盖（委托给 core）。

    优先级: Default > Yaml > CLI
    """
    if yaml_path is not None:
        yaml_path = str(resolve_config(yaml_path))
    return load_config_from_cli(
        yaml_path,
        overrides,
        factory=from_dict,
        override_map=_OVERRIDE_MAP,
        defaults=Config(),
    )


# Internal helpers


def _parse_train(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    early_stopping = data.pop("early_stopping", {})
    return {**data, "early_stopping": EarlyStoppingConfig(**early_stopping)}


# 以下函数已删除，由 core.apply_overrides / core.set_nested 替代
